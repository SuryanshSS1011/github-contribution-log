# Contribution 5: Exempt `sys.version` family comparisons from PLR2004

**Contribution Number:** 5
**Student:** Suryansh Sijwali
**Issue:** https://github.com/astral-sh/ruff/issues/25588
**Pull Request:** https://github.com/astral-sh/ruff/pull/25743
**Status:** **Merged** on 2026-06-10 by ntBre after one review round.

---

## Why I Chose This Issue

ruff is the Astral team's Rust-based Python linter. Same engineering
culture as uv (fast review, strict CI, clear scope guidance from
maintainers), but a different subsystem so it diversifies my portfolio
away from package-manager work.

The issue itself is small and self-contained: PLR2004 (magic-value
comparison) currently fires on patterns like
`sys.implementation.version[0] >= 3`. The literal `3` there is not an
unnamed constant — it is a Python major version tag — and asking users
to define `PY3 = 3` aliases adds zero readability.

I picked it after live-verifying several other candidates and finding
them taken or blocked:
- IREE #24179 (ONNX DepthToSpace CRD) — root cause was upstream in
  torch-mlir; PR #4544 already in flight.
- Enzyme-JAX #2141 — was being closed by my own already-submitted
  PR #2524 plus Pangoraw's stalled #2158.
- Enzyme-JAX #1957 — avik-pal (the issue author) had a draft PR #2439
  in flight himself.
- setuptools #5196 — PR #5202 was already open with the exact fix the
  maintainer had suggested.

ruff #25588 was the first remaining clean candidate: open, unassigned,
no in-flight PR, and the maintainer (MichaReiser) had already blessed
the direction in the issue thread.

---

## Understanding the Issue

### Problem Description

PLR2004 (`magic-value-comparison`) flags literal values used in
comparisons. It has a small allow-list (`0`, `1`, `""`, `"__main__"`,
and the boolean/None family) plus a user-configurable
`lint.pylint.allow-magic-value-types` setting that can broaden the
allow-list to whole literal types.

What it doesn't have is a way to recognize that a literal is being
compared against a *version-derived* operand. The reporter
(jonathandung) hit this with:

```python
if sys.implementation.version[0] >= 3:
    ...
```

This is a perfectly conventional Python-major-version guard, but ruff
flags the `3`.

### Maintainer Discussion

ntBre pushed back: "I'm not sure we should start adding exceptions like
this to the rule. I think it's fine to use a `noqa` here or add 3 to
the configuration option once #18961 lands."

MichaReiser overruled: "Ruff has the same issue with `sys.version`. I'd
be fine adding an exception for `sys.version` and maybe
`sys.implementation.version`, if it's not too hard."

That set the scope precisely: `sys.version` and
`sys.implementation.version`, no broader. In particular, `sys.version_info`
was deliberately excluded — Micha did not mention it, and the existing
ecosystem uses `sys.version_info` for version checks heavily enough that
broadening the exemption could surprise users with reduced linting
coverage.

### Current Behavior

On ruff 0.15.16 (current master):

```python
import sys

if sys.implementation.version[0] >= 3:  # PLR2004 fires
    ...
if sys.version >= "3.10":               # PLR2004 fires
    ...
```

### Affected Component

`crates/ruff_linter/src/rules/pylint/rules/magic_value_comparison.rs`,
specifically the `magic_value_comparison` function which is called from
the AST checker for `Expr::Compare` nodes.

---

## Reproduction Process

### Steps to Reproduce the Bug

Save the following as `repro.py`:

```python
import sys

if sys.implementation.version[0] >= 3:
    print("py3")
```

Run:

```
ruff check --select PLR2004 repro.py
```

Output:

```
repro.py:3:30: PLR2004 Magic value used in comparison, consider replacing `3` with a constant variable
```

### Why This Matters

`sys.implementation.version[0]` is the standard way to discriminate
between CPython major versions when `sys.version_info` is unavailable
or when comparing against a non-CPython implementation's version tuple.
Forcing users to either `noqa` every such guard or invent a `PY3 = 3`
constant for `3` adds noise without adding clarity.

---

## Solution Approach

### Analysis

I first looked at how other ruff rules already detect `sys.<name>`
operands. Two patterns showed up:

1. `crates/ruff_linter/src/rules/flake8_2020/helpers.rs:5` — `is_sys`
   helper using `semantic.resolve_qualified_name` against
   `["sys", <target>]`.
2. `crates/ruff_linter/src/rules/flake8_pyi/rules/unrecognized_version_info.rs:144`
   — `resolve_qualified_name(map_subscript(left))` to recognize a
   `sys.version_info` operand whether or not it was being subscripted.

The `map_subscript` helper (`ruff_python_ast::helpers::map_subscript`)
unwraps one level of `x[…]` to `x`, leaving non-subscript expressions
alone. Combined with `resolve_qualified_name`, this gives me a way to
recognize `sys.version`, `sys.implementation.version`, and any
subscript or attribute access on either.

### Proposed Solution

Add one helper, `is_sys_version_comparand`, that returns `true` if the
expression resolves (after one `map_subscript` unwrap) to a qualified
name beginning with `["sys", "version"]` or
`["sys", "implementation", "version"]`. Using `segments.get(..N)`
prefix-matching means `sys.version.major`, `sys.implementation.version[0]`,
and the bare forms all match correctly.

Then in `magic_value_comparison`, for each literal comparand, check
whether either adjacent operand in the comparison is a sys-version
comparand. If yes, skip the diagnostic for that literal. This works
for both simple comparisons (`x == 3`) and chained ones
(`0 < x < 3`).

### Why Skip `sys.version_info`

MichaReiser only blessed `sys.version` and `sys.implementation.version`.
There is a separate in-flight PR (#18961 by DaniBodor) adding a
user-configurable `pylint.allow_magic_values` setting that would let
users opt in to a custom allow-list. My change is complementary: it
gives the two clearly-meaningful version operands an automatic exemption,
while leaving the broader configurability to that other PR.

---

## Implementation

### Code Changes

**File:** `crates/ruff_linter/src/rules/pylint/rules/magic_value_comparison.rs`

1. **Added imports** for `map_subscript` and `SemanticModel`.

2. **Updated the rule docstring** to document the new exemption with
   `sys.implementation.version[0] >= 3` as the motivating example.

3. **Added the helper:**

   ```rust
   fn is_sys_version_comparand(expr: &Expr, semantic: &SemanticModel) -> bool {
       let Some(qualified_name) = semantic.resolve_qualified_name(map_subscript(expr)) else {
           return false;
       };
       let segments = qualified_name.segments();
       matches!(segments.get(..2), Some(["sys", "version"]))
           || matches!(
               segments.get(..3),
               Some(["sys", "implementation", "version"])
           )
   }
   ```

4. **Rewrote the diagnostic loop** to collect operands once into a
   `Vec<&Expr>` and, for each literal, check its previous and next
   neighbors via `checked_sub(1)` and `index + 1` before reporting.
   The early-return for two-literals-in-a-comparison
   (`R0133: comparison-of-constants`) is preserved.

### Fixture Changes

**File:** `crates/ruff_linter/resources/test/fixtures/pylint/magic_value_comparison.py`

- Moved `import sys` to the top of the file (followed standard Python
  convention).
- Added 8 cases at the bottom: one `sys.version_info[0] >= 3` case
  (intentionally still flagged to encode the deliberate scope omission)
  and seven `# correct` cases covering the newly-exempt shapes
  (`sys.implementation.version[0]`, `sys.implementation.version.major`,
  `sys.version[0]`, `sys.version`, both RHS and LHS literal positions,
  and a chained comparison).

### Snapshot Changes

Both `PLR2004_magic_value_comparison.py.snap` and
`allow_magic_value_types.snap` were regenerated by `cargo insta`. The
main snapshot diff is exactly one new violation added (the
`sys.version_info[0] >= 3` case), which confirms that all seven
exempt cases pass silently. The `allow_magic_value_types` snapshot
diff is line-number shifts only (because of the `import sys` move).

---

## Testing Strategy

### Tests Run

- `cargo test --package ruff_linter --lib pylint` — **184 passed, 0 failed.**
- `cargo fmt --check` — clean.
- `cargo clippy --package ruff_linter -- -D warnings` — clean.
- `cargo dev generate-all` — reports no schema regeneration needed
  (the change doesn't add settings).

### Coverage Reasoning

- The `# [magic-value-comparison]` annotation on the `sys.version_info`
  case in the fixture is the regression test for the deliberate
  scope omission. If someone later widens `is_sys_version_comparand`
  to include `sys.version_info` without revisiting the design
  decision, this case would silently start passing and the snapshot
  would surface it.
- The chained-comparison case (`0 < sys.implementation.version[0] < 4`)
  exercises the `index ± 1` neighbor-check logic for the middle
  position.
- Both RHS (`sys.version[0] >= "3"`) and LHS (`3 <= sys.implementation.version[0]`)
  literal positions are covered.

---

## Pull Request

PR #25743 was opened on 2026-06-08. ntBre (who had pushed back on the
issue) was both auto-assigned and self-requested as reviewer.

**AI Policy incident:** the initial PR body I posted was drafted with
AI help, then posted without enough rewriting in my own voice. ntBre
flagged it: "Could you confirm that you've read and adhered to our AI
Policy? The summary here gives me the impression that it was written
by an LLM." I acknowledged honestly (the code is mine, but the body
prose wasn't), read the AI Policy, and ntBre asked me to edit the
summary and mark the PR ready for review. I rewrote the summary in
my own words and re-submitted.

This is the right operating discipline going forward: AI for the
investigation and code-level decisions is fine and useful, but
maintainer-facing prose (PR body, issue replies, review responses)
needs to be in my own voice.

---

## Learnings & Reflections

### What went well

- The verification discipline carried over from prior sessions paid
  off. Live-checking PR-collision and hardware-constraint reduced
  the candidate set from "10+ vetted by an agent hunt" to "4 actually
  clean" quickly.
- Finding the existing `is_sys` / `map_subscript` patterns first
  saved me from inventing a worse solution. The pattern was already
  blessed by the codebase; I just applied it to a new rule.
- The fixture's `# [magic-value-comparison]` annotation convention
  let me encode the deliberate `sys.version_info` scope omission
  directly in the test file — a future reader doesn't have to wonder
  why that case still fires.

### What I'd do differently

- The PR body should have been written in my own voice from the
  start. AI-drafted prose is a real tell to maintainers, especially
  at Astral; ntBre catching it immediately was a clear signal of
  what the bar is. I will type all maintainer-facing prose myself
  on future PRs.
- The scope-justification ("why not `sys.version_info`") should
  have been in the PR body explicitly. I put it in my own head and
  the AI-flavored body had a parenthetical aside about it that got
  diluted; the rewrite is clearer about the deliberate scope.

---

## Review and merge

ntBre reviewed on 2026-06-09 with five substantive points:

1. Replace `segments.get(..N)` with slice-with-rest patterns
   (`["sys", "version", ..]`) in `is_sys_version_comparand`.
2. Migrate the new tests to mdtest format
   (`crates/ruff_linter/resources/mdtest/pylint/`).
3. Include `sys.version_info` in the exemption: "I also think we
   should just handle `sys.version_info`. Is there a reason not
   to?" This expanded the scope I had originally taken from
   MichaReiser's wording.
4. Trim the docstring paragraph.
5. Drop the `Vec<&Expr>::collect()` and index arithmetic; use a
   `peekable()` iterator with a `previous` slot instead.

Addressed all five in a single force-pushed revision. ntBre
approved and merged on 2026-06-10. The merged diff is +93 / -11
lines across `magic_value_comparison.rs` and the new mdtest.

The scope-expansion to `sys.version_info` is the only design call
the review surfaced. My original strict reading of Micha's words
("`sys.version` and *maybe* `sys.implementation.version`") was
defensible but turned out to be over-conservative — `sys.version_info`
is the canonical Python version-check site and excluding it would
have left users with the same friction the issue was filed about,
just on a different attribute. ntBre's question made that obvious
and I should have caught it before posting.

---

## Resources Used

- ruff issue thread: https://github.com/astral-sh/ruff/issues/25588
- ruff PR: https://github.com/astral-sh/ruff/pull/25743
- Astral AI Policy:
  https://github.com/astral-sh/.github/blob/main/AI_POLICY.md
- Related in-flight PR: https://github.com/astral-sh/ruff/pull/18961
  (DaniBodor's user-configurable `pylint.allow_magic_values` setting)
- Existing pattern reference:
  `crates/ruff_linter/src/rules/flake8_2020/helpers.rs`
  `crates/ruff_linter/src/rules/flake8_pyi/rules/unrecognized_version_info.rs`
