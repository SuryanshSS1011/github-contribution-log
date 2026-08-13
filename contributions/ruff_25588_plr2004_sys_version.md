# Exempt `sys.version` family comparisons from PLR2004

**Project:** [astral-sh/ruff](https://github.com/astral-sh/ruff)
**Issue:** [#25588](https://github.com/astral-sh/ruff/issues/25588) · **Pull request:** [#25743](https://github.com/astral-sh/ruff/pull/25743) · **Branch:** `plr2004-sys-version-exemption`
**Status:** **Merged** 2026-06-10 (merge commit `7fbef674f`). Issue closed.

---

## The issue

### Why I picked it

ruff is Astral's Rust-based Python linter. It has the same engineering culture as uv, with fast review, strict CI and clear scope guidance from maintainers, but it is a different subsystem, which diversifies my work away from package-manager internals. My learning goal was to understand how a linter decides that a literal is meaningful rather than magic, which is a semantic-model question rather than a syntax one.

The issue itself is small and self-contained. PLR2004 (magic-value comparison) fires on patterns like `sys.implementation.version[0] >= 3`, where the literal `3` is not an unnamed constant but a Python major-version tag, and asking users to define a `PY3 = 3` alias adds zero readability.

I picked it after live-verifying several other candidates and finding them taken or blocked:

- IREE #24179 (ONNX DepthToSpace CRD), where the root cause was upstream in torch-mlir and PR #4544 was already in flight.
- Enzyme-JAX #2141, which was being closed by my own already-submitted PR #2524 plus Pangoraw's stalled #2158.
- Enzyme-JAX #1957, where `avik-pal` (the issue author) had a draft PR #2439 in flight himself.
- setuptools #5196, where PR #5202 was already open with the exact fix the maintainer had suggested.

ruff #25588 was the first remaining clean candidate, being open, unassigned, free of an in-flight PR, and with a maintainer who had already blessed the direction in-thread.

### What was wrong

PLR2004 flags literal values in comparisons, but its allow-list has no way to recognize that the other operand is version-derived, so a conventional Python-version guard like `sys.implementation.version[0] >= 3` gets flagged as a magic value. Users are then pushed toward a `noqa` or an invented `PY3 = 3` constant, and both of those make the code worse rather than clearer. It matters because version guards are extremely common in real Python, so the false positive is high-frequency and pushes people to disable a useful rule. I chose it because two maintainers had already argued the scope out in the thread, which meant I could work on the semantics instead of negotiating the design.

### Before I started

Open, unassigned, with no in-flight PR touching `magic_value_comparison.rs`. I did not post a claim comment, because the maintainers had already settled the direction in-thread and ruff's norm for a single-rule change of this size is to open the PR. In hindsight a one-line "taking this" would have cost nothing, which I cover under What I'd do differently.

### Prior discussion

The scope was decided in the issue thread before I arrived, and reading it carefully is what set my boundaries.

`ntBre` pushed back on [2026-06-03](https://github.com/astral-sh/ruff/issues/25588):

> I'm not sure we should start adding exceptions like this to the rule. I think it's fine to use a `noqa` here or add 3 to the configuration option once #18961 lands.

`MichaReiser` overruled:

> Ruff has the same issue with `sys.version`. I'd be fine adding an exception for `sys.version` and maybe `sys.implementation.version`, if it's not too hard.

That set the scope precisely as `sys.version` and `sys.implementation.version` and nothing broader. I read `sys.version_info` as deliberately excluded, since Micha did not name it and the ecosystem uses it heavily enough that a silent exemption could reduce linting coverage in a surprising way. There was also related work in [PR #18961](https://github.com/astral-sh/ruff/pull/18961) by DaniBodor, adding a user-configurable `pylint.allow_magic_values` setting, which is complementary rather than overlapping.

### What counts as done

1. `sys.version` and `sys.implementation.version` comparands exempt their literal operand from PLR2004.
2. Subscripts and attribute access on either, such as `sys.version[0]` and `sys.implementation.version.major`, are recognized.
3. Both operand positions work, covering `x >= 3` and `3 <= x`, including chained comparisons.
4. There is no behavior change for any other literal comparison.
5. Snapshot or mdtest coverage exists, and `cargo test`, `cargo fmt` and `cargo clippy` are clean.

---

## Diagnosis and plan

### Environment setup

I used ruff's `CONTRIBUTING.md` for the test and snapshot workflow, and read the CI workflow to match the exact lint invocations.

- The Rust toolchain was already configured from the uv work, including Homebrew rustup's proxy model and the toolchain bin dir on `PATH`.
- **Challenge (snapshot workflow):** rule changes are verified through `cargo insta` snapshots rather than hand-written assertions, so a change surfaces as a snapshot diff that you have to read as the real test result. Getting comfortable with reading `.snap` diffs as the primary signal was the main ramp-up cost.
- **Challenge (generated files):** ruff regenerates schemas and docs via `cargo dev generate-all`, and CI fails if generated output drifts. I ran it to confirm nothing needed regeneration, since this change adds no settings.
- The test loop is fast, because `cargo test --package ruff_linter --lib pylint` runs the whole pylint rule set in seconds.

### Steps to reproduce

1. Save this as `repro.py`:
   ```python
   import sys

   if sys.implementation.version[0] >= 3:
       print("py3")
   ```
2. Build ruff from `main`, which was 0.15.16 at the time.
3. Run `ruff check --select PLR2004 repro.py`.

### Expected vs. actual

**Actual:**

```
repro.py:3:30: PLR2004 Magic value used in comparison, consider replacing `3` with a constant variable
```

**Expected:** no diagnostic. `sys.implementation.version[0]` is the standard way to discriminate CPython major versions when `sys.version_info` is unavailable or when comparing a non-CPython implementation's version tuple, so the `3` is a version tag rather than a magic number. The same applies to `sys.version >= "3.10"`.

### Root cause

`magic_value_comparison` in `crates/ruff_linter/src/rules/pylint/rules/magic_value_comparison.rs` examines each comparand in isolation. It has a small hard-coded allow-list of `0`, `1`, `""`, `"__main__"` and the boolean and None family, plus a user-configurable `lint.pylint.allow-magic-value-types` that widens by type. There is no mechanism to consider the literal's neighbor, so it cannot know the literal is being compared against a version-derived operand. The fix therefore has to add neighbor-awareness to the diagnostic loop rather than another entry to the allow-list.

### The plan

**Understand:** A literal compared against a version operand is meaningful rather than magic, so the rule needs to look at the operand next to the literal before reporting.

**Match:** Two existing ruff patterns do exactly this kind of `sys.<name>` recognition:

1. `crates/ruff_linter/src/rules/flake8_2020/helpers.rs:5`, an `is_sys` helper using `semantic.resolve_qualified_name` against `["sys", <target>]`.
2. `crates/ruff_linter/src/rules/flake8_pyi/rules/unrecognized_version_info.rs:144`, using `resolve_qualified_name(map_subscript(left))` to recognize a `sys.version_info` operand whether or not it is subscripted.

`ruff_python_ast::helpers::map_subscript` unwraps one level of `x[…]` to `x` and leaves non-subscripts alone, so combining it with `resolve_qualified_name` recognizes `sys.version`, `sys.implementation.version`, and any subscript or attribute access on either. The pattern was already blessed by the codebase, so I just applied it to a new rule.

**Plan:**
1. Add `is_sys_version_comparand(expr, semantic)`, returning true when the expression resolves after one `map_subscript` unwrap to a qualified name prefixed `["sys", "version"]` or `["sys", "implementation", "version"]`.
2. In `magic_value_comparison`, for each literal comparand, check whether either adjacent operand is a sys-version comparand and skip the diagnostic for that literal if so.
3. Preserve the existing early return for two literals in a comparison, which is `R0133: comparison-of-constants`.
4. Update the rule docstring with `sys.implementation.version[0] >= 3` as the motivating example.
5. Add fixture cases covering both operand positions, chained comparisons and the deliberate `sys.version_info` omission.
6. Run `cargo test`, `cargo insta`, `fmt`, `clippy` and `cargo dev generate-all`.

**Implement:** Branch `plr2004-sys-version-exemption`.

**Review:** My self-checklist was that the neighbor logic handles first and last positions without panicking, chained comparisons are covered, the `R0133` early return is unchanged, there are no new settings so no schema regeneration is needed, and the docstring example matches the issue.

**Evaluate:** The snapshot diff should show exactly one added violation, the deliberately still-flagged `sys.version_info[0] >= 3` case, which proves the seven exempt cases now pass silently.

### Edge cases

- **Chained comparisons** such as `0 < sys.implementation.version[0] < 4`, where the literal has two neighbors, so the check must look both directions from a middle position.
- **Literal on either side**, covering `sys.version[0] >= "3"` and `3 <= sys.implementation.version[0]`.
- **Boundary positions**, since a literal first or last in the operand list has only one neighbor and the index arithmetic must not underflow. I handled that with `checked_sub(1)` in the first cut, and the reviewer's `peekable()` rewrite handles it in the final version.
- **Deliberate non-exemption of `sys.version_info`**, encoded as a fixture case that must still flag, so that widening the helper later cannot happen silently.

---

## Implementation

### Commits and files

| Commit | Date | Message |
|---|---|---|
| `a68487fc5` | 2026-06-08 | `[magic-value-comparison]` Exempt `sys.version` family comparisons (PLR2004) |

I force-pushed once on 2026-06-09 to fold in all five review points, and it merged as `7fbef674f`.

**Files modified (merged diff, +93 / −11):**

| File | Δ |
|---|---|
| `crates/ruff_linter/src/rules/pylint/rules/magic_value_comparison.rs` | +38 / −11 |
| `crates/ruff_linter/resources/mdtest/pylint/magic-value-comparison.md` | +55 (new) |

The helper as merged:

```rust
fn is_sys_version_comparand(expr: &Expr, semantic: &SemanticModel) -> bool {
    let Some(qualified_name) = semantic.resolve_qualified_name(map_subscript(expr)) else {
        return false;
    };
    matches!(
        qualified_name.segments(),
        ["sys", "version", ..] | ["sys", "implementation", "version", ..]
    )
}
```

Before review this used `segments.get(..N)` prefix matching, and ntBre asked for slice-with-rest patterns, which read better and let clippy nest them cleanly.

**Test-location change during review:** my first cut extended the existing fixture and snapshot pair, which meant `resources/test/fixtures/pylint/magic_value_comparison.py` plus regenerating `PLR2004_magic_value_comparison.py.snap` and `allow_magic_value_types.snap`. ntBre asked for the new cases in ruff's newer mdtest format instead, so the merged PR adds `resources/mdtest/pylint/magic-value-comparison.md` and leaves the legacy fixture alone.

### What was hard

The hard part was reading scope out of two conflicting maintainer comments. `ntBre` opposed the exception entirely while `MichaReiser` allowed it for `sys.version` and "maybe" `sys.implementation.version`. I took the strict reading and deliberately excluded `sys.version_info`, then encoded that omission as a fixture case so it could not be widened by accident.

In review, ntBre, the maintainer who had originally pushed back, asked to include `sys.version_info` after all. My strict reading was defensible but over-conservative, because `sys.version_info` is the canonical Python version-check site and excluding it left users with the same friction the issue was filed about on a different attribute. I widened the helper and the tests in the same revision.

### Testing

- `cargo test --package ruff_linter --lib pylint` gives **184 passed, 0 failed.**
- `cargo fmt --check` is clean and `cargo clippy --package ruff_linter -- -D warnings` is clean.
- `cargo dev generate-all` reports no regeneration needed, since the change adds no settings.
- **Coverage reasoning:**
  - The still-flagged `sys.version_info` case in the first cut was a regression test for the deliberate scope omission, and when the scope widened in review the case flipped to exempt, which is exactly what the snapshot diff showed.
  - The chained case `0 < sys.implementation.version[0] < 4` exercises the neighbor logic from a middle position.
  - Both literal positions, LHS and RHS, are covered.
- **Before and after evidence from CI:** ruff's ecosystem bot ran the build against 56 real projects and reported 0 added and 2 removed violations in 1 project with 55 projects unchanged, so the change removes exactly the two intended false positives in the wild and alters nothing else.

---

## Review and outcome

### The pull request

**[astral-sh/ruff#25743](https://github.com/astral-sh/ruff/pull/25743)**, opened 2026-06-08 against `astral-sh/ruff:main`, referencing the issue with a close keyword. `ntBre` was both auto-assigned and self-requested as reviewer.

**AI-policy incident:** the PR body I first posted had been drafted with AI help and posted without enough rewriting in my own voice. ntBre flagged it the same day, asking "Could you confirm that you've read and adhered to our AI Policy? The summary here gives me the impression that it was written by an LLM." I acknowledged it honestly, since the code was mine and the prose was not, read the [AI Policy](https://github.com/astral-sh/.github/blob/main/AI_POLICY.md), and he asked me to rewrite the summary and mark the PR ready. I rewrote it in my own words and re-submitted. The operating rule I took from it is that AI for investigation and code-level decisions is fine, while every maintainer-facing word in a PR body, issue reply or review response is one I type myself.

### Maintainer feedback

| Date | From | Feedback | My response |
|---|---|---|---|
| 2026-06-08 | ntBre | AI-policy question about the PR body's prose. | Acknowledged honestly the same day, read the policy, rewrote the summary in my own voice and marked it ready for review. |
| 2026-06-09 | ntBre | Five points, covering (1) using slice-with-rest patterns instead of `segments.get(..N)`, (2) migrating the new tests to mdtest, (3) "I also think we should just handle `sys.version_info`. Is there a reason not to?", (4) trimming the docstring paragraph, and (5) dropping the `Vec<&Expr>::collect()` and index arithmetic in favor of a `peekable()` iterator with a `previous` slot. | Addressed all five in one force-pushed revision, with a per-point summary comment on the thread, including that clippy wanted the slice patterns nested. |
| 2026-06-10 | ntBre | **Approved** with "Nice, thank you!" | Merged the same day as `7fbef674f`. |

### What I learned

**Technical:** The reusable insight is that a magic value is a property of the comparison rather than of the literal, so the fix had to add neighbor-awareness to the diagnostic loop, which is a different shape from extending an allow-list. Finding `is_sys` and `map_subscript` first saved me from inventing a worse mechanism, because the codebase had already solved "recognize a `sys.*` operand through subscripts" twice and the right move was to apply the blessed pattern rather than write a third one. I also learned to treat the ecosystem-check bot as real before-and-after evidence, since 0 added and 2 removed violations across 56 projects is a much stronger statement than any fixture I could write.

**Process & collaboration:** Two things stand out. ntBre spotted LLM-flavored prose immediately, and the only correct response was to say plainly what happened and fix it, which cost one comment where being defensive would have cost the PR. The reviewer who opposed the issue is also the one who asked me to widen its scope, which is a good reminder that in-thread positions are opinions at a moment rather than fixed constraints, and that asking "is there a reason not to?" is how good reviewers test a boundary.

**What I'd do differently:** I would have written the PR body myself from the first keystroke, and that is now a standing rule. I would have put the scope justification for excluding `sys.version_info` explicitly in the PR body instead of only in my head, because had it been visible, ntBre's question would have been answered before he asked it and I might have noticed my own reasoning was thin. I would also post a one-line "taking this" on the issue even when the repo's norm is PR-first, since it costs nothing and makes the work visible to anyone else reading the thread.

### References

- Issue thread: https://github.com/astral-sh/ruff/issues/25588
- PR: https://github.com/astral-sh/ruff/pull/25743
- Astral AI Policy: https://github.com/astral-sh/.github/blob/main/AI_POLICY.md
- Related in-flight PR [#18961](https://github.com/astral-sh/ruff/pull/18961), DaniBodor's user-configurable `pylint.allow_magic_values` setting.
- Existing pattern references in `crates/ruff_linter/src/rules/flake8_2020/helpers.rs` and `crates/ruff_linter/src/rules/flake8_pyi/rules/unrecognized_version_info.rs`.
