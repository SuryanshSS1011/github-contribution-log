# Contribution 3: Hint when a build failure is caused by a requires-python mismatch

**Contribution Number:** 3
**Student:** Suryansh Sijwali
**Issue:** https://github.com/astral-sh/uv/issues/7035
**Pull Request:** https://github.com/astral-sh/uv/pull/19673
**Status:** Phase III — PR submitted, awaiting maintainer review

---

## Why I Chose This Issue

uv is the Rust-based Python package and project manager from Astral (the
Ruff team). It's one of the strongest engineering-culture repos in active
open source — fast iteration, careful review, strict CI — and contributing
there is a learning opportunity in itself regardless of the specific issue.

Issue #7035 asks for a targeted error hint when a transitive build failure
is caused by the failing package's declared `requires-python` excluding the
active interpreter. This is a common confusing failure mode: a user pins
`numba==0.53.1` (released ~early 2021) on Python 3.12, and gets a deep
build-backend `RuntimeError` somewhere inside `llvmlite`'s `setup.py` that
doesn't say "you're using the wrong Python." The targeted hint would
shortcut a lot of "why doesn't this work?" diagnosis.

I picked it after going through several other uv issues this session: most
that looked promising turned out to be either (a) phantoms — already
implemented but the issue hadn't been closed (uv #7040, #4711's prior
attempts), (b) design-unresolved at the maintainer level (uv #4711 itself,
where I posted a form-factor question first), or (c) Windows-only
verification (uv #12142). #7035 was the first one that was clearly
available, had real engineering substance, and had a verifiable gap on
current master.

---

## Understanding the Issue

### Problem Description

When a transitive dependency of a user's requirement fails to build, the
underlying build-backend error (typically a Python `SyntaxError` from a
package using Python 2 `print` statements, or a `RuntimeError` from a
hand-rolled version guard in the package's `setup.py`) is what gets
surfaced. The actual root cause — the package's declared `requires-python`
doesn't include the active interpreter — is information uv already has
cached (it's in the dist's registry metadata), but never tells the user.

The issue body specifically mentions this is the most common reason for
`pyannote-audio → numba → llvmlite` build failures in the field.

### Expected Behavior

When `uv pip install` or `uv pip sync` fails to build a package whose
declared `requires-python` excludes the active interpreter, surface a
targeted hint identifying the version mismatch as the likely root cause —
before the existing derivation-chain hint and the generic build-failure
hint, since it diagnoses the actual problem.

### Current Behavior (verified on uv 0.11.19, current master)

Reproduction:

```toml
[project]
name = "test"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["numba==0.53.1"]
```

```
$ uv pip install -e .
... (llvmlite build failure with python-version-guard RuntimeError) ...

hint: `llvmlite` (v0.36.0) was included because `test` (v0.1.0) depends
      on `numba` (v0.53.1) which depends on `llvmlite`
hint: Build failures usually indicate a problem with the package or the
      build environment
```

The two existing hints establish the derivation chain (already a real
contribution, landed by an earlier PR) and a generic build-failure note,
but neither identifies the `requires-python` mismatch.

### Affected Components

- `crates/uv-errors/src/lib.rs` — the `Hints` API (only had `push`/append;
  needed `prepend` to insert the new diagnostic at the front).
- `crates/uv-distribution-types/src/requested.rs` — added a `file()`
  accessor on `RequestedDist`, mirroring the existing one on `Dist`.
- `crates/uv/src/commands/diagnostics.rs` — the central hint-rendering
  layer for distribution-failure errors. Threaded the active
  `python_version` through `OperationDiagnostic` and added the new hint
  function `requires_python_hint`.
- `crates/uv/src/commands/pip/install.rs` and `pip/sync.rs` — two call
  sites each, wired to pass `interpreter.python_version()` into the
  diagnostic.

---

## Reproduction Process

### Environment Setup

- macOS arm64 (Apple Silicon).
- Installed Rust via Homebrew's `rustup` formula. Note: the formula uses
  rustup's proxy model — `cargo`/`rustc` aren't symlinked into
  `/opt/homebrew/bin`; binaries live under
  `~/.rustup/toolchains/stable-aarch64-apple-darwin/bin/`. Have to add
  that to PATH.
- Forked `astral-sh/uv` on GitHub, cloned my fork at `~/oss-work/uv`.
- First clean build: 2 min 31 sec.
- Incremental rebuild after the patch: 35 seconds.

### Steps to Reproduce the Bug

1. `mkdir uv_test && cd uv_test && uv venv .venv --python 3.12`
2. Write a `pyproject.toml` listing `numba==0.53.1` (or any package whose
   transitive dependency has a `requires-python` not including 3.12).
3. `uv pip install -e .` against master uv.
4. Observe: build fails with `llvmlite`'s `RuntimeError: Cannot install on
   Python version 3.12.4; only versions >=3.6,<3.10 are supported.`
5. Observe: the two existing hints appear, but none identify the
   `requires-python` mismatch.

### Reproduction Evidence

The "before" output is captured in the PR description and the
`Understanding the Issue` section above.

---

## Solution Approach

### Analysis

The information needed to generate the hint is already in scope at error-
rendering time:

- The failing distribution carries its registry metadata via `Dist::file()`
  (or `RequestedDist::file()` after this PR), which includes the
  `requires_python: Option<VersionSpecifiers>` field set by the package
  author.
- The active interpreter's Python version is available via
  `interpreter.python_version() -> &Version` on the `Interpreter` already
  bound in `pip_install` / `pip_sync` before the call to
  `OperationDiagnostic::report`.

So no network calls, no metadata refetch — just plumbing two pieces of
already-computed state through to the renderer.

### Proposed Solution

Three logical changes:

1. Add `Hints::prepend` to the `uv-errors` crate so the new hint can be
   inserted at the front of the hint list (since it diagnoses the root
   cause and should render before contextual hints).
2. Expose `file()` on `RequestedDist`, mirroring the existing one on
   `Dist`, so the diagnostic layer can reach registry metadata uniformly.
3. Thread `python_version: Option<Version>` through `OperationDiagnostic`
   via a new `with_python_version()` builder. In the existing
   `dist_error` / `requested_dist_error` renderers, call the new
   `requires_python_hint` function, which returns `Some(hint_string)`
   exactly when the dist has a `requires-python` and it excludes the
   active interpreter version. Wire `with_python_version` into the two
   call sites each in `pip_install` and `pip_sync`.

### Implementation Plan

Using UMPIRE framework (adapted):

**Understand:** Failing transitive build → user gets a deep build-backend
error with no indication that the *real* problem is requires-python.
Surface a targeted hint when the mismatch can be detected from cached
metadata.

**Match:** uv already has `DerivationChain` (computed in
`uv-distribution-types`, rendered via `format_chain` in `diagnostics.rs`)
that does exactly this kind of "extract structured context from a failure
and format a hint" work. The new hint follows the same pattern: extract a
structured property (`requires-python` mismatch) from a failure, format a
human-readable hint string, prepend to the hint list.

**Plan:**
1. Read `diagnostics.rs`, `preparer.rs`, and `dist_error.rs` end-to-end
   to map the existing infrastructure.
2. Verify on current master that the gap is real (the numba/llvmlite
   repro emits no requires-python-aware hint).
3. Add `Hints::prepend` (commit 1).
4. Add `RequestedDist::file()` (commit 2).
5. Thread `python_version` through `OperationDiagnostic`, add
   `requires_python_hint`, wire it into the two error renderers, and
   pass the interpreter version at the four call sites in
   `pip_install`/`pip_sync` (commit 3).
6. Build, run the existing `resolve_derivation_chain` test, manually
   verify the patch on the numba repro and a known-good install.
7. `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D
   warnings`.
8. Match uv's doc-comment house style: one-line `///` per function/field
   unless a multi-paragraph block is actually needed.
9. Open PR; iterate on review.

**Implement:** Branch `pip-install-requires-python-hint` on my fork;
three commits at `72da414` (HEAD).

**Review:** Self-review checklist before opening PR:
- [x] Single-line `///` docs match house style (verified against
  `Hints::push`, `Dist::file`, neighboring functions in
  `diagnostics.rs`).
- [x] Diff scoped to `pip install` / `pip sync` only — flagged the broader
  set of `OperationDiagnostic` callers in the PR description as a possible
  follow-up.
- [x] No new dependencies, no new public types beyond the one accessor.
- [x] Existing `resolve_derivation_chain` test still passes (wsgiref has
  no `requires-python` metadata, so the new hint correctly doesn't fire).
- [x] `cargo fmt --check` clean.
- [x] `cargo clippy --workspace --all-targets -- -D warnings` clean.

**Evaluate:** Manually run the numba/llvmlite repro on Python 3.12 →
observe the new hint as the first hint in the output, followed by the
existing chain hint and generic hint. Run a known-good install (`uv pip
install rich`) → observe no spurious hint.

---

## Testing Strategy

### Unit Tests

Not applicable to this contribution — the change lives at the integration
layer (rendering hints attached to distribution-failure errors), not at a
unit boundary.

### Integration Tests

The PR currently ships **without a new integration test**. Reasons:

- The existing pattern in `crates/uv/tests/it/pip_install.rs` for hint
  tests uses real package installs (`wsgiref==0.1.2` for the chain test).
  A new test using `numba==0.53.1` would be analogous, but the
  classification feedback I got was that hitting PyPI for a specific old
  sdist version is brittle.
- A local-sdist test (generate a sdist with `requires-python = ">=3.6,
  <3.10"` and a guarded `setup.py`) is cleaner but adds test
  infrastructure beyond the patch scope.
- I flagged this explicitly in the PR description and offered to add a
  test in whichever shape the maintainer prefers.

### Manual Testing

- [x] `numba==0.53.1` install on Python 3.12 → new hint appears as the
  first hint, before the chain hint, before the generic hint.
- [x] `rich` install on Python 3.12 (compatible package) → no spurious
  hint.
- [x] `wsgiref==0.1.2` install on Python 3.12 (no `requires-python`
  metadata) → the new hint correctly doesn't fire; the existing chain
  hint still appears.

---

## Implementation Notes

### Code Changes

- **Files modified:**
  - `crates/uv-distribution-types/src/requested.rs` (+9 / −1)
  - `crates/uv-errors/src/lib.rs` (+5)
  - `crates/uv/src/commands/diagnostics.rs` (+54 / −7)
  - `crates/uv/src/commands/pip/install.rs` (+2)
  - `crates/uv/src/commands/pip/sync.rs` (+2)
- **Branch:** `pip-install-requires-python-hint` on
  `SuryanshSS1011/uv`.
- **Commits:**
  - `76deb21` — Add Hints::prepend for inserting root-cause hints first.
  - `ac497f6` — Expose File access on RequestedDist.
  - `72da414` — Hint when a build failure is caused by a requires-python
    mismatch. (Closes #7035.)
- **Approach decision — why scope to `pip install`/`pip sync` only:**
  The same hint would fire usefully on many other `OperationDiagnostic`
  call sites (about 30 across the codebase: project sync, add, run, tool,
  audit, version, etc.). Threading the `python_version` through all of
  them at once would have doubled the diff size and increased the chance
  of touching a code path with subtler error flows. Scoped to the two
  sites named in the issue body for review; flagged the follow-up
  explicitly in the PR description.
- **Approach decision — why `prepend` and not `push`:** The new hint
  diagnoses the *likely root cause* (`requires-python` mismatch). The
  existing chain hint provides *context* (where the failing dist came
  from). Convention in compiler-style diagnostics is root cause first,
  context after. Prepending matches this.

---

## Pull Request

**PR Link:** https://github.com/astral-sh/uv/pull/19673

**Maintainer Feedback:** None yet (submitted ~minutes ago).

**Status:** PR open, CI passing on completed checks (fmt, lint subsets,
typos, lockfiles), several long-running checks still in progress (clippy
linux/windows, cargo test on linux/windows, dev binary builds).
Awaiting first reviewer assignment.

---

## Learnings & Reflections

To be filled in after review and merge.

Open observations as of submission:

- **uv's CI is unusually thorough** — ~30 checks including platform
  variants, multiple lint tools, generated-file consistency, doc build,
  benchmark suites. Tells me the maintainers expect contributions to
  hold up under scrutiny. Worth running `cargo fmt --check` and `cargo
  clippy -- -D warnings` locally before pushing.
- **House style for `///` comments is strictly one-line in this
  codebase**, even on private functions. Verified by reading neighbors
  before settling on the final form. My first draft had multi-line docs
  with rationale and issue links; revised to match.
- **The "phantom issue" pattern is real and common in fast-moving
  repos.** Of the candidates I evaluated, four turned out to already be
  fixed: AMReX #3848, Enzyme-JAX #152 (the OptimizationBarrierOp
  checklist), llama.cpp XIELU/TOP_K, and uv #7040. The right discipline
  is to actually reproduce the gap on current master before writing
  code, every time.

---

## Resources Used

- Issue thread: https://github.com/astral-sh/uv/issues/7035
- uv `AGENTS.md` (root of repo) — coding conventions, especially around
  doc comments, tests, and import patterns.
- uv `CONTRIBUTING.md` — local dev setup, CI invocations.
- Existing `resolve_derivation_chain` test in
  `crates/uv/tests/it/pip_install.rs` — pattern for hint regression
  tests.
- `crates/uv-distribution-types/src/dist_error.rs` — `DerivationChain`
  implementation, which served as the structural template for the new
  hint.
