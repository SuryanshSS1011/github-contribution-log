# Hint when a build failure is caused by a `requires-python` mismatch

**Student:** Suryansh Sijwali ([@SuryanshSS1011](https://github.com/SuryanshSS1011))
**Project:** [astral-sh/uv](https://github.com/astral-sh/uv) · **My fork:** https://github.com/SuryanshSS1011/uv
**Issue:** [#7035](https://github.com/astral-sh/uv/issues/7035) · **Pull request:** [#19673](https://github.com/astral-sh/uv/pull/19673) · **Branch:** `pip-install-requires-python-hint`
**Status:** Open. CI green, reworked after review, parked on an upstream refactor.

---

## Phase I: Issue Selection

### Why I Chose This Issue

uv is the Rust-based Python package and project manager from Astral, the team behind Ruff. It is one of the strongest engineering-culture repos in active open source, with fast iteration, careful review and strict CI, so contributing there is a learning opportunity in itself regardless of the specific issue. My learning goal was to work on diagnostics rather than features, meaning I wanted to figure out what information a tool already holds at failure time and how to surface it without lying to the user. That turned out to be exactly the hard part here.

Issue #7035 asks for a targeted error hint when a transitive build failure is caused by the failing package's declared `requires-python` excluding the active interpreter. This is a common and confusing failure mode. A user pins `numba==0.53.1`, released in early 2021, on Python 3.12 and gets a deep build-backend `RuntimeError` from inside `llvmlite`'s `setup.py` that never says "you're using the wrong Python."

I picked it after going through several other uv issues. Most of the promising ones turned out to be phantoms that were already implemented but not closed (uv #7040 and #4711's prior attempts), or design-unresolved at the maintainer level (uv #4711 itself, where I posted a form-factor question first), or Windows-only to verify (uv #12142). #7035 was the first one that was clearly available, had real engineering substance, and had a verifiable gap on current `main`.

### Problem Summary

When a transitive dependency fails to build, uv surfaces the build backend's own error, usually a `SyntaxError` from Python-2-era code or a `RuntimeError` from a hand-rolled version guard, and never mentions that the package simply does not support the running interpreter. uv already has that fact cached in the distribution's metadata, so the user is left reading a stack trace to rediscover something the tool knows. It matters because this is the single most common cause of the `pyannote-audio` to `numba` to `llvmlite` class of failures that end up filed as uv bugs. I chose it because it is a real diagnostic-quality problem in a repo with a high review bar, and the gap was reproducible on current `main`.

### Issue Vetting

Open, unassigned, no in-flight PR touching the hint path, and no blocking labels. The only prior comment is from `hauntsaninja` on [2024-09-08](https://github.com/astral-sh/uv/issues/7035#issuecomment-2337003074), asking for a related special case when the package version predates the target Python's first release candidate. That is adjacent, unimplemented, and explicitly out of scope for this PR.

### Where It Lives

The final surface after review:

- `crates/uv-distribution/src/source/mod.rs`, the build-failure site, where the build environment's interpreter and the distribution's `requires-python` are both in scope.
- `crates/uv-errors/src/lib.rs`, the `Hints` API, which only had `push` (append) and needed `prepend` so a root-cause hint renders before contextual hints.
- `crates/uv-types/src/traits.rs`, build-context plumbing for the interpreter.

The first implementation instead targeted `crates/uv/src/commands/diagnostics.rs` plus the four `pip_install` and `pip_sync` call sites. The Maintainer Feedback Log explains why that moved.

### Acceptance Criteria

1. A build failure whose distribution declares a `requires-python` that excludes the interpreter used for the build emits a hint naming both ranges.
2. The hint renders before the existing derivation-chain hint and the generic build-failure hint.
3. No spurious hint appears on a compatible package, or on a distribution with no `requires-python` metadata.
4. The wording does not claim causation it cannot prove, and does not tell the user to move up or down a version when either direction can be correct.
5. `cargo fmt` and `cargo clippy` are clean, and snapshot coverage is added.

---

## Phase II: Reproduce & Plan

### Environment Setup

I used uv's `CONTRIBUTING.md` for the local dev loop and its root `AGENTS.md` for conventions around doc comments, tests and import patterns, plus the CI workflow files for the exact lint invocations the bots run.

- macOS arm64 (Apple Silicon).
- **Challenge (Rust toolchain not on `PATH`):** I installed Rust via Homebrew's `rustup` formula, which uses rustup's proxy model, so `cargo` and `rustc` are not symlinked into `/opt/homebrew/bin`. I resolved it by putting `~/.rustup/toolchains/stable-aarch64-apple-darwin/bin/` on `PATH`.
- **Challenge (build cost):** first clean build took 2 min 31 s and an incremental rebuild after a patch takes about 35 s. That is cheap enough to iterate on, unlike the other compiled repos I work in.
- **Challenge (snapshot format drift after a rebase):** after rebasing onto `main`, previously green snapshot tests went red. The cause was upstream PR #20630 changing the `uv_snapshot!` header format, since the `success:` line and empty stdout block are dropped and the exit code renders differently. That was not my change, so I regenerated the snapshots and flagged it on the PR to save the reviewer the diagnosis time.
- A `CHANGELOG.md` conflict during the same rebase was resolved by taking upstream's version wholesale, because my change adds no changelog entry.

### Steps to Reproduce

1. `mkdir uv_test && cd uv_test && uv venv .venv --python 3.12`
2. Write this `pyproject.toml`:
   ```toml
   [project]
   name = "test"
   version = "0.1.0"
   requires-python = ">=3.12"
   dependencies = ["numba==0.53.1"]
   ```
3. Build uv from `main` and run `uv pip install -e .` with it.
4. Observe the build fail with `llvmlite`'s `RuntimeError: Cannot install on Python version 3.12.4; only versions >=3.6,<3.10 are supported.`
5. Read the hints printed underneath.

### Expected vs. Actual

**Actual (uv 0.11.19, `main`):**

```
hint: `llvmlite` (v0.36.0) was included because `test` (v0.1.0) depends
      on `numba` (v0.53.1) which depends on `llvmlite`
hint: Build failures usually indicate a problem with the package or the
      build environment
```

The two hints establish the derivation chain and a generic build-failure note. Neither mentions `requires-python`, even though `llvmlite==0.36.0` declares `>=3.6,<3.10` and uv has that string in hand.

**Expected:**

```
hint: The build requires Python >=3.6, <3.10, but Python 3.12 is used.
hint: `llvmlite` (v0.36.0) was included because ...
hint: Build failures usually indicate a problem with the package or the
      build environment
```

### Root Cause

This is not a logic bug but a missing diagnostic, plus a layering mistake in my own first fix. The information needed is already in scope at failure time, since the distribution carries `requires_python: Option<VersionSpecifiers>` in its registry metadata or statically in a source tree's `pyproject.toml`, and the interpreter is bound in the build context. Nothing renders it.

Review surfaced the deeper layering point. The resolve interpreter is not necessarily the interpreter that built the distribution, and build failures also occur under `uv lock` and `uv pip compile` rather than only under `pip install` and `pip sync`. So the hint belongs at the build-failure site in `uv-distribution`, where the build environment's interpreter is the one actually in scope, and not in the command-level diagnostics layer.

### Plan (UMPIRE)

**Understand:** A failing transitive build gives the user a deep build-backend error with no indication that the real problem is `requires-python`. Surface a targeted hint whenever that mismatch is detectable from metadata uv already holds.

**Match:** uv already has `DerivationChain`, computed in `uv-distribution-types` and rendered via `format_chain`, doing exactly this shape of work by extracting structured context from a failure, formatting a hint string and attaching it. After review I matched a second and closer precedent in PR #20157, which attaches context to the build error itself and produces the hint in the error's `Hint` implementation, leaving the error rendering otherwise untouched.

**Plan:**
1. Read `diagnostics.rs`, `preparer.rs` and `dist_error.rs` end to end to map the existing infrastructure.
2. Verify on `main` that the gap is real, since the numba and llvmlite repro emits no `requires-python`-aware hint.
3. Add `Hints::prepend` so a root-cause hint can render before contextual hints.
4. Compute the mismatch where the build fails, in `uv-distribution/src/source/mod.rs`, using the build environment's interpreter and `build_requires_python`.
5. Attach the context to the build error and emit the hint from its `Hint` impl, following the #20157 shape.
6. Add snapshot coverage in the `pip_install` and `pip_sync` integration tests.
7. Run `cargo fmt --check` and `cargo clippy --workspace --all-targets -- -D warnings`.
8. Match uv's house style of one-line `///` per item unless a block is genuinely needed.

**Implement:** Branch `pip-install-requires-python-hint` on my fork, with the final state as a single squashed commit, `5346e25ac`.

**Review:** My self-checklist before pushing was that single-line `///` docs are verified against neighbors, there are no new dependencies, there are no new public types beyond what the hint needs, `fmt` and `clippy` are clean, and the hint does not fire on dists without `requires-python`.

**Evaluate:** Run the numba and llvmlite repro on 3.12 and confirm the new hint renders first, run `uv pip install rich` and confirm no spurious hint, then run `wsgiref==0.1.2` (which has no `requires-python` metadata) and confirm the hint stays silent while the existing chain hint still appears.

### Edge Cases Considered

- **Interpreter below the range rather than above it:** zanieb raised this. Telling the user to upgrade would be wrong when they are on Python 3.8 and the package needs `>=3.10`. I checked that `RequiresPython::from_specifiers(...).range()` exposes `lower()` and `upper()` so the direction is detectable, then deliberately kept the wording neutral, as in "The build requires Python X, but Python Y is used", rather than shipping a directional suggestion in this PR.
- **Universal resolution:** Build environments could in principle use different interpreter versions. I confirmed that `pip install` and `pip sync` do not support `--universal` and both resolve with `ResolverEnvironment::specific`, so there is one well-defined version in scope. Moving the hint to the build site made the point moot anyway, since the build environment's own interpreter is now used.
- **No `requires-python` available:** `build_requires_python` returns `None` for a source tree with no readable or parseable `pyproject.toml` (legacy `setup.py`-only projects), one with no `requires-python` or a `dynamic` one, and one with a malformed specifier that gets logged at `debug!`. In all three the original build error is returned unchanged with no hint attached.
- **Minor-version display:** `requires-python` is expressed in minor versions, so the hint shows `3.12` rather than `3.12.4`. I flagged this in the PR body as an easy change if the maintainer prefers the full version.

---

## Phase III: Build

### Implementation Progress

| Commit | Date | Message |
|---|---|---|
| `5346e25ac` | 2026-07-28 | Hint when a build failure is caused by a requires-python mismatch |

The branch previously carried three commits from the resolve-side design, covering `Hints::prepend`, a `RequestedDist::file()` accessor and then the hint itself wired through `OperationDiagnostic`. That design was replaced during review, the accessor became unnecessary once the hint moved to the build site, and the history was squashed on rebase.

**Files modified (final diff):**

| File | Δ |
|---|---|
| `crates/uv-distribution/src/source/mod.rs` | +100 / −7 |
| `crates/uv-errors/src/lib.rs` | +5 |
| `crates/uv-types/src/traits.rs` | +18 / −3 |
| `crates/uv/tests/pip_install/pip_install.rs` | +39 |
| `crates/uv/tests/pip/pip_sync.rs` | +2 |

**Design decision (why `prepend` and not `push`):** the new hint diagnoses the likely root cause while the existing chain hint provides context, and compiler-style diagnostics put root cause first with context after.

**Design decision (why the build site and not the command):** this is covered under Root Cause. The short version is that scoping to two commands was a smaller diff but the wrong layer, and the reviewer was right.

### Challenges Faced

The concrete blocker was testing. In the first design the hint only fired for registry distributions, the ones that carry a `File` with `requires-python`, and I could not get uv's fixtures to line that up. `packse` packages build successfully so nothing triggers the failure, and the path-dependency and custom-build-backend tests never go through a registry `File`. Rather than guess, I said so on the review thread and asked whether there was a reference test for the registry-sdist-build-failure case.

zanieb's answer reframed the problem. The registry-only restriction was itself the mistake, because `requires-python` is available for all distribution types. Moving the hint to the build-failure site removed the restriction and made the change testable with the custom-build-backend approach at the same time, so the testing blocker and the design flaw had the same fix.

### Testing

- **New:** insta snapshot coverage for the hint in the integration tests, covering `build_requires_python_hint` plus the touched `require_hashes_find_links_no_hash`, following the existing `uv_snapshot!` pattern used by the neighboring hint tests in `crates/uv/tests/pip_install/pip_install.rs`. uv's own test-inventory bot confirmed the diff as 1 test added and 0 removed.
- **Existing suite:** CI is green with 49 checks passing, 8 skipped and 1 cancelled as of 2026-08-04. uv runs about 30 distinct checks including platform variants, multiple lint tools, generated-file consistency and doc builds.
- **Manual:** `numba==0.53.1` on Python 3.12 renders the new hint first, followed by the chain hint and the generic hint. `rich` on 3.12 produces no spurious hint. `wsgiref==0.1.2`, which has no `requires-python`, correctly leaves the hint silent with the chain hint unaffected.
- `cargo fmt --check` is clean and `cargo clippy` is clean on the touched crates.

---

## Phase IV: Submit & Iterate

### Pull Request

**[astral-sh/uv#19673](https://github.com/astral-sh/uv/pull/19673)**, opened 2026-06-04 against `astral-sh/uv:main`, referencing `Closes #7035`.

The body opens with the user-facing problem of confused bug reports caused by an unexplained build failure, then the change, then before and after console output for the numba and llvmlite case, then an explicit scope note about the neutral wording and the minor-version display, and finally a checked test plan. Reviewer `zanieb` is engaged directly on the thread.

### Maintainer Feedback Log

| Date | From | Feedback | My response |
|---|---|---|---|
| 2026-06-28 | zanieb | "I'm not sure we can say it's the most likely cause unless we also have sniffed for a SyntaxError in the text." | Reworded to state the mismatch neutrally and drop the causation claim. |
| 2026-06-28 | zanieb | How does this behave under universal resolution, where build environments may use different interpreters? | Investigated and answered on-thread that `pip install` and `pip sync` do not support universal resolution and resolve with `ResolverEnvironment::specific`, so one interpreter is in scope. Made moot by the later move to the build site. |
| 2026-06-28 | zanieb | "We'll want some sort of test coverage for this... by creating a test package with a custom build backend." | Added unit coverage on the message builder immediately, explained the registry-`File` blocker for an end-to-end test and asked for a reference fixture. Resolved by the rework. |
| 2026-06-28 | zanieb | What if `python_version` is below the `requires-python` range, wouldn't the suggestion be backwards? | Confirmed the direction is detectable via `RequiresPython::range()`'s `lower()` and `upper()`, and kept the wording direction-neutral instead of shipping a possibly wrong suggestion. |
| 2026-06-29 | zanieb | "These build failures could happen in `uv lock` or `uv pip compile` too. Are we perhaps attaching the hint in the wrong location?" and "Why do we only fire the hint for a registry distribution?" | Agreed, and proposed generating the hint at the build-failure site in `uv-distribution` using the build environment's interpreter so it covers every distribution type. zanieb confirmed that matched his suggestion. |
| 2026-06-30 to 2026-07-13 | Me | Own updates, no maintainer feedback in this window. | Pushed the full rework, posted a summary of what changed, and bumped once after two weeks of silence. |
| 2026-07-18 | zanieb | "I'll try to take a look next week, feel free to ping me if I forget again 😬" | Posted a short note on what had changed since he last looked so the context would be fresh, then pinged on 2026-07-22 as invited. |
| 2026-07-28 | zanieb | Doc-comment placement on `build_requires_python`, and a question about when we can fail to determine `requires-python`. | Moved the prose to inline comments per the suggestion and enumerated the three `None` cases, which are an unparseable or absent `pyproject.toml`, an absent or `dynamic` `requires-python`, and a malformed specifier, noting the error is returned unchanged in each. |
| 2026-07-28 | zanieb | "The structure of this is not quite what I'd expect, I'm looking into a refactor to share." | Acknowledged, and pre-emptively flagged that the earlier red CI came from #20630 changing the `uv_snapshot!` header format after my rebase rather than from the patch, so it would not cost him diagnosis time. Currently waiting on his refactor. |

### Learnings & Reflections

**Technical:** The reviewer's central point was that I had attached the hint at the wrong layer, and that is the thing I will carry forward. My version was correct for the two commands named in the issue and wrong for the program, because `uv lock` and `uv pip compile` hit the same failure and the interpreter that matters is the build environment's rather than the resolver's. The tell was there in my own scope note about "about 30 other `OperationDiagnostic` call sites", because when the honest description of a change is that it is right for 2 of 30 call sites, that is usually a signal about layering rather than about scope discipline.

**Process & collaboration:** Two habits paid off. When I hit the end-to-end test blocker I described precisely what I had tried and asked a specific question instead of shipping an untested patch or silently dropping the test, and the answer solved a design problem rather than only the test problem. I also kept the thread warm without being a nuisance, with one bump after three weeks of silence, a note on what had changed when he said he would look next week, and a ping only when he explicitly invited one. Flagging the #20630 snapshot breakage myself, before he could waste time on it, is also the cheapest possible way to be easy to review.

**What I'd do differently:** I would spend the first hour asking where the failure actually originates instead of where it is currently rendered. I started from the rendering layer because that is where the output I wanted to change lives, and it cost a full redesign. I would also stop treating "the issue body names two commands" as a scope boundary, because issue bodies describe symptoms and the fix belongs wherever the cause is.

### Resources Used

- Issue thread: https://github.com/astral-sh/uv/issues/7035
- uv `AGENTS.md` in the repo root, for conventions on doc comments, tests and imports.
- uv `CONTRIBUTING.md`, for local dev setup and CI invocations.
- PR [#20157](https://github.com/astral-sh/uv/pull/20157), the error-context and `Hint`-impl pattern the final design follows.
- `crates/uv-distribution-types/src/dist_error.rs`, holding `DerivationChain`, the structural template for the first design.
- Existing hint tests in `crates/uv/tests/pip_install/pip_install.rs`, for the snapshot-test pattern.
