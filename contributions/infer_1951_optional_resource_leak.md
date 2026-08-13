# Model `java.util.Optional` so Pulse catches resource leaks through it

**Project:** [facebook/infer](https://github.com/facebook/infer)
**Issue:** [#1951](https://github.com/facebook/infer/issues/1951) · **Pull request:** [#2068](https://github.com/facebook/infer/pull/2068) · **Branch:** `pulse-model-optional-resource-leak`
**Status:** **Merged** 2026-07-03 (merge commit `f123c84`). Issue closed.

---

## The issue

### Why I picked it

Infer is Meta's static analyzer and Pulse is its memory-and-resource lifetime checker. I wanted a contribution in the static-analysis and program-analysis space, which is the closest OSS analogue to my research interests, and my learning goal was to understand how an abstract interpreter models library types, meaning the mechanism by which a checker knows what `Optional.get()` does without analyzing the JDK.

This one was a clean fit, because a maintainer (`davidpichardie`) had already confirmed the false negative on the issue, invited a PR, and pointed at the exact file and a sibling model to copy:

> We have very limited bandwidth these days and we would welcome a PR to solve this problem. The best approach is to *model* the `java.util.Optional` class, as we did for the `java.util.Boolean` class. [...] Want to have a try? I'm happy to answer any question.

A confirmed bug with a prescribed approach and a worked example is the combination that makes an OSS PR actually land, which made it a strong pick despite the repo being OCaml with a heavy build.

### What was wrong

Pulse attaches a `JavaResource` obligation when a resource is constructed and reports `PULSE_RESOURCE_LEAK` if it becomes unreachable without being closed, but it has no model for `java.util.Optional`, so a resource wrapped in one such as `Optional.of(new FileInputStream(...))` becomes invisible to that reachability reasoning and the leak goes unreported. It matters because this is a false negative in a leak checker, and silent under-reporting is worse than noise since nobody knows to look. I chose it because a maintainer had already confirmed it and named both the file and the model to mirror, so the work was implementation and verification rather than negotiation.

### Before I started

Before starting I re-verified the issue was still open, unassigned and free of an in-flight PR, and following my own rule about old issues I checked that `java.util.Optional` was genuinely still unmodeled on current `main` rather than fixed but unclosed. I then posted a scoping comment on the issue on 2026-06-29 rather than just claiming it, because on close reading the shape did not look identical to `Boolean`:

> Following your suggestion I started from the `java.lang.Boolean` model in `PulseModelsJava.ml`. While scoping it I think the precise shape here may be a bit different from `Boolean` [...] so I wanted to check the approach with you before opening a PR.

### Where it lives

- `infer/src/pulse/PulseModelsJava.ml`, the Java model table, where the `Boolean` model around line 570 is the structural template.
- `infer/tests/codetoanalyze/java/pulse/ResourceLeaks.java`, the Pulse Java test corpus.
- `infer/tests/codetoanalyze/java/pulse/issues.exp`, the expected-output file that encodes which reports must fire.

### What counts as done

1. The issue's repro reports `PULSE_RESOURCE_LEAK`.
2. A wrapped resource that is closed via `get().close()` stays clean, with no false positive.
3. Both `Optional.of` and `Optional.ofNullable` are covered.
4. There is no regression anywhere else in the Pulse Java corpus.

---

## Diagnosis and plan

### Environment setup

I used infer's `INSTALL.md` and `Makefile` targets plus `infer.opam.locked` as the authority on toolchain versions, and built on Penn State's ROAR Collab HPC cluster rather than locally, since I have no root there and my home directory is near quota.

- **Challenge (no root for system libraries):** the build needs GMP for zarith and a couple of other system libs that are normally installed by the package manager. I resolved it by building GMP from source into a prefix and running opam with `depext=false` so it would not try to install system packages it cannot.
- **Challenge (toolchain must match the lockfile exactly):** `ppxlib` in particular, because a newer version breaks the ppx-generated parsetree and the build fails deep in generated code with errors that do not point at the cause. I resolved it by pinning strictly to `infer.opam.locked`.
- **Challenge (long builds over an SSH session):** a full infer build outlives a login session, so builds were detached with `setsid` and polled rather than run in the foreground.
- The scratch filesystem rather than home holds the opam root and build tree, since home is capped and a full infer build does not fit.

### Steps to reproduce

1. Build infer from source at a known commit, using a stock build with no patch.
2. Save the issue's repro as `Main.java`:
   ```java
   void foo(boolean b) throws IOException {
       Optional<FileInputStream> stream;
       if (b) {
           stream = Optional.of(new FileInputStream("file.txt")); // leak
       }
   }
   ```
3. Run `infer run --pulse-only -- javac Main.java`.
4. Read the report.

### Expected vs. actual

**Actual, on a stock build:** `No issues found`. The resource is allocated, wrapped and never closed, and Pulse says nothing.

**Expected:** `Main.java:9: error: PULSE_RESOURCE_LEAK`, the same report Pulse produces when the identical `FileInputStream` is held in a plain local variable.

**After the model, on the same binary and commit:** `Main.java:9: error: PULSE_RESOURCE_LEAK`. Running the before and after against a stock build at the same commit is what makes the fix causally responsible rather than incidentally correlated.

### Root cause

Pulse's reasoning is reachability-based, since the `JavaResource` attribute lives on the resource's own allocation and a leak is reported when that allocation becomes unreachable without a close. `Optional` is an ordinary library class with no model, so Pulse cannot relate `Optional.of(x)` to `x`, which makes the wrapper opaque, the wrapped value's reachability untrackable through it, and the obligation neither discharged nor reported. The gap is a missing model in `PulseModelsJava.ml` rather than a defect in the leak logic.

### The plan

**Understand:** A wrapper type with no model breaks the reachability chain that leak detection depends on.

**Match:** The `Boolean` model in the same file is a value box, meaning a backing field plus `init`, `valueOf` and `booleanValue`. `Optional.of` and `ofNullable` are static factories that behave like `Boolean.value_of` by allocating a fresh object and storing the argument into a backing field, and `Optional.get` is the load, so the same shape applies with a `__infer_model_backing_optional` field.

**Plan:**
1. Re-verify the false negative on current `main` with a stock build.
2. Post the scoping question to the maintainer, since `Optional`'s API surface is wider than `Boolean`'s.
3. Add an `Optional` module with `init`, `of_` and `get` in `PulseModelsJava.ml`, in the same `DSL.Syntax` style as `Boolean`.
4. Register three matcher rows gated on `PatternMatch.Java.implements "java.util.Optional"` for `of`, `ofNullable` and `get`.
5. Add leak cases and a must-stay-clean closed case to `ResourceLeaks.java`, then regenerate `issues.exp`.
6. Run the Pulse Java test suite and check `ocamlformat` at the repo-pinned version.

**Implement:** Branch `pulse-model-optional-resource-leak`, single commit.

**Review:** My self-checklist was that the model reads like `Boolean` rather than like a new invention, that matcher rows are gated on the interface predicate the file already uses, that test imports are placed alphabetically per the file's convention, and that the leak-reporting logic itself is untouched.

**Evaluate:** Run before and after on the same binary and commit, then a whole-corpus regression run.

### Edge cases

- **The closed case must stay silent:** `optional.get().close()` has to discharge the obligation, or the model trades a false negative for a false positive, which in a leak checker is the worse trade. This is covered by `optionalOfClosedOk`.
- **`ofNullable` as well as `of`:** Both are factories in real code, so modeling only `of` would leave half the pattern unreported.
- **No separate release machinery needed:** Because the `JavaResource` attribute already lives on the resource's own allocation, forwarding the value through `get` is sufficient. I checked that rather than assuming it, since adding release or delegation logic would have been a much larger and riskier change.

---

## Implementation

### Commits and files

| Commit | Date | Message |
|---|---|---|
| `33da46a5a` | 2026-07-02 | [infer][pulse] model java.util.Optional to catch resource leaks through it |

**Files modified (+53):**

| File | Δ |
|---|---|
| `infer/src/pulse/PulseModelsJava.ml` | +31 |
| `infer/tests/codetoanalyze/java/pulse/ResourceLeaks.java` | +20 |
| `infer/tests/codetoanalyze/java/pulse/issues.exp` | +2 |

The model is an `Optional` module with `init`, `of_` and `get` over a `__infer_model_backing_optional` field, written in the same `DSL.Syntax` style as `Boolean`, with three matcher rows gated on `PatternMatch.Java.implements "java.util.Optional"`.

### What was hard

The real cost of this contribution was not the 31-line model but getting a working infer build without root on a shared cluster. Three things went wrong in sequence. zarith needs GMP and there is no package manager available, so GMP had to be built from source with opam run at `depext=false`. opam's default solver happily installs a newer `ppxlib` than `infer.opam.locked` pins, which breaks the ppx-generated parsetree and fails deep inside generated OCaml with errors that do not name the cause. A full build also outlives an SSH session, so builds had to be detached and polled. None of that is interesting engineering, but all of it had to be solved before the first line of the model could be validated, and it is the reason I re-used this environment for the later infer work on #1937.

I put the design question, meaning whether `Optional` really is a `Boolean`-shaped value box, to the maintainer before writing the PR rather than guessing, since `Optional`'s API surface with `isPresent`, `orElse` and `map` is much wider and it was not obvious how much of it needed modeling for the leak case. The answer turned out to be only the three methods on the allocation and observation path.

### Testing

- **New tests**, following the corpus's `…Bad` and `…Ok` naming convention which encodes the expected verdict in the method name:
  - `optionalOfNotClosedBad` and `optionalOfNullableNotClosedBad`, which must report `PULSE_RESOURCE_LEAK`.
  - `optionalOfClosedOk`, closed via `get().close()`, which must stay clean.
- `issues.exp` was regenerated with exactly the two new expected reports, and the diff being only those two lines is itself the proof that nothing else in the corpus changed verdict.
- `make -C infer/tests/codetoanalyze/java/pulse test` passes, and no other test in the Pulse Java corpus regresses.
- **Before and after evidence:** a stock build at the same commit reports `No issues found`, and with the model it reports `Main.java:9: error: PULSE_RESOURCE_LEAK`.
- `ocamlformat` at the repo-pinned version reports the model file format-clean, and the test file's import was placed in alphabetical order to match the file's convention.

---

## Review and outcome

### The pull request

**[facebook/infer#2068](https://github.com/facebook/infer/pull/2068)**, opened 2026-07-02 against `facebook/infer:main`, referencing the issue with a close keyword. The body leads with the false negative and why a leak checker under-reporting is the serious direction of error, then the value-box model mirroring `Boolean`, then the before and after evidence. My CLA was already signed from earlier ExecuTorch work.

I @-mentioned `davidpichardie`, the maintainer who had confirmed the bug and prescribed the approach, directly on the PR so it would reach the person with context:

> @davidpichardie This implements the Optional model we discussed in #1951 (of/ofNullable/get), mirroring the Boolean model. Verified the repro reports PULSE_RESOURCE_LEAK after the change and stays clean when closed via get().

### Maintainer feedback

| Date | From | Feedback | My response |
|---|---|---|---|
| 2025-11-04 | davidpichardie (on the issue) | Confirmed the false negative, invited a PR, prescribed modeling `Optional` the way `java.util.Boolean` is modeled, and linked the exact source line. | Followed the prescription and used `Boolean` as the structural template. |
| 2026-06-29 | Me | Own update, no maintainer feedback in this window. | Posted a scoping question on the issue before implementing, since `Optional`'s API surface is wider than `Boolean`'s and I wanted the approach confirmed first. |
| 2026-07-02 | Me | Own update, no maintainer feedback in this window. | Opened PR #2068 (`33da46a5a`) and @-mentioned davidpichardie with the before and after result. |
| 2026-07-03 | dulmarod | Merged as `f123c84` via Meta's internal import flow, with no change requests. | n/a |

No review comments were left on the PR, because infer's maintainers import approved external PRs through Meta's internal system, so a merge with no public review round is the normal healthy outcome here rather than a sign the PR was skimmed.

### What I learned

**Technical:** Modeling a wrapper for resource-lifetime purposes is really about keeping the wrapped obligation reachable rather than about tracking the wrapper. Because the `JavaResource` attribute already lives on the resource's own allocation, forwarding the value through `get` was sufficient and no release or delegation machinery was needed to get the low-false-positive behavior. That is a much smaller change than I expected going in, and I only found it by asking what the leak logic actually depends on instead of trying to model `Optional` faithfully.

**Process & collaboration:** Copying an existing model exactly is what made this land in a day. The `Boolean` model gave me the DSL idioms, the matcher-row style and the backing-field convention, and had I invented a shape, a reviewer would have had to evaluate both the semantics and the form. Asking the scoping question first was also right even though the answer was that it really is that simple, because the question cost one comment and removed the risk of building a much larger model than the issue needed.

**What I'd do differently:** I would have started the environment work a day earlier and treated it as its own task rather than as setup overhead. Three separate build blockers, covering GMP, ppxlib pinning and session lifetime, each cost hours, and none were visible from the issue. For any OCaml or compiler-scale repo I now budget the build as the first deliverable rather than as a prerequisite, which is exactly what made the follow-up contribution on #1937 much faster.

### References

- Issue #1951, for the confirmed bug and prescribed approach.
- `infer/src/pulse/PulseModelsJava.ml`, for the `Boolean` model as structural template.
- `infer/tests/codetoanalyze/java/pulse/ResourceLeaks.java` and `issues.exp`, for the Pulse Java test conventions.
- `infer.opam.locked`, the authority on toolchain versions.
