# Model C variadic out-parameters to fix a false `MEMORY_LEAK_C`

**Project:** [facebook/infer](https://github.com/facebook/infer) · **My fork:** https://github.com/SuryanshSS1011/infer
**Issue:** [#1937](https://github.com/facebook/infer/issues/1937) · **Pull request:** [#2078](https://github.com/facebook/infer/pull/2078) · **Branch:** `pulse-c-va-arg`
**Status:** Open, awaiting review. Scoped to the leak only, so it does not carry `Fixes #1937`.

---

## Phase I: Issue Selection

### Why I Chose This Issue

Infer is Meta's static analyzer and Pulse is its memory-and-resource lifetime checker. After landing a Pulse model fix in #1951 I wanted a meatier contribution in the same area, and the appeal here was precisely its scale, because it turned out to be rooted in a documented `TODO` inside Pulse's interprocedural call handling rather than a shallow model gap, so it touches core call dispatch rather than a lookup table. My learning goal was how an abstract interpreter passes information across a call boundary, and what it structurally cannot pass.

It is a genuine false positive that still reproduces on current `main`, filed by a real user, the mariadb-connector-c author, against real code.

### Problem Summary

Pulse falsely reports `MEMORY_LEAK_C` when memory is allocated and stored through a variadic out-parameter, because the `va_arg` result is disconnected from the caller's argument, so `*p = malloc(...)` looks like a store into a fresh location that immediately leaks. A plain non-variadic `char** p` out-parameter is handled correctly, so the defect is specific to the variadic path. It matters because false positives are what get a static analyzer switched off, and this pattern is idiomatic C in exactly the kind of low-level library that most benefits from Pulse, so the report is unactionable. I chose it because it is a real false positive with a live reporter, and because the root cause sits deep enough to be worth the effort.

### Issue Vetting

Open, unassigned, with no in-flight PR. Following my rule about old infer issues, where pre-2024 false positives are frequently already fixed but not closed, I built infer with the clang frontend enabled and confirmed the false positive still reproduces on current `main` before doing anything else.

I posted my root-cause analysis to the issue on 2026-07-12 rather than a claim comment. The reporter `grooverdan`, the mariadb-connector-c author, and another commenter both responded supportively within a day, which confirmed the direction was worth building.

### Where It Lives

- `infer/src/pulse/PulseCallOperations.ml`, holding `trim_actuals_if_var_arg`, which drops the extra variadic actuals at the interprocedural boundary and carries its own `TODO` noting this is unimplemented.
- `infer/src/pulse/PulseModelsC.ml`, where `va_start`, `va_arg` and `va_end` models belong.
- `infer/src/pulse/PulseSpecialization.ml` and `infer/src/IR/Specialization.ml(i)`, the specialization mechanism.
- `infer/src/pulse/Pulse.ml`, holding `prepare_args_if_hack_variadic`, the precedent for variadic handling in another language.
- `infer/tests/codetoanalyze/c/pulse/var_arg.c` and `issues.exp`, the C Pulse regression corpus.

### Acceptance Criteria

1. The minimal repro and the common malloc-per-out-parameter shapes stop reporting `MEMORY_LEAK_C`.
2. A genuine leak, meaning an out-parameter that is never freed, still reports, so the fix must not weaken real detection.
3. There is no regression in the C Pulse corpus, including existing value-returning variadic tests.
4. Anything not fixed is stated explicitly rather than implied.

---

## Phase II: Reproduce & Plan

### Environment Setup

I reused the infer build environment established for #1951 on Penn State's ROAR Collab cluster, which is the single biggest reason this contribution was feasible at all, with one addition.

- **Challenge (the clang frontend is required here):** #1951 was a Java-model change while this one analyses C, so the build needs infer's clang frontend enabled, which is a substantially longer build than the default.
- Carried over from #1951 are GMP built from source since I have no root for system packages, `depext=false`, strict pinning to `infer.opam.locked` and particularly `ppxlib` where a newer version breaks the ppx-generated parsetree, and detached builds that outlive an SSH session.

### Steps to Reproduce

1. Build infer from current `main` with the clang frontend enabled.
2. Save this minimal C file:
   ```c
   void set_ptr(int n, ...) {
     va_list a; va_start(a, n);
     char** p = va_arg(a, char**);
     *p = malloc(4);
     va_end(a);
   }
   void caller() { char* v; set_ptr(1, &v); free(v); }
   ```
3. Run `infer run --pulse-only -- clang -c <file>.c`.
4. For contrast, run the same thing with a non-variadic out-parameter:
   ```c
   void set(char** p) { *p = malloc(4); }
   void caller() { char* v; set(&v); free(v); }
   ```

### Expected vs. Actual

**Actual:** the variadic version reports `MEMORY_LEAK_C` on `caller` even though `v` is freed, while the non-variadic version is clean. The issue's original `ma_multi_malloc` from mariadb-connector-c reports as well.

**Expected:** both are clean, because the allocation escapes through the caller's `v` and is freed.

I also verified the other direction, that a genuine leak with an out-parameter never freed still reports, because the fix must not buy a cleared false positive with a lost true positive.

### Root Cause

Two mechanisms compound:

1. `va_arg` lowers to an unmodeled `__builtin_va_arg`, which Pulse havocs, so the returned pointer is a fresh unconstrained value disconnected from the caller's `&v`.
2. `trim_actuals_if_var_arg` in `PulseCallOperations.ml` drops the extra variadic actuals at the interprocedural boundary, and its own comment documents this as unimplemented with "currently there is no support for handling the remaining arguments."

With the actuals dropped there is nothing left to reconnect the `va_arg` read to, so `*p = malloc(...)` writes into a location nothing else can reach, and Pulse correctly concludes from its own state that the allocation leaks.

The contrast with the non-variadic case is the key. There, Pulse binds the callee formal `p` to the caller actual `&v` in `materialize_pre_for_parameters`, so the callee's post-state, which says `*p` now points at malloc'd memory and is initialized, is reflected back onto the caller's real `v`. A C variadic has no formal parameter for the extras, and that is the structural difference that makes this hard.

### Plan (UMPIRE)

**Understand:** The extra actuals are discarded at the boundary and `va_arg` is havoc'd, so the allocation lands somewhere unreachable.

**Match:** Two precedents mattered. `Pulse.ml`'s `prepare_args_if_hack_variadic` shows infer already solves this shape for Hack, and Pulse's specialization mechanism already re-analyzes a callee with call-site context injected, which is how aliasing and dynamic types are handled. Rather than invent new machinery, the fix rides that.

**Plan:**
1. Add a `variadic_actuals` field to `Specialization.Pulse.t` to carry the caller's extra actuals.
2. Have `PulseSpecialization.apply` seed them into a well-known global bridge array in the callee's initial state during the specialized re-analysis.
3. Model `va_start`, `va_arg` and `va_end` in `PulseModelsC.ml`, with `va_arg` reading successive elements of that array.
4. Trigger the specialization from `PulseCallOperations.ml` when an `is_clang_variadic` callee is called with extra actuals.
5. Add regression tests for the cleared false positives, a still-reported genuine leak, and the known limitations.

**Implement:** Branch `pulse-c-va-arg`.

**Review:** My self-checklist was that real leak detection is unweakened, that existing value-returning variadic tests are unchanged, that the `issues.exp` diff is reviewed line by line, that `ocamlformat` is clean at the repo-pinned version, and that the PR body states both limitations explicitly and omits `Fixes #1937`.

**Evaluate:** Run the whole C Pulse regression suite, plus the specific before and after on the repro shapes.

### Edge Cases Considered

- **Value-returning variadics must be unaffected:** A summing `sum(int n, ...)` reads `int`s rather than out-parameter pointers. My first version perturbed an existing true positive there, which the regression suite caught, and I resolved it by type-gating the `va_arg` read to pointer results so value-returning variadics are left completely alone.
- **Genuine leaks must still report**, so an out-parameter that is never freed is kept as a positive test.
- **Loop shapes:** Straight-line, two-argument and fixed-count-loop out-parameter patterns are handled, and the one-block-split case is recorded as a known limitation rather than quietly failing.
- **The issue's own `ma_multi_malloc`** is a harder shape and still reports, which the Scope section covers.

---

## Phase III: Build

### Implementation Progress

| Commit | Date | Message |
|---|---|---|
| `ae51630d7` | 2026-07-13 | [pulse] model C variadic out-parameters to fix false `MEMORY_LEAK_C` |

**Files modified (7 files, +223 / −9):**

| File | Δ |
|---|---|
| `infer/tests/codetoanalyze/c/pulse/var_arg.c` | +79 |
| `infer/src/pulse/PulseModelsC.ml` | +55 |
| `infer/src/pulse/PulseSpecialization.ml` | +37 / −1 |
| `infer/src/pulse/PulseCallOperations.ml` | +22 / −1 |
| `infer/src/IR/Specialization.ml` | +18 / −6 |
| `infer/src/IR/Specialization.mli` | +7 / −1 |
| `infer/tests/codetoanalyze/c/pulse/issues.exp` | +5 |

In detail, there is a new `VariadicActuals` submodule and `variadic_actuals` field on the Pulse specialization, a `seed_variadic_actuals` that writes each caller actual into `__infer_va_args_global[k]` in the callee's initial state, `va_start`, `va_arg` and `va_end` models plus an `eval_read_global` helper with `va_arg` type-gated to pointer reads, and the call-site trigger for `is_clang_variadic` callees with extra actuals.

### Challenges Faced

**Pushing past two architectural dead-end conclusions:** The fix looked impossible twice, because the obvious approaches of binding through the pre-condition or rooting a heap path on a callee formal genuinely do not work for C variadics, since there is no formal parameter for the extras. Both times the honest read of the code was that this cannot be done from here. The unlock was recognizing that Pulse's specialization mechanism is a third path that sidesteps the constraint, because it re-analyzes the callee with call-site context injected, which is enough to make the allocation escape into something stable and reachable even though it is not the caller's actual variable.

**A regression that only the suite caught:** The first version weakened an unrelated true positive on a value-returning variadic, `sum(int n, ...)`. Nothing about the change looked like it should touch that path, and I would not have found it by reasoning. Type-gating the `va_arg` read to pointer results resolved it cleanly, and the episode is the reason I trust the corpus more than my own model of what a change affects.

**Proving what could not be done:** After the leak fix worked, I prototyped the natural extension for the `PULSE_UNINITIALIZED_VALUE` side, which was to keep the extra actuals, add a synthetic formal for each and bind them at the boundary, then built it and ran it. It does not work, because the binding is materialized against the callee's summary precondition and a formal invented only at the call site is not in it, so nothing binds. The global-bridge path has the same limitation, since a global binds only to itself in the caller rather than to `&v`. Building the rejected approach and reporting the negative result is what let me scope the PR honestly rather than promising something I could not deliver.

### Testing

- **New regression tests** in `codetoanalyze/c/pulse/var_arg.c`, following the corpus's `FP_` and `FN_` prefix convention that encodes expected verdicts in the test name:
  - single, two-argument and looped out-parameter patterns, where the leak is now cleared
  - a genuine unfreed out-parameter, where the leak is still reported
  - the one-block-split case, marked as a known limitation rather than silently omitted
  - the remaining `PULSE_UNINITIALIZED_VALUE` cases marked `FP_` for follow-up
- `issues.exp` was regenerated, the full C Pulse regression suite passes with no other changes, and the pre-existing `sum` and NPE entries are unchanged, which is the evidence that the type gate does what it claims.
- `ocamlformat` is clean at the repo-pinned version.
- **CI:** one red job, the macOS OCaml build. Its failures are in `build_systems/gradle` and `build_systems/merge-capture`, where a worker terminates on a signal and a buck/javac capture step fails, rather than in the C Pulse tests this PR touches.

---

## Phase IV: Submit & Iterate

### Pull Request

**[facebook/infer#2078](https://github.com/facebook/infer/pull/2078)**, opened 2026-07-13 against `facebook/infer:main`. The body states the false positive, the two-mechanism root cause, the specialization-based fix, and both limitations explicitly. My CLA was already signed.

It deliberately omits a `Fixes #1937` keyword so the issue stays open for its remaining symptoms, which is a scope decision I also announced on the issue thread the same day so nobody would expect a full close:

> Just to set expectations, PR #2078 clears the `MEMORY_LEAK_C` false positive but intentionally leaves the `PULSE_UNINITIALIZED_VALUE` ones open.

### Scope and Known Limitations

1. **The companion `PULSE_UNINITIALIZED_VALUE` false positives are not addressed:** Clearing them requires reflecting the callee's write onto the caller's real variable, which needs true caller-actual binding, and I confirmed the specialization mechanism cannot carry that. The actuals at the call site are opaque abstract values rather than heap paths, `apply` runs in the callee frame without access to caller values, and specialization keys must be stable heap paths. That is precisely why the leak is fixable through this mechanism, since it only needs the allocation to escape somewhere reachable, and the uninitialized-value false positive is not.
2. **The issue's exact `ma_multi_malloc` still reports:** It does a single `malloc(tot_length)` and distributes pointers into that one block across a second `va_start` pass with `*ptr = res; res += length`, with sizes computed in the first pass. That aliasing-through-arithmetic shape, and the two-pass counting and assigning idiom, are each harder than the common malloc-per-out-parameter pattern this PR handles.

### Maintainer Feedback Log

| Date | From | Feedback | My response |
|---|---|---|---|
| 2026-07-12 | Me | Own update, no maintainer feedback in this window. | Posted the root-cause analysis to issue #1937, covering the `__builtin_va_arg` havoc plus the `trim_actuals_if_var_arg` TODO, rather than a claim comment. |
| 2026-07-12 | grooverdan (reporter, mariadb-connector-c author) | Supportive, saying that force-resolving variadic names "conceptually seems to put in a model where all other static analysis functions can cleanly identify faults (and non-faults) in an equivalent way to standard args." | Confirmed the direction was worth building, so I proceeded to the PR. |
| 2026-07-13 | nilokmiss-max | "The proposed direction for modeling C variadics in Pulse sounds reasonable [...] I look forward to reviewing the PR and the regression tests." | n/a |
| 2026-07-13 | Me | Own update, no maintainer feedback in this window. | Opened PR #2078 and posted the scope note on the issue so the partial fix was not mistaken for a full one. |
| 2026-07-30 | Me | Own update, no maintainer feedback in this window. | Posted a full design write-up on the issue for the uninitialized-value side, covering where both false positives come from, why #2078 clears one and not the other, the prototype I built and why it failed since the binding materializes against the callee's summary precondition so a call-site-invented formal never binds, and the direction that looks right, which is synthetic formals for C variadic extras in the frontend with `va_arg` lowering to read them. I ended with a direct question about whether that is welcome as an external contribution or is a frontend change they would rather do internally, and I am awaiting the call. |

There are no review comments yet on the PR itself.

### Learnings & Reflections

**Technical:** The most useful thing I learned is why the two false positives have different difficulty, which only becomes clear once you can state what crosses a call boundary. The leak needs the allocation to escape into something stable and reachable, so a disconnected but stable cell suffices, while the uninitialized-value false positive needs the callee's write reflected onto the caller's actual variable, which requires a real formal-to-actual binding, and C variadics have no formal to bind. Same symptom, same root cause, structurally different fixability. Getting to that statement is what made the scope decision obvious instead of arbitrary.

**Process & collaboration:** Knowing what you cannot do is as valuable as the fix, but only if you prove it rather than assume it, so I built the rejected approach, ran it and reported the negative result. That is what let me scope the PR honestly, omit the `Fixes` keyword and mark the remaining cases `FP_` in the corpus instead of over-promising. Asking the frontend-versus-external question before building the next phase is the same discipline applied forward, because a frontend change to C variadic lowering is exactly the kind of thing a maintainer team may want to own, and finding that out costs one comment versus weeks of work.

**What I'd do differently:** I would run the full regression suite earlier, before the design felt finished, because the value-returning-variadic regression was invisible to reasoning and obvious to the corpus, and finding it earlier would have shaped the type gate as part of the design rather than as a repair. I would also have posted the uninitialized-value design question at the same time as the PR rather than seventeen days later, since the PR's scope note said what I was not doing but not what I proposed doing about it, and that gap is latency I created.

### Resources Used

- Issue #1937 and its `ma_multi_malloc` reproduction from mariadb-connector-c.
- `Pulse.ml`, for `prepare_args_if_hack_variadic`, the Hack variadic precedent.
- `PulseSpecialization.ml`, `IR/Specialization.ml` and `PulseInterproc.ml`, the specialization mechanism.
- `PulseCallOperations.ml`, for `trim_actuals_if_var_arg` and its documented TODO.
- `materialize_pre_for_parameters`, for how normal formal-to-actual binding works and therefore what variadics lack.
