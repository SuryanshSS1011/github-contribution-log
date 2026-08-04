# Constant-propagate `stablehlo.reduce` in Enzyme-JAX

**Student:** Suryansh Sijwali ([@SuryanshSS1011](https://github.com/SuryanshSS1011))
**Project:** [EnzymeAD/Enzyme-JAX](https://github.com/EnzymeAD/Enzyme-JAX) · **My fork:** https://github.com/SuryanshSS1011/Enzyme-JAX
**Issue:** [#1084](https://github.com/EnzymeAD/Enzyme-JAX/issues/1084) · **Pull request:** [#2524](https://github.com/EnzymeAD/Enzyme-JAX/pull/2524) · **Branch:** `reduce-const-prop`
**Status:** Open. Reviewed, blocked on a scope decision about extending the fold to add and mul.

---

## Phase I: Issue Selection

### Why I Chose This Issue

Enzyme-JAX is the MLIR/StableHLO front end for the Enzyme autodiff framework from Will Moses' group at MIT, and it is the bridge from JAX-level programs into Enzyme's gradient generation over StableHLO. Of every candidate I evaluated, this is the closest match to my Causality-Aware-RL and SecureCodeRL research thread, since it covers ML compilers, autodiff-adjacent transforms and real production users. My learning goal was to get hands-on with MLIR's pattern-rewrite model, and specifically with how independent rewrite patterns cascade to produce a fold that no single pattern performs.

Issue #1084 asks for constant propagation on `stablehlo.reduce`, so that when the inputs are constants and the body is a pure binary op, the result is determined at compile time. The issue body points at stablehlo's reference interpreter as the canonical semantics, which gives a clear correctness oracle.

I picked this after eliminating several other candidates with hard phantom checks. The `OptimizationBarrierOp` item from Enzyme-JAX #152's checklist already worked via the generic batch-op fallback, several llama.cpp CUDA candidates were already-implemented phantoms, and uv #4824 turned out to be closed by a prior merged PR.

### Problem Summary

`stablehlo.reduce` over splat-constant inputs with a pure binary body such as `and` or `or` is algebraically determined at compile time, but Enzyme's default `--enzyme-hlo-opt` pipeline leaves it in the IR. Upstream StableHLO's `stablehlo-aggressive-folder` pass already folds exactly this shape, so Enzyme is leaving a known and free simplification on the table in the pipeline that every JAX-through-Enzyme program runs. It matters because an unfolded reduce blocks downstream shape and constant reasoning rather than only costing one instruction. I chose it because it is a bounded compiler-optimization task in the ML-compiler space I want to work in, with an upstream implementation available as a correctness reference.

### Issue Vetting

Open and unassigned. Before writing code I built `enzymexlamlir-opt` from `main` and confirmed the fold does not happen under `--enzyme-hlo-opt`, then posted a design question rather than a claim comment, since Enzyme-JAX has no assignment culture and the norm is to open a PR first.

### Where It Lives

- `src/enzyme_ad/jax/Passes/EnzymeHLOOpt.cpp`, the roughly 36,000-line file holding nearly all of Enzyme's HLO optimization patterns and `EnzymeHLOOptPass::runOnOperation`.
- `src/enzyme_ad/jax/TransformOps/TransformOps.td` and `src/enzyme_ad/jax/Integrations/c/EnzymeXLA.cpp`, for tablegen and pipeline registration, added on maintainer request.
- `test/lit_tests/reduce_const_prop.mlir`, a new lit test.

### Maintainer Context

I opened with a design question on the issue on [2026-06-04](https://github.com/EnzymeAD/Enzyme-JAX/issues/1084), asking whether to wire the whole upstream pass in or import the relevant patterns inline, with the probe results attached. `wsmoses`, the Enzyme lead, replied in 67 minutes:

> feel free to import the stablehlo-aggressive-folder pattern of relevance here!

That single sentence set the scope as patterns rather than the whole pass, and "of relevance" rather than everything.

### Acceptance Criteria

1. The issue's exact MLIR folds to `stablehlo.constant dense<true> : tensor<i1>` plus `return` under `--enzyme-hlo-opt`.
2. The sibling `or` and `false` case folds identically.
3. A reduce with a non-constant input is left untouched.
4. Patterns are registered where Enzyme's other reduce patterns are, including the tablegen path.
5. There is no regression in the existing reduce, const and and lit tests.

---

## Phase II: Reproduce & Plan

### Environment Setup

I used Enzyme-JAX's `DEVDOCS.md` for the Bazel invocation and lit-test layout, and read the CI workflow to match the build configuration.

- I built from source on Penn State's ROAR Collab HPC cluster, which runs Slurm with A100 and A40 nodes, because the build is far too heavy for my laptop.
- **Challenge (home-directory quota):** ROAR home is capped at 16 GB and the Bazel cache alone runs about 10 GB. I resolved it by redirecting the Bazel output root to `/storage/work/sss6371/.cache/bazel` on scratch via `--output_user_root`.
- **Challenge (toolchain pinning):** Bazel 7.7.0 is pinned through `USE_BAZEL_VERSION` and gcc 13.2.0 through `module load gcc/13.2.0`, and without the pins the build picks up the cluster default and fails.
- **Cost:** the first clean build took about 87 minutes, produced about 10 GB of cache and a 256 MB binary. An incremental rebuild after touching `EnzymeHLOOpt.cpp` takes about 5 minutes, and that ratio is what made this issue tractable at all.

### Steps to Reproduce

1. On a ROAR compute node, in `Enzyme-JAX/`:
   ```sh
   module load gcc/13.2.0
   export USE_BAZEL_VERSION=7.7.0
   bazel --output_user_root="$HOME_BAZEL_CACHE_DIR" build -c opt :enzymexlamlir-opt
   ```
2. Save the issue's MLIR as `/tmp/probe.mlir`:
   ```mlir
   module {
     func.func @main() -> tensor<i1> {
       %c_0 = stablehlo.constant dense<true> : tensor<1x140xi1>
       %c_7 = stablehlo.constant dense<true> : tensor<i1>
       %0 = stablehlo.reduce(%c_0 init: %c_7) applies stablehlo.and
            across dimensions = [0, 1]
            : (tensor<1x140xi1>, tensor<i1>) -> tensor<i1>
       return %0 : tensor<i1>
     }
   }
   ```
3. `./bazel-bin/enzymexlamlir-opt /tmp/probe.mlir --enzyme-hlo-opt`
4. For contrast, run the same file through `--canonicalize` and through
   `--pass-pipeline="builtin.module(func.func(stablehlo-aggressive-folder))"`.

### Expected vs. Actual

| Pipeline | Result |
|---|---|
| `--canonicalize` | reduce unchanged |
| `--enzyme-hlo-opt` | reduce unchanged, which is the gap |
| `stablehlo-aggressive-folder` | folds correctly to `dense<true>` |

Expected from `--enzyme-hlo-opt`:

```mlir
%c = stablehlo.constant dense<true> : tensor<i1>
return %c : tensor<i1>
```

The three-way comparison is what localized the problem, because it shows the fold is implemented upstream and simply never reached from Enzyme's default pipeline.

### Root Cause

Enzyme's `EnzymeHLOOptPass` does not include the upstream folder's reduce patterns. Reading `StablehloAggressiveFolder.cpp` showed the fold is not one pattern but a cascade of three steps:

1. `LowerBoolSplatConstantsIntoReduceOpRegion` matches a reduce with splat-constant inputs and an `and` or `or` body, replacing uses of the body's block arguments with constants, so the body turns from `and(%arg0, %arg1)` into `and(const_true, const_true)`.
2. A binop fold collapses `and(true, true)` to `true`, leaving the body as a single constant.
3. `FoldReduceOpToConstantInitializer` matches a reduce whose body returns a constant and replaces the reduce with it.

Step 2 is load-bearing and Enzyme already provides it. I verified that separately with a dedicated probe, which showed that `--enzyme-hlo-opt` does already fold `and(true, true)` to `true` via the existing `AndSimplify` and `OrSimplify`. So importing patterns 1 and 3 closes the cascade without importing a binop folder.

### Plan (UMPIRE)

**Understand:** A specific class of `stablehlo.reduce` with splat-constant inputs and a single binop body should compile-time-fold. Upstream implements it as a cascade, and Enzyme's default pipeline never invokes the two ends of that cascade.

**Match:** Enzyme already has many reduce patterns in `EnzymeHLOOpt.cpp`, including `ReduceToReshape`, `ReducePad` and `ReduceConcat`, all built on `CheckedOpRewritePattern<stablehlo::ReduceOp, …>`. The imports slot in beside them, and the only adaptation is turning upstream's `FoldOpRewritePattern<OpTy>` and `matchAndRewrite` into Enzyme's CRTP base and `matchAndRewriteImpl`. `ReduceToReshape` at `EnzymeHLOOpt.cpp:3962` served as the in-file boilerplate template, and `test/lit_tests/and_const_prop.mlir` as the test template.

**Plan:**
1. Post the design question on the issue before writing code.
2. Locate the exact upstream patterns and analyze the cascade.
3. Probe that Enzyme already folds `and(true,true)` and `or(false,false)`, so no binop folder is needed.
4. Insert the two pattern structs before `struct ReduceToReshape final`, each with a one-line provenance comment naming the upstream source.
5. Register both in the inline `patterns.add<…>(context, PatternBenefit(65000))` block.
6. Register in tablegen (`TransformOps.td`) and the default pipeline, per the dev docs.
7. Add `test/lit_tests/reduce_const_prop.mlir` with positive, sibling and negative cases.
8. Rebuild, taking about 5 minutes incrementally, and run the new test plus a broad sweep of existing reduce, const and and lit tests.

**Implement:** Branch `reduce-const-prop`, with two functional commits described in Phase III.

**Review:** My self-checklist covered provenance named in source comments and the PR body, adaptation to `CheckedOpRewritePattern` staying faithful to upstream semantics, out-of-scope upstream patterns (`FoldReduceOpReducingZeroDims` and `FoldReduceOpWithRedundantResults`) deliberately not imported per the "of relevance" framing, a lit test covering positive, sibling and negative cases, 23 existing lit tests passing, no new compiler warnings, and comments kept to one line each focused on provenance rather than restating the code.

**Evaluate:** The issue's exact MLIR collapses to `stablehlo.constant` plus `return`, `bazel test //test/lit_tests:reduce_const_prop.mlir.test` passes, and 23 sibling tests are unaffected.

### Edge Cases Considered

- **Non-constant input** must be left alone, which is encoded as a negative lit case.
- **Idempotency of the body op:** This is the one that turned into an open design question. The upstream pattern only handles `and` and `or`, and the cascade only terminates correctly for those, because they are idempotent and `f(x, x) = f(x, …, x)` regardless of element count. For `add` and `mul` the result depends on the number of reduced elements, so `reduce(splat<2>, init=0){add}` over 4 elements must fold to `8` and not `2`. Extending to add/mul therefore needs a new pattern computing the closed form (`init + N*x` and `init * x^N`) with integer-overflow and float-precision handling, rather than a wider import.

---

## Phase III: Build

### Implementation Progress

| Commit | Date | Message |
|---|---|---|
| `d7503076e` | 2026-06-04 | Import reduce-constant-prop patterns from stablehlo aggressive folder |
| `449c37c45` | 2026-06-04 | Register reduce-constant-prop patterns in tablegen and default pipeline |
| `709726e8e` | 2026-07-23 | Apply clang-format |
| `0a9fae107`, `81c68c9a7` | 2026-07-23, 2026-07-29 | Merge `main` to keep the branch current while the design question is open |

**Files modified:**

| File | Δ |
|---|---|
| `src/enzyme_ad/jax/Passes/EnzymeHLOOpt.cpp` | +100 / −2 |
| `src/enzyme_ad/jax/TransformOps/TransformOps.td` | +8 |
| `src/enzyme_ad/jax/Integrations/c/EnzymeXLA.cpp` | +2 |
| `test/lit_tests/reduce_const_prop.mlir` | +44 (new) |

**Design decision (import the patterns inline rather than wire in the whole `stablehlo-aggressive-folder` pass):** wsmoses' guidance was to import the pattern of relevance, singular. Wiring the full pass would broaden the diff considerably, since it carries many unrelated log, exp and sqrt folders, and it risks changing unrelated pipeline behavior.

**Design decision (two patterns plus existing simplifications, not one monolithic pattern):** the fold structurally is a cascade. A single pattern doing all three steps would duplicate `AndSimplify` and `OrSimplify` logic and drift from it over time, so cascading through the existing infrastructure is more faithful to MLIR's rewrite model.

### Challenges Faced

The hard part was verifying a cascade rather than a single pattern. Importing patterns 1 and 3 only works if Enzyme independently supplies step 2, and nothing in the upstream file says so. Reading the code was not conclusive, so I built a separate probe that ran `and(true, true)` through `--enzyme-hlo-opt` and confirmed the fold. That one measurement was load-bearing for the whole design, because it is why the PR imports two patterns instead of three or more, and it is what let me answer wsmoses' scope framing precisely.

The second challenge is unresolved and is why this PR is still open, and it is the idempotency edge case above. Rather than guess at semantics for `add` and `mul`, I posted the analysis and asked for a decision.

### Testing

- **New lit test** `test/lit_tests/reduce_const_prop.mlir` with three FileCheck cases following `and_const_prop.mlir`'s style:
  - `reduce_and_splat_true`, where the issue's exact IR must fold to `dense<true>`.
  - `reduce_or_splat_false`, the sibling case for `stablehlo.or`.
  - `reduce_and_nonconst`, where a non-constant input must be left unchanged.
- **Regression sweep** of 23 existing lit tests via `bazel test`, all passing: `addreduceslicefusion`, `addreduceslicefusion2`, `and_const_prop`, `and_pad_pad`, `binop_const_lift_computation`, `binopcomplexconstsimplify`, `broadcastreduce`, `concatreduce`, `concatreduce2`, `concatreduce3`, `constpadconcat_to_concat`, `constpropthroughbarrier`, `convert_to_splatted_constants`, `convertconst`, `elementwise_reduce_slice_fuse`, `elementwise_reduce_slice_fuse2`, `foldgather`, `foldpad`, `fullreduce_nocrash`, `gatherconstprop`, `is_finite_const_prop`, `log_const_prop`, `math_const_prop`.
- **Manual:** the issue's exact MLIR produces the expected single-constant output, the `or` and `false` sibling folds identically, and a non-constant reduce is untouched.
- Enzyme-JAX uses lit tests rather than C++ unit tests for pattern verification, so there is no unit-test layer to add to.

---

## Phase IV: Submit & Iterate

### Pull Request

**[EnzymeAD/Enzyme-JAX#2524](https://github.com/EnzymeAD/Enzyme-JAX/pull/2524)**, opened 2026-06-04 against `EnzymeAD/Enzyme-JAX:main`, referencing `Closes #1084`. The body leads with the missing fold and the three-pipeline probe that localized it, then the cascade analysis, then the two imported patterns with upstream provenance, then the lit-test cases. wsmoses was already engaged on the issue, so the PR landed directly in his queue and he reviewed three minutes after submission.

Codecov reports all modified and coverable lines covered.

### Maintainer Feedback Log

| Date | From | Feedback | My response |
|---|---|---|---|
| 2026-06-04 | wsmoses (on the issue) | "feel free to import the stablehlo-aggressive-folder pattern of relevance here!" | Scoped to two patterns rather than the whole pass, and opened the PR the same day. |
| 2026-06-04 | wsmoses (review) | "make sure to also add this to tablegen [see the dev docs]" | Done the same day in `449c37c45`, covering `TransformOps.td` and `EnzymeXLA.cpp` registration. |
| 2026-06-04 | wsmoses (review) | "can we extend this to also support add/mul, including non constants? [and appropriate tests]" | Answered with the idempotency analysis, that the upstream pattern only handles idempotent body ops and the cascade only terminates for those, and that add/mul need a closed-form pattern (`init + N*x` and `init * x^N`) with overflow and float-precision handling. Asked for a decision on that scope before building it. |
| 2026-06-28 | wsmoses | "On travel right now adding some other reviewers", in reply to my bump. | Kept the branch rebased on `main` with merges on 2026-07-23 and 2026-07-29 and clang-formatted it, and I am still awaiting the add/mul decision. |

### Learnings & Reflections

**Technical:** MLIR folds are frequently emergent rather than implemented, since no single pattern in the upstream file performs this fold and the fold I wanted is what three independent rewrites produce when composed. That changes how you verify a change, because the useful experiment was not whether my pattern matches but whether the pipeline already supplies the middle of the cascade. The second technical lesson is that idempotency is the hidden precondition of the whole design, since `and` and `or` fold without knowing the element count while `add` and `mul` cannot. That distinction is invisible until you try to generalize, and it is exactly the kind of thing worth surfacing to a maintainer rather than guessing.

**Process & collaboration:** Asking a design question before writing code got actionable direction in 67 minutes and shaped the scope of the entire PR. The same discipline is why the PR is currently parked rather than wrong. When wsmoses asked for add/mul, the easy move was to implement something and let review sort it out, but instead I wrote up why the requested extension is a different pattern with real numeric hazards and asked. That is the right call even though it has cost weeks of latency, because a fast wrong fold in a compiler pipeline is much worse than a slow correct one. What I would add is that I should have bumped sooner than three weeks the first time.

**What I'd do differently:** I would fold the tablegen registration into the first commit. The dev docs cover it, and it was flagged in review three minutes after I opened, so a reviewable-quality PR should not need a maintainer to point at documentation I could have read first. I would also lead the PR body with the idempotency limitation rather than leave it for the review thread, since it is the single most important thing a reviewer needs to know about how far this generalizes.

### Resources Used

- Issue thread: https://github.com/EnzymeAD/Enzyme-JAX/issues/1084
- Upstream patterns: [`StablehloAggressiveFolder.cpp`](https://github.com/openxla/stablehlo/blob/main/stablehlo/transforms/optimization/StablehloAggressiveFolder.cpp)
- Enzyme-JAX `DEVDOCS.md`, for the Bazel invocation, lit-test layout and pattern registration.
- `EnzymeHLOOpt.cpp:3962` (`ReduceToReshape`), the in-file template for `CheckedOpRewritePattern` boilerplate.
- `test/lit_tests/and_const_prop.mlir`, the style template for the new lit test.
