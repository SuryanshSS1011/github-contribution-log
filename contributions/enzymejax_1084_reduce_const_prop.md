# Contribution 4: Reduce constant prop in Enzyme-JAX

**Contribution Number:** 4
**Student:** Suryansh Sijwali
**Issue:** https://github.com/EnzymeAD/Enzyme-JAX/issues/1084
**Pull Request:** https://github.com/EnzymeAD/Enzyme-JAX/pull/2524
**Status:** Phase III — PR submitted, awaiting maintainer review

---

## Why I Chose This Issue

Enzyme-JAX is the MLIR/StableHLO front for the Enzyme autodiff framework
(Will Moses' group at MIT) — the bridge from JAX-level programs into
Enzyme's gradient generation over StableHLO. Contributing here is the
closest match to my Causality-Aware-RL / SecureCodeRL research thread of
any candidate I evaluated: ML compilers, autodiff-adjacent transforms,
real production users (JAX → Enzyme is how serious autodiff over XLA
gets done).

Issue #1084 asks for constant propagation on `stablehlo.reduce` — when
the inputs are constants and the body is a pure binary op, the result
can be folded at compile time. The issue body pointed at stablehlo's
reference interpreter implementation as the canonical semantics. The
patch directly fills a small but real gap in Enzyme's default
optimization pipeline.

I picked this after eliminating several other candidates with hard
phantom checks: the OptimizationBarrierOp item from Enzyme-JAX #152's
checklist turned out to already work via the generic batch-op fallback,
several llama.cpp CUDA candidates were already-implemented phantoms, and
uv #4824 turned out to be closed by a prior merged PR. Verifying the
gap on current `main` first is the discipline this issue passed.

---

## Understanding the Issue

### Problem Description

When `stablehlo.reduce` takes splat-constant inputs and the body is a
pure binary op like `stablehlo.and` or `stablehlo.or`, the result is
algebraically determined at compile time. Enzyme's default optimization
pipeline (`--enzyme-hlo-opt`) leaves these reduces in place, missing a
clear simplification.

Example from the issue body:

```
%c_0 = stablehlo.constant dense<true> : tensor<1x140xi1>
%c_7 = stablehlo.constant dense<true> : tensor<i1>
%0 = stablehlo.reduce(%c_0 init: %c_7) applies stablehlo.and
     across dimensions = [0, 1]
     : (tensor<1x140xi1>, tensor<i1>) -> tensor<i1>
return %0
```

This should fold to a single `stablehlo.constant dense<true> : tensor<i1>` + return.

### Expected Behavior

Running the example through `--enzyme-hlo-opt` should produce:

```
%c = stablehlo.constant dense<true> : tensor<i1>
return %c : tensor<i1>
```

### Current Behavior (before patch)

I built `enzymexlamlir-opt` from current `main` and ran the issue's
exact snippet through three pipelines:

- `--canonicalize`: does not fold.
- `--enzyme-hlo-opt`: does not fold.
- `--pass-pipeline="builtin.module(func.func(stablehlo-aggressive-folder))"`:
  **does** fold correctly.

So the upstream `stablehlo-aggressive-folder` pass already implements
the fold, but Enzyme-JAX's default pipeline doesn't invoke it for this
case. The fix is to bring the relevant pattern(s) into Enzyme directly.

### Affected Components

- `src/enzyme_ad/jax/Passes/EnzymeHLOOpt.cpp` — the ~36,000-line file
  holding nearly all of Enzyme's HLO optimization patterns and the
  `EnzymeHLOOptPass::runOnOperation` registration.
- `test/lit_tests/reduce_const_prop.mlir` — new lit test asserting the
  fold on the issue's exact MLIR.

---

## Reproduction Process

### Environment Setup

- Built Enzyme-JAX from source on Penn State's ROAR Collab HPC cluster
  (Slurm + A100/A40 nodes). Bazel 7.7.0 (pinned via `USE_BAZEL_VERSION`),
  gcc 13.2.0, Bazel cache redirected to `/storage/work/sss6371/.cache/bazel`
  to avoid the 16 GB home quota.
- First clean build: ~87 minutes, ~10 GB cache, 256 MB binary.
- Incremental rebuild after touching `EnzymeHLOOpt.cpp`: ~5 minutes.

### Steps to Reproduce

```sh
# In Enzyme-JAX/ on a compute node:
module load gcc/13.2.0
export USE_BAZEL_VERSION=7.7.0
bazel --output_user_root="$HOME_BAZEL_CACHE_DIR" build -c opt :enzymexlamlir-opt

# Save the issue's MLIR to /tmp/probe.mlir, then:
./bazel-bin/enzymexlamlir-opt /tmp/probe.mlir --enzyme-hlo-opt
```

### Reproduction Evidence

Before the patch, `--enzyme-hlo-opt` returns the reduce unchanged.
After the patch, output collapses to `stablehlo.constant dense<true> :
tensor<i1>` plus `return`.

---

## Solution Approach

### Analysis

The maintainer (wsmoses, Enzyme lead) responded to my design-question
comment with a clear direction:

> feel free to import the stablehlo-aggressive-folder pattern of
> relevance here!

Reading the upstream `StablehloAggressiveFolder.cpp`, the fold is
produced by a *cascade* of two patterns + Enzyme's existing
`AndSimplify`/`OrSimplify`:

1. **`LowerBoolSplatConstantsIntoReduceOpRegion`** matches a reduce
   whose inputs are splat constants and whose body is `and`/`or`. It
   replaces uses of the body's block arguments with constants, turning
   the body from `and(%arg0, %arg1)` into `and(const_true, const_true)`.
2. Enzyme's existing `AndSimplify` then folds `and(const_true,
   const_true)` to `const_true`, simplifying the body to a single
   constant.
3. **`FoldReduceOpToConstantInitializer`** matches a reduce whose body
   returns a constant, and replaces the reduce with that constant.

Verifying the third step's dependency on the second: I ran a separate
probe confirming that `--enzyme-hlo-opt` already folds `and(true, true)
→ true`. So the cascade is closed using only the two new patterns +
existing Enzyme machinery.

### Proposed Solution

Lift both `LowerBoolSplatConstantsIntoReduceOpRegion` and
`FoldReduceOpToConstantInitializer` near-verbatim from upstream into
Enzyme's `EnzymeHLOOpt.cpp`, adapting them to Enzyme's
`CheckedOpRewritePattern<OpTy, Child>` CRTP base (replacing upstream's
`FoldOpRewritePattern<OpTy>`) and `matchAndRewriteImpl` signature
(replacing upstream's `matchAndRewrite`).

Register both patterns inline alongside `ReduceToReshape` and the other
existing reduce-related patterns in `EnzymeHLOOptPass::runOnOperation`.

### Implementation Plan

Using UMPIRE framework (adapted):

**Understand:** A specific class of `stablehlo.reduce` (splat-constant
inputs, single binop body) should compile-time-fold to a constant.
Upstream's aggressive folder handles this via two patterns + an
elementwise binop fold; Enzyme's default pipeline doesn't invoke them.

**Match:** Enzyme already has many reduce-related patterns in
`EnzymeHLOOpt.cpp` (e.g. `ReduceToReshape`, `ReducePad`,
`ReduceConcat`), all built on `CheckedOpRewritePattern<stablehlo::ReduceOp,
…>`. The two new patterns slot in beside them with the same boilerplate
shape. Upstream uses `FoldOpRewritePattern` and `matchAndRewrite`; the
adaptation is small.

**Plan:**

1. Post a design-question comment on the issue (asking maintainer
   whether to wire the whole upstream pass or import the patterns
   inline).
2. After wsmoses replied with "import the pattern of relevance," locate
   the exact upstream patterns and analyze the cascade.
3. Verify Enzyme already folds `and(true, true)` and `or(false,
   false)` via existing patterns, so we don't need to also import a
   binop folder.
4. Insert the two pattern structs in `EnzymeHLOOpt.cpp` before
   `struct ReduceToReshape final`, with one-line provenance comments
   pointing at the upstream source.
5. Register both in the inline `patterns.add<…>(context,
   PatternBenefit(65000))` block at the registration site.
6. Add a new lit test `test/lit_tests/reduce_const_prop.mlir` covering
   the issue's exact MLIR, a sibling `or`/`false` case, and a negative
   case (non-constant input must be left untouched).
7. Rebuild Enzyme-JAX (incremental, ~5 min).
8. Run new lit test and a broad set of existing reduce/const/and lit
   tests via `bazel test` to confirm no regressions.
9. Commit and push.

**Implement:** Branch `reduce-const-prop` on
`SuryanshSS1011/Enzyme-JAX`, single commit `103d90b` (post-amend).

**Review:** Self-review checklist before opening PR:

- [x] Pattern provenance is named in source comments and PR body.
- [x] Adaptation to Enzyme's `CheckedOpRewritePattern` base is minimal
  and faithful to upstream semantics.
- [x] Out-of-scope upstream patterns (`FoldReduceOpReducingZeroDims`,
  `FoldReduceOpWithRedundantResults`) intentionally not imported, per
  wsmoses' "of relevance" framing.
- [x] Lit test covers positive, sibling, and negative cases.
- [x] 23 existing lit tests touching reduce/const/and pass via
  `bazel test`.
- [x] Build clean, no new compiler warnings introduced.
- [x] Comments match house style (one line each, focused on
  provenance/why, not on what the code does).

**Evaluate:** Manual probe shows the issue's exact MLIR collapses to
the expected `stablehlo.constant + return`. `bazel test
//test/lit_tests:reduce_const_prop.mlir.test` passes. 23 sibling tests
unaffected.

---

## Testing Strategy

### Unit Tests

Not applicable — Enzyme-JAX uses lit tests, not C++ unit tests, for
pattern verification.

### Lit Tests

New file `test/lit_tests/reduce_const_prop.mlir` with three FileCheck
cases:

- `reduce_and_splat_true` — `reduce(splat<true>, init=true) and across
  [0,1]` must fold to `stablehlo.constant dense<true>`.
- `reduce_or_splat_false` — sibling case for `stablehlo.or`.
- `reduce_and_nonconst` — `reduce(arg, init=true) and across [0,1]`
  with a non-constant `arg` must be left unchanged (negative case).

### Regression Tests

Ran the following 23 existing lit tests via `bazel test` to confirm no
regressions:

`addreduceslicefusion`, `addreduceslicefusion2`, `and_const_prop`,
`and_pad_pad`, `binop_const_lift_computation`,
`binopcomplexconstsimplify`, `broadcastreduce`, `concatreduce`,
`concatreduce2`, `concatreduce3`, `constpadconcat_to_concat`,
`constpropthroughbarrier`, `convert_to_splatted_constants`,
`convertconst`, `elementwise_reduce_slice_fuse`,
`elementwise_reduce_slice_fuse2`, `foldgather`, `foldpad`,
`fullreduce_nocrash`, `gatherconstprop`, `is_finite_const_prop`,
`log_const_prop`, `math_const_prop`. All 23 pass.

### Manual Testing

- Issue's exact MLIR through `--enzyme-hlo-opt` produces the expected
  single-constant output.
- A sibling `or`/`false` case folds identically.
- A reduce with non-constant input is left untouched.
- A small set of existing reduce-related lit tests (3 in initial smoke
  test, 23 in broad regression sweep) all pass.

---

## Implementation Notes

### Code Changes

- **Files modified:**
  - `src/enzyme_ad/jax/Passes/EnzymeHLOOpt.cpp` (+99 / −1)
  - `test/lit_tests/reduce_const_prop.mlir` (+42, new file)
- **Branch:** `reduce-const-prop` on `SuryanshSS1011/Enzyme-JAX`.
- **Commit:** `103d90b` — Import reduce-constant-prop patterns from
  stablehlo aggressive folder. (Closes #1084.)
- **Approach decision — why import the patterns inline rather than
  wire the whole `stablehlo-aggressive-folder` pass:** wsmoses'
  guidance was "import the pattern of relevance," singular. Wiring the
  full pass would broaden the diff considerably (it includes many
  unrelated patterns — log/exp/sqrt/etc. folders) and risks changing
  unrelated pipeline behavior. Inline import keeps the change scoped
  exactly to the issue.
- **Approach decision — why two patterns + existing simplifications,
  not one custom pattern:** The fold structurally happens as a cascade
  (splat-lowering → binop-fold → reduce-fold). Writing a single
  monolithic pattern that does all three steps would duplicate
  AndSimplify/OrSimplify logic and risk drift. Cascading via the
  existing infrastructure is more faithful to MLIR's pattern-rewrite
  model.

---

## Pull Request

**PR Link:** https://github.com/EnzymeAD/Enzyme-JAX/pull/2524

**Maintainer Feedback:** None yet (just submitted).

**Status:** PR open, awaiting CI and maintainer review. wsmoses already
engaged on the issue (provided the design direction), so the PR is
likely to land in his review queue.

---

## Learnings & Reflections

To be filled in after review and merge.

Open observations as of submission:

- **Design-question-first paid off again.** The session has seen
  several maintainer interactions that validated this discipline (uv
  #4711 closed-as-not-planned, uv #7040 already-fixed). Here it
  produced the opposite outcome — clear actionable direction in 67
  minutes — but the underlying principle is the same: ask before
  writing for any non-trivial scope.

- **Verifying via cascade is harder than verifying a single pattern.**
  The probe that showed `--enzyme-hlo-opt` folds `and(true,true) →
  true` was load-bearing for the design — it's why we didn't need to
  also import a binop folder. Catching that early avoided either
  over-scoping (bringing in folders we don't need) or under-scoping
  (importing patterns that wouldn't actually close the cascade).

- **trl's "consistency over correctness" is a useful framing.** Even
  on Enzyme-JAX (which doesn't publish that maxim), it's still right
  to lift patterns near-verbatim from upstream with a provenance
  comment, rather than rewrite them in Enzyme's voice. Future
  maintainers can grep for `LowerBoolSplatConstantsIntoReduceOpRegion`
  and trace it back upstream in one step.

---

## Resources Used

- Issue thread: https://github.com/EnzymeAD/Enzyme-JAX/issues/1084
- Upstream patterns:
  https://github.com/openxla/stablehlo/blob/main/stablehlo/transforms/optimization/StablehloAggressiveFolder.cpp
- Enzyme-JAX `DEVDOCS.md` — Bazel build invocation, lit test layout.
- Enzyme-JAX `EnzymeHLOOpt.cpp` line 3962 (`ReduceToReshape`) — used
  as the in-file template for `CheckedOpRewritePattern` boilerplate.
- Existing lit test `test/lit_tests/and_const_prop.mlir` — used as
  the style template for the new lit test.
