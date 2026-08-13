# Simplify `real`, `imag`, and `abs` of a complex `exp`

**Project:** [EnzymeAD/Enzyme-JAX](https://github.com/EnzymeAD/Enzyme-JAX)
**Issue:** [#1990](https://github.com/EnzymeAD/Enzyme-JAX/issues/1990) · **Pull request:** [#2635](https://github.com/EnzymeAD/Enzyme-JAX/pull/2635) · **Branch:** `feat/1990-real-imag-abs-exp`
**Status:** Open. One review round addressed the same day, awaiting a second look.

---

## The issue

### Why I picked it

This is my second contribution to Enzyme-JAX after the reduce-constant-prop patterns in #1084, and it was chosen deliberately as a second one. I already knew the file, the `CheckedOpRewritePattern` boilerplate, the lit-test conventions and the build, so the marginal cost was low and I could spend the effort on the part that is actually interesting, which is the cost model. My learning goal was writing algebraic rewrites where the rewrite is only a win under conditions the pattern has to check itself.

It was filed by `avik-pal`, a maintainer, as a terse note to self, with three lines of math, no prose, the label `complex`, unassigned, and zero comments since 2026-01-27. Maintainer-filed with an unambiguous spec is about as low-risk as a compiler-optimization task gets.

### What was wrong

Taking `real`, `imag` or `abs` of a complex `exp` leaves the whole computation in the complex domain, when the standard identities let it be done in real arithmetic:

```
real(exp(x)) = exp(real(x)) * cos(imag(x))
imag(exp(x)) = exp(real(x)) * sin(imag(x))
abs(exp(x))  = exp(real(x))
```

For `x = a + bi`, `exp(x) = exp(a)(cos b + i sin b)`, so `|exp(x)| = exp(a)`. It matters because `abs(exp(x))` in particular collapses from a complex transcendental plus a magnitude computation to a single real `exp`, and complex ops are substantially more expensive than their real counterparts in the generated code. I chose it because the specification is exact, the templates exist in-repo, and the one genuinely interesting decision, which is when the rewrite is not a win, is a real engineering judgment rather than transcription.

### Before I started

Open, unassigned, zero comments, and no in-flight PR. I deliberately did not post a claim comment, and I checked the repo's norms before deciding, since recent merges are all COLLABORATOR or MEMBER accounts (`Pangoraw`, `wsmoses`, `luraess`, `ImanHosseini`) plus `enzymead-bot`, and there is no "I'd like to work on this" culture because contributors open PRs directly. A claim comment on a six-month-old issue would have added latency and nothing else. The one open question I had, which was C++ versus tablegen since PR #2568 migrated some patterns, was better asked on the PR with code attached than preemptively on the issue.

### Where it lives, verified against `main` before writing anything

In `src/enzyme_ad/jax/Passes/EnzymeHLOOpt.cpp`, which is 36,635 lines:

- **`RealOpCanon` at :15054** and **`ImagOpCanon` at :15082**, the exact templates to mirror. Both use `CheckedOpRewritePattern<stablehlo::XOp, XOpCanon>` plus `matchAndRewriteImpl`, and both already match on `op.getOperand().getDefiningOp<stablehlo::ComplexOp>()`, which is the same `getDefiningOp<...>()` idiom needed to match `exp` as the producer.
- **`AbsPositiveSimplify` at :22229**, the closest abs precedent, and the first place to read for how neighbouring patterns handle multi-use.
- **The registration list**, with `ImagOpCanon` at :36487 and `RealOpCanon` at :36490.
- **Helpers available**, being `guaranteedPurelyRealResult` and `guaranteedPurelyImagResult` at :15071, :15095, :26168, :26616 and :35474 to :35480.
- **Gap confirmed**, since `stablehlo::ExpOp` appears at :7655, :7664, :7698, :7707 (binop simplifications), :27754 and :36225 (`UnaryConstProp`), and none touches the real/imag-of-exp case.

### What counts as done

1. `real`, `imag` and `abs` of a complex `exp` are rewritten into the real-arithmetic forms above.
2. Non-complex `real` and `imag` still go to `RealOpCanon` and `ImagOpCanon`, and `abs` on a real value is untouched.
3. The rewrite does not fire when it would add work rather than remove it.
4. Lit tests cover all three patterns plus the negative case.

---

## Diagnosis and plan

### Environment setup

I used Enzyme-JAX's `DEVDOCS.md` plus the in-repo precedents, and verified through upstream CI rather than a local build.

- **Challenge (the build is the blocker, not the code):** Enzyme-JAX builds LLVM/MLIR and XLA from source via Bazel, a first build runs several hours with no incremental shortcut, and as of 2026-07-16 my ROAR account had no Enzyme-JAX clone or Bazel setup for it because the #1084 environment had aged out.
- **How I resolved it:** rather than spend hours of cluster time before writing three patterns, I verified everything that can be verified without compiling, meaning the math numerically, the MLIR construction and type inference against existing `RealOp` and `ImagOp` call sites in the same file, and the lit-test form against the repo's conventions, then let upstream CI, which has the toolchain cached, compile and run it. This is a deliberate trade, and I state it plainly rather than implying a local run.
- **Convention check:** the lit-test form was verified from `test/lit_tests/conj_real.mlir`, and the neighbouring complex tests `abspositive.mlir`, `conj_complex.mlir`, `binopcomplexconstsimplify.mlir` and `diff_exp_complex.mlir` were checked for naming and style.

### Steps to reproduce the missing optimization

1. Build `enzymexlamlir-opt` from `main`, or use CI.
2. Write a function taking `real`, `imag` or `abs` of a complex `stablehlo.exponential`:
   ```mlir
   func.func @abs_of_exp(%arg0: tensor<complex<f64>>) -> tensor<f64> {
     %0 = stablehlo.exponential %arg0 : tensor<complex<f64>>
     %1 = stablehlo.abs %0 : (tensor<complex<f64>>) -> tensor<f64>
     return %1 : tensor<f64>
   }
   ```
3. Run it through `--enzyme-hlo-opt`.

### Expected vs. actual

**Actual:** the complex `stablehlo.exponential` survives, followed by `stablehlo.abs` on a complex value, so the whole computation stays in the complex domain.

**Expected:** `abs(exp(x))` becomes a single real `exp(real(x))`, and `real(exp(x))` and `imag(exp(x))` become real-arithmetic `mul(exp(real(x)), cos/sin(imag(x)))`.

### Root cause

There is no missing capability, only a missing pattern. Enzyme's HLO optimizer has canonicalizers for `real` and `imag` over `stablehlo.complex` in `RealOpCanon` and `ImagOpCanon`, and an abs simplification for positive values in `AbsPositiveSimplify`, but nothing recognizes `exp` as the producer, so the identities are never applied. I confirmed this by enumerating every `stablehlo::ExpOp` reference in the file, with the line numbers above, rather than inferring it from behavior.

### The plan

**Understand:** Three known identities are unimplemented, and the producer-matching idiom needed is already used by the neighbouring canonicalizers.

**Match:** `RealOpCanon` at :15054 and `ImagOpCanon` at :15082 are the structural templates, with the same base class, the same `matchAndRewriteImpl` signature and the same `getDefiningOp<...>()` producer match. `AbsPositiveSimplify` at :22229 is the abs precedent and the place to check how the file handles multi-use.

**Plan:**
1. Add three `CheckedOpRewritePattern`s after `ImagOpCanon`, being `RealOfExpSimplify`, `ImagOfExpSimplify` and `AbsOfExpSimplify`.
2. Guard each on the operand's element type actually being complex, match the producer with `getDefiningOp<stablehlo::ExpOp>()`, then rebuild with `RealOp`, `ImagOp`, `ExpOp`, `CosineOp`, `SineOp` and `MulOp::create`, relying on element-type inference from complex to real exactly as the existing `RealOp::create` sites do.
3. Register all three in the pattern list next to the other real, imag and complex canonicalizers.
4. Add `test/lit_tests/realimagabsexp.mlir` with one `func.func` per identity.
5. Ask on the PR whether they would prefer this ported to tablegen, since PR #2568 migrated some patterns and the neighbours around `RealOpCanon` are still C++.

**Implement:** Branch `feat/1990-real-imag-abs-exp`.

**Review:** My self-checklist was that naming follows the file's convention, using `*Simplify` for algebraic rewrites and `*Canon` for canonicalizations, that the `// real(exp(x)) -> ...` arrow comments match `ConjComplexNegate` and neighbours, that registration is grouped by family since the list is family-grouped rather than strictly alphabetical, and that there is no overlap with `AbsPositiveSimplify`, which bails on complex operands.

**Evaluate:** CI compiles and runs the lit tests, and the math was verified numerically beforehand.

### Edge cases

The one real judgment call was identified in planning before review:

> **Guard against duplicating a transcendental when `exp` has other uses:** If the `exp` result is consumed elsewhere, rewriting `real(exp x)` into `exp(real x)·cos(imag x)` does not remove the original complex `exp`, it adds a second one. Net loss.

I flagged this in my implementation plan as the thing a reviewer will look for, and `wsmoses` asked for exactly that guard on review. It is now enforced by a single-use check on all three patterns, with a negative lit test where the `exp` is also returned directly.

I also considered that non-complex operands must fall through to `RealOpCanon` and `ImagOpCanon` and leave real `abs` untouched, which is what the element-type guard is for.

---

## Implementation

### Commits and files

| Commit | Date | Message |
|---|---|---|
| `5c4df4934` | 2026-07-18 | Simplify real, imag, and abs of complex exp |
| `a6e4657c7` | 2026-07-18 | Apply clang-format |
| `2e8edb732` | 2026-07-18 | Skip exp simplification when the exp has other users, a review fix |

**Files modified:**

| File | Δ |
|---|---|
| `src/enzyme_ad/jax/Passes/EnzymeHLOOpt.cpp` | +87 |
| `test/lit_tests/realimagabsexp.mlir` | +53 (new) |

The three patterns are added after `ImagOpCanon` and registered next to the other real, imag and complex canonicalizers:

- `RealOfExpSimplify`, turning `stablehlo.real` of a complex `stablehlo.exponential` into `exp(real(x)) * cos(imag(x))`.
- `ImagOfExpSimplify`, turning `stablehlo.imag` of a complex `exp` into `exp(real(x)) * sin(imag(x))`.
- `AbsOfExpSimplify`, turning `stablehlo.abs` of a complex `exp` into `exp(real(x))`.

Each one guards with `if (!isa<ComplexType>(op.getOperand().getType().getElementType())) return failure();`, matches the producer via `getDefiningOp<stablehlo::ExpOp>()`, checks the single-use condition, then rebuilds through `RealOp`, `ImagOp`, `ExpOp`, `CosineOp`, `SineOp` and `MulOp::create`.

**Consistency notes:** `*Simplify` is the dominant suffix in this file for algebraic rewrites as opposed to `*Canon` for canonicalizations, the `// real(exp(x)) -> ...` arrow comments match `ConjComplexNegate` and neighbouring algebraic patterns, the registration list is grouped by family rather than strictly alphabetical so the three sit together after `ImagOpCanon`, and there is no overlap with `AbsPositiveSimplify` since it bails on complex operands.

### What was hard

**Verifying a compiler pattern without a local compiler:** A first Bazel build here is a multi-hour LLVM/MLIR/XLA compile with no incremental shortcut, and the environment from my earlier Enzyme-JAX work had aged out. Rather than block on that for a roughly 90-line change, I split verification. The identities were checked numerically, the MLIR construction and element-type inference from complex to real were checked against the existing `RealOp::create` call sites in the same file which do exactly the same inference, the lit test was written to the conventions of `conj_real.mlir` and the neighbouring complex tests, and compilation plus test execution were left to upstream CI which has the toolchain cached. The trade-off is slower feedback on typos, which is a fair price, but what it must not become is an unstated assumption, so the PR says where verification happened.

**Reading the cost model rather than just the identity:** The math is unconditionally true, but the rewrite is not unconditionally profitable. Working that out before submitting, and writing it into the plan as the thing a reviewer would look for, is what made the review round a one-line ask I could turn around the same day.

### Testing

- **New lit test** `test/lit_tests/realimagabsexp.mlir`, with `func.func` cases `real_of_exp`, `imag_of_exp` and `abs_of_exp` over `tensor<complex<f64>>`, using `CHECK-LABEL` plus `CHECK-DAG` so the checks are robust to operand ordering, following `conj_real.mlir`'s form.
- **Negative test added in review**, a case where the `exp` is also returned directly, asserting the rewrite does not fire and the original `exp` is left untouched.
- **CI status as of 2026-08-04:** 14 checks green, 2 skipped and 7 red. The red ones are the `Reactant_jll`, GB-25 and TPU integration matrix, and every other open PR in the repo currently carries 2 to 5 failures in that same matrix, so they are repo-wide rather than specific to this patch. I checked that rather than asserting it.

---

## Review and outcome

### The pull request

**[EnzymeAD/Enzyme-JAX#2635](https://github.com/EnzymeAD/Enzyme-JAX/pull/2635)**, opened 2026-07-18 against `EnzymeAD/Enzyme-JAX:main`, referencing #1990. The body states the three identities and their derivation, the patterns and where they sit relative to the existing canonicalizers, the multi-use cost-model concern, and explicitly that verification is via CI rather than a local build and why.

### Maintainer feedback

| Date | From | Feedback | My response |
|---|---|---|---|
| 2026-07-18 | wsmoses (review) | "can you add an additional check that the only user of the exp is this realop? [eg if its used elsewhere we might as well avoid the redundant work]" | Addressed the same day in `2e8edb732` by adding the single-user guard, so the rewrite only fires when the `exp` feeds nothing but this op and otherwise bails and leaves the original `exp` untouched. I applied it to the `imag` and `abs` patterns too since the same redundant-work argument holds, and added a negative test where the `exp` is also returned directly. |

I am awaiting a second look, and the branch is otherwise idle by design rather than stalled on my side.

### What I learned

**Technical:** An algebraic identity being true is not the same as the rewrite being profitable. `real(exp x)` becoming `exp(real x)·cos(imag x)` removes a complex transcendental only if the original `exp` becomes dead, and if it is consumed elsewhere the rewrite adds one. That is the whole content of the review round, and it generalizes, because any pattern that duplicates a producer needs a use-count condition. I also learned the small but real discipline of extending a reviewer's point beyond what they asked, since wsmoses only mentioned the `real` case and the same argument applies verbatim to `imag` and `abs`.

**Process & collaboration:** Deciding not to comment on the issue was itself a judgment call, made by looking at who actually merges in the repo and how they behave rather than applying a generic etiquette rule. Second contributions to the same repo are much cheaper than first ones, because knowing the file, the conventions and the maintainer's review style meant the work went into the interesting decision instead of the boilerplate. Writing the risky part into the plan ahead of time also meant the review round cost hours rather than days.

**What I'd do differently:** I would have shipped the single-use guard in the first commit. I had identified it in my own plan as the thing a reviewer would look for and then submitted without it, which is the worst of both worlds, because I paid the analysis cost and still took the review round. If I have written down that something is the reviewer's first question, that is a signal to implement it rather than to mention it. I would also have set the build environment up in parallel with writing the patterns, so CI was a confirmation rather than the only check.

### References

- Issue #1990, for the three identities as filed by `avik-pal`.
- `EnzymeHLOOpt.cpp`, specifically `RealOpCanon` at :15054, `ImagOpCanon` at :15082, `AbsPositiveSimplify` at :22229 and the registration list at :36487 to :36490.
- `test/lit_tests/conj_real.mlir` and the neighbouring complex tests, for lit-test conventions.
- PR #2568, the tablegen migration, which is the reason I asked about C++ versus tablegen on the PR.
