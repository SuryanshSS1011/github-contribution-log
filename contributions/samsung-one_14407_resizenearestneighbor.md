# Contribution 2: [one-optimize] Fold ResizeNearestNeighbor Op

**Contribution Number:** 2
**Student:** Suryansh Sijwali
**Issue:** https://github.com/Samsung/ONE/issues/14407
**Status:** Phase I In Progress

---

## Why I Chose This Issue

Samsung/ONE is an ML compiler and runtime stack targeting on-device inference,
and `one-optimize` is its graph-level optimization driver. This issue asks for
a constant-folding pass for the ResizeNearestNeighbor op, motivated by a
real-world model pattern: DETR's positional-embedding code lowers to a
ResizeNearestNeighbor over a constant absolute-position tensor, which is a
prime candidate for compile-time folding. A sibling issue (#12046, ResizeBilinear
folding) was already merged, giving me a proven landing template to model
the implementation on.

I picked this as my second contribution because it is a contained
compiler-pass change in an active ML compiler — directly aligned with my
research interests in ML compilers and optimization passes (Causality-Aware-RL,
SecureCodeRL) — and because Samsung/ONE is a slow-moving repo, which makes it
safe to reserve ahead of time without the issue being sniped while I'm
building up the toolchain on other contributions. The end-to-end shape of the
work (read existing pass → understand the IR → implement folding → add unit
tests) maps cleanly to standard compiler-optimization practice.

---

## Understanding the Issue

### Problem Description

When a model graph contains a ResizeNearestNeighbor op whose input tensor is a
compile-time constant, the op can be evaluated at compile time and replaced
with its computed constant output. This eliminates a runtime op (and the
allocation/copy it implies) without changing model semantics. The
`one-optimize` driver supports this kind of constant-folding pass for many
ops, but not yet for ResizeNearestNeighbor. The motivating example is DETR's
absolute-positional-embedding code path (shown in the issue body), where
`F.interpolate(..., mode="nearest")` on a constant `abs_pos` tensor lowers to
a ResizeNearestNeighbor that ONE currently leaves un-folded.

### Expected Behavior

`one-optimize` should detect ResizeNearestNeighbor ops with constant inputs
and replace them with the equivalent constant tensor at compile time. Output
shape/values must match the runtime ResizeNearestNeighbor kernel exactly for
all attribute combinations (`align_corners`, `half_pixel_centers`, target
size).

### Current Behavior

ResizeNearestNeighbor ops survive the optimization pipeline even when all
their inputs are constants, leaving the op (and its allocation) in the final
optimized graph.

### Affected Components

To be confirmed by reading the codebase, but expected to include:

- The `one-optimize` driver and its pass registration (similar to where
  ResizeBilinear folding from #12046 was registered).
- A new folding pass for ResizeNearestNeighbor (probably under the same
  directory structure as the existing ResizeBilinear pass).
- Unit tests for the new pass (matching the test convention used in #12046).

---

## Reproduction Process

### Environment Setup

To be filled in once I have the Samsung/ONE build environment set up. ONE
uses CMake and ships build scripts under `infra/`. The build is heavier than
a docs change but lighter than llama.cpp + OpenCL.

### Steps to Reproduce

1. Build ONE from source.
2. Construct (or use an existing) test model containing a
   ResizeNearestNeighbor op with constant inputs.
3. Run `one-optimize` over it.
4. Observed: ResizeNearestNeighbor op survives in the optimized graph
   instead of being replaced by a constant.

### Reproduction Evidence

To be added after environment is set up:
- **Commit showing reproduction:** TBD
- **Screenshots/logs:** Before/after `circle-dump` output of an optimized
  model showing the un-folded op.
- **My findings:** TBD

---

## Solution Approach

### Analysis

The sibling pass for ResizeBilinear (merged via #12046) provides the
structural template. The new pass needs to:

1. Pattern-match ResizeNearestNeighbor ops whose input tensor (and shape/size
   operand, if separate) are constants.
2. Compute the nearest-neighbor-upsampled output tensor at pass time,
   matching the attribute semantics the runtime kernel uses.
3. Replace the original op with a new constant tensor in the graph.
4. Cover the attribute combinations the runtime kernel supports
   (`align_corners`, `half_pixel_centers`).

### Proposed Solution

Implement a new folding pass `FoldResizeNearestNeighborPass` (name TBD per
repo conventions) that mirrors the structure of the ResizeBilinear folding
pass from #12046. Register it in the `one-optimize` pass list. Add unit
tests covering:

- The DETR positional-embedding shape pattern from the issue body.
- `align_corners` and `half_pixel_centers` attribute combinations.
- Edge cases (1×1 input, identity resize, non-square targets).

### Implementation Plan

Using UMPIRE framework (adapted):

**Understand:** Constant-fold ResizeNearestNeighbor so DETR-style positional
embeddings don't carry a runtime resize op when the input is constant at
compile time.

**Match:** The merged ResizeBilinear folding pass (PR for #12046) is the
exact structural template. The runtime ResizeNearestNeighbor kernel in ONE
is the semantic reference for the folding math.

**Plan:**
1. Post claim comment on issue #14407 to lock in the work and ask one scope
   question (single pass vs extending an existing folding pass).
2. Set up the Samsung/ONE build per `infra/` scripts; confirm the existing
   test suite is green on a clean checkout.
3. Read the merged ResizeBilinear folding pass and the runtime
   ResizeNearestNeighbor kernel; map the kernel's semantics onto the pass's
   constant-evaluation logic.
4. Write a failing unit test for the DETR shape pattern from the issue body
   (verifies the gap before patching, same discipline as a regression
   test).
5. Implement the folding pass following the ResizeBilinear template; cover
   the attribute combinations the runtime supports.
6. Run the full unit-test suite plus the new tests; verify the failing
   test now passes and nothing else regressed.
7. Open PR referencing #14407 (mention only, no closing keyword unless the
   issue is explicitly a single-PR fix). Include before/after `circle-dump`
   evidence of a sample model.

**Implement:** Branch link to be added after step 2.

**Review:** Self-review checklist before opening PR:
- [ ] Pass structure matches the ResizeBilinear folding pass conventions.
- [ ] All ResizeNearestNeighbor attribute combinations the runtime supports
  are handled, with tests for each.
- [ ] Folding output matches the runtime kernel bit-for-bit on test inputs.
- [ ] No regressions in the existing test suite.
- [ ] PR description references #14407 and includes before/after evidence.

**Evaluate:** The unit tests assert exact-match folding output against
reference values; running `one-optimize` on a DETR-shaped model shows the
ResizeNearestNeighbor op disappears from the optimized graph.

---

## Testing Strategy

### Unit Tests

- [ ] Test case 1: DETR positional-embedding pattern from the issue body —
  asserts the op is folded and the resulting constant matches the runtime
  kernel output.
- [ ] Test case 2: `align_corners = true` attribute combination.
- [ ] Test case 3: `half_pixel_centers = true` attribute combination.
- [ ] Test case 4: Identity resize (input size == output size) — should
  fold to the input constant unchanged.
- [ ] Test case 5: Non-square target size.
- [ ] Test case 6: Non-constant input — pass must leave the op untouched
  (negative test).

### Integration Tests

- [ ] Full `one-optimize` test suite green after the patch.
- [ ] `circle-dump` of a sample model containing a constant-input
  ResizeNearestNeighbor shows the op is gone from the optimized output.

### Manual Testing

- [ ] Run the pass on the DETR shape pattern end-to-end; compare optimized
  graph against the runtime-evaluated reference.

---

## Implementation Notes

### Week 1 Progress

To be filled in as work proceeds.

### Code Changes

- **Files modified:** TBD.
- **Key commits:** TBD.
- **Approach decisions:** TBD.

---

## Pull Request

**PR Link:** Not yet opened.

**PR Description:** Draft below; will refine before submitting.

> Implements constant folding for ResizeNearestNeighbor in `one-optimize`,
> closing the gap noted in #14407. Pass structure mirrors the existing
> ResizeBilinear folding pass (#12046). Covers the `align_corners` and
> `half_pixel_centers` attribute combinations and matches the runtime
> kernel's output on unit-test inputs. Verified on a DETR-shaped
> positional-embedding pattern from the issue body — the
> ResizeNearestNeighbor op is folded out of the optimized graph.

**Maintainer Feedback:** None yet.

**Status:** Phase I — issue identified, claim comment to be posted.

---

## Learnings & Reflections

To be filled in after submission and review cycle.

---

## Resources Used

- Sibling merged pass: PR for #12046 (ResizeBilinear folding) — structural
  template.
- Issue body and DETR motivation:
  https://github.com/Samsung/ONE/issues/14407
- Samsung/ONE repo: https://github.com/Samsung/ONE
- `one-optimize` driver documentation in the ONE repo (path TBD after
  initial exploration).
