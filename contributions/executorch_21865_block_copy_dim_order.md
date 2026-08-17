# Reject non-default dim order in the portable kernels that copy by block

**Project:** [pytorch/executorch](https://github.com/pytorch/executorch)
**Issue:** none, self-sourced, follow-up to [#21828](https://github.com/pytorch/executorch/pull/21828) · **Pull request:** [#21865](https://github.com/pytorch/executorch/pull/21865) · **Branch:** `fix/dim-order-gate-block-copy-ops`
**Status:** Open, awaiting review.

---

## The issue

### Why I picked it

This is the second half of the same sweep that produced [#21828](https://github.com/pytorch/executorch/pull/21828). While reducing the element-indexed kernels to one line in `coordinateToIndex`, I found a second group that shares the assumption but cannot be fixed the same way. Splitting them rather than shipping one large PR was deliberate, since the two need genuinely different fixes and a reviewer can evaluate each on its own terms. My learning goal here was knowing when not to support a layout, which is a different judgment from knowing how to.

### What was wrong

Five more portable kernels accept a channels-last input and return wrong data without erroring, and three more share the defect while being masked by an unrelated check. These kernels copy in blocks rather than indexing element by element, taking the elements around the operating dimension as a contiguous run, and that run only exists in the default dim order. It matters for the same reason as #21828, since the corruption is silent and the output tensor is labelled correctly, but the resolution has to be different because supporting the layout would mean restructuring every loop. I picked it because leaving half a discovered family unfixed would have been the worse outcome.

Checked against eager PyTorch with only the memory format differing:

| kernel | contiguous input | channels-last input |
|---|---|---|
| `aten.cat` | matches | wrong, max abs diff 4.141 |
| `aten.pixel_unshuffle` | matches | wrong, max abs diff 3.941 |
| `aten.slice_scatter` | matches | wrong, max abs diff 3.376 |
| `aten.topk` | matches | wrong, max abs diff 2.581 |
| `aten.constant_pad_nd` | matches | wrong, max abs diff 2.526 |
| `aten.cumsum` | matches | rejected, status 0x12 |
| `aten.split_copy` | matches | rejected, status 0x12 |
| `aten.split_with_sizes_copy` | matches | rejected, status 0x12 |

### How I found it

Self-sourced, from the same differential sweep against eager PyTorch that produced #21828. The three rejected kernels are masked by `tensors_have_same_dim_order`, and that check fires only because the exported graph happens to give them differing dim orders. Nothing guards the arithmetic itself, which is why the unit tests in this PR reach those kernels directly and prove the guard is doing the work rather than the accident.

### Where it lives

- `kernels/portable/cpu/op_cat.cpp`, `op_constant_pad_nd.cpp`, `op_cumsum.cpp`, `op_pixel_unshuffle.cpp`, `op_slice_scatter.cpp`, `op_split_copy.cpp`, `op_split_with_sizes_copy.cpp` and `op_topk.cpp`.
- The shared pattern is `getLeadingDims` and `getTrailingDims` taken around the operating dim as the length of a contiguous run.
- `op_addmm`, `op_bmm` and `op_avg_pool2d`, which already carry the guard this PR adds and are therefore the in-repo precedent.

### What counts as done

1. All eight kernels reject a non-default dim order rather than returning wrong numbers.
2. The guard is the same idiom the codebase already uses elsewhere.
3. One test per kernel, each reaching the new guard specifically rather than an existing check.
4. Contiguous behavior is untouched.

---

## Diagnosis and plan

### Environment setup

Same setup as [#21828](https://github.com/pytorch/executorch/pull/21828), with the repo's `.venv` and pybindings on macOS CPU for runtime differential testing, and the C++ gtest targets built on ROAR with the module ordering and `-DPYTHON_EXECUTABLE` workarounds recorded there.

- **Challenge (proving a guard is what rejects the input):** these kernels already carry `tensors_have_same_dim_order`, so a careless test would pass a mixed set of tensors, get rejected by the existing check, and appear to validate a guard that was never reached. I made every test pass a uniformly channels-last set, so the existing check passes and only the new guard can reject.
- **Challenge (one test file was not registered):** `op_pixel_unshuffle_test.cpp` was missing from `kernels/test/CMakeLists.txt`, while `op_pixel_shuffle_test.cpp` and every other op in that directory were present, so those tests never ran in a CMake build and my new one would not have either.

### Steps to reproduce

1. Install ExecuTorch with pybindings.
2. Export a one-op model for any of the five silently wrong kernels and run it on a channels-last input, as in the reproduction in #21828.
3. Compare against eager PyTorch with the same values in contiguous format as a control.
4. For the three masked kernels, call the kernel directly from a gtest with a uniformly channels-last tensor set, which bypasses the graph-level dim order mismatch that normally rejects them.

### Expected vs. actual

**Actual:** the five report wrong values with no error, at the magnitudes in the table above. The three masked kernels are rejected with status 0x12, but only because their dim orders differ, so the rejection is incidental rather than a real defense.

**Expected:** either the correct values, or a clear rejection that comes from the kernel's own requirements.

### Root cause

These kernels copy in blocks, taking the leading and trailing extents around the operating dim as the length of a contiguous run:

```cpp
const size_t outer = getLeadingDims(out, dim);
const size_t dim_stride = getTrailingDims(out, dim);
```

That run only exists in the default dim order. Under channels-last the elements after `dim` are not adjacent in memory, so the arithmetic walks the wrong bytes.

This is the same assumption as #21828, but it needs a different fix. There the kernels index element by element, so reading the tensor's real strides is sufficient and no control flow changes. Here the algorithm depends on contiguity itself, since the whole point of the loop is that a block of elements can be copied in one go, so supporting the layout would mean restructuring each loop rather than correcting an index expression.

### The plan

**Understand:** Eight kernels treat a range around the operating dim as contiguous, which is false for any non-default dim order.

**Match:** `op_addmm`, `op_bmm` and `op_avg_pool2d` already carry exactly the guard these need:

```cpp
ET_KERNEL_CHECK(ctx, tensor_is_default_dim_order(in), InvalidArgument, out);
```

Using the established idiom means the change reads as applying an existing rule consistently rather than introducing a new policy. `pixel_shuffle` already carries this exact pair of checks, so `pixel_unshuffle` missing them looks like an oversight rather than a decision.

**Plan:**
1. Add the guard to all eight kernels.
2. Give `cat` the check on each input as well as the output, since it had no dim order check at all.
3. Add a `NonDefaultDimOrderDies` test per kernel, each passing uniformly channels-last tensors so only the new guard can reject.
4. Register `op_pixel_unshuffle_test.cpp` in the CMake source list.

**Implement:** Branch `fix/dim-order-gate-block-copy-ops`, three commits grouped by op family.

**Review:** My self-checklist was that the guard matches the existing idiom exactly, that each test reaches the new guard rather than an existing check, and that the user-visible consequence is stated plainly in the body.

**Evaluate:** Revert the eight kernel sources while keeping the tests, and confirm all eight fail.

### Edge cases

- **This makes some working programs fail.** `cat` and `split` appear in nearly every model, so a channels-last program that runs today would start erroring instead of returning wrong numbers. I flagged that explicitly in the PR, argued it is still better than silent corruption, and offered to restructure the loops to support the layout instead if the maintainers prefer.
- **Kernels that look affected and are not.** `select_scatter` and `native_batch_norm` use the same helpers but are safe, because their auxiliary tensors are lower rank, so the existing same-dim-order check already forces the input contiguous.
- **One I could not attribute.** `unfold_copy` is likely affected, but the wrong result I measured came from a multi-op graph and I could not isolate it to the kernel, so I left it out rather than claim it.
- **Adjacent instances handled elsewhere.** The same block arithmetic appears in the optimized `layer_norm`, which became [#21866](https://github.com/pytorch/executorch/pull/21866), and in the quantized `dequantize`, which [#21517](https://github.com/pytorch/executorch/pull/21517) covers.
- **Landing order.** This and #21828 touch no files in common and can land in either order, which I stated up front so neither blocks the other.

---

## Implementation

### Commits and files

| Commit | Date | Message |
|---|---|---|
| `260928591` | 2026-08-14 | Reject non-default dim order in `cat`, `constant_pad_nd` and `slice_scatter` |
| `e87b6acdb` | 2026-08-14 | Reject non-default dim order in `cumsum` and the split copies |
| `54aa30a41` | 2026-08-15 | Reject non-default dim order in `pixel_unshuffle` and `topk` |

The commits are grouped by op family rather than one per kernel, so each is a reviewable unit.

**Files modified:** 8 kernel sources for a total of +30, 8 test files for a total of +115, plus `kernels/test/CMakeLists.txt` at +1. The largest single kernel change is `op_topk.cpp` at +8 and `op_cat.cpp` at +6, since those needed checks on more than one tensor.

### What was hard

The genuinely hard part was accepting that the right fix here is worse for users than the fix in the sibling PR. #21828 makes channels-last work. This one makes it fail loudly. Writing a PR that knowingly breaks programs which currently run is uncomfortable, and the temptation was to attempt the loop restructuring so both halves of the family got real support.

I did not, because the restructuring is a different and much larger change for each of eight kernels, and shipping it half-verified would trade silent corruption for a subtler class of bug in code I had rewritten. Stating the consequence plainly in the body, with an offer to restructure instead, puts the decision where it belongs, which is with the maintainers who know how many channels-last programs exist in the wild.

The second difficulty was making the tests prove anything. Every one of these kernels already carries `tensors_have_same_dim_order`, so passing a mixed set of tensors gets you a rejection that has nothing to do with the new guard. Making each test pass a uniformly channels-last set is what forces the existing check to pass and leaves only the new guard able to reject.

### Testing

- **A `NonDefaultDimOrderDies` test per kernel**, eight in total, each passing a uniformly channels-last tensor set so the existing same-dim-order check passes and only the new guard can reject.
- **Negative control:** reverting the eight kernel sources while keeping the tests fails all eight, which is what shows the tests pin the guard rather than passing vacuously.
- Discovery evidence is the eight-row differential table, measured against eager PyTorch with only the memory format differing.
- The PR also registers `op_pixel_unshuffle_test.cpp` in `kernels/test/CMakeLists.txt`, without which neither the existing tests in that file nor the new one would run in a CMake build.

---

## Review and outcome

### The pull request

**[pytorch/executorch#21865](https://github.com/pytorch/executorch/pull/21865)**, opened 2026-08-15 against `pytorch/executorch:main`, self-labeled `release notes: ops & kernels`. No `Fixes` keyword, since it is self-sourced.

The body opens by placing it against #21828, stating that the two touch no files in common and can land in either order, then gives the eight-row differential table, then explains why the same defect needs a different fix here. It names the kernels that look affected and are not, admits the one I could not attribute, points at the two adjacent PRs covering the optimized and quantized instances, and flags the breaking consequence for `cat` and `split` with an offer to restructure instead.

### Maintainer feedback

| Date | From | Feedback | My response |
|---|---|---|---|
| 2026-08-15 | Copilot review | Unable to review, quota limit reached on the requesting account. | n/a, so the extra manual pass stands in for it. |

No human review yet.

### What I learned

**Technical:** The same wrong assumption can require opposite fixes depending on what the code does with it. Reading an element through a row-major stride is a correctable index expression, but copying a block because you believe it is contiguous is a structural dependency on the layout, and no amount of index fixing repairs it. Recognizing which of the two you are looking at is the whole decision, and it is visible in whether the loop touches one element or a run.

**Process & collaboration:** Splitting one discovery into two PRs with different fixes was better than one large one, because the reviewer for the stride change does not have to also weigh a breaking guard. Saying plainly that this makes currently-running programs fail, rather than burying it, is the part I would most want to repeat, since a maintainer cannot make that trade without knowing it exists. Admitting `unfold_copy` as suspected but unattributed is the same instinct, because an unproven claim in a bug report costs credibility that the proven ones need.

**What I'd do differently:** I would design the tests before writing the guards. I initially wrote a test that got rejected by the pre-existing `tensors_have_same_dim_order` check and briefly believed my guard worked, which is exactly the vacuous-pass failure I have run into before in a different form. For any change whose effect is a rejection, the first question is which check does the rejecting.

### References

- `kernels/portable/cpu/op_addmm.cpp`, `op_bmm.cpp` and `op_avg_pool2d.cpp`, the existing users of this guard.
- `op_pixel_shuffle.cpp`, which already carries the pair of checks its inverse was missing.
- Sibling [#21828](https://github.com/pytorch/executorch/pull/21828), the element-indexed half of the same defect.
- Adjacent [#21866](https://github.com/pytorch/executorch/pull/21866) for the optimized `layer_norm` and [#21517](https://github.com/pytorch/executorch/pull/21517) for the quantized `dequantize`.
- Issue [#16429](https://github.com/pytorch/executorch/issues/16429), for the standing request that these layouts be supported rather than rejected.
