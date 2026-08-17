# Check dim order in the optimized `layer_norm` as the portable one does

**Project:** [pytorch/executorch](https://github.com/pytorch/executorch)
**Issue:** none, self-sourced, related to [#21828](https://github.com/pytorch/executorch/pull/21828) and [#21865](https://github.com/pytorch/executorch/pull/21865) · **Pull request:** [#21866](https://github.com/pytorch/executorch/pull/21866) · **Branch:** `fix/optimized-layer-norm-dim-order`
**Status:** Open, awaiting review. CI green on the completed jobs.

---

## The issue

### Why I picked it

This is the smallest of the three PRs from one dim-order sweep, and the one with the sharpest evidence, because the correct behavior already exists a few directories away. While checking whether the block-copy assumption in [#21865](https://github.com/pytorch/executorch/pull/21865) reached beyond the portable library, I found the optimized kernel library carrying the same arithmetic without the guard its portable counterpart has, and with a `TODO` in the portable one explaining exactly why the guard is needed. My learning goal was the relationship between the two kernel libraries, since ExecuTorch ships a portable reference implementation and an optimized one that is swapped in by build flag.

### What was wrong

`aten.native_layer_norm` returns wrong data on a channels-last input and nothing errors, but only in builds that use the optimized kernel library. The portable kernel is correct, because it carries a default-dim-order check with the reason written next to it, while the optimized kernel has neither that check nor the matching same-dim-order one. It matters because `native_layer_norm.out` is mapped to the optimized implementation in `optimized.yaml`, so any build with `EXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON` silently gets the ungated version, and layer norm is in essentially every transformer. I picked it because the fix is to copy two lines the project already wrote for itself.

Measured against eager PyTorch, the optimized kernel is off by 3.192 on a channels-last input where it matches exactly on contiguous input.

### How I found it

Self-sourced, by asking whether the block-copy assumption I had just documented in the portable library appeared in the optimized one too. The portable `native_layer_norm` carries this, with the reason stated:

```cpp
// Only support default dim order for now.
// TODO: Support other dim orders.
ET_KERNEL_CHECK(
    ctx, tensor_is_default_dim_order(input), InvalidArgument, ret_val);
```

Finding a guard in one library and its absence in the other, for the same op with the same arithmetic, is a strong enough signal that I went straight to measuring it.

### Where it lives

- `kernels/optimized/cpu/op_native_layer_norm.cpp`, missing both checks.
- The portable `op_native_layer_norm.cpp`, which has them and is the reference for what the optimized one should do.
- `optimized.yaml`, where `native_layer_norm.out` is mapped to `torch::executor::opt_native_layer_norm_out`, which is what makes the gap reachable.

### What counts as done

1. The optimized kernel rejects a non-default dim order rather than returning wrong numbers.
2. It carries the same two checks as the portable kernel, so the two libraries agree.
3. A test that reaches the new check specifically rather than an existing one.
4. The test also runs against the portable kernel, where it should already pass.

---

## Diagnosis and plan

### Environment setup

Same as the sibling PRs, with the repo's `.venv` and pybindings on macOS CPU for the differential measurement, and the gtest targets built on ROAR using the module ordering and `-DPYTHON_EXECUTABLE` workarounds recorded in [#21517](https://github.com/pytorch/executorch/pull/21517).

- **Challenge (the two libraries have to be exercised separately):** which kernel you get depends on `EXECUTORCH_BUILD_KERNELS_OPTIMIZED`, so a measurement that does not control that flag tells you nothing about which implementation produced the number.
- **Challenge (reaching the new check rather than an existing one):** the kernel requires `mean` and `rstd` to share the input's rank with the normalized dims set to 1, and it checks that before it reaches any dim order check, so a test that gets those shapes wrong fails for the wrong reason.

### Steps to reproduce

1. Build ExecuTorch with `EXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON`.
2. Export a model whose forward is `torch.nn.functional.layer_norm` and run it on a channels-last input.
3. Compare the output against eager PyTorch on the same values.
4. Repeat with the same values in contiguous format as a control.

### Expected vs. actual

**Actual:** the channels-last run is off by 3.192 against eager PyTorch, with no error raised, while the contiguous control matches exactly.

**Expected:** either matching values, or the same clear `InvalidArgument` the portable kernel gives.

### Root cause

The optimized kernel splits the buffer into `M` rows of `N` contiguous elements:

```cpp
const size_t M = getLeadingDims(input, dim);
const size_t N = getTrailingDims(input, dim) * dim_size;
```

That layout only exists in the default dim order, so under channels-last the row boundaries do not correspond to anything real and the normalization statistics are computed over the wrong elements. This is the same block-copy assumption as [#21865](https://github.com/pytorch/executorch/pull/21865), in a different kernel library.

The portable kernel makes the same assumption and is safe only because it guards against it. The optimized kernel was written without the guard, and since `optimized.yaml` maps the op to the optimized implementation, the guarded version is the one that does not run.

### The plan

**Understand:** Two implementations of one op share an arithmetic assumption, and only one of them checks the precondition.

**Match:** The portable kernel is the reference. Copying its two checks across, rather than inventing a new form of validation, makes the libraries agree and means the change is reviewable by diffing against a file already in the tree.

**Plan:**
1. Add `tensor_is_default_dim_order(input)` to the optimized kernel.
2. Add the matching `tensors_have_same_dim_order` check, which is also absent.
3. Add a test passing a uniformly channels-last set, so the same-dim-order check passes and only the new default-dim-order check can reject.
4. Give `mean` and `rstd` the input's rank with normalized dims set to 1, so the test reaches the dim order check rather than failing on shape validation first.

**Implement:** Branch `fix/optimized-layer-norm-dim-order`, single commit.

**Review:** My self-checklist was that the checks are byte-equivalent to the portable ones, that the test reaches the new guard specifically, and that the test file being shared by both libraries is called out rather than left as a surprise.

**Evaluate:** Run the test against both kernel libraries, where it should newly pass against the optimized one and already pass against the portable one.

### Edge cases

- **Guard rather than stride-aware indexing.** This kernel depends on contiguity structurally, in the same way as the block-copy kernels in #21865, so reading the tensor's real strides would not help. Supporting the layout would mean restructuring the row loop, which is out of scope for a correctness fix.
- **Test-shape preconditions.** The kernel validates `mean` and `rstd` shapes before any dim order check, so the test has to satisfy those first or it never reaches the code under test.
- **Shared test file.** `kernels/test/op_native_layer_norm_test.cpp` is compiled against both libraries, so the new test also runs against the portable kernel, where it passes already. That is a feature rather than a problem, since it pins the two implementations to the same contract.

---

## Implementation

### Commits and files

| Commit | Date | Message |
|---|---|---|
| `128d5cbd5` | 2026-08-15 | Check dim order in the optimized layer_norm as the portable one does |

**Files modified:**

| File | Δ |
|---|---|
| `kernels/test/op_native_layer_norm_test.cpp` | +26 |
| `kernels/optimized/cpu/op_native_layer_norm.cpp` | +10 |

Ten lines of kernel change, of which the substance is two `ET_KERNEL_CHECK` calls copied from the portable implementation.

### What was hard

Nothing about the fix was hard, which is the interesting part. The difficulty was entirely in noticing, because the optimized library is not where you look when the portable kernel is correct, and the portable kernel is correct. The `TODO` comment in the portable file is what made the gap visible, since it states that only the default dim order is supported for now, and that statement is false for the implementation the build actually selects.

The one real trap was writing a test that proves anything. The kernel checks `mean` and `rstd` shapes before it reaches any dim order check, so an incorrectly shaped test fails on shape validation and looks like the guard working. Passing tensors that are uniformly channels-last, with `mean` and `rstd` at the input's rank and the normalized dims set to 1, is what leaves the new check as the only thing that can reject.

### Testing

- **`OpNativeLayerNormTest.NonDefaultDimOrderDies`** passes a channels-last input with `out`, `mean` and `rstd` all channels-last, so the same-dim-order check passes and only the new default-dim-order check can reject. `mean` and `rstd` share the input's rank with the normalized dims set to 1, which the kernel requires before it reaches any dim order check.
- The test file is shared by both kernel libraries, so it also runs against the portable kernel, where it already passes. That makes it a contract test across the two implementations rather than a single-kernel regression test.
- Discovery evidence is the differential measurement against eager PyTorch, off by 3.192 on channels-last and exact on contiguous.
- CI is green on the completed jobs.

---

## Review and outcome

### The pull request

**[pytorch/executorch#21866](https://github.com/pytorch/executorch/pull/21866)**, opened 2026-08-15 against `pytorch/executorch:main`, self-labeled `release notes: ops & kernels`. No `Fixes` keyword, since it is self-sourced.

The body places it against #21828 and #21865 as the same underlying assumption in the optimized library, gives the measured error, then quotes the portable kernel's existing guard including its `TODO`, which is the strongest single piece of evidence, since the project has already written down that this op needs the check. It then shows the block arithmetic that makes the guard necessary, explains why stride-aware indexing would not work here, and notes that the shared test file means the test also covers the portable kernel.

### Maintainer feedback

| Date | From | Feedback | My response |
|---|---|---|---|
| 2026-08-15 | Copilot review | Unable to review twice, quota limit reached on the requesting account. | n/a, so the manual pass stands in for it. |

No human review yet.

### What I learned

**Technical:** A guard in one implementation is evidence about every other implementation of the same operation. The portable kernel's comment says only the default dim order is supported, and that sentence is a specification for the op rather than for the file it sits in, so the optimized kernel violating it is a bug even though nothing in the optimized file looks wrong on its own. Reading across parallel implementations is a cheap and productive way to find gaps, because any asymmetry is either a bug or a deliberate difference somebody should have documented.

**Process & collaboration:** Quoting the project's own `TODO` back to it is the most efficient argument available, since it removes any debate about whether the check is wanted. Splitting this out from #21865 rather than folding it in was also right, because it touches a different kernel library with different owners, and a small PR against the optimized library can be reviewed by whoever owns that library without them having to read eight portable kernels.

**What I'd do differently:** I would check the optimized library at the same time as the portable one rather than as a follow-up thought. I had already documented the block-copy assumption in eight portable kernels before asking whether the optimized library shared it, and there was no reason that question had to come second. When a defect is about an assumption rather than a line of code, the search should cover every implementation of the affected ops from the start.

### References

- The portable `kernels/portable/cpu/op_native_layer_norm.cpp`, its guard and its `TODO`.
- `optimized.yaml`, the mapping that makes the ungated kernel the one that runs.
- Siblings [#21828](https://github.com/pytorch/executorch/pull/21828) and [#21865](https://github.com/pytorch/executorch/pull/21865) from the same sweep.
- [#21517](https://github.com/pytorch/executorch/pull/21517), the same assumption in the quantized `dequantize` path.
