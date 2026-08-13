# Allow transposed convolution weights with a non-default dim order

**Project:** [pytorch/executorch](https://github.com/pytorch/executorch) · **My fork:** https://github.com/SuryanshSS1011/executorch
**Issue:** [#20804](https://github.com/pytorch/executorch/issues/20804) · **Pull request:** [#21035](https://github.com/pytorch/executorch/pull/21035) · **Branch:** `fix/20804-transposed-conv-out-channels-1`
**Status:** **Merged** 2026-07-22 (merge commit `c46dc2726`). Issue closed.

---

## Phase I: Issue Selection

### Why I Chose This Issue

ExecuTorch is PyTorch's on-device inference runtime, and this is a real portable-CPU kernel bug rather than a refactor. It is my second contribution to the repo and a deliberate step up from the build-infra work in #20556 into the kernel layer I actually want to work in. My learning goal was tensor layout, covering dim orders, strides, and how a kernel's indexing does or does not depend on memory layout.

It was filed by `novak-vaclav` while developing the NXP backend, where an `aten.convolution` with `transposed=True` and `output_channels == 1` fails in the portable kernel with a dim-order check error.

### Problem Summary

`check_convolution_args` validates that a convolution's weight tensor has a default or channels-last dim order, but a transposed conv weight is laid out `(in_channels, out_channels / groups, kH, kW)`, so when `out_channels == 1` the singleton dimension yields a dim order like `[1, 0, 2, 3]`, which is neither, and a perfectly valid model fails to run with status 18. It matters because the failure is silent-looking and total, since a legitimate `conv_transpose2d` simply cannot execute on the portable CPU backend and the error describes a layout rather than the real constraint. I chose it because it is a genuine kernel-correctness question in the ML-systems space, and because the failing path is orthogonal to the reporter's own work, so it was their lane versus mine.

### Issue Vetting

Open, unassigned, with no in-flight PR. Because the reporter is an active contributor, unassigned was not enough, so I checked what he was actually working on and confirmed his recent merged PRs touch only `backends/nxp/**` while the bug lives in `kernels/portable/cpu`. That is a different lane, so picking it up would not duplicate or step on his work. I did not post a claim comment here, and in hindsight a one-liner noting I was taking the portable-kernel side would have made that lane separation visible to him rather than only to me.

### Where It Lives

- `kernels/portable/cpu/util/kernel_ops_util.cpp`, holding `check_convolution_args`, which calls `tensor_is_default_or_channels_last_dim_order(weight)` unconditionally.
- `kernels/portable/cpu/op_convolution.cpp`, the kernel whose indexing correctness had to be established before relaxing the check.
- `kernels/test/op_convolution_test.cpp`, the gtest suite, including the `make_with_dimorder` helper.
- `runtime/core/exec_aten/util/dim_order_util.h` and `tensor_util.h`, holding `dim_order_to_stride_nocheck` and `calculate_linear_index`.

### Acceptance Criteria

1. A transposed conv with a non-default weight dim order runs instead of failing with status 18.
2. It computes the correct result on that layout, verified against a PyTorch reference rather than just not erroring.
3. Forward, meaning non-transposed, convolution validation is unchanged.
4. There is a regression test that genuinely fails before the change.

---

## Phase II: Reproduce & Plan

### Environment Setup

I reused the ExecuTorch checkout and CMake presets from #20556, plus the repo's gtest layout for `kernels/test`. CI is the end-to-end judge here, since ExecuTorch runs the real C++ build and gtest on Meta-internal infra.

- **Challenge (the failing input could not be produced locally):** stock `to_edge` export keeps the weight at `[0,1,2,3]`, and the failing `[1,0,2,3]` layout only arises through the reporter's NXP passes, which I could not trigger, so I could not reproduce end to end from a model. I resolved it by reproducing at the level that mattered, constructing the weight in the failing dim order directly in a C++ test via `make_with_dimorder`, and separately proving kernel correctness with a standalone harness described below.
- **Challenge (a wall of red CI that had nothing to do with me):** the branch was 223 commits behind `main`, so arm, qnn and cortex-m jobs failed on infrastructure grounds with a ci-docker-hash mismatch and `git describe` failures. I resolved it by merging `upstream/main` cleanly with no conflicts and re-reading which jobs failed, after which only lintrunner's comment-wrapping complaint remained, plus one flaky OOM.

### Steps to Reproduce

1. Build ExecuTorch's portable CPU kernels with tests enabled.
2. Construct a transposed convolution whose weight has `out_channels == 1`, so shape `(in_channels, 1, kH, kW)`, with dim order `[1, 0, 2, 3]` as produced by the reporter's NXP passes, which in a test is done via `make_with_dimorder`.
3. Invoke the portable `convolution` op with `transposed = true`.
4. Observe the return status.

### Expected vs. Actual

**Actual:** the op fails in `check_convolution_args` with status 18, rejected by `tensor_is_default_or_channels_last_dim_order(weight)`. With `out_channels > 1` the dim order is `[0,1,2,3]` and the check passes, which matches exactly what the reporter observed and is what pins the cause to the singleton dimension.

**Expected:** the op runs and produces the same output as `torch.nn.ConvTranspose2d` for the same weights and inputs.

### Root Cause

`check_convolution_args` applies the forward-convolution layout assumption to transposed convolutions. A forward conv weight is `(out_channels, in_channels / groups, kH, kW)` while a transposed conv weight is `(in_channels, out_channels / groups, kH, kW)`, so when `out_channels == 1` that singleton dim produces a dim order such as `[1, 0, 2, 3]`, which is neither contiguous nor channels-last, and the unconditional check rejects a tensor the kernel can handle perfectly well. The validation is wrong rather than the tensor.

### The correctness question, and how I answered it

The subtlety that made this more than a one-line change is that the check clearly rejects the tensor, but relaxing it blindly would be wrong if the transposed kernel then miscomputed on that layout, and a silent wrong answer in a conv kernel is far worse than a clean failure.

So rather than assume, I proved the kernel is dim-order-correct. I built a standalone C++ harness using ExecuTorch's verbatim `dim_order_to_stride_nocheck` and `calculate_linear_index` from `dim_order_util.h` and `tensor_util.h`. For a `(2,1,3,3)` transposed weight, all 18 weight lookups returned identical values for dim order `[1,0,2,3]` versus `[0,1,2,3]`, so the strides genuinely differ yet every lookup matches, because the kernel indexes the weight through strides derived from its own dim order. That established the correct fix is to relax the check rather than normalize the weight upstream.

### Plan (UMPIRE)

**Understand:** A layout assumption valid for forward convolution is applied unconditionally to transposed convolution, where the weight's dimension roles are swapped.

**Match:** `check_convolution_args` already receives `transposed` as a parameter and already branches on it for other checks, so gating this one check on it is the shape the function already uses rather than a new mechanism.

**Plan:**
1. Reproduce the rejection with a weight built in the failing dim order.
2. Prove the kernel indexes correctly on that layout via the standalone stride and index harness.
3. Guard the weight dim-order check on `transposed` in `check_convolution_args`, with a comment explaining why the kernel does not depend on the layout.
4. Add a gtest that fails before the change, asserted against a `torch.nn.ConvTranspose2d` reference.
5. Push, then work the CI signal.

**Implement:** Branch `fix/20804-transposed-conv-out-channels-1`.

**Review:** My self-checklist was that forward-conv validation is untouched, that the change is one behavioral line plus a comment, that the test genuinely exercises the differing-strides case, and that the test class and helpers match the file's conventions.

**Evaluate:** The test fails before and passes after, and CI's real C++ build and gtest run on Meta infra as the end-to-end judge.

### Edge Cases Considered

- **Degenerate test shapes:** With both channel dims size-1, dim order `[1,0,2,3]` collapses to the same strides as the default, so such a test passes even unpatched and proves nothing. This is exactly the trap I fell into, which I cover under Challenges.
- **`out_channels > 1`**, where the dim order is `[0,1,2,3]` and the check already passes, so behavior must be unchanged.
- **Forward convolutions**, which must keep the strict check, so the relaxation is gated on `transposed` precisely so forward-conv validation is not weakened.

---

## Phase III: Build

### Implementation Progress

| Commit | Date | Message |
|---|---|---|
| `1433bf17e` | 2026-07-18 | Allow transposed convolution weights with a non-default dim order |
| `211235f69` | 2026-07-20 | Wrap test comments to satisfy clang-format |
| `356be3c55`, `7467d849f`, `b936ea380`, `e38cec5ba` | 2026-07-20 | Merge `upstream/main` to clear the stale-base CI failures |

**Files modified:**

| File | Δ |
|---|---|
| `kernels/portable/cpu/util/kernel_ops_util.cpp` | +7 / −2 |
| `kernels/test/op_convolution_test.cpp` | +42 |

The behavioral change is one guarded check plus the comment explaining why the kernel does not depend on the weight's layout, and everything else is the test.

### Challenges Faced

**The test that tested nothing:** My first draft used a `(1,1,2,2)` weight. Copilot's automated review caught that with both channel dims size-1, the dim order `[1,0,2,3]` collapses to the same strides as the default, so the test passed even against unpatched source. It was a real defect, because I would have shipped a regression test that could never regress. I moved to a `(2,1,2,2)` weight where the strides genuinely differ, at `[4,8,2,1]` versus the default `[4,4,2,1]`, and recomputed the torch reference output for the new shape. Since Copilot only reviews once and later runs hit a quota limit, I also did a manual pass on top and fixed the test class from `OpConvOutTest` to `OpConvCorrectnessTest` along with a non-idiomatic tensor helper.

**Distinguishing my red CI from the repo's:** An early run showed a wall of failures. The cause was a stale base 223 commits behind `main`, so arm, qnn and cortex-m jobs failed on infra grounds with a ci-docker-hash mismatch and `git describe` failures, plus lintrunner wanting my test comments wrapped. After merging `upstream/main` cleanly and wrapping the comments, everything cleared except one flaky OOM (exit 137) on an unrelated arm TOSA model test.

### Testing

- **New:** `TransposedWeightNonDefaultDimOrder` in `kernels/test/op_convolution_test.cpp`, a transposed conv with a `(2,1,2,2)` weight built in dim order `[1,0,2,3]` via the file's existing `make_with_dimorder` helper, asserted against a `torch.nn.ConvTranspose2d` reference output. It fails before the change because the weight is rejected, and passes after.
- **Patterns followed:** it is placed in the existing `OpConvCorrectnessTest` class alongside the other correctness cases, using the suite's own tensor-construction helpers rather than hand-rolled ones.
- **Standalone correctness evidence:** the stride and index harness described above, with 18 of 18 weight lookups identical between dim orders `[1,0,2,3]` and `[0,1,2,3]` despite differing strides.
- **Existing suite:** ExecuTorch CI runs the real C++ build and gtest on Meta-internal infra, and it was green after the rebase apart from one flaky OOM (exit 137) on an unrelated arm TOSA model test.

---

## Phase IV: Submit & Iterate

### Pull Request

**[pytorch/executorch#21035](https://github.com/pytorch/executorch/pull/21035)**, opened 2026-07-18 against `pytorch/executorch:main`, referencing the issue with a close keyword and self-labeled `release notes: ops & kernels` via `@pytorchbot`. The body opens with the user-visible failure, a valid `conv_transpose2d` rejected with status 18, and why the layout assumption is wrong for transposed weights, then the correctness argument including the 18 of 18 harness result, then the one-line change, then the test and its failing-before evidence.

Maintainer routing worked through the issue and PR threads, since `metascroy` asked `rascani` to look on the issue on 2026-07-21, and `nil-is-all` tagged him on the PR with "Thanks for the PR, @SuryanshSS1011. CI looks good. Tagging @rascani to review."

### Maintainer Feedback Log

| Date | From | Feedback | My response |
|---|---|---|---|
| 2026-07-18 | Copilot review | The `(1,1,2,2)` test weight is degenerate, since with both channel dims size-1 the non-default dim order collapses to the same strides, so the test passes unpatched. | A real defect. I moved to `(2,1,2,2)` where strides genuinely differ, at `[4,8,2,1]` versus `[4,4,2,1]`, recomputed the torch reference, and did an additional manual pass that also fixed the test class name and a non-idiomatic tensor helper. |
| 2026-07-20 | CI (lintrunner) | Test comments not wrapped per clang-format. | Fixed in `211235f69`, and I merged `upstream/main` in the same window to clear the stale-base infra failures. |
| 2026-07-21 | metascroy (on the issue) | "@rascani can you have a look?" | n/a |
| 2026-07-22 | nil-is-all | "CI looks good. Tagging @rascani to review." | n/a |
| 2026-07-22 | rascani | **Approved.** | Merged as `c46dc2726`, and the issue closed as completed. |

### Learnings & Reflections

**Technical:** When relaxing a validation check, prove the code behind it is actually correct for the newly-allowed input before touching the check. The standalone harness, which reuses ExecuTorch's verbatim stride and index helpers, answered that in minutes without needing the full build or the reporter's NXP pipeline, and it turned "this check seems too strict" into "the kernel indexes through its own dim order, so the layout is irrelevant to it." That evidence is what made a one-line change defensible in a kernel.

**Process & collaboration:** Two habits mattered. Read automated reviews as real reviews, because Copilot caught a test that could not fail, which is the worst kind of test to ship, and always do a manual pass on top anyway, because the tool reviews once and its later runs were quota-limited. A wall of red CI is also usually a stale base rather than your change, so merging `upstream/main` and re-reading which jobs fail, separating infra from your files, sorts signal from noise fast. Checking lane separation before starting, since the reporter's merged PRs were all `backends/nxp/**`, is also what made this safe to pick up without stepping on the person who filed it.

**What I'd do differently:** I would have designed the test around the invariant it must break rather than around the smallest convenient shape. Picking `(1,1,2,2)` was a reflex toward minimality, and minimal in shape turned out to be degenerate in strides. The right question up front is what makes this test fail on unpatched source, and asking it would have caught the degenerate case before review. I would also have posted a short note on the issue confirming I was taking the portable-kernel side, so the reporter could see the lane separation I had verified.

### Resources Used

- Issue #20804 and the reporter's NXP PRs, used to confirm lane separation.
- `kernels/portable/cpu/util/kernel_ops_util.cpp` and `op_convolution.cpp`.
- `runtime/core/exec_aten/util/dim_order_util.h` and `tensor_util.h`, whose `dim_order_to_stride_nocheck` and `calculate_linear_index` I reused verbatim in the harness.
- `kernels/test/op_convolution_test.cpp` and its `make_with_dimorder` helper.
- `torch.nn.ConvTranspose2d`, the reference output oracle.
