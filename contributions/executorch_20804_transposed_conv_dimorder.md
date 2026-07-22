# Contribution 10: Allow transposed convolution weights with a non-default dim order

**Contribution Number:** 10
**Student:** Suryansh Sijwali
**Issue:** https://github.com/pytorch/executorch/issues/20804
**Pull Request:** https://github.com/pytorch/executorch/pull/21035
**Status:** **Merged** on 2026-07-22 by rascani (merge commit `c46dc2726`);
issue #20804 closed as completed.

---

## Why I Chose This Issue

ExecuTorch is PyTorch's on-device inference runtime. This was a real portable-CPU
kernel bug (not a refactor), filed by `novak-vaclav` while developing the NXP
backend: an `aten.convolution` with `transposed=True` and `output_channels == 1`
fails in the portable kernel with a dim-order check error. It fit the ML-systems
/ kernel space I wanted to work in, and the failing path (`kernels/portable/cpu`)
was orthogonal to the reporter's own NXP work, so it was genuinely their lane vs.
mine. I confirmed it was open, unassigned, had no in-flight PR, and that the
reporter's recent merged PRs touched only `backends/nxp/**`, before starting.

## Understanding the Issue

`check_convolution_args` in `kernels/portable/cpu/util/kernel_ops_util.cpp` calls
`tensor_is_default_or_channels_last_dim_order(weight)` unconditionally. A forward
conv weight is laid out `(out_channels, in_channels / groups, kH, kW)`, but a
**transposed** conv weight is `(in_channels, out_channels / groups, kH, kW)`. When
`out_channels == 1`, that singleton dimension produces a dim order such as
`[1, 0, 2, 3]`, which is neither contiguous (`[0,1,2,3]`) nor channels-last, so the
check fails with status 18. For `out_channels > 1` the dim order is `[0,1,2,3]` and
the check passes, which matched exactly what the reporter observed.

## Reproduction and the Correctness Question

The subtlety: the check clearly rejects the tensor, but relaxing it blindly would be
wrong if the transposed kernel then miscomputed on that layout (a silent wrong answer
is worse than a clean failure). Stock `to_edge` export keeps the weight at
`[0,1,2,3]`; the failing `[1,0,2,3]` only arises through the reporter's NXP passes,
which I could not trigger locally.

So rather than assume, I proved the kernel is dim-order-correct. I built a standalone
C++ harness using ExecuTorch's verbatim `dim_order_to_stride_nocheck` and
`calculate_linear_index` (from `dim_order_util.h` / `tensor_util.h`). For the
`(2,1,3,3)` transposed weight, all 18 weight lookups returned identical values for
dim order `[1,0,2,3]` vs `[0,1,2,3]` — the kernel indexes the weight through strides
derived from its own dim order, so it reads correctly regardless of layout (the
strides even differ, yet every lookup matches). That established the fix was safe to
relax the check, not normalize the weight upstream.

## The Fix

Guard the weight dim-order check on `transposed` in `check_convolution_args`: skip
`tensor_is_default_or_channels_last_dim_order(weight)` when `transposed` is true (the
parameter is already available), with a comment explaining why the kernel does not
depend on that layout. One-line behavioral change plus the comment.

## Testing Strategy

Added `TransposedWeightNonDefaultDimOrder` in `kernels/test/op_convolution_test.cpp`:
a transposed conv with a `(2,1,2,2)` weight built in dim order `[1,0,2,3]` via
`make_with_dimorder`, asserted against a `torch.nn.ConvTranspose2d` reference output.
The test fails before the change (weight rejected) and passes after.

## Review and Merge

Copilot's automated review caught a genuine test-validity bug: my first draft used a
`(1,1,2,2)` weight, but with both channel dims size-1 the dim order `[1,0,2,3]`
collapses to the same strides as the default, so the test passed even unpatched and
did not actually exercise the fix. I moved to `(2,1,2,2)`, where the strides genuinely
differ (`[4,8,2,1]` vs default `[4,4,2,1]`), and recomputed the torch reference. Since
Copilot only reviews once, I also did a manual pass and fixed the test class
(`OpConvOutTest` -> `OpConvCorrectnessTest`) and a non-idiomatic tensor helper.

ExecuTorch CI runs the real C++ build and gtest on Meta-internal infra, so CI was the
end-to-end judge. An early run showed a wall of red, but it was a stale base: the
branch was 223 commits behind main, so arm/qnn/cortex-m jobs failed on infra
(ci-docker-hash mismatch, `git describe` failures) and lintrunner wanted my test
comments wrapped. After merging upstream/main (clean, no conflicts) and wrapping the
comments, everything cleared except one flaky OOM (exit 137) on an unrelated arm TOSA
model test. `rascani` merged it.

## Learnings & Reflections

- When relaxing a validation check, prove the code behind it is actually correct for
  the newly-allowed input before touching the check. The standalone harness answered
  that without needing the full build.
- Always read agentic reviews (like Copilot's): it caught a degenerate test shape that would have
  shipped a test that did not test anything. And always do a manual style + correctness
  pass on top regardless.
- A wall of red CI is often a stale base, not your change. Merging upstream/main and
  re-reading which jobs fail (infra vs. your files) separates signal from noise.

## Resources Used

- Issue #20804 and the reporter's NXP PRs (to confirm lane separation)
- `kernels/portable/cpu/util/kernel_ops_util.cpp`, `op_convolution.cpp`,
  `dim_order_util.h`, `tensor_util.h`
- `kernels/test/op_convolution_test.cpp` and `make_with_dimorder`
- `torch.nn.ConvTranspose2d` for the reference output
