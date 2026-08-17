# Fix out-of-bounds traversal in `reduce_util` for non-contiguous tensors

**Project:** [pytorch/executorch](https://github.com/pytorch/executorch)
**Issue:** [#16429](https://github.com/pytorch/executorch/issues/16429) · **Pull request:** [#21517](https://github.com/pytorch/executorch/pull/21517) · **Branch:** `fix/16429-reduce-util-noncontiguous`
**Status:** Open, approved by `JacobSzwejbka` on 2026-08-13 and awaiting merge. One red check, the Arduino library job, unrelated to CPU kernels.

---

## The issue

### Why I picked it

This is my third ExecuTorch kernel contribution, after the MLX build isolation in #20585 and the transposed-convolution dim-order fix in #21035. The dim-order family is where I had built up the most context, and #16429 was reported as wrong output from `dequantize_per_channel` on a channels-last input, which is the same shape of bug as #21035 but one layer deeper. My learning goal was to get from a reported symptom to the actual memory error underneath it, rather than stopping at the kernel that appears in the traceback.

The issue was filed by `MartinPavella` in January 2026 and had sat open, which is a long time for a report of numerically wrong output in a quantized kernel.

### What was wrong

`dequantize_per_channel` returns wrong values on a channels-last input, but the wrong values are a symptom and the real defect is a memory error in a shared reduction helper. `apply_on_flat_ix_with_dim_mask_and_base` in `reduce_util.h` rewinds its cursor using an identity that only holds for contiguous tensors, so on a channels-last tensor the walk leaves the output buffer entirely. It matters because it is an out-of-bounds write in a kernel that ships by default, and the visible symptom (wrong numbers) is much less alarming than the actual behavior, which under the existing tests produces `double free or corruption (!prev)`. I picked it because the report pointed at the quantized kernel while the bug is not in the quantized kernel at all.

### Before I started

Open, unassigned, and with no in-flight PR after seven months. I confirmed the fault still reproduced on current `main` before writing anything, and I widened the check beyond the reported op, since a bug in a shared helper is rarely confined to the one caller somebody noticed.

### Where it lives

- `kernels/portable/cpu/util/reduce_util.h`, holding `apply_on_flat_ix_with_dim_mask_and_base` and the carry-over line that does the rewind.
- The three callers that actually reach it with a non-contiguous tensor, which are `dequantize_per_channel`, `op_any` and `op_var_mean`.
- The five reduction ops that never reach it, `op_amax`, `op_amin`, `op_mean`, `op_sum` and `op_var`, because they all guard with `tensor_is_default_dim_order(in)` first.

### What counts as done

1. The reported `dequantize_per_channel` case produces correct values with no out-of-bounds access.
2. `any.dims_out`, which nobody reported, is also fixed, since it shares the helper.
3. Contiguous tensors are provably unaffected.
4. The fix is not on the per-element path, so reductions do not get slower.
5. Tests fail before the change and pass after it.

---

## Diagnosis and plan

### Environment setup

I used the repo's CMake presets and its `kernels/test` gtest layout, with the heavier builds on Penn State's ROAR Collab cluster and quick iteration locally on macOS.

- **Challenge (module system silently broken by a full home directory):** ROAR home was at 100%, so Lmod could not write its cache and `module load` failed without saying why. Setting `XDG_CACHE_HOME=$HOME/scratch/.cache` first fixed it.
- **Challenge (module load order matters):** `gcc/13.4.0` has to be loaded before `cmake/3.26.6`, because loading gcc afterwards reorders the hierarchical module tree and drops cmake from `PATH`.
- **Challenge (the repo's own test script does not configure cleanly):** `test/run_oss_cpp_tests.sh` does not pass `-DPYTHON_EXECUTABLE`, and cmake's `find_package(Python3)` ignores `PATH` order, so it picks a python without torch and configure dies in `get_torch_base_path`. I injected the flag locally rather than fighting the script.
- **Challenge (the script's `all` target cannot run):** it fails generating `.pte` test resources, which needs the full ExecuTorch Python package, so I built the specific test targets directly and skipped that entirely.

### Steps to reproduce

1. Build the quantized and portable kernel test targets.
2. Construct a `(2, 2, 3, 3)` int8 tensor in channels-last memory format, whose strides are `[18, 1, 6, 2]`.
3. Run `dequantize_per_channel` on it with `axis=1`.
4. Compare the output against per-channel dequantization computed from each element's channel alone.
5. Run it under the existing kernel tests and watch for allocator errors on exit.

### Expected vs. actual

**Actual:** the walk covers output indices 0 to 17, skips 18 to 34 entirely, and writes 17 times past the end of a 36 element output. Under the existing kernel tests that surfaces as wrong values followed by `double free or corruption (!prev)`.

**Expected:** every output index is visited exactly once and nothing is written outside the buffer.

I also confirmed a second victim that nobody had reported. `any.dims_out` reads out of bounds on the same path and returns a wrong answer, because the skipped range is exactly where its data sits.

### Root cause

`apply_on_flat_ix_with_dim_mask_and_base` walks the elements mapping to one output index. When the innermost dimension is exhausted it rewinds the cursor and carries into the next dimension with:

```cpp
curr_index -= strides[curr_dim - 1];
```

The comment directly above that line states the assumption, which is that `strides[curr_dim - 1]` equals `in.size(curr_dim) * strides[curr_dim]`. That identity holds only for a contiguous tensor. For the `(2, 2, 3, 3)` channels-last case the strides are `[18, 1, 6, 2]`, and the two expressions disagree at `curr_dim = 1`, where they are 2 against 18, and again at `curr_dim = 2`, where they are 18 against 1. The rewind is then wrong by exactly that difference and the walk leaves the buffer.

So the code and its own comment disagree, and the comment is right.

### The plan

**Understand:** A shared reduction helper rewinds using a contiguity identity, so any caller that admits a non-contiguous tensor walks off the buffer.

**Match:** The fix is to subtract the value the existing comment already describes. That keeps the helper's contract as documented rather than introducing a new one, and for a contiguous tensor the two expressions evaluate to the same number, so that path is untouched by construction.

**Plan:**
1. Change the carry-over to compute `in.size(curr_dim) * strides[curr_dim]` as the comment specifies.
2. Confirm the change sits in the carry-over rather than the inner loop, so it is not on the per-element path.
3. Add a test for the reported `dequantize_per_channel` case.
4. Add a test for `any.dims_out`, the unreported second victim.
5. Audit which reduction ops can actually reach the helper with a non-contiguous tensor.

**Implement:** Branch `fix/16429-reduce-util-noncontiguous`.

**Review:** My self-checklist was that contiguous behavior is provably identical, that the diff stays inside the helper rather than spreading guards across kernels, and that both new tests fail before the change.

**Evaluate:** Full runs of `quantized_kernels_test` and `portable_kernels_test`, plus reverting the helper while keeping the tests.

### Edge cases

- **Which ops actually reach this.** Most reduction ops never do, since `op_amax`, `op_amin`, `op_mean`, `op_sum` and `op_var` all guard with `tensor_is_default_dim_order(in)` and reject non-contiguous input before the traversal runs. The ones that accept it are `op_any`, `op_var_mean` and `dequantize_per_channel`, and I confirmed the fault in two of the three.
- **Guard versus fix.** Adding a `tensor_is_default_dim_order` guard to `dequantize_per_channel`, matching the other five, would stop the corruption in one line. I chose not to, because it leaves the traversal wrong for `op_any` and `op_var_mean`, and because #16429 asks for non-contiguous dim order to be supported rather than rejected. I said so in the PR and offered to switch.
- **Scope discipline.** Relaxing the five existing guards, so those ops also accept non-contiguous input, is a larger behavior change and is deliberately not part of this PR.
- **A build-system gap found on the way.** `op_dequantize_test.cpp` is listed in `kernels/quantized/test/targets.bzl` but was missing from `_quantized_kernels_test_sources` in `kernels/test/CMakeLists.txt`, while every other op in that directory is present. So those tests never ran in a CMake build, and my new one would not have either.

---

## Implementation

### Commits and files

| Commit | Date | Message |
|---|---|---|
| `36674e605` | 2026-07-31 | Fix out-of-bounds traversal in reduce_util for non-contiguous tensors |
| `79c3303ac` | 2026-08-04 | Clarify contiguous memory format wording and add reduce_util test |
| `9df2973c0` | 2026-08-13 | Merge `main` |

**Files modified:**

| File | Δ |
|---|---|
| `kernels/quantized/test/op_dequantize_test.cpp` | +51 |
| `kernels/portable/cpu/util/test/reduce_test.cpp` | +38 |
| `kernels/test/op_any_test.cpp` | +26 |
| `kernels/portable/cpu/util/reduce_util.h` | +3 / −4 |
| `kernels/test/CMakeLists.txt` | +1 |

The behavioral change is a net one line in a header. Everything else is tests and the CMake registration that makes one of those test files run at all.

### What was hard

The hard part was refusing the obvious fix. The issue names `dequantize_per_channel`, the failing test is in the quantized kernels, and a one-line `tensor_is_default_dim_order` guard on that kernel would have closed the report and matched what five sibling ops already do. It would also have left the actual out-of-bounds write in place for two other callers, one of which I had already confirmed returns wrong answers.

Working out which callers could reach the helper was what settled it. Five of the eight reduction ops guard before the traversal, so they were never at risk, and the three that do not are exactly the three that admit non-contiguous input. Once the population was that small it was clear the defect belonged to the helper rather than to any one kernel.

The second thing worth recording is that the fix was already written down. The comment above the offending line describes the correct expression, and the code beneath it does something else. Reading the comment as a specification rather than as decoration is what made this a one-line change instead of a redesign.

### Testing

- **`OpDequantizeOutTest.DequantizePerChannelChannelsLast`** dequantizes a `(2, 2, 3, 3)` channels-last int8 tensor per channel on `axis=1`. Each expected value is derived from the element's channel alone, so it passes only if the kernel honors the tensor's dim order.
- **`OpAnyOutTest.ChannelsLastMultiDimReduction`** reduces a channels-last tensor over dims `{0, 2, 3}` with `keepdim=true`. The single true value sits at channels-last physical index 19, inside the range the old traversal skipped, so before the change channel 1 was reported false while channel 0 picked up out-of-bounds memory and was reported true.
- **`ReduceUtilTest.ApplyOverDimListChannelsLast`** was added at the reviewer's request, covering the helper directly rather than through a kernel.
- Both original tests fail before the change and pass after it.
- **Full runs after the change:** `quantized_kernels_test` 73 passed, and `portable_kernels_test` 1624 ran with 1520 passed and 0 failures.
- CI is green apart from `test-arduino-library`, which does not build CPU kernels.

---

## Review and outcome

### The pull request

**[pytorch/executorch#21517](https://github.com/pytorch/executorch/pull/21517)**, opened 2026-07-31 against `pytorch/executorch:main`, referencing `Fixes #16429`, self-labeled `release notes: ops & kernels` via `@pytorchbot`, and cc'ing `@larryliu0820`, `@manuelcandales` and `@JakeStevens`.

The body leads with the correction that the reported symptom is wrong values but the underlying problem is a memory error and it is not in the quantized kernel, then shows the offending line against its own comment, then gives the concrete index arithmetic for the reported tensor. It states which ops can and cannot reach the helper, explains why I chose the traversal fix over a per-kernel guard, offers to switch if the maintainers prefer, and flags the missing CMake registration as a separate thing the PR also fixes.

### Maintainer feedback

| Date | From | Feedback | My response |
|---|---|---|---|
| 2026-07-31 | nil-is-all | "Thanks for the PR. Running CI currently", cc'ing MartinPavella, metascroy, Gasoonjia, JacobSzwejbka and JakeStevens. | n/a |
| 2026-08-03 | Gasoonjia | "Thanks for your great work! Please add an extra test here", pointing at `kernels/portable/cpu/util/test/reduce_test.cpp`, plus a comment-wording note. | Added `ReduceUtilTest.ApplyOverDimListChannelsLast` so the helper is covered directly rather than only through kernels, and rephrased the marked comment to name contiguous memory format unambiguously. Both in `79c3303ac`. |
| 2026-08-13 | JacobSzwejbka | **Approved** with "lgtm. Thanks!" | Merged `main` into the branch the same day to keep it current while it waits. |

### What I learned

**Technical:** When a comment and the code below it disagree, the comment is often the specification and the code is the bug. That was literally true here, since the fix is to compute the expression the comment already describes. The wider lesson is about where a defect lives. The issue named a quantized kernel, the wrong numbers appeared in a quantized kernel, and the defect was in a shared reduction helper that three unrelated ops call. Enumerating the callers, and noticing that five of eight guard themselves out of the problem, is what turned a plausible one-kernel patch into the correct one-line fix.

**Process & collaboration:** Stating the alternative I rejected, and offering to switch, is what I would repeat. The guard fix was genuinely defensible and a maintainer might have preferred it, so putting that choice in the body with the reasoning meant nobody had to guess whether I had considered it. The review that followed was one substantive request, for a test at the helper level rather than only at the kernel level, which was a better placement than mine and cost one commit to satisfy.

**What I'd do differently:** I would check the build registration of any test file I add to before assuming it runs. I found by accident that `op_dequantize_test.cpp` was missing from the CMake source list, which means my new test would have silently never executed in a CMake build. That is the same category of failure as the bug itself, something that quietly does nothing, and I should look for it deliberately rather than stumble on it.

### References

- Issue [#16429](https://github.com/pytorch/executorch/issues/16429), reported by `MartinPavella`.
- `kernels/portable/cpu/util/reduce_util.h`, and the comment above the carry-over line that describes the correct expression.
- My earlier [#21035](https://github.com/pytorch/executorch/pull/21035), the transposed-convolution dim-order fix, which is the same family of bug.
- `kernels/quantized/test/targets.bzl` against `kernels/test/CMakeLists.txt`, where the registration gap showed up.
