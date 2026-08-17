# Index portable kernels by the tensor's strides instead of assuming row-major

**Project:** [pytorch/executorch](https://github.com/pytorch/executorch)
**Issue:** none, self-sourced, related to [#16429](https://github.com/pytorch/executorch/issues/16429) · **Pull request:** [#21828](https://github.com/pytorch/executorch/pull/21828) · **Branch:** `fix/dim-order-gate-flip-gather-scatter`
**Status:** Open, awaiting review. CI green on the completed jobs.

---

## The issue

### Why I picked it

I found this myself while working the dim-order family in ExecuTorch, after the transposed-convolution fix in #21035 and the `reduce_util` traversal fix in #21517. Both of those were reported bugs, and by the third one I had enough context to go looking rather than wait. My learning goal was differential testing as a discovery method, meaning running the runtime against eager PyTorch across a matrix of ops with only the memory format varying, and letting the mismatches tell me where to read.

That approach found six kernels in one sweep, three silently wrong and three saved only by accident.

### What was wrong

Three portable CPU kernels accept a channels-last input, return a tensor correctly labelled channels-last, and fill it with wrong data, and nothing errors. Three more share the same defect but happen to be masked by an unrelated check that rejects them at runtime. It matters because the failure is silent numerical corruption in ops that appear in ordinary models, and because the tensor is labelled correctly on the way out, so nothing downstream has any signal that the values are wrong. I picked it because a single line in a shared header explains all six, which makes it fixable rather than merely reportable.

Measured against eager PyTorch, same op and same values with only the memory format differing:

| kernel | contiguous input | channels-last input |
|---|---|---|
| `aten.flip` | matches | wrong, max abs diff 32 |
| `aten.gather` | matches | wrong, max abs diff 30 |
| `aten.scatter` | matches | wrong, max abs diff 30 |
| `aten.scatter_add` | matches | rejected, status 0x12 |
| `aten.roll` | matches | rejected, status 0x12 |
| `aten.permute_copy` | matches | rejected, status 0x12 |

### How I found it

Self-sourced on 2026-08-13 by differential testing against eager PyTorch, then root-caused in the kernel source and confirmed by reading across the whole affected family. Each kernel passes its own contiguous control, so the harness is validated per op rather than in aggregate, which matters because a harness bug would otherwise look exactly like a kernel bug. I confirmed no existing issue or PR covered it, and that all the affected files were byte-identical to `upstream/main`.

### Where it lives

- `runtime/core/exec_aten/util/tensor_util.h`, holding `coordinateToIndex` and `memoizeTrailingDims`, which is the single line the whole family reduces to.
- `kernels/portable/cpu/op_flip.cpp`, `op_roll.cpp` and `op_permute_copy.cpp`, which additionally write their output linearly by flat index.
- `kernels/portable/cpu/op_scatter_add.cpp`, carrying a guard that only existed because the indexing was row-major.

### What counts as done

1. The three silently wrong kernels produce the same values on channels-last input as they do contiguously.
2. The three incidentally rejected kernels work rather than merely continuing to error.
3. Contiguous behavior is unchanged by construction.
4. One test per kernel, each failing before the change.

---

## Diagnosis and plan

### Environment setup

Runtime verification used the repo's own `.venv` with ExecuTorch pybindings and torch 2.12.0 on macOS CPU, which is enough to export a model and execute it. The heavier C++ test builds ran on ROAR.

- **Challenge (the discovery harness has to be trusted before its results are):** a differential test that reports every op as wrong is far more likely to be a broken harness than six broken kernels. I made each op run its own contiguous control in the same harness, so a mismatch only counts when the contiguous case matches exactly and the channels-last case does not.
- **Challenge (the branch point moves under you):** I built against a fresh clone of upstream `main` at a tip newer than my branch point, and confirmed the patch still applied cleanly, which also confirmed the kernels were unchanged upstream while I worked.
- Carried over from #21517 are the ROAR module gotchas, which are setting `XDG_CACHE_HOME` before `module load` because home is full, loading `gcc/13.4.0` before `cmake/3.26.6`, injecting `-DPYTHON_EXECUTABLE` because the repo's test script does not, and building specific test targets rather than the `all` target that needs the full Python package.

### Steps to reproduce

1. Install ExecuTorch with pybindings.
2. Export a one-op model and run it on a channels-last input:
   ```python
   import torch
   from executorch.exir import to_edge_transform_and_lower
   from executorch.runtime import Runtime

   class M(torch.nn.Module):
       def forward(self, x): return torch.flip(x, [1])

   base = torch.arange(2*3*4*4, dtype=torch.float32).reshape(2,3,4,4)
   x    = base.to(memory_format=torch.channels_last)

   prog = to_edge_transform_and_lower(torch.export.export(M().eval(), (x,))).to_executorch()
   meth = Runtime.get().load_program(prog.buffer).load_method("forward")

   print(torch.equal(meth.execute([x])[0], M()(x)))
   ```
3. Repeat with the same tensor in contiguous format as a control.
4. Repeat for `gather`, `scatter`, `scatter_add`, `roll` and `permute_copy`.

### Expected vs. actual

**Actual:** the channels-last run prints `False` with no error raised, while the contiguous control prints `True`. For `scatter_add`, `roll` and `permute_copy` the run fails with status 0x12 instead.

**Expected:** both print `True`, because the memory format of a tensor should not change the values an op computes.

### Root cause

One line in `coordinateToIndex`:

```cpp
index += coordinate[d] * getTrailingDims(tensor, d);
```

`getTrailingDims` is the product of the sizes after `d`, which is the row-major stride rather than the tensor's real stride. A `(2, 3, 4, 4)` channels-last tensor has strides `[48, 1, 12, 3]`, but that expression yields `[48, 16, 4, 1]`, so the buffer is walked in the wrong order.

The reason nothing rejects it is a subtly wrong predicate. `flip_out` does check dim order, but with `tensors_have_same_dim_order(in, out)`, and here the input and output are both channels-last, so they agree and the check passes. What the kernel actually needed was that the dim order be the default one, because its index math is row-major only. `gather` and `scatter` check neither. `roll` and `permute_copy` survive only incidentally, since their output dim order differs from the input, so the same-dim-order check happens to fail and they error out rather than corrupt.

### The plan

**Understand:** A shared coordinate-to-index helper derives strides from sizes, so every kernel that indexes through it walks a non-contiguous buffer in the wrong order.

**Match:** The kernels here index element by element, which means reading the tensor's real strides is sufficient and no loop needs restructuring. `tensor.strides()` is already what the rest of the runtime uses, so this makes the helper agree with the rest of the system rather than introducing anything new.

**Plan:**
1. Make `coordinateToIndex` read `tensor.strides()`, and `memoizeTrailingDims` likewise for its one caller, `permute_copy`.
2. Map the output writes in `flip`, `roll` and `permute_copy` through strides too, since those wrote linearly by flat index, which is correct only when the output is contiguous.
3. Drop the `scatter_add` index guard, which existed only because the indexing was row-major.
4. Add one channels-last correctness test per kernel.

**Implement:** Branch `fix/dim-order-gate-flip-gather-scatter`, six commits.

**Review:** My self-checklist was that contiguous tensors are unaffected by definition, since for them the two expressions are equal, that no per-element cost is added, and that `getTrailingDims` itself is left alone because elsewhere it legitimately means the length of a contiguous run.

**Evaluate:** Seven tests, each comparing a channels-last run against the same values run contiguously, all failing before the change.

### Edge cases

- **Guard versus support.** My own findings note originally proposed adding a `tensor_is_default_dim_order` guard to each kernel, matching `op_addmm`, `op_bmm` and others. I rejected my earlier plan, because the same assumption appears in a dozen more places and #16429 asks for non-contiguous dim order to be supported rather than rejected. I put that choice in the PR and offered to switch.
- **Where stride-awareness is not enough.** `getTrailingDims` is unchanged here on purpose, because elsewhere it means the length of a contiguous run, for the block copies in `cat`, `split` and others. Those share the defect but need a different fix, which became [#21865](https://github.com/pytorch/executorch/pull/21865).
- **Output writes are a separate bug from input reads.** Fixing the helper alone repairs `gather`, `scatter` and `scatter_add`, but `flip`, `roll` and `permute_copy` also write output linearly, so they needed the second change as well.
- **Performance.** The new form is cheaper rather than more expensive, since `getTrailingDims` recomputes a product with an overflow check per dimension while a stride read does not. A later commit also skips the output index mapping entirely when the output is contiguous.

---

## Implementation

### Commits and files

| Commit | Date | Message |
|---|---|---|
| `64769fe13` | 2026-08-14 | Make `coordinateToIndex` use the tensor's strides instead of assuming row-major |
| `0ba31a776` | 2026-08-14 | Map the output write through strides in `roll` and `permute_copy` |
| `035e6b6f7` | 2026-08-14 | Drop the index dim order restriction in `scatter_add` now that indexing is stride-aware |
| `d5088eaa8` | 2026-08-15 | Skip the output index mapping when the output is contiguous |
| `a8b5c9b7a` | 2026-08-15 | Cover `permute_copy` and `scatter.value` with channels-last correctness tests |
| `e762e15c3` | 2026-08-14 | Merge `main` |

**Files modified:**

| File | Δ |
|---|---|
| `kernels/test/op_scatter_test.cpp` | +35 |
| `kernels/test/op_roll_test.cpp` | +32 |
| `kernels/test/op_flip_test.cpp` | +24 |
| `kernels/test/op_scatter_add_test.cpp` | +18 |
| `kernels/test/op_gather_test.cpp` | +16 |
| `kernels/test/op_permute_copy_test.cpp` | +16 |
| `kernels/portable/cpu/op_roll.cpp` | +16 / −14 |
| `kernels/portable/cpu/op_flip.cpp` | +15 / −17 |
| `kernels/portable/cpu/op_permute_copy.cpp` | +12 / −1 |
| `runtime/core/exec_aten/util/tensor_util.h` | +5 / −6 |
| `kernels/portable/cpu/op_scatter_add.cpp` | −3 |

Three of the four source files shrink, and the header does too.

### What was hard

The hard part was deciding to reverse my own conclusion. My findings note from the day before had settled on adding a default-dim-order guard to each affected kernel, which is the established idiom in this codebase and is what `op_addmm`, `op_bmm`, `op_avg_pool2d` and half a dozen others already do. It is defensible, it turns silent corruption into a clear `InvalidArgument`, and it is a much smaller change.

What changed my mind was counting how far the assumption spreads. The same row-major arithmetic appears in a dozen more places, so guarding each site is an unbounded amount of work that never actually supports the layout, and the issue this family traces back to asks for support rather than rejection. Once I could see that the element-indexed kernels needed only a stride read, the guard looked like the wrong default even though it was the safer-looking one.

The second difficulty was that fixing the read did not fix everything. `gather`, `scatter` and `scatter_add` were repaired by the helper change alone, but `flip`, `roll` and `permute_copy` also wrote their output by linear flat index, which is a separate instance of the same assumption on the other side of the operation. That only showed up because the tests compared against a contiguous control rather than merely checking the ops stopped erroring.

### Testing

- **Seven tests, one per kernel**, each running a channels-last input and checking it against the result the same values produce contiguously, so a test passes only if the kernel honors the tensor's dim order. All seven fail before the change.
- The comparison-against-contiguous shape is deliberate, because a test that only asserted "does not error" would have passed against the broken kernels for `flip`, `gather` and `scatter`, which never errored in the first place.
- Discovery evidence is the differential matrix in the summary table, measured against eager PyTorch with only the memory format differing, and each op passing its own contiguous control.
- CI is green on the completed jobs.

---

## Review and outcome

### The pull request

**[pytorch/executorch#21828](https://github.com/pytorch/executorch/pull/21828)**, opened 2026-08-14 against `pytorch/executorch:main`, self-labeled `release notes: ops & kernels`. It carries no `Fixes` keyword, since it is self-sourced and #16429 is a broader request that this only partly serves.

The body opens with the six-row differential table, so the reviewer sees the measured behavior before any explanation, then reduces all six to the single line in `coordinateToIndex` with the concrete strides for a `(2, 3, 4, 4)` tensor. It states that contiguous behavior is unchanged by definition, notes the change is also cheaper, explains why `getTrailingDims` itself is deliberately left alone, and names the guard alternative with an offer to switch.

An automated review from Copilot confirmed the diagnosis on 2026-08-14 and generated no comments on a second pass over all twelve files.

### Maintainer feedback

| Date | From | Feedback | My response |
|---|---|---|---|
| 2026-08-14 | Copilot review | Summarized the change as addressing silent data corruption in the portable implementations of `flip`, `gather` and `scatter` under non-default dim orders, and on a second pass over all twelve files generated no new comments. | n/a |
| 2026-08-15 | Me | Own updates, no maintainer feedback yet. | Added the contiguous-output fast path and extended coverage to `permute_copy` and `scatter.value`. |

No human review yet.

### What I learned

**Technical:** A single arithmetic assumption can be a whole family of bugs, and the useful question is not which kernels are broken but which expression they all share. Reducing six failing ops to one line in `coordinateToIndex` is what made the fix small. The second lesson is that the same assumption can appear on both sides of an operation, since reading input through row-major strides and writing output by linear flat index are the same mistake in two places, and fixing only the read leaves half the kernels wrong.

**Process & collaboration:** Differential testing against a reference implementation is a much better discovery tool than reading code, but only if each case carries its own control. Running the contiguous variant of every op through the identical harness is what let me trust six simultaneous failures instead of assuming my harness was broken. Writing up the alternative I rejected, including that it was my own earlier plan, also seems right, since a maintainer who prefers the guard can say so without me having hidden that it was on the table.

**What I'd do differently:** I would have written the contiguous control into the harness from the first run rather than adding it once the results looked implausible. I also spent a day committed to the guard approach before counting how many sites shared the assumption, and that count is what actually decided the design, so it should have come first.

### References

- Issue [#16429](https://github.com/pytorch/executorch/issues/16429), which asks for non-contiguous dim order support and is the reason this fixes rather than guards.
- `runtime/core/exec_aten/util/tensor_util.h`, `coordinateToIndex` and `memoizeTrailingDims`.
- My [#21517](https://github.com/pytorch/executorch/pull/21517), the same class of bug in `reduce_util`, and [#21035](https://github.com/pytorch/executorch/pull/21035), the transposed-convolution case.
- Follow-ups [#21865](https://github.com/pytorch/executorch/pull/21865) for the block-copy kernels and [#21866](https://github.com/pytorch/executorch/pull/21866) for the optimized `layer_norm`.
