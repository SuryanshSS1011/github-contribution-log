# Fix wrapped packing duplicating data when the input table is sliced

**Project:** [huggingface/trl](https://github.com/huggingface/trl)
**Issue:** [#6669](https://github.com/huggingface/trl/issues/6669), filed by me · **Pull request:** [#6670](https://github.com/huggingface/trl/pull/6670) · **Branch:** `fix-pack-wrapped-sliced-table`
**Status:** Open, awaiting review. CI green.

---

## The issue

### Why I picked it

I found this one by reading `trl/data_utils.py` looking for exactly this shape of defect, which is a silent data-integrity bug in a preprocessing path where nothing errors and the shapes stay right. Packing is a good place to look, because it rewrites the contents of a dataset while preserving row and token counts, so a mistake there produces plausible-looking output. My learning goal was PyArrow's zero-copy slicing model, specifically the difference between an array's logical view and the child buffer underneath it, which is the kind of detail that only bites when data arrives sliced.

### What was wrong

`_pack_wrapped` sizes its output from the array's sliced view but reads the token data from the raw child buffer, so when the input table is a slice it re-reads from the start of the buffer instead of from the slice. `datasets.map(batched=True)` hands each batch to the packing function as a slice, so every batch after the first can come back holding data from the beginning of the dataset. It matters because row counts and token counts are unchanged, so nothing errors and nothing downstream looks wrong, while the model trains on duplicated data with the rest silently dropped. I picked it because it is a correctness bug in a public, documented API that has shipped in every release from v1.0.0 through v1.9.0.

```python
ds = Dataset.from_dict({"input_ids": [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]})
pack_dataset(ds, seq_length=4, strategy="wrapped", map_kwargs={"batch_size": 3})["input_ids"]
# before: [[1, 2, 3, 4], [5, 6], [1, 2, 3, 4], [5, 6]]
# after:  [[1, 2, 3, 4], [5, 6], [7, 8, 9, 10], [11, 12]]
```

### How I found it

Self-sourced on 2026-08-05 by code reading, then confirmed with a minimal reproduction and bisected to the commit that introduced it. I filed [#6669](https://github.com/huggingface/trl/issues/6669) with the reproduction and opened the PR against it the same day, rather than opening a PR with no issue, because a silent data-integrity bug in a shipped API is worth having a citable report for even if the fix lands immediately.

### Where it lives

- `trl/data_utils.py`, `_pack_wrapped`, where the slice correction is computed into a local and then never used.
- `trl/sft_trainer.py:1619`, which calls `pack_dataset` and is how the bug can reach training.
- `tests/test_data_utils.py`, whose two existing `wrapped` tests use 3-row datasets and therefore only ever exercise a single batch.

### What counts as done

1. Multi-batch packing returns the dataset's own data rather than a repeat of the first batch.
2. Output is byte-identical to current behavior when the table is not sliced, so no correct result changes.
3. Both `list` and `large_list` types work, with int32 and int64 offsets.
4. Throughput is unchanged, since the code this regressed was a performance change.

---

## Diagnosis and plan

### Environment setup

Local CPU work with python 3.12, pyarrow 25.0.0, datasets 5.x and torch 2.13.0, against `main` at `d242e584`.

- **Challenge (the bug hides depending on how the dataset was built):** triggering it needs the table to actually be sliced. A table whose chunk boundaries line up with the map batch size gets concatenated by `combine_chunks` into a fresh array, which masks it, and an indices mapping from `shuffle`, `filter` or `select` makes `datasets` materialize batches via `take`, which masks it too. So a casual repro attempt can easily conclude the code is fine.
- **Challenge (most of the test file needs network):** the remaining 507 tests in `tests/test_data_utils.py` need tokenizer downloads, so I ran the offline subset covering Pack, Truncate, Conversational, Unpair, ExtractPrompt and Chatml, and noted that the untested remainder exercises chat templates, which this change does not touch.
- I matched TRL's pinned ruff 0.13.3 rather than whatever was in my environment, so lint results would match CI.

### Steps to reproduce

1. Install trl from `main` with pyarrow and datasets.
2. Build a six-row dataset directly with `Dataset.from_dict`, so no indices mapping exists.
3. Pack it with a batch size smaller than the dataset:
   ```python
   pack_dataset(ds, seq_length=4, strategy="wrapped", map_kwargs={"batch_size": 3})["input_ids"]
   ```
4. Compare against packing the same dataset in a single batch.

### Expected vs. actual

**Actual:** `[[1, 2, 3, 4], [5, 6], [1, 2, 3, 4], [5, 6]]`. Rows 3 to 5 are replaced by a second copy of rows 0 to 2. Row and token counts are unchanged, so nothing errors.

**Expected:** `[[1, 2, 3, 4], [5, 6], [7, 8, 9, 10], [11, 12]]`.

At 2500 rows with the default map batch size, 6000 of 10000 tokens are lost and replaced with duplicates.

I also measured which configurations actually corrupt:

| configuration | result |
|---|---|
| default `batch_size=1000` on a map-produced dataset | ok |
| `num_proc=4` | corrupt, 1000 of 10000 tokens lost |
| `batch_size=333` | corrupt, 2664 of 10000 lost |
| `batch_size=1500` | ok |
| `Dataset.from_dict` with no prior map, default batch size | corrupt, 6000 of 10000 lost |

### Root cause

`_pack_wrapped` computes the slice correction and then discards it:

```python
offsets, values = columns[0].offsets, columns[0].values
values = values[offsets[0].as_py() : offsets[-1].as_py()]  # slice-corrected
num_elements = len(values)                                 # uses the correction
offsets = np.arange(0, num_elements, seq_length, dtype=...)
offsets = np.concatenate((offsets, [num_elements]))
columns = [
    type(column).from_arrays(offsets.astype(...), column.values)  # ignores it
    for column in columns
]
```

`column.values` is the whole child buffer of the underlying table, so it still holds every row preceding the array's own offset. The new offsets are therefore applied from position 0 rather than from the slice start, and the function returns rows from the beginning of the buffer. The `values` local is computed correctly and is dead.

This is a regression rather than an original defect. The pre-2026 implementation passed the slice-corrected `values` and was correct. PR #5189 (`6c0fccd3`, 2026-03-14, a 35% packing speedup) rewrote the loop as a list comprehension over all columns and swapped in `column.values`, which left the correction stranded above it.

### The plan

**Understand:** The output is sized from the sliced view and filled from the unsliced buffer, so a non-zero slice offset shifts every read back to the start of the dataset.

**Match:** PyArrow already has the right primitive. `flatten()` respects the list array's own offset while `values` does not, so the fix is to use the accessor that means what the code intends, per column since each carries its own offsets.

**Plan:**
1. Replace `column.values` with `column.flatten()`, computed per column and zipped with the columns when rebuilding.
2. Confirm output is byte-identical on the non-sliced path.
3. Add a regression test that packs across two map batches.
4. Check `bfd` and `bfd_split`, which use the same pattern.
5. Benchmark, since the code being corrected was a performance PR.

**Implement:** Branch `fix-pack-wrapped-sliced-table`, single commit.

**Review:** My self-checklist was that no correct output changes, that both offset widths work, and that the performance claim of the PR I am partially reverting is not undone.

**Evaluate:** An independent reference implementation, checked over hundreds of randomized shapes as well as targeted edge cases.

### Edge cases

- **Only `wrapped` is affected.** `bfd` and `bfd_split` hit the same pattern but are protected by the `pc.filter` and `pc.take` that run before it, which materialize fresh arrays with offset 0.
- **Reachability from `SFTTrainer`.** Exactly one combination corrupts, which is `shuffle_dataset=False` with `dataset_num_proc` set. The default path is safe, but only because the pre-packing shuffle forces materialization, which is luck rather than design. The public `pack_dataset` API is broken independently of the trainer, and it is exported from `trl/__init__.py` and documented with examples.
- **Offset widths.** Checked on `large_list` with int64 offsets as well as `list` with int32.
- **Performance.** Since #5189 was a performance PR, I benchmarked rather than assumed. `flatten()` is metadata work like `values`, so at 500k rows by 16 tokens the median of 7 runs is 0.08 ms before and 0.09 ms after, which is within noise.

---

## Implementation

### Commits and files

| Commit | Date | Message |
|---|---|---|
| `4507eab72` | 2026-08-06 | Fix wrapped packing duplicating data when the input table is sliced |

**Files modified:**

| File | Δ |
|---|---|
| `tests/test_data_utils.py` | +15 |
| `trl/data_utils.py` | +6 / −5 |

### What was hard

The hard part was not finding the bug but proving the fix is complete, because the failure is silent and the surface is combinatorial. A test that packs two batches shows the specific reported case is fixed, and says nothing about ragged rows, empty rows, zero-row tables, multi-chunk tables, `seq_length` of 1, or `seq_length` larger than the whole dataset.

So I wrote an independent reference implementation, which concatenates each column's rows in order and then re-chunks, and checked `_pack_wrapped` against it over 600 randomized shapes plus targeted edge cases covering slices at every offset, the degenerate `seq_length` values, ragged and empty rows, zero-row tables, multi-chunk tables, 1 to 3 columns, both list types and both offset widths. All pass with the fix and 127 of them fail without it. That number is the useful one, because it says the reference and the implementation genuinely disagree before the change rather than the harness being vacuous.

The second difficulty was scoping the claim about who is affected. It would have been easy to say `SFTTrainer` corrupts training data, which is alarming and mostly wrong. Measuring the four combinations showed the default path is safe because the pre-packing shuffle happens to materialize the table, and only `shuffle_dataset=False` with `dataset_num_proc` set actually corrupts. Reporting that accurately matters more than reporting it dramatically.

### Testing

- **`TestPackDatasetWrapped::test_with_multiple_map_batches`** packs across two map batches. It fails on `main` with the duplication above and passes with the fix. The two existing `wrapped` tests use 3-row datasets, so they only ever exercise a single batch, which is why this was never caught.
- **Reference-implementation cross-check** over 600 randomized shapes plus the targeted edge cases described above. All pass with the fix and 127 fail without it.
- **End-to-end round trip** through `pack_dataset` across batch sizes 1 to 777, `num_proc` of 2 and 4, and all three strategies, with `input_ids`, `attention_mask` and `labels` staying aligned.
- `pytest tests/test_data_utils.py -k "Pack"` gives 10 passed, 548 deselected. The wider offline subset gives 51 passed.
- Byte-identical output to current behavior on the non-sliced path, so no existing correct result changes.
- `ruff check` and `ruff format --check` clean with TRL's pinned ruff 0.13.3.
- CI green.

---

## Review and outcome

### The pull request

**[huggingface/trl#6670](https://github.com/huggingface/trl/pull/6670)**, opened 2026-08-06 against `huggingface/trl:main`, referencing `Fixes #6669`, with `@qgallouedec` and `@albertvillanova` requested for review.

The body opens with the mechanism, that the output is sized from the sliced view and read from the raw buffer, then the before-and-after reproduction as executable code, then the origin in #5189 with the specific commit. It states that only `wrapped` is affected and why `bfd` and `bfd_split` are not, confirms byte-identical output on the non-sliced path, and reports the benchmark, since the code being corrected came from a performance PR. TRL's template asks for an AI usage disclosure, which I filled in as AI-assisted.

An automated reviewer, Cursor Bugbot, independently summarized the same diagnosis and rated it medium risk as a silent data-integrity bug in a core preprocessing path.

### Maintainer feedback

| Date | From | Feedback | My response |
|---|---|---|---|
| 2026-08-06 | Cursor Bugbot | Confirmed the diagnosis, that later batches arrive as sliced tables while the old code read `column.values` from offset 0, and noted the fix preserves byte-identical output for non-sliced inputs. | n/a, it agreed with the analysis. |

No human review yet.

### What I learned

**Technical:** In PyArrow, an array's logical view and its child buffer are different things, and the accessors that expose them look interchangeable. `values` returns the whole child buffer while `flatten()` respects the array's own offset, so code that mixes them works perfectly until something upstream hands it a slice. The tell in this case was a computed local that nothing reads, since a dead variable holding a correction usually means the correction was meant to be applied and no longer is.

**Process & collaboration:** Bisecting to the commit that introduced a regression is worth the time, because it changes the story from "this code is wrong" to "this specific rewrite dropped a correction", which is far easier for a reviewer to confirm. Measuring the reachability matrix instead of asserting that the trainer is affected is the other thing I would repeat, because the honest answer, that the default path is saved by an unrelated shuffle, is more useful than the alarming one and is what tells a maintainer how urgent this is.

**What I'd do differently:** I would build the reference implementation first and use it to find the bug, rather than reading code to find the bug and then building the reference to prove the fix. The reference took an hour and found 127 disagreements, so it would have located this on its own, and it would also have covered the cases I had to think of by hand.

### References

- Issue [#6669](https://github.com/huggingface/trl/issues/6669), which I filed with the reproduction.
- `trl/data_utils.py`, `_pack_wrapped`, and PR #5189 (`6c0fccd3`), which introduced the regression.
- PyArrow documentation on `ListArray.flatten` against `ListArray.values`.
- `trl/sft_trainer.py:1619` and `:1404`, for how packing is reached from training.
