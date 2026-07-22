# Contribution: Make the dataset fingerprint independent of Arrow chunking

**Student:** Suryansh Sijwali
**Issue:** https://github.com/huggingface/datasets/issues/8327
**Pull Request:** https://github.com/huggingface/datasets/pull/8339
**Status:** Merged 2026-07-22 by `lhoestq` (PR #8339, commit 3bfbc67ce). Issue
#8327 closed as completed.

---

## Why I Chose This Issue

I had already filed the bisection on this issue in an earlier session, so the
diagnostic work was mine and the maintainer was engaged. It is a pure-Python
`datasets` bug with a measurable, reproducible symptom (an OOM), and the fix is a
targeted change to how the dataset fingerprint is computed. No GPU, no model
downloads, and the repo merges outside contributors on a reasonable cadence.

## Understanding the Issue

The report says `Dataset.from_pandas` allocates ~1 MB per Arrow chunk, so a
DataFrame that has been through `shuffle().to_pandas()` (which produces one Arrow
chunk per row) blows up to tens of GB and the process is killed. The reporter
correctly narrowed it to "something in `Dataset`" but not to a specific function.

My bisection (posted to the issue) showed the allocation is not in the Arrow
conversion at all. On `datasets` 5.0.0:

- `pa.Table.from_pandas(df)` peak: ~0 MB
- `Dataset.from_pandas(df)` peak: 1499 MB
- `generate_fingerprint(ds)` alone: 1496 MB

So ~99.8% of the allocation is in fingerprinting. Holding the data constant at
600 rows and varying only the chunk count (50 / 200 / 600 chunks) made the
fingerprint peak scale linearly with chunk count, confirming the cost tracks
chunking rather than data size.

## Root Cause

`generate_fingerprint` (`src/datasets/fingerprint.py`) hashes `dataset.__dict__`,
which includes the underlying Arrow table. The hasher serializes objects with
`datasets`' dill-based `dumps`. There was no custom reducer for `pa.Table` /
`pa.ChunkedArray`, and `Dataset._data` is an `InMemoryTable` wrapper whose
`__dict__` also carries a `_batches` list (one `RecordBatch` per chunk). Default
pickling of the wrapper and the table both serialize chunk-by-chunk, so the
pickle stream — and therefore the hashing cost — grows with the number of chunks.
Measured: the pickled size of a 600-row table went from 1.3 MB (50 chunks) to
15.9 MB (600 chunks) for identical data.

## The Fix

`lhoestq` preferred making the table hash chunk-count-independent (rather than
calling `combine_chunks()` on the stored data, which would add a copy). Two
coordinated changes:

1. `InMemoryTable.__getstate__` / `__setstate__` (in `table.py`): serialize via
   the underlying `self.table` only. The `_batches` / `_offsets` index is derived
   and rebuilt in `__init__`, so it doesn't need pickling. This mirrors how
   `MemoryMappedTable` and `ConcatenationTable` already define their own
   `__getstate__`.
2. Dill reducers for `pa.Table` and `pa.ChunkedArray` (in `utils/_dill.py`) that
   serialize a chunk-count-independent form by combining each column's chunks one
   at a time (bounded memory, never a whole-table copy).

Both are needed: the first routes `InMemoryTable` pickling through `self.table`,
the second makes that table hash chunk-independently and cheaply.

## Verification

Ran on a Slurm node with real RSS measurement (not `tracemalloc`, which cannot
see pyarrow's C allocations):

- `Dataset.from_pandas` on the 2000-chunk repro: RSS delta dropped from a ~1500 MB
  blowup to ~4 MB.
- Fingerprint is stable across different chunkings of identical data, and still
  distinguishes different data (no collisions).
- `pickle` round-trip of a dataset reconstructs correctly.
- `tests/test_fingerprint.py` and `tests/test_table.py` pass (306 tests; the two
  failures, `test_hash_torch_compiled_module` and `test_move_script_doesnt_change_hash`,
  also fail on clean `main` in my environment and are unrelated).
- Added `test_hash_arrow_table_is_independent_of_chunking`, which fails on
  unpatched source and passes with the fix.

## What I Learned

The interesting part was that the report's stated cause (`from_pandas`) was wrong,
and confirming the real one required measuring rather than reading. The fix also
had a false first attempt: I initially targeted `pa.Table` only, but the real
object is the `InMemoryTable` wrapper, and a naive whole-table `combine_chunks()`
was actually more memory-expensive for pathological chunking. Matching the
existing sibling-class serialization pattern was the clean resolution.

## Code / Style Notes

Followed the repo's own conventions after a review pass: the dill reducers use the
nested `create_X` closure form like the existing `create_torchGenerator` /
`create_spacyLanguage` reducers, the test imports pyarrow inline (matching the
other tests in that file), and the `__getstate__` comment is kept terse to match
the sibling wrappers.
