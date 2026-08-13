# Make the dataset fingerprint independent of Arrow chunking

**Project:** [huggingface/datasets](https://github.com/huggingface/datasets)
**Issue:** [#8327](https://github.com/huggingface/datasets/issues/8327) · **Pull request:** [#8339](https://github.com/huggingface/datasets/pull/8339) · **Branch:** `fix-fingerprint-chunk-independent-8327`
**Status:** **Merged** 2026-07-22 (merge commit `3bfbc67ce`). Issue closed.

---

## The issue

### Why I picked it

`datasets` is the data layer under most of the HuggingFace training stack, so a memory bug in it affects an enormous number of pipelines. I had already filed the bisection on this issue in an earlier session, so the diagnostic work was mine and the maintainer was already engaged with my findings, which made finishing it the obvious next step. My learning goal was measurement discipline, meaning proving where memory actually goes in a mixed Python and Arrow (C++) system, where the usual Python tooling lies to you.

It is a pure-Python bug with a measurable, reproducible symptom in the form of an OOM, the fix is a targeted change to how the fingerprint is computed, and it needs no GPU or model downloads.

### What was wrong

`Dataset.from_pandas` allocates roughly 1 MB per Arrow chunk, so a DataFrame that has been through `shuffle().to_pandas()`, which produces one Arrow chunk per row, balloons to tens of GB and gets the process killed. The reported cause was wrong though, because the allocation is not in the Arrow conversion but in fingerprinting, which dill-pickles the underlying table chunk by chunk so that cost tracks chunk count rather than data size. It matters because the trigger is an ordinary `shuffle()` and the failure is an OOM kill with no indication of where the memory went. I chose it because I had already bisected it, and because the fix sits at a nice depth as a real serialization design question rather than a one-line guard.

### Before I started

Open, unassigned, with no in-flight PR. I posted my bisection to the issue on 2026-07-16 as measurements rather than a claim, and `lhoestq`, the maintainer, responded the same day endorsing the approach and naming the constraint he cared about:

> making the table hash chunk-count-independent makes sense and would avoid the extra copy from `combine_chunks()`. Though at the same time I understand it's maybe not as easy to do, let me know how this goes if you plan to look into it :)

That comment is what set the design, which is chunk-count-independent hashing without an extra whole-table copy.

### Where it lives

- `src/datasets/fingerprint.py`, holding `generate_fingerprint`, which hashes `dataset.__dict__`.
- `src/datasets/table.py`, holding `InMemoryTable`, the wrapper whose `__dict__` carries the per-chunk `_batches` list, where `MemoryMappedTable` and `ConcatenationTable` already define their own `__getstate__`.
- `src/datasets/utils/_dill.py`, the dill reducer registry with `create_torchGenerator` and `create_spacyLanguage`.

### What counts as done

1. The repro's memory blowup disappears, measured in real RSS rather than `tracemalloc`.
2. The fingerprint is stable across different chunkings of identical data.
3. It still distinguishes different data, so no collisions are introduced.
4. No whole-table copy is added, which was `lhoestq`'s explicit constraint.
5. A `pickle` round-trip of a dataset still reconstructs correctly.

---

## Diagnosis and plan

### Environment setup

I used the repo's editable install and pytest layout, and ran the measurement on a Slurm compute node so a multi-GB allocation could be observed without disturbing my laptop.

- **Challenge (the obvious measurement tool cannot see the bug):** `tracemalloc` only tracks Python allocations and is blind to pyarrow's C++ allocations, which is most of what is being measured here. I resolved it by measuring real RSS deltas around each call instead.
- **Challenge (separating size of data from number of chunks):** the two are normally correlated, so the natural repro cannot distinguish them. I resolved it by holding the row count fixed at 600 and varying only the chunk count across 50, 200 and 600.
- The run was on a Slurm node with `datasets` 5.0.0, where a roughly 1.5 GB spike is comfortably observable.

### Steps to reproduce

1. Install `datasets` 5.0.0.
2. Build a DataFrame with many Arrow chunks, for example `ds.shuffle().to_pandas()`, which yields one chunk per row.
3. Measure peak memory around each of these separately:
   ```python
   pa.Table.from_pandas(df)      # Arrow conversion alone
   Dataset.from_pandas(df)       # the reported symptom
   generate_fingerprint(ds)      # fingerprinting alone
   ```
4. Then hold the data constant at 600 rows and vary only the chunk count across 50, 200 and 600, re-measuring the fingerprint peak each time.

### Expected vs. actual

**Actual**, on `datasets` 5.0.0:

| Step | Peak |
|---|---|
| `pa.Table.from_pandas(df)` | ~0 MB |
| `Dataset.from_pandas(df)` | 1499 MB |
| `generate_fingerprint(ds)` alone | 1496 MB |

So about 99.8% of the allocation is fingerprinting rather than the Arrow conversion the issue blamed. Holding data constant and varying only chunk count made the fingerprint peak scale linearly with chunk count, which confirms the cost tracks chunking rather than data size. Measured directly, the pickled size of a 600-row table grows from 1.3 MB at 50 chunks to 15.9 MB at 600 chunks for identical data.

**Expected:** fingerprinting a dataset should cost a function of the data rather than of how that data happens to be chunked, and `Dataset.from_pandas` should not allocate about 1.5 GB for a 600-row frame.

### Root cause

`generate_fingerprint` hashes `dataset.__dict__`, which includes the underlying Arrow table, and the hasher serializes with `datasets`' dill-based `dumps`. Two things compound:

1. There was no custom reducer for `pa.Table` or `pa.ChunkedArray`, so default pickling walks chunk by chunk.
2. `Dataset._data` is an `InMemoryTable` wrapper whose `__dict__` also carries a `_batches` list with one `RecordBatch` per chunk, and that gets pickled too.

Both the wrapper and the table therefore serialize per chunk, so the pickle stream, and with it the hashing cost, grows with the number of chunks.

### The plan

**Understand:** Hash cost is proportional to chunk count because serialization is per chunk in two independent places.

**Match:** `MemoryMappedTable` and `ConcatenationTable`, siblings of `InMemoryTable` in the same file, already define their own `__getstate__` and `__setstate__`. The reducer side has an established form too, since the existing `create_torchGenerator` and `create_spacyLanguage` reducers in `_dill.py` use a nested `create_X` closure. Both halves of the fix have an in-repo template.

**Plan:**
1. Give `InMemoryTable` a `__getstate__` and `__setstate__` that serialize via `self.table` only, since `_batches` and `_offsets` are derived and rebuilt in `__init__` so they need not be pickled.
2. Add dill reducers for `pa.Table` and `pa.ChunkedArray` that serialize a chunk-count-independent form by combining each column's chunks one at a time, which keeps memory bounded and never makes a whole-table copy.
3. Verify with real RSS, chunk-count independence and collision behavior.
4. Add a regression test that fails on unpatched source.

**Implement:** Branch `fix-fingerprint-chunk-independent-8327`.

**Review:** My self-checklist was that the reducers follow the existing `create_X` closure form, the `__getstate__` comment stays terse to match the sibling wrappers, the test imports pyarrow inline like the other tests in that file, and there is no whole-table copy anywhere.

**Evaluate:** Re-run the RSS measurement on the 2000-chunk repro, check fingerprint stability and distinctness, and round-trip a pickled dataset.

### Edge cases

- **Both halves are required:** The `__getstate__` alone would route pickling through `self.table` but leave the table itself hashing per chunk, and the reducers alone would not stop `_batches` from being pickled. I verified each independently rather than assuming the pair.
- **Memory during the fix itself:** A naive `combine_chunks()` on the whole table is more memory-expensive for pathological chunking, which is the thing being fixed, so combining per column one chunk at a time keeps the peak bounded.
- **Fingerprint distinctness:** Making the hash chunk-independent must not make it collide across different data, which I tested explicitly.
- **Cache invalidation:** The change alters fingerprint values once, which invalidates existing caches. That is a user-visible consequence, so I flagged it to the maintainer on the issue and in the PR body rather than letting it be discovered.
- **Duplicate column names:** Arrow permits them, and this surfaced in review.

---

## Implementation

### Commits and files

| Commit | Date | Message |
|---|---|---|
| `543ee3e4e` | 2026-07-16 | Make the dataset fingerprint independent of Arrow chunking |
| `f719dc953` | 2026-07-18 | Refactor `create_arrowTable` to use `from_arrays`, a review fix preserving duplicate field names |

**Files modified:**

| File | Δ |
|---|---|
| `src/datasets/utils/_dill.py` | +32 |
| `src/datasets/table.py` | +9 |
| `tests/test_fingerprint.py` | +15 |

### What was hard

The first challenge was that the issue's stated cause was wrong and the standard tool could not show it. `tracemalloc` reports almost nothing here because the memory is in pyarrow's C++ allocator, so the only way to find the real cost was RSS measurement around isolated calls, followed by a second experiment with fixed rows and varying chunks to prove the cost tracked chunking rather than size. Reading the code would not have gotten there, since the fingerprinting path is not where anyone looks first.

The second was a false start in the fix. I initially wrote a reducer for `pa.Table` only, which did not help much, because the object actually being pickled is the `InMemoryTable` wrapper and its `_batches` list. My first instinct of `combine_chunks()` on the whole table was actively counterproductive, because for the pathological chunking this issue is about, that copy is exactly the memory blowup being fixed. `lhoestq` had pre-empted this in his issue comment about avoiding the extra copy from `combine_chunks()`, which is why reading the maintainer's constraint carefully mattered more than moving fast.

The resolution in both cases was to match the existing sibling pattern rather than invent one, since `MemoryMappedTable` and `ConcatenationTable` already showed how a table wrapper should define its own state.

### Testing

- **New:** `test_hash_arrow_table_is_independent_of_chunking` in `tests/test_fingerprint.py`, which fails on unpatched source and passes with the fix. It imports pyarrow inline, matching the other tests in that file.
- **Existing suite:** `tests/test_fingerprint.py` and `tests/test_table.py` give 306 tests passing. Two failures, `test_hash_torch_compiled_module` and `test_move_script_doesnt_change_hash`, also fail on clean `main` in my environment and are unrelated to this change.
- **Before and after evidence:** on the 2000-chunk repro, the `Dataset.from_pandas` RSS delta drops from a roughly 1500 MB blowup to about 4 MB.
- **Behavioral checks:** the fingerprint is stable across different chunkings of identical data and still distinguishes different data with no collisions, and a `pickle` round-trip of a dataset reconstructs correctly.
- The PR merged past one pre-existing red CI check unrelated to the change.

---

## Review and outcome

### The pull request

**[huggingface/datasets#8339](https://github.com/huggingface/datasets/pull/8339)**, opened 2026-07-16 against `huggingface/datasets:main`, referencing the issue with a close keyword. The body opens with the user-visible OOM and the measurement showing the reported cause was wrong, then the two coordinated changes and why both are needed, then the RSS before and after, and an explicit callout that fingerprint values change once and existing caches are invalidated. I also posted that consequence back on the issue thread so it would not be discovered later:

> It takes the repro from a ~1.5 GB allocation down to a few MB. One thing worth flagging is that it changes fingerprint values once, so it invalidates existing caches.

### Maintainer feedback

| Date | From | Feedback | My response |
|---|---|---|---|
| 2026-07-16 | lhoestq (on the issue) | Endorsed making the table hash chunk-count-independent specifically to avoid the extra copy from `combine_chunks()`. | Designed to that constraint, going per column one chunk at a time and never making a whole-table copy. |
| 2026-07-18 | Sanjays2402 (review) | "Arrow allows duplicate field names, but this dict comprehension collapses them. `dumps()` will then fail in `.cast(schema)` because the reconstructed table has fewer columns; `pa.Table.from_arrays(columns, schema=schema)` preserves duplicates." | A real bug, so I switched `create_arrowTable` to positional `pa.Table.from_arrays(columns, schema=schema)` in `f719dc953` and thanked him on the thread. |
| 2026-07-22 | lhoestq | **Approved** with "lgtm :)" | Merged as `3bfbc67ce`, and the issue closed as completed. |

### What I learned

**Technical:** The report's stated cause of `from_pandas` was wrong, and confirming the real one required measuring rather than reading, with the right instrument, since `tracemalloc` is blind to pyarrow's C++ allocations. The experiment that actually settled it was the controlled one, holding the data fixed and varying only the chunk count to watch the cost scale, and that is the shape I want to reach for whenever two plausible causes are correlated in the natural repro. The second lesson is that the obvious fix of `combine_chunks()` was the bug in disguise, since a whole-table copy is precisely the memory spike being eliminated, so "make it canonical" has to be weighed against what peak it costs.

**Process & collaboration:** Posting measurements to the issue rather than a claim comment got a maintainer design constraint the same day, and that constraint shaped the whole implementation. Community review mattered too, because `Sanjays2402` is not a maintainer and his catch about duplicate Arrow field names was a genuine correctness bug that would have failed at `.cast(schema)` on real data. Taking a non-maintainer review as seriously as a maintainer's is straightforwardly worth it. Flagging the cache-invalidation consequence proactively also meant the maintainer could weigh it rather than find out afterward.

**What I'd do differently:** I would have looked for the wrapper before writing a reducer for the wrapped type. I targeted `pa.Table` first because it is the type named in the symptom, when the object actually being pickled was `InMemoryTable`, and the sibling classes in the same file were already documenting the correct approach. Reading the neighbors of the class you are modifying, before modifying it, would have saved the whole false start.

### References

- Issue #8327 and my bisection posted to the thread.
- `src/datasets/table.py`, for `MemoryMappedTable` and `ConcatenationTable` `__getstate__` as the pattern to mirror.
- `src/datasets/utils/_dill.py`, for `create_torchGenerator` and `create_spacyLanguage` reducers as the form to follow.
- `tests/test_fingerprint.py`, for existing test conventions such as the inline pyarrow import.
