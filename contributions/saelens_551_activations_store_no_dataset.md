# Contribution: Don't load the dataset when reading cached activations

**Student:** Suryansh Sijwali
**Issue:** https://github.com/decoderesearch/SAELens/issues/551
**Pull Request:** https://github.com/decoderesearch/SAELens/pull/716
**Status:** Merged 2026-07-20 by `chanind` (PR #716).

---

## Why I Chose This Issue

SAELens (the sparse-autoencoder library, org since renamed from `jbloomAus` to
`decoderesearch`) is a small but genuinely alive repo where the maintainer merges
outside contributors on a days-to-weeks cadence. This issue is a concrete,
CPU-only performance bug with a bounded fix, filed by the Neuronpedia author who
also gave a workaround. Although it is tagged `[Proposal]`, the underlying ask is
factual and unambiguous, so it did not carry the design risk that most proposals
in that repo do.

## Understanding the Issue

`ActivationsStore.from_cache_activations` constructs a store that reads
pre-computed activations from disk. But `__init__` unconditionally loads the
Hugging Face dataset (`load_dataset`) and iterates a sample from it to detect
tokenization, even though nothing on the cached-activations path ever uses the
dataset. The reporter's workaround was to pass a dummy dataset just to satisfy the
constructor.

The key observation is that the dataset argument is already marked as a no-op on
this path: in `from_cache_activations` it sits directly under the maintainer's own
`# NOOP` comment, next to `prepend_bos`, `hook_head_index`, and `streaming`. The
code declared it a no-op but did not actually skip it.

## Reproduction / Premise Check

Verified against v6.46.0. Instrumenting `load_dataset` and constructing a store
via `from_cache_activations` showed a full dataset load happening before the cache
was ever touched. On the cached path this is pure wasted time and memory (and can
fail outright if the configured `dataset_path` is unreachable).

I also confirmed the fix is bounded: `__init__` only touches the dataset in two
places (the `load_dataset` call and the `next(iter(self.dataset))` tokenization
probe), and the two consumption paths are cleanly separated. Cached reads go
through `cached_activation_dataset`; live reads go through `iterable_sequences`
(a generator). They never cross, so on the cache path the dataset is loaded,
probed once, and then never read again.

## The Fix

Made `dataset` optional in `ActivationsStore.__init__` and added an early return
from the tokenization probe when it is `None`, then pass `dataset=None` from
`from_cache_activations`. `load_dataset` needed no separate guard, since
`isinstance(None, str)` is already `False`. Also added a guard that raises a clear
`ValueError` when neither a dataset nor a `cached_activations_path` is provided,
and asserted the dataset is present in the three internal methods that iterate it
so the now-optional type stays sound.

## Verification

- Cache path with `dataset=None`: the store constructs and serves real batches
  (`next_batch()` returns the expected shape) with zero `load_dataset` calls.
- Live path (dataset provided) is unchanged: `is_dataset_tokenized` and
  `tokens_column` resolve as before.
- Added `test_load_cached_activations_does_not_load_the_dataset` (monkeypatches
  `load_dataset` to fail if called) and
  `test_activations_store_requires_dataset_or_cache_path` for the guard. The first
  fails on unpatched source and passes with the fix.
- CI green on the PR.

## Review Follow-up (Copilot + type-check)

Copilot's automated review flagged that the class-level `dataset` attribute
annotation still said `HfDataset` while I now assign `None` to it, which would
fail the repo's `pyright` type-check. That was a real miss (my local pyright ran
standalone and could not resolve imports, so it did not catch it). Fixed the
annotation to `HfDataset | None`, and pyright then flagged three internal
`self.dataset` reads on the newly-optional type, which I narrowed with
`assert self.dataset is not None`. Copilot's second point (a guard for the
neither-dataset-nor-cache case) was a fair hardening and is the `ValueError` above.

## Strategy Note

This is intended as an entry-credential contribution. SAELens's issue board is
largely inactive (the newest open issue is weeks old, and the healthy merge rate
comes from contributors self-sourcing their own PRs rather than working the
board). Landing this one small, clean fix establishes standing to then contribute
self-sourced work directly, the way the most active outside contributor there
(`danra`, ~19 PRs vs 3 issues) does.
