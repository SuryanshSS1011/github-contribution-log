# Don't load the dataset when reading cached activations

**Project:** [decoderesearch/SAELens](https://github.com/decoderesearch/SAELens), the org renamed from `jbloomAus`
**Issue:** [#551](https://github.com/decoderesearch/SAELens/issues/551) · **Pull request:** [#716](https://github.com/decoderesearch/SAELens/pull/716) · **Branch:** `activations-store-optional-dataset`
**Status:** **Merged** 2026-07-20 (merge commit `f34ac827`). Issue closed.

---

## The issue

### Why I picked it

SAELens is the sparse-autoencoder library used across mechanistic-interpretability work, which is directly adjacent to the interpretability side of my research interests. It is a small but genuinely alive repo where the maintainer merges outside contributions on a days-to-weeks cadence. My learning goal was narrower than usual and deliberately so, which was to land a small unambiguous fix cleanly in an interpretability codebase to establish standing, then contribute self-sourced work there.

The issue is a concrete, CPU-only performance bug with a bounded fix, filed by the Neuronpedia author, who also supplied a workaround. Although it is tagged `[Proposal]`, the underlying ask is factual and unambiguous, so it did not carry the design risk most proposals in that repo do.

### What was wrong

`ActivationsStore.from_cache_activations` builds a store that reads pre-computed activations from disk, but `__init__` unconditionally calls `load_dataset` and iterates a sample from it to detect tokenization, even though nothing on the cached path ever touches the dataset. Users end up waiting on and paying memory for a full HuggingFace dataset load just to read activations already on disk, and the constructor fails outright if the configured `dataset_path` is unreachable, so the reporter's workaround was to pass a dummy dataset purely to satisfy the constructor. It matters because the cached-activations path exists specifically to make repeated runs cheap, and this defeats that. I chose it because the fix is bounded, verifiable on CPU, and the code already declares the intended behavior.

### Before I started

Open and unassigned with no in-flight PR. The strongest signal was in the source rather than the thread, because the `dataset` argument is already marked as a no-op on this path. In `from_cache_activations` it sits directly under the maintainer's own `# NOOP` comment, next to `prepend_bos`, `hook_head_index` and `streaming`. The code declared it a no-op but did not actually skip it, which means the intended behavior was never in question.

### Where it lives

- `sae_lens/training/activations_store.py`, holding `ActivationsStore.__init__` with the `load_dataset` call and the `next(iter(self.dataset))` tokenization probe, plus `from_cache_activations`.
- `tests/test_cache_activations_runner.py`, the cached-activations test surface.

### What counts as done

1. Constructing a store via `from_cache_activations` performs zero `load_dataset` calls.
2. The cached path still serves real batches.
3. The live path with a dataset provided is behaviorally unchanged.
4. A configuration with neither a dataset nor a cache path fails with a clear error rather than a confusing one.
5. The repo's `pyright` type-check stays clean.

---

## Diagnosis and plan

### Environment setup

I used the repo's `CONTRIBUTING.md` and Makefile targets for the test and type-check loop, and everything here is CPU-only with no GPU or model download needed for this path.

- Verified against v6.46.0.
- **Challenge (local `pyright` disagreed with CI's):** running `pyright` standalone in my environment could not resolve the project's imports, so it silently under-reported and missed a real type error that CI's configured run would catch. This bit me, because the class-level `dataset` annotation issue below got through my local check. The fix is to run the type-checker through the repo's own task or config rather than bare, so it sees the same environment CI does.

### Steps to reproduce

1. Install SAELens v6.46.0, and CPU is fine.
2. Instrument `load_dataset` by monkeypatching it to record calls, or to raise.
3. Construct a store from cached activations:
   ```python
   store = ActivationsStore.from_cache_activations(cfg, model)
   ```
4. Observe whether `load_dataset` was called before any cached file was read.

### Expected vs. actual

**Actual:** a full HuggingFace dataset load happens during `__init__`, before the cache is touched at all, plus one iteration of the dataset for the tokenization probe. On the cached path this is pure wasted time and memory, and it fails outright if `dataset_path` is unreachable.

**Expected:** zero dataset loads. The dataset is already documented as a no-op on this path, so it should not be fetched, iterated or required.

### Root cause

`ActivationsStore.__init__` treats the dataset as mandatory infrastructure, since it calls `load_dataset` and then runs `next(iter(self.dataset))` to detect whether the data is pre-tokenized, while `from_cache_activations` labels the same argument a no-op. The two disagree, and `__init__` wins.

I confirmed the fix is genuinely bounded before writing it. `__init__` touches the dataset in exactly two places, being the `load_dataset` call and the tokenization probe, and the two consumption paths are cleanly separated, since cached reads go through `cached_activation_dataset` and live reads through the `iterable_sequences` generator. They never cross, so on the cache path the dataset is loaded, probed once and then never read again, which is what makes skipping it safe rather than merely deferred.

### The plan

**Understand:** A constructor unconditionally acquires an expensive resource that one of its two code paths never uses.

**Match:** The repo already expresses "not used on this path" as the `# NOOP` group in `from_cache_activations`, so the fix makes the code match that existing maintainer-authored declaration rather than introducing a new concept.

**Plan:**
1. Make `dataset` optional in `ActivationsStore.__init__`.
2. Early-return from the tokenization probe when it is `None`.
3. Pass `dataset=None` from `from_cache_activations`.
4. Add a guard raising a clear `ValueError` when neither a dataset nor a `cached_activations_path` is given.
5. Keep the now-optional type sound in the internal methods that iterate the dataset.
6. Test that the cache path performs zero `load_dataset` calls.

**Implement:** Branch `activations-store-optional-dataset`, single commit.

**Review:** My self-checklist was that the live path is untouched, that there is no public API break since the parameter becomes optional rather than removed, and that type annotations are updated to match the new optionality.

**Evaluate:** Bite-test it, so the new test must fail on unpatched source and pass with the fix.

### Edge cases

- **`load_dataset` needs no separate guard**, because `isinstance(None, str)` is already `False`. I checked that rather than assuming it, which kept the diff smaller.
- **Neither dataset nor cache path:** Previously this produced a confusing downstream failure, and now it raises a clear `ValueError` at construction.
- **Type soundness after the change:** Making `dataset` optional means every internal `self.dataset` read is now on an optional value, and there are three of them, each needing narrowing or the type-check fails.

---

## Implementation

### Commits and files

| Commit | Date | Message |
|---|---|---|
| `7aa610724` | 2026-07-16 | fix: don't load the dataset when reading cached activations |

**Files modified:**

| File | Δ |
|---|---|
| `sae_lens/training/activations_store.py` | +16 / −3 |
| `tests/test_cache_activations_runner.py` | +52 |

This includes the `HfDataset | None` annotation fix and the three `assert self.dataset is not None` narrowings added in response to review.

### What was hard

The obstacle was one I created. My local `pyright` ran standalone and could not resolve the project's imports, so it reported clean while the repo's configured run would not have. Copilot's automated review caught the consequence, which was that the class-level `dataset` attribute annotation still said `HfDataset` while I was now assigning `None` to it, and that would fail the repo's type-check.

Fixing the annotation to `HfDataset | None` then surfaced three internal `self.dataset` reads that were no longer sound on an optional type, which I narrowed with `assert self.dataset is not None` at each. That is the correct fix rather than a `# type: ignore`, because those three methods genuinely only run on the live path, so the assertion documents the invariant instead of suppressing it.

The general lesson is to verify with the project's own tooling configuration rather than a bare invocation of the same tool, because a type-checker that cannot resolve imports does not fail loudly and just stops finding things.

### Testing

- **New tests** in `tests/test_cache_activations_runner.py`, following that file's existing monkeypatch-based style:
  - `test_load_cached_activations_does_not_load_the_dataset`, which monkeypatches `load_dataset` to fail if called, so the test is the acceptance criterion. It fails on unpatched source and passes with the fix.
  - `test_activations_store_requires_dataset_or_cache_path`, covering the new `ValueError` guard.
- **Manual and behavioral verification:**
  - The cache path with `dataset=None` constructs and serves real batches, so `next_batch()` returns the expected shape with zero `load_dataset` calls.
  - The live path with a dataset provided is unchanged, and `is_dataset_tokenized` and `tokens_column` resolve as before.
- CI is green on the PR, including the repo's `pyright` run after the annotation fix.

---

## Review and outcome

### The pull request

**[decoderesearch/SAELens#716](https://github.com/decoderesearch/SAELens/pull/716)**, opened 2026-07-16 against `decoderesearch/SAELens:main`, referencing the issue with a close keyword. The body opens with the user-visible cost, meaning a full dataset load before reading activations already on disk plus a hard failure if `dataset_path` is unreachable, notes that the code already declares this argument a no-op on the path, then describes the change and the guard, and lists the before and after verification.

### Maintainer feedback

| Date | From | Feedback | My response |
|---|---|---|---|
| 2026-07-16 | Copilot review | The class-level `dataset` annotation still says `HfDataset` while `None` is now assigned, which would fail the repo's `pyright`. | A real catch, and one my standalone `pyright` had missed. Changed it to `HfDataset \| None`, after which pyright flagged three internal `self.dataset` reads on the optional type, which I narrowed with `assert self.dataset is not None`. |
| 2026-07-16 | Copilot review | Suggested guarding the case where there is neither a dataset nor a cache path. | Fair hardening, so I added the `ValueError` with a clear message, plus a test. |
| 2026-07-16 | Copilot review (2nd pass) | No new comments. | n/a |
| 2026-07-20 | chanind | **Approved** with "Makes sense! Thanks for this fix!" | Merged as `f34ac827`. |

### What I learned

**Technical:** The `# NOOP` comment was the most valuable thing in the diff, because the maintainer had already written down the intended behavior and the bug was simply that the code did not honor it. When a comment and the code disagree, the comment is often the specification, and that makes the change nearly un-arguable in review. The second lesson is that making a parameter optional is never a one-line change in a typed codebase, since every downstream read inherits the optionality, so following the annotation through to the three narrowing sites is the actual work.

**Process & collaboration:** Automated reviewers are worth taking seriously. Copilot's catch was a real defect that my local tooling had hidden, and treating it as a genuine review by investigating rather than dismissing is what kept the PR to a single round with the human maintainer. The strategic framing also mattered, because this was chosen as an entry-credential contribution. SAELens's issue board is largely inactive, with the newest open issue weeks old, and the healthy merge rate comes from contributors self-sourcing their own PRs, since the most active outside contributor there, `danra`, has about 19 PRs against 3 issues. Landing one small clean fix from the board is how you earn standing to work that way.

**What I'd do differently:** I would run the repo's own type-check invocation before pushing rather than a bare `pyright`. More generally, when a tool reports clean, I should confirm it was actually looking, because a checker that cannot resolve imports produces a green result that means nothing. That is a cheap sanity check, and it would have removed the only review round on this PR.

### References

- Issue #551, for the report and the reporter's dummy-dataset workaround.
- `sae_lens/training/activations_store.py`, specifically the `# NOOP` group in `from_cache_activations` that documents the intended behavior.
- `tests/test_cache_activations_runner.py`, for existing monkeypatch-style test conventions.
