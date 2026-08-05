# Run the LLM autocast and the activation buffer on the model's device

**Student:** Suryansh Sijwali ([@SuryanshSS1011](https://github.com/SuryanshSS1011))
**Project:** [decoderesearch/SAELens](https://github.com/decoderesearch/SAELens) · **My fork:** https://github.com/SuryanshSS1011/SAELens
**Issue:** none, self-sourced · **Pull request:** [#722](https://github.com/decoderesearch/SAELens/pull/722) · **Branch:** `fix/activations-store-autocast-device`
**Status:** **Merged** 2026-08-05 (merge commit `928d44fb5`). Self-sourced, so there is no upstream issue.

---

## Phase I: Issue Selection

### Why I Chose This Issue

This is the second half of a deliberate two-step plan in SAELens. The first was [#551](https://github.com/decoderesearch/SAELens/issues/551), a small clean fix taken off the issue board purely to establish standing. SAELens's board is largely inactive, and the healthy merge rate there comes from contributors self-sourcing their own PRs rather than working the board, which is how the most active outside contributor (`danra`, roughly 19 PRs against 3 issues) operates. With one merge already landed, self-sourcing was the way to keep contributing.

I went looking in `ActivationsStore` because I had just read it end to end for #551 and knew the class well. My learning goal was device-placement bugs, which are a category I wanted to understand better because they fail silently rather than loudly. A wrong device usually does not raise. It warns, or it quietly copies, and the cost shows up as a number that is subtly wrong or a run that is slower than it should be.

### Problem Summary

`ActivationsStore.get_activations` assumes CUDA in two independent places, so on Apple Silicon, CPU and XPU the `autocast_lm=True` flag silently does nothing and the activation buffer is allocated on the default device rather than where the model actually ran. Neither failure raises, so a user who opts into bfloat16 autocast gets a full-precision forward pass with no indication, and every batch takes an unnecessary round trip off the accelerator and back. It matters because SAELens is used heavily on Macs for local interpretability work, and because the sibling method in the same class already does both things correctly, so the bug reads as an oversight rather than a design choice. I chose it because it is a real correctness and performance bug that I could prove with measurements on two different accelerators.

### How I Found It, and Why There Is No Issue

Self-sourced while reading `ActivationsStore` for the earlier contribution. I deliberately did not open an issue first, for two reasons drawn from how the repo actually behaves. Nobody there files claim comments, and the PRs that died closed and unmerged were the ones framed as speculative features rather than fixes. So I framed this as a bug fix throughout, opened the PR directly against `upstream/main` at `8be14080` (version 6.47.0), and re-checked upstream at final validation to confirm no rebase was needed.

### Where It Lives

- `sae_lens/training/activations_store.py`, specifically `get_activations`, which holds both CUDA assumptions, and its sibling `get_multi_hook_activations`, which already avoids both and is therefore the in-repo precedent.
- `sae_lens/training/activations_store.py`, `get_raw_llm_batch`, which does the redundant `.to(self.device)` that the buffer fix makes into a no-op.
- `tests/training/test_activations_store.py`, where lines 174 and 206 already assert `activations.device == store.device`.

### Acceptance Criteria

1. `autocast_lm=True` actually engages on MPS, CPU and XPU rather than being silently disabled.
2. `autocast_lm` still engages on CUDA exactly as before.
3. Activations are bit-identical before and after on CUDA, where autocast already worked.
4. The returned tensor lands on `self.device` rather than on the default device.
5. No new test failures on either CPU or CUDA.

---

## Phase II: Reproduce & Plan

### Environment Setup

I needed two machines for this, because the bug behaves differently on CUDA and non-CUDA hardware and the strongest evidence is only obtainable on a GPU. Local work was on Apple Silicon (MPS) and the CUDA verification ran on Penn State's ROAR Collab cluster on an A100-PCIE-40GB in a 2g.10gb MIG slice with torch 2.13.0+cu126.

- **Challenge (finding a GPU queue I could actually submit to):** the `mgc-open` partition only allows the `mgc_mri` and `mgc_nih` accounts and `sla-prio` only allows `reserved_allocation`, neither of which I have. The GPU queue reachable with the free `open` account is `standard`, covering a100, v100, a40 and p100. I found this with `sbatch --test-only` to probe account and partition combinations rather than burning failed submissions.
- **Challenge (home directory at zero bytes free):** the virtual environment alone is 5.6 GB, so `PIP_CACHE_DIR`, `TMPDIR` and `HF_HOME` all had to be redirected to `~/scratch` before anything would install.
- **Challenge (compute nodes have no network):** every model and dataset has to be pre-cached on the login node and then used with `HF_HUB_OFFLINE=1`. Caching only gpt2 produced 41 spurious fixture errors, and adding `tiny-stories-1M` and `NeelNanda/c4-10k` dropped that to 2.
- **Challenge (local linters disagreed with CI):** my `.venv` had ruff 0.15.20 against the repo's pinned `^0.7.4`, which produced 26 phantom lint errors repo-wide. Installing `ruff==0.7.4` and `pyright==1.1.365` to match the pins fixed it.
- **Challenge (pyright resolving the wrong interpreter):** run bare it reported 269 phantom errors on `main`, because it could not see `transformer_lens`. With `--pythonpath .venv/bin/python` it reports 0 errors on both touched files and only 6 repo-wide, all of them optional dependencies that will not install on a Mac (`mamba_lens`, `sparsify`, `dictionary_learning`). CI does not hit this, because `make check-type` runs pyright through poetry where the poetry environment is the interpreter.
- One HuggingFace gotcha worth recording: repeated unauthenticated gpt2 pulls got me rate-limited, which surfaced as 16 errors in an unrelated test module. Setting `HF_TOKEN` before benchmarking avoids it.

### Steps to Reproduce

1. Install SAELens 6.47.0 on a non-CUDA machine, so Apple Silicon or CPU.
2. Build an `ActivationsStore` on `tiny-stories-1M` with `autocast_lm=True`.
3. Call `get_activations` and keep the result.
4. Rebuild the same store with `autocast_lm=False` and call `get_activations` again.
5. Compare the two tensors elementwise, and watch stderr while step 3 runs.
6. Separately, on a CUDA machine, build a store with `act_store_device="cuda"` and check `.device` on the tensor `get_activations` returns.

### Expected vs. Actual

| Check | Expected | Actual on `main` |
|---|---|---|
| `autocast_lm=True` vs `False` on MPS | activations differ by bfloat16 rounding | bit-identical, so autocast never ran |
| stderr during step 3 | nothing | `UserWarning: CUDA is not available or torch_xla is imported. Disabling autocast.` |
| `.device` with `act_store_device="cuda"` | `cuda:0` | `cpu` |
| buffer step timing, 537 MB batch on MPS | no host round trip | 68.8 ms |

After the fix the activations on MPS differ by 1.3e-3, which is bfloat16 rounding and is the flag doing what it promises. The tensor lands on `cuda:0`. The buffer step drops to 21.4 ms.

### Root Cause

Two independent CUDA assumptions in `get_activations`, both already avoided by `get_multi_hook_activations` in the same class.

The first is a hardcoded autocast device:

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.autocast_lm):
```

On a non-CUDA device PyTorch does not raise here. It emits a warning and turns autocast off, so `autocast_lm=True` silently runs the model in full precision on MPS, CPU and XPU.

The second is a staging buffer allocated with no device:

```python
stacked_activations = torch.zeros((n_batches, n_context, self.d_in))
```

It lands on the default device no matter where the model ran, so the slice assignments below it copy the activations off the accelerator, and `get_raw_llm_batch` immediately moves them straight back with `.to(self.device)`. The existing tests at `test_activations_store.py:174` and `:206` already assert `activations.device == store.device`, and they pass on `main` only because they happen to run on CPU, where the default device and the store device coincide.

### Plan (UMPIRE)

**Understand:** Two device assumptions that fail quietly, one costing precision and one costing a round trip, in a method whose sibling already handles both.

**Match:** `get_multi_hook_activations`, in the same class and file, uses `_get_input_token_device(self.model).type` for autocast and does not allocate an undeviced buffer. Rather than invent an approach, the fix is to make the two methods agree, which also makes the change trivially reviewable because the reviewer can diff two neighbours.

**Plan:**
1. Replace the hardcoded `device_type="cuda"` with `_get_input_token_device(self.model).type`, the exact expression the sibling uses.
2. Allocate the staging buffer on `self.device`.
3. Verify on MPS that autocast now engages, and on CUDA that nothing changes numerically.
4. Add a test asserting the buffer lands on the store's device.
5. Run the suite unpatched and patched on both machines and compare test by test rather than by count.

**Implement:** Branch `fix/activations-store-autocast-device` off `upstream/main` at `8be14080`, single commit.

**Review:** My self-checklist was that CUDA behavior is bit-identical, that the diff stays at two lines of source plus a test, that the PR body states the one real behavior change honestly, and that `make check-ci` passes with the repo-pinned linter versions.

**Evaluate:** Four configurations on CUDA, covering store on CUDA and on CPU with autocast off and on, compared by float64 checksum, plus a named test-by-test comparison of the suite before and after.

### Edge Cases Considered

- **Which device for the buffer.** Allocating on `layerwise_activations.device` looks equivalent to `self.device` and is worse. With `act_store_device="cpu"` and the model on the GPU it builds the buffer in VRAM and then ships all of it to the host anyway, which measured 141 ms against 55 ms on `main` at 537 MB, and it holds an extra copy in VRAM while doing so. That hurts exactly the users who chose that setting to save memory. I wrote this version first, measured it, and threw it away.
- **Deleting the buffer entirely** to match the multi-hook path would change behavior twice over. The buffer pins the output to `float32`, and its fixed shape turns a hook whose width disagrees with `d_in` into a `RuntimeError` instead of a silently wrong result, since without it a 128-wide hook against `d_in=64` returns width 128 and nobody notices.
- **`torch.empty` instead of `torch.zeros`.** All three branches fully overwrite the buffer today, and the saving is real at 35% on MPS (19.95 ms to 12.88 ms) and 31% on an A100 (4.487 ms to 3.106 ms) for a 537 MB batch. I left it out anyway, because it is an optimization rather than the bug, it is only about 1.4% of an end-to-end `get_activations` call on gpt2, and it silently becomes unsafe the day a future branch stops fully overwriting.
- **Meta tensors.** I wondered whether the new autocast device expression needed a guard for a model with parameters on `meta`, which happens under accelerate disk offload. I tested it rather than guessing, and the realistic configuration fails on both `main` and the fix, with `NotImplementedError: Cannot copy out of meta tensor` before and `RuntimeError: unsupported autocast device_type 'meta'` after. No working configuration is broken and only the message is less descriptive, so I left it unguarded.

---

## Phase III: Build

### Implementation Progress

| Commit | Date | Message |
|---|---|---|
| `35bf4f99c` | 2026-07-31 | fix: run LLM autocast and the activation buffer on the model's device |

**Files modified:**

| File | Δ |
|---|---|
| `tests/training/test_activations_store.py` | +22 |
| `sae_lens/training/activations_store.py` | +4 / −2 |

The source change is two lines, which is the point. Almost all the work was proving that those two lines are correct and that the two alternatives I considered are worse.

### Challenges Faced

**The measurement had to happen on hardware I do not own.** The single best piece of evidence for the buffer half of this fix is that `act_store_device="cuda"` returns a tensor on `cpu` on `main` and on `cuda:0` with the change, and that is impossible to observe on a Mac. Getting a CUDA run meant working out which ROAR queue my free account could actually reach, moving a 5.6 GB environment onto scratch because home had zero bytes free, and pre-caching both models and datasets because compute nodes have no network. That was most of the elapsed time on this contribution.

**My first version of the buffer fix was wrong in a way that only measurement caught.** Allocating on `layerwise_activations.device` reads as the more natural choice, since it puts the buffer where the data already is. It is a 2.5x regression for `act_store_device="cpu"` users, and I only knew that because I timed it. Writing the plausible version first and then measuring it is what turned a guess into a defensible choice, and it is why the PR body has an explicit section on alternatives I ruled out.

**A predicted result came out right for the wrong reason.** I expected the `torch.empty` saving to be negligible on CUDA, reasoning that the memset would be dwarfed by the PCIe transfer. The measurement disagreed, showing 31%, because once the buffer stays on the GPU there is no PCIe transfer left to dwarf it. My conclusion to leave it out survived on absolute size rather than on the mechanism I had assumed, and noticing that the right answer came from wrong reasoning is worth more than the answer.

### Testing

- **New test** in `tests/training/test_activations_store.py`, asserting the returned activations land on the store's device, following the existing device assertions at lines 174 and 206 in the same file rather than introducing a new style.
- **CUDA verification** on ROAR, four configurations covering the store on CUDA and on CPU with autocast off and on. All four are bit-identical between `main` and the fix, compared by float64 checksum at 14111.1516722143 and 14110.7792109251, which confirms the change is a no-op where autocast already worked.
- **Suite comparison by name rather than by count**, run unpatched and then patched on the same machine. Seven tests were already failing in both runs, zero were newly failing, and passes went from 45 to 46, which is the new test. Comparing by name matters here, because an unchanged failure count can still hide one test breaking while another starts passing.
- **Before and after evidence:** MPS activations go from bit-identical with `autocast_lm=False` to differing by 1.3e-3, the buffer step on a 537 MB batch goes from 68.8 ms to 21.4 ms, and the returned tensor moves from `cpu` to `cuda:0` under `act_store_device="cuda"`.
- `make check-ci` passes with the repo-pinned ruff 0.7.4 and pyright 1.1.365, and pyright reports 0 errors on both touched files.

---

## Phase IV: Submit & Iterate

### Pull Request

**[decoderesearch/SAELens#722](https://github.com/decoderesearch/SAELens/pull/722)**, opened 2026-07-31 against `decoderesearch/SAELens:main`. It carries no `Fixes #` line, because it is self-sourced and there is no issue to close.

The body follows the repo's `PULL_REQUEST_TEMPLATE.md` and mirrors the shape of my merged #716, with a prose `# Description`, small code blocks showing each of the two offending lines, then the `Type of change` and `Checklist` sections completed, including the `make check-ci` confirmation. I dropped the template's "Performance Check" section, as #716 did, since this is not a training-algorithm change.

Two things in the body were deliberate. There is a section on alternatives I ruled out with the measurements behind each, so the reviewer can see the two plausible-looking fixes were considered and rejected on evidence rather than overlooked. And there is an explicit statement that this is not a no-op for existing users, since anyone already setting `autocast_lm=True` on MPS or CPU will now get the bfloat16 forward pass the flag promises, so their activations shift by bfloat16 rounding and their memory use drops. The flag defaults to `False`, so nobody who has not opted in is affected.

### Maintainer Feedback Log

| Date | From | Feedback | My response |
|---|---|---|---|
| 2026-07-31 | Copilot review | Unable to review, since the requesting account had hit its quota. | n/a, so I did an extra manual pass in place of it. |
| 2026-08-05 | chanind | **Approved** with "Great find, thank you for this fix!" | Merged as `928d44fb5`. |

No changes were requested. I read that as the payoff for the body doing the reviewer's work in advance, since the alternatives section answers the two questions a careful reviewer would have asked, and the CUDA checksums answer the third.

### Learnings & Reflections

**Technical:** Device bugs hide because the failure modes are polite. `torch.autocast` with an unavailable device type does not raise, it warns and disables itself, so a flag that promises bfloat16 quietly delivers float32 and the only symptom is that your numbers are slightly different from a colleague's on a different machine. An undeviced `torch.zeros` behaves the same way, costing a silent round trip rather than an error. The general lesson is that when a call takes a device and you do not pass one, or you pass a constant, that is a bug waiting for someone on different hardware, and the fastest way to find such bugs is to look for a sibling function that already got it right.

**Process & collaboration:** Self-sourcing worked because I had already banked a merge on the board. The two-step plan was to land #551 to establish standing, then contribute the way the productive contributors there actually do, which is by opening PRs for things they found themselves. Reading how the repo treats different kinds of PR mattered more than any etiquette rule, since the closed and unmerged ones were speculative features, so I framed this strictly as a bug fix and never as a proposal. Putting the ruled-out alternatives and their measurements in the body is what I would repeat most, because it converted what could have been two review rounds into a single approval.

**What I'd do differently:** I would set up the CUDA environment before writing the fix rather than after. I built and measured on MPS first, and the strongest evidence in the whole PR, that the tensor comes back on `cpu` when the user asked for `cuda`, only became available once ROAR was working. Doing that first would have shaped the framing earlier. I would also be quicker to write the plausible-looking version of a change purely to measure it, since I nearly shipped `layerwise_activations.device` on the reasoning that it read better, and it was 2.5x slower for the users who care most about memory.

### Resources Used

- `sae_lens/training/activations_store.py`, particularly `get_multi_hook_activations`, the sibling that already handled both cases correctly.
- `tests/training/test_activations_store.py`, lines 174 and 206, the existing device assertions that the buffer fix finally makes meaningful.
- The repo's `PULL_REQUEST_TEMPLATE.md`, and my earlier merged PR #716 as the model for how much of it to fill in.
- PyTorch autocast documentation, for the behavior of an unavailable device type.
