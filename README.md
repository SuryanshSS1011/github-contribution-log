# Open-Source Contribution Log

Tracking workspace for my open-source contributions across ML systems and program-analysis tooling, adjacent to my research.

Nine merged. The In review table lists a selection of the open pull requests rather than all of them. In both tables, each entry in the "What I fixed" column links to its full writeup in [`contributions/`](contributions/).

## Merged

| Project | Issue | What I fixed | Pull request | Merged |
|---|---|---|---|---|
| pytorch/executorch | [#16429](https://github.com/pytorch/executorch/issues/16429) | [A reduction helper walked out of bounds on non-contiguous tensors](https://github.com/SuryanshSS1011/github-contribution-log/blob/main/contributions/executorch_21517_reduce_util_noncontiguous.md) | [#21517](https://github.com/pytorch/executorch/pull/21517) | 2026-08-17 |
| decoderesearch/SAELens | self-sourced | [`autocast_lm` silently did nothing off CUDA, and the activation buffer landed on the wrong device](https://github.com/SuryanshSS1011/github-contribution-log/blob/main/contributions/saelens_722_activations_store_device.md) | [#722](https://github.com/decoderesearch/SAELens/pull/722) | 2026-08-05 |
| huggingface/datasets | [#8327](https://github.com/huggingface/datasets/issues/8327) | [`Dataset.from_pandas` OOM caused by a fingerprint that scaled with Arrow chunk count](https://github.com/SuryanshSS1011/github-contribution-log/blob/main/contributions/datasets_8327_fingerprint_chunking.md) | [#8339](https://github.com/huggingface/datasets/pull/8339) | 2026-07-22 |
| pytorch/executorch | [#20804](https://github.com/pytorch/executorch/issues/20804) | [Portable CPU kernel rejected valid transposed convolution weights with a non-default dim order](https://github.com/SuryanshSS1011/github-contribution-log/blob/main/contributions/executorch_20804_transposed_conv_dimorder.md) | [#21035](https://github.com/pytorch/executorch/pull/21035) | 2026-07-22 |
| decoderesearch/SAELens | [#551](https://github.com/decoderesearch/SAELens/issues/551) | [`ActivationsStore` loaded the whole dataset even when reading cached activations](https://github.com/SuryanshSS1011/github-contribution-log/blob/main/contributions/saelens_551_activations_store_no_dataset.md) | [#716](https://github.com/decoderesearch/SAELens/pull/716) | 2026-07-20 |
| astral-sh/ty | [#3674](https://github.com/astral-sh/ty/issues/3674) | [Renaming an attribute left the matching `__slots__` string stale, which breaks the class](https://github.com/SuryanshSS1011/github-contribution-log/blob/main/contributions/ty_3674_slots_rename.md) | [ruff#26438](https://github.com/astral-sh/ruff/pull/26438) | 2026-07-06 |
| facebook/infer | [#1951](https://github.com/facebook/infer/issues/1951) | [Pulse missed resource leaks through `java.util.Optional` because the type was unmodeled](https://github.com/SuryanshSS1011/github-contribution-log/blob/main/contributions/infer_1951_optional_resource_leak.md) | [#2068](https://github.com/facebook/infer/pull/2068) | 2026-07-03 |
| pytorch/executorch | [#20556](https://github.com/pytorch/executorch/issues/20556) | [MLX submodule build patched a pinned third-party `CMakeLists.txt` at configure time](https://github.com/SuryanshSS1011/github-contribution-log/blob/main/contributions/executorch_20556_mlx_externalproject.md) | [#20585](https://github.com/pytorch/executorch/pull/20585) | 2026-06-29 |
| astral-sh/ruff | [#25588](https://github.com/astral-sh/ruff/issues/25588) | [PLR2004 flagged ordinary Python version guards such as `sys.implementation.version[0] >= 3`](https://github.com/SuryanshSS1011/github-contribution-log/blob/main/contributions/ruff_25588_plr2004_sys_version.md) | [#25743](https://github.com/astral-sh/ruff/pull/25743) | 2026-06-10 |

## In review

| Project | Issue | What I fixed | Pull request | Status |
|---|---|---|---|---|
| huggingface/trl | [#6669](https://github.com/huggingface/trl/issues/6669) | [Wrapped packing re-read from the start of the buffer on sliced tables](https://github.com/SuryanshSS1011/github-contribution-log/blob/main/contributions/trl_6670_pack_wrapped_sliced.md) | [#6670](https://github.com/huggingface/trl/pull/6670) | CI green, awaiting review. |
| decoderesearch/SAELens | self-sourced | [A BOS token id of 0 read as absent, so Pythia runs silently got no BOS](https://github.com/SuryanshSS1011/github-contribution-log/blob/main/contributions/saelens_724_bos_token_id_zero.md) | [#724](https://github.com/decoderesearch/SAELens/pull/724) | Awaiting review, CI failures not attributable to the change. |
| facebook/infer | [#1937](https://github.com/facebook/infer/issues/1937) | [Pulse falsely reported a leak through a C variadic out-parameter](https://github.com/SuryanshSS1011/github-contribution-log/blob/main/contributions/infer_1937_variadic_out_param_leak.md) | [#2078](https://github.com/facebook/infer/pull/2078) | CI green, awaiting review. |
| huggingface/diffusers | [#14037](https://github.com/huggingface/diffusers/issues/14037) | [A stale `HookRegistry` cache broke caching enabled after a warmup pass](https://github.com/SuryanshSS1011/github-contribution-log/blob/main/contributions/diffusers_14037_hook_registry_cache.md) | [#14093](https://github.com/huggingface/diffusers/pull/14093) | CI green, test rework pushed after four review rounds. |
| astral-sh/uv | [#7035](https://github.com/astral-sh/uv/issues/7035) | [No hint when a build failure is caused by a `requires-python` mismatch](https://github.com/SuryanshSS1011/github-contribution-log/blob/main/contributions/uv_7035_requires_python_hint.md) | [#19673](https://github.com/astral-sh/uv/pull/19673) | Reworked onto the maintainer's refactor, awaiting review. |

## Filed bug reports

| Project | Issue | What I reported | Status |
|---|---|---|---|
| huggingface/transformers | [#47487](https://github.com/huggingface/transformers/issues/47487) | [CodeLlama tokenizer dropped leading whitespace on decode, refiled with a fresh reproduction after #46491 was stale-closed](https://github.com/SuryanshSS1011/github-contribution-log/blob/main/contributions/transformers_47488_codellama_leadingspace.md) | My [#47488](https://github.com/huggingface/transformers/pull/47488) merged, then reverted by [#47861](https://github.com/huggingface/transformers/pull/47861) in favour of a decoder-only fix in [#47862](https://github.com/huggingface/transformers/pull/47862), still open. |
| huggingface/transformers | [#46489](https://github.com/huggingface/transformers/issues/46489) | DeepSeek-Coder v1 tokenizer produced wrong output on v5 and later, a gap left by PR #44801 | Resolved by upstream [#46091](https://github.com/huggingface/transformers/pull/46091), issue closed. |

## About the writeups

Each file in [`contributions/`](contributions/) follows the same structure, so any entry can be read end to end without context:

- **The issue:** why I picked it, what was wrong, how I checked it was still available, the specific files involved, and what I decided would count as done.
- **Diagnosis and plan:** environment setup with the concrete obstacles I hit, numbered reproduction steps, expected versus actual behavior, the root cause, and the plan I worked to.
- **Implementation:** commit hashes and dates, files changed with line counts, what turned out to be hard, and what was tested manually and automatically.
- **Review and outcome:** the pull request, a dated log of maintainer feedback and how I responded, and what I learned about the problem, the collaboration, and what I would do differently.
