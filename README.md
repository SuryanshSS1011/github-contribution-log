# Open-Source Contribution Log

Tracking workspace for my open-source contributions across ML systems and program-analysis tooling, adjacent to my research.

Eight merged and three in review, across upstream projects. In the tables below, each entry in the "What I fixed" column links to its full writeup in [`contributions/`](contributions/).

## Merged

| Project | Issue | What I fixed | Pull request | Merged |
|---|---|---|---|---|
| decoderesearch/SAELens | self-sourced | [`autocast_lm` silently did nothing off CUDA, and the activation buffer landed on the wrong device](https://github.com/SuryanshSS1011/github-contribution-log/blob/main/contributions/saelens_722_activations_store_device.md) | [#722](https://github.com/decoderesearch/SAELens/pull/722) | 2026-08-05 || huggingface/datasets | [#8327](https://github.com/huggingface/datasets/issues/8327) | [`Dataset.from_pandas` OOM caused by a fingerprint that scaled with Arrow chunk count](https://github.com/SuryanshSS1011/github-contribution-log/blob/main/contributions/datasets_8327_fingerprint_chunking.md) | [#8339](https://github.com/huggingface/datasets/pull/8339) | 2026-07-22 |
| pytorch/executorch | [#20804](https://github.com/pytorch/executorch/issues/20804) | [Portable CPU kernel rejected valid transposed convolution weights with a non-default dim order](https://github.com/SuryanshSS1011/github-contribution-log/blob/main/contributions/executorch_20804_transposed_conv_dimorder.md) | [#21035](https://github.com/pytorch/executorch/pull/21035) | 2026-07-22 |
| decoderesearch/SAELens | [#551](https://github.com/decoderesearch/SAELens/issues/551) | [`ActivationsStore` loaded the whole dataset even when reading cached activations](https://github.com/SuryanshSS1011/github-contribution-log/blob/main/contributions/saelens_551_activations_store_no_dataset.md) | [#716](https://github.com/decoderesearch/SAELens/pull/716) | 2026-07-20 |
| astral-sh/ty | [#3674](https://github.com/astral-sh/ty/issues/3674) | [Renaming an attribute left the matching `__slots__` string stale, which breaks the class](https://github.com/SuryanshSS1011/github-contribution-log/blob/main/contributions/ty_3674_slots_rename.md) | [ruff#26438](https://github.com/astral-sh/ruff/pull/26438) | 2026-07-06 |
| facebook/infer | [#1951](https://github.com/facebook/infer/issues/1951) | [Pulse missed resource leaks through `java.util.Optional` because the type was unmodeled](https://github.com/SuryanshSS1011/github-contribution-log/blob/main/contributions/infer_1951_optional_resource_leak.md) | [#2068](https://github.com/facebook/infer/pull/2068) | 2026-07-03 |
| pytorch/executorch | [#20556](https://github.com/pytorch/executorch/issues/20556) | [MLX submodule build patched a pinned third-party `CMakeLists.txt` at configure time](https://github.com/SuryanshSS1011/github-contribution-log/blob/main/contributions/executorch_20556_mlx_externalproject.md) | [#20585](https://github.com/pytorch/executorch/pull/20585) | 2026-06-29 |
| astral-sh/ruff | [#25588](https://github.com/astral-sh/ruff/issues/25588) | [PLR2004 flagged ordinary Python version guards such as `sys.implementation.version[0] >= 3`](https://github.com/SuryanshSS1011/github-contribution-log/blob/main/contributions/ruff_25588_plr2004_sys_version.md) | [#25743](https://github.com/astral-sh/ruff/pull/25743) | 2026-06-10 |

## In review

| Project | Issue | What I fixed | Pull request | Status |
|---|---|---|---|---|
| astral-sh/uv | [#7035](https://github.com/astral-sh/uv/issues/7035) | [No hint when a build failure is caused by a `requires-python` mismatch](https://github.com/SuryanshSS1011/github-contribution-log/blob/main/contributions/uv_7035_requires_python_hint.md) | [#19673](https://github.com/astral-sh/uv/pull/19673) | CI green, reworked after review, parked on an upstream refactor. |
| huggingface/diffusers | [#14037](https://github.com/huggingface/diffusers/issues/14037) | [A stale `HookRegistry` cache broke caching enabled after a warmup pass](https://github.com/SuryanshSS1011/github-contribution-log/blob/main/contributions/diffusers_14037_hook_registry_cache.md) | [#14093](https://github.com/huggingface/diffusers/pull/14093) | CI green, test rework pushed after four review rounds. |
| EnzymeAD/Enzyme-JAX | [#1084](https://github.com/EnzymeAD/Enzyme-JAX/issues/1084) | [`stablehlo.reduce` over splat constants was not folded at compile time](https://github.com/SuryanshSS1011/github-contribution-log/blob/main/contributions/enzymejax_1084_reduce_const_prop.md) | [#2524](https://github.com/EnzymeAD/Enzyme-JAX/pull/2524) | Reviewed, blocked on a scope decision about add and mul. |

## Filed bug reports

| Project | Issue | What I reported | Status |
|---|---|---|---|
| huggingface/transformers | [#47487](https://github.com/huggingface/transformers/issues/47487) | CodeLlama tokenizer dropped leading whitespace on decode, refiled with a fresh reproduction after [#46491](https://github.com/huggingface/transformers/issues/46491) was stale-closed | My [#47488](https://github.com/huggingface/transformers/pull/47488) merged, later superseded by a more optimal decoder-only fix in [#47862](https://github.com/huggingface/transformers/pull/47862). |
| huggingface/transformers | [#46489](https://github.com/huggingface/transformers/issues/46489) | DeepSeek-Coder v1 tokenizer produced wrong output on v5 and later, a gap left by PR #44801 | Resolved by upstream [#46091](https://github.com/huggingface/transformers/pull/46091), issue closed. |

## About the writeups

Each file in [`contributions/`](contributions/) follows the same four phases, so any entry can be read end to end without context:

- **Phase I, Issue Selection:** why I picked the issue, a short problem summary, how I verified it was still live and claimable, the specific files involved, and the acceptance criteria I held myself to.
- **Phase II, Reproduce & Plan:** environment setup with the concrete obstacles I hit, numbered reproduction steps, expected versus actual behavior, the root cause, and a UMPIRE plan.
- **Phase III, Build:** commit hashes and dates, files changed with line counts, the obstacles that came up during implementation, and what was tested manually and automatically.
- **Phase IV, Submit & Iterate:** the pull request, a dated log of maintainer feedback and how I responded, and reflections on the technical lesson, the collaboration, and what I would do differently.
