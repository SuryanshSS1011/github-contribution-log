# Open-Source Contribution Log

Tracking workspace for my open-source contributions. Focus areas: ML compilers, kernel/op implementation, inference performance, HPC profiling tooling, and developer tools.

## Active contributions

| # | Repo | Issue | Topic | Status |
|---|---|---|---|---|
| [1](https://github.com/SuryanshSS1011/github-contribution-log/blob/main/contributions/amrex_4160_tinyprofileparser.md) | AMReX-Codes/amrex | [#4160](https://github.com/AMReX-Codes/amrex/issues/4160) | Document TinyProfileParser | Phase I — claim comment posted |
| ~~[2](https://github.com/SuryanshSS1011/github-contribution-log/blob/main/contributions/samsung-one_14407_resizenearestneighbor.md)~~ | ~~Samsung/ONE~~ | ~~[#14407](https://github.com/Samsung/ONE/issues/14407)~~ | ~~Fold ResizeNearestNeighbor in `one-optimize`~~ | Blocked — [repo merge freeze](https://github.com/Samsung/ONE/issues/16479) |
| [3](https://github.com/SuryanshSS1011/github-contribution-log/blob/main/contributions/uv_7035_requires_python_hint.md) | astral-sh/uv | [#7035](https://github.com/astral-sh/uv/issues/7035) | Hint when a build failure is caused by a `requires-python` mismatch | Phase III — [PR #19673](https://github.com/astral-sh/uv/pull/19673) submitted |
| [4](https://github.com/SuryanshSS1011/github-contribution-log/blob/main/contributions/enzymejax_1084_reduce_const_prop.md) | EnzymeAD/Enzyme-JAX | [#1084](https://github.com/EnzymeAD/Enzyme-JAX/issues/1084) | Reduce constant prop | Phase III — [PR #2524](https://github.com/EnzymeAD/Enzyme-JAX/pull/2524) submitted |
| [5](https://github.com/SuryanshSS1011/github-contribution-log/blob/main/contributions/ruff_25588_plr2004_sys_version.md) | astral-sh/ruff | [#25588](https://github.com/astral-sh/ruff/issues/25588) | Exempt `sys.version` family comparisons from PLR2004 | **Merged** — [PR #25743](https://github.com/astral-sh/ruff/pull/25743) |
| [6](https://github.com/SuryanshSS1011/github-contribution-log/blob/main/contributions/executorch_20556_mlx_externalproject.md) | pytorch/executorch | [#20556](https://github.com/pytorch/executorch/issues/20556) | Isolate the MLX submodule build with ExternalProject | **Merged** — [PR #20585](https://github.com/pytorch/executorch/pull/20585) |
| [7](https://github.com/SuryanshSS1011/github-contribution-log/blob/main/contributions/diffusers_14037_hook_registry_cache.md) | huggingface/diffusers | [#14037](https://github.com/huggingface/diffusers/issues/14037) | Invalidate stale `HookRegistry` child-registries cache on enable/disable cache | [PR #14093](https://github.com/huggingface/diffusers/pull/14093) — under review; maintainer greenlit test approach (round 3) |
| [8](https://github.com/SuryanshSS1011/github-contribution-log/blob/main/contributions/infer_1951_optional_resource_leak.md) | facebook/infer | [#1951](https://github.com/facebook/infer/issues/1951) | Model `java.util.Optional` so Pulse catches resource leaks through it | **Merged** — [PR #2068](https://github.com/facebook/infer/pull/2068) |
| [9](https://github.com/SuryanshSS1011/github-contribution-log/blob/main/contributions/ty_3674_slots_rename.md) | astral-sh/ty | [#3674](https://github.com/astral-sh/ty/issues/3674) | Update `__slots__` string when renaming an attribute | **Merged** — [PR #26438](https://github.com/astral-sh/ruff/pull/26438) |

Per-contribution writeups live in [`contributions/`](contributions/).

## Filed bug reports

Issues I investigated and filed; maintainers are actioning the fix.

| # | Repo | Issue | Topic | Status |
|---|---|---|---|---|
| 1 | huggingface/transformers | [#46489](https://github.com/huggingface/transformers/issues/46489) | DeepSeek-Coder v1 tokenizer wrong output on v5+ (gap in PR #44801) | Fix in flight by maintainer ([PR #46091](https://github.com/huggingface/transformers/pull/46091)) covering the scope my coverage probe expanded |
| 2 | huggingface/transformers | [#46491](https://github.com/huggingface/transformers/issues/46491) | CodeLlama tokenizer strips leading space on round-trip (v4 regression) | Open; analysis with proposed fix paths posted, local implementation ready |