# Invalidate the `HookRegistry` child-registries cache on enable/disable cache

**Student:** Suryansh Sijwali ([@SuryanshSS1011](https://github.com/SuryanshSS1011))
**Project:** [huggingface/diffusers](https://github.com/huggingface/diffusers) · **My fork:** https://github.com/SuryanshSS1011/diffusers
**Issue:** [#14037](https://github.com/huggingface/diffusers/issues/14037) · **Pull request:** [#14093](https://github.com/huggingface/diffusers/pull/14093) · **Branch:** `fix/14037-hook-registry-cache-invalidation`
**Status:** Open. CI green, test rework pushed after four review rounds.

---

## Phase I: Issue Selection

### Why I Chose This Issue

diffusers is HuggingFace's diffusion-model library. I wanted a self-contained correctness bug in core infrastructure that is verifiable on CPU without a GPU, and this one sits in the hook and caching system that every diffusion accelerator in the library relies on, covering `FirstBlockCache`, `FasterCache` and friends. My learning goal was stale-cache and invalidation bugs in a framework with a plugin-style hook registry, which is a class of bug where the wrong fix location is much more tempting than the right one.

The issue ships a fully self-contained reproduction, so I could confirm the failure before writing anything.

### Problem Summary

`HookRegistry._get_child_registries()` memoizes the child-module registries it discovers by walking `named_modules()`, and nothing ever invalidates that memo. `enable_cache()` and `disable_cache()` add and remove block-level hooks, changing which modules carry a `_diffusers_hook`, so a registry cached before caching was enabled is wrong afterwards and the next cached forward dies with `ValueError: No context is set`. It matters because the trigger is an ordinary usage order, meaning you run the pipeline once and then turn caching on, and the error message points nowhere near the actual cause. I chose it because it is a real correctness bug in shared infrastructure, reproducible on CPU in seconds, and genuinely unclaimed.

### Issue Vetting

I verified it was open and unassigned across all PR states, and scanned open PRs by topic rather than only by issue number, since competing PRs often reference an issue by subject line. I posted a claim comment on the issue on 2026-06-29 before opening the PR:

> Hey, I have been looking into this and have a fix planned, so I can take it on if no one else is currently working on it. Will follow up with a PR containing the fix soon!

### Where It Lives

- `src/diffusers/hooks/hooks.py`, holding `HookRegistry._get_child_registries()` and `_child_registries_cache`, which is populated and never cleared by `register_hook` or `remove_hook`.
- `src/diffusers/models/cache_utils.py`, holding `enable_cache()` and `disable_cache()`, which mutate the hook tree from the root module.
- `tests/hooks/test_hooks.py` and `tests/pipelines/test_pipelines_common.py` (`FirstBlockCacheTesterMixin`), the two test surfaces this ended up touching.

### Acceptance Criteria

1. The issue's exact reproduction stops raising `ValueError: No context is set`.
2. Normal-order usage, meaning `enable_cache` then `cache_context`, and the `disable_cache` path are unaffected.
3. A regression test reproduces the reported failure rather than only the internal mechanism.
4. `make quality` is clean and the diff is scoped to the fix and its tests.

---

## Phase II: Reproduce & Plan

### Environment Setup

I used diffusers' `CONTRIBUTING.md` for the editable install and `make quality` loop, and read the PR template, which has an explicit AI-assistance disclosure section, before opening.

- CPU-only local environment on `diffusers==0.39.0.dev0` from an editable install of `main`. The repro builds a tiny randomly-initialized `FluxTransformer2DModel`, so there is no weights download and no GPU.
- **Challenge (`make quality` is stricter than it looks):** a bare `ruff check` passes while CI still fails, because `make quality` also runs `doc-builder style`, which wraps docstrings at 119 columns, and `check_doc_toc`. My first push failed `check_code_quality` on a docstring line-length restyle. I resolved it by running the full `make quality` before pushing and reverting unrelated files that `make style` reformats, so the diff stays scoped.
- **Challenge (CI is gated for first-time contributors):** only the metadata checks (`fixes-issue`, `missing-tests`, `label` and `size-label`) run until a maintainer approves the heavier workflows, so early red or absent CI is not necessarily signal.

### Steps to Reproduce

1. Install `diffusers` from `main` in editable mode, and CPU is fine.
2. Build a small randomly-initialized `FluxTransformer2DModel`, which the issue's script does.
3. Enter a cache context and run a forward before enabling caching:
   ```python
   with torch.no_grad(), model.cache_context("cond"):
       model(**make_inputs())
   ```
4. Now enable caching with `model.enable_cache(FirstBlockCacheConfig(threshold=0.2))`.
5. Run the same context-wrapped forward again.

### Expected vs. Actual

**Actual:** step 5 raises `ValueError: No context is set`.

**Expected:** the second forward runs with caching active, because enabling caching after a warmup pass is an ordinary usage order rather than a misuse.

### Root Cause

`HookRegistry._get_child_registries()` caches the list of child registries it finds by walking `named_modules()`. When `cache_context()` is first entered while no block hooks exist, which is the warmup pass, the parent registry caches an incomplete child list. `enable_cache(FirstBlockCacheConfig(...))` then registers block hooks, but `_set_context()` still iterates the stale cached list, so the new block `StateManager`s never receive a context and the next cached forward raises.

The subtle part is where to invalidate. The staleness originates in `register_hook` and `remove_hook`, but those run on the child block registries, which cannot reach the parent registry whose cache is stale. `enable_cache` and `disable_cache` operate on the root module, so they are the only place with the whole subtree in scope.

### Plan (UMPIRE)

**Understand:** A memo of which children have hooks outlives the event that changes which children have hooks.

**Match:** The codebase already treats `enable_cache` and `disable_cache` in `cache_utils.py` as the root-level entry points for mutating the hook tree, so putting invalidation there matches the existing division of responsibility rather than introducing a new one.

**Plan:**
1. Reproduce the issue's script on `main` before writing any fix.
2. Confirm the buggy path exists in source, since `_child_registries_cache` is populated in `hooks.py` and never cleared in `register_hook` or `remove_hook`.
3. Add a public `HookRegistry.invalidate_child_registries_cache()` that clears the cache across the module subtree.
4. Call it from `enable_cache` and `disable_cache` after hooks are added or removed.
5. Surface the placement choice explicitly in the PR so the reviewer can weigh in on parent versus child.
6. Add a regression test, then run `make quality` and the hook suite.

**Implement:** Branch `fix/14037-hook-registry-cache-invalidation`.

**Review:** My self-checklist was that there is no public API change beyond the one helper, that I do not touch the unrelated copyright-year bump or other files' pre-existing style drift, and that the fix does not alter behavior when caching was enabled first.

**Evaluate:** Bite-test both directions, so revert the source fix and confirm the new test fails with the original `ValueError`, then restore it and confirm it passes.

### Edge Cases Considered

- **Normal order**, meaning `enable_cache` then `cache_context`, must be unaffected.
- **`disable_cache`** leaves a cache that is stale in the other direction, with children that no longer have hooks, so it needs the same invalidation rather than only `enable_cache`.
- **Where the cache lives:** Invalidating only the registry you are holding is not enough, because the stale entry is on the parent, so the helper has to clear across the subtree.

---

## Phase III: Build

### Implementation Progress

| Commit | Date | Message |
|---|---|---|
| `424f75264` | 2026-06-29 | Invalidate HookRegistry child-registries cache on enable/disable cache |
| `a71bf7a34` | 2026-07-03 | Test the enable-cache-after-inference flow through a pipeline |
| `d80bba85e` | 2026-07-13 | Make the FirstBlockCache regression test drive two explicit `cache_context` forwards |
| `e4def1b32`, `61b378ab2`, `607fc8d72` | 2026-06-30 to 2026-07-22 | Merge `main` to keep the branch current across review rounds |

**Files modified:**

| File | Δ |
|---|---|
| `src/diffusers/hooks/hooks.py` | +14 |
| `src/diffusers/models/cache_utils.py` | +9 |
| `tests/pipelines/test_pipelines_common.py` | +46 |
| `tests/hooks/test_hooks.py` | +20 |

The source fix has been stable since the first commit, and every subsequent commit is test work driven by review.

### Challenges Faced

The obstacle here was not the fix but writing a test that actually reproduces the reported failure, and it took three rounds to get right.

1. My first test, `test_child_registries_cache_invalidation`, verified the invalidation mechanism at the registry level. `sayakpaul` asked whether the context path should be exercised end to end as reported in the issue, and he was right, because the mechanism test would pass even if `enable_cache` never called the helper.
2. My second attempt ran the issue's `FluxTransformer2DModel` reproduction through the public caching API, then moved into `FirstBlockCacheTesterMixin` as `test_first_block_cache_enabled_after_inference` so it would cover every pipeline in that suite, meaning Flux, LTX, Mochi, CogVideoX and Hunyuan. `sayakpaul` questioned whether entering `cache_context("cond")` before caching is enabled is a practical flow, since it looks like a no-op.
3. I explained that pipelines call `cache_context()` unconditionally around every transformer forward, since `pipeline_qwenimage.py` does `with self.transformer.cache_context("cond"):` on every denoise step regardless of whether caching is on, so the pre-`enable_cache` context mirrors a real pipeline run. It is a caching no-op that still builds the parent registry's child cache, which is exactly the bug's precondition. He accepted the reasoning with "Yeah let's do that".
4. He then caught that the test was still implicitly relying on the pipeline to enter the context and did not include the second context entry from the issue. That was a fair call, so I agreed on the thread and reworked it to drive the two `cache_context("cond")` forwards explicitly, with the second being where `No context is set` was raised, and to assert the pipeline actually called the transformer.

The through-line is that a regression test has to fail for the reported reason through the reported path, and each round removed one more layer of indirection between the test and the issue.

### Testing

- **New, in final form:** an end-to-end regression test in `FirstBlockCacheTesterMixin` in `tests/pipelines/test_pipelines_common.py` that drives the issue's two explicit `cache_context("cond")` forwards through a real pipeline, calling `pipe(**inputs)` with caching off, then `enable_cache()`, then `pipe(**inputs)` again, and asserts the transformer was actually called. Because it lives in the mixin it runs for every FirstBlockCache pipeline. There is also a unit test of the invalidation helper in `tests/hooks/test_hooks.py`.
- **Test patterns followed:** the mixin placement is diffusers' own convention for behavior that must hold across all pipelines in a family, and `sayakpaul` specifically asked for it there rather than as a standalone test.
- **Bite-tested both directions:** with the source fix reverted to upstream the test fails with the original `ValueError: No context is set`, and with the fix it passes across all the FirstBlockCache pipelines.
- **Existing suite:** the full hook suite passes except one pre-existing failure, `test_skip_layer_internal_block`, which is a torch-version error-string mismatch that also fails on clean `main`.
- **Manual:** I confirmed normal-order caching and the `disable_cache` path are unaffected.
- CI is 4 of 4 green as of 2026-08-04.

---

## Phase IV: Submit & Iterate

### Pull Request

**[huggingface/diffusers#14093](https://github.com/huggingface/diffusers/pull/14093)**, opened 2026-06-29 against `huggingface/diffusers:main`, referencing the issue with a close keyword, and the `fixes-issue` metadata check passes. The body opens with the user-visible failure and the stale-cache diagnosis, then the fix, then the placement question I deliberately surfaced for the reviewer, which is why invalidation belongs at the root `enable_cache` and `disable_cache` rather than in `register_hook` and `remove_hook`. The diffusers template's AI-assistance disclosure section is filled in honestly.

An automated reviewer (`sergereview`) also assessed the change on 2026-07-02 and agreed with the stale-cache diagnosis and the targeting.

### Maintainer Feedback Log

| Date | From | Feedback | My response |
|---|---|---|---|
| 2026-06-30 | sayakpaul | "Should there also be testing when the context path is exercised, as reported in #14037?" | Agreed, and added an end-to-end test reproducing the issue's flow, plus fixed the `check_code_quality` docstring-restyle failure in the same push. |
| 2026-07-02 | sayakpaul | "Help me understand why would this be a practical flow though... does this reflect how caching is to be used actually in practice?" | Explained that pipelines call `cache_context()` unconditionally around every transformer forward, using `pipeline_qwenimage.py` as the concrete example, so the pre-`enable_cache` context mirrors a real run and is exactly what builds the stale cache. He replied "Yeah let's do that". |
| 2026-07-03 | Me | Own update, no maintainer feedback in this window. | Pushed `a71bf7a34`, moving the end-to-end test into `FirstBlockCacheTesterMixin` per his suggestion so it covers Flux, LTX, Mochi, CogVideoX and Hunyuan, and removed the standalone test. |
| 2026-07-13 | sayakpaul | The test still assumes the pipeline enters the context internally and omits the issue's second context entry, so "I think this test is still incomplete." | Agreed on the thread, then pushed `d80bba85e`, which drives the two `cache_context("cond")` forwards explicitly, asserts the transformer was called, and which I confirmed fails on `main` and passes with the fix across all FirstBlockCache pipelines. |
| 2026-07-24 | Me | Own update, no maintainer feedback in this window. | Gentle bump summarizing the rework, and I am awaiting the next review. |

### Learnings & Reflections

**Technical:** Memoization bugs are located by asking what event invalidates this and then who can see the whole structure when that event happens, and those are usually two different places. Here the staleness originates in `register_hook` and `remove_hook` but can only be fixed from `enable_cache` and `disable_cache`, because the child registry cannot reach the parent whose cache is wrong. Fixing it at the origin would have been the intuitive move and would not have worked.

**Process & collaboration:** The reviewer was right four times in a row about the test and never commented on the fix, which taught me more than an approval would have. The specific lesson is that a regression test which verifies the mechanism you implemented is nearly worthless, because it is shaped by your fix rather than by the bug, and it has to fail for the reported reason through the reported path before it means anything. I also learned when to push back with evidence rather than just comply, because when he questioned whether the pre-`enable_cache` context was a practical flow, the right answer was a concrete pipeline in the codebase doing exactly that rather than a rewrite.

**What I'd do differently:** I would write the end-to-end test first, from the issue's script, and only then the unit test, because reversing that order would have compressed four rounds into roughly one. I would also verify against the repo's real quality gate, meaning `make quality` rather than `ruff check`, before the first push, since the first CI failure was self-inflicted and cost a round trip.

### Resources Used

- Issue #14037 and its self-contained reproduction script.
- `src/diffusers/hooks/hooks.py` and `src/diffusers/models/cache_utils.py`.
- `tests/pipelines/test_pipelines_common.py`, holding `FirstBlockCacheTesterMixin`, the convention for cross-pipeline behavior tests.
- `src/diffusers/pipelines/qwenimage/pipeline_qwenimage.py`, the evidence that pipelines enter `cache_context()` unconditionally.
