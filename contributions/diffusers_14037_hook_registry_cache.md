# Contribution 7: Invalidate the HookRegistry child-registries cache on enable/disable cache

**Contribution Number:** 7
**Student:** Suryansh Sijwali
**Issue:** https://github.com/huggingface/diffusers/issues/14037
**Pull Request:** https://github.com/huggingface/diffusers/pull/14093
**Status:** Open, under active review as of 2026-07-02. Maintainer (sayakpaul)
has reviewed across three rounds; latest exchange resolved a question about the
end-to-end test's flow. CI is gated behind first-time-contributor approval; the
metadata checks pass and the earlier `check_code_quality` failure was fixed.

---

## Why I Chose This Issue

diffusers is HuggingFace's diffusion-model library. I wanted a self-contained
correctness bug in core infrastructure, verifiable on CPU without a GPU. This one
is in the hook/caching system (`FirstBlockCache` / `FasterCache` and friends),
which every diffusion accelerator relies on, and the issue ships a fully
self-contained reproduction. It was genuinely unclaimed — verified across all PR
states and by scanning open PRs by topic, not just by issue number.

## Understanding the Issue

`HookRegistry._get_child_registries()` caches the child-module registries it finds
by walking `named_modules()`, and never invalidates that cache. `enable_cache()` /
`disable_cache()` add and remove block-level hooks, which changes which modules
carry a `_diffusers_hook`. If `cache_context()` is first entered while no block
hooks exist (a warmup pass with caching disabled), the parent registry caches an
incomplete child list. A later `enable_cache(FirstBlockCacheConfig(...))`
registers block hooks, but `_set_context()` still iterates the stale cache, so the
new block `StateManager`s never receive a context and the next cached forward
raises `ValueError: No context is set`.

## Reproduction Process

The issue's self-contained script builds a tiny randomly-initialized
`FluxTransformer2DModel` on CPU, enters `cache_context()` before `enable_cache()`,
and the second cached forward raises the error. I reproduced it on `main` before
writing any fix (`diffusers==0.39.0.dev0`, CPU). I also confirmed the buggy code
path exists on `main` — `_child_registries_cache` is populated in
`hooks.py` and never cleared in `register_hook` / `remove_hook`.

## Solution Approach

Invalidate the cached child-registry list when the set of hooks in the module tree
changes. The staleness originates in `register_hook` / `remove_hook`, but those run
on the *child* block registries, which can't reach the *parent* registry whose
cache is stale. `enable_cache` / `disable_cache` operate on the root module, so I
added a public `HookRegistry.invalidate_child_registries_cache()` helper (clears
the cache on every registry in the tree) and call it from `enable_cache` /
`disable_cache` after hooks are added/removed. I surfaced this placement choice in
the PR so the reviewer could weigh in.

I kept the change minimal and did not touch the unrelated copyright-year bump or
other files' pre-existing style drift.

## Implementation Notes

- `HookRegistry.invalidate_child_registries_cache()` in `hooks.py` clears
  `_child_registries_cache` across the module subtree.
- Called from `enable_cache` and `disable_cache` in `models/cache_utils.py`.
- No change to public API beyond the new helper.

Diff: `hooks.py` +14, `cache_utils.py` +9, `tests/hooks/test_hooks.py` +64.

## Testing Strategy

- Reproduced the issue's exact failure on `main`, confirmed it passes with the fix.
- Confirmed normal-order caching (`enable_cache` then `cache_context`) and the
  `disable_cache` path still work — no regression.
- `test_child_registries_cache_invalidation` — unit test of the cache-invalidation
  helper.
- The full hook suite passes except one pre-existing failure
  (`test_skip_layer_internal_block`, a torch-version error-string mismatch that
  fails on clean `main` too).

## Pull Request

Opened as PR #14093 against `main`. The diffusers PR template has an explicit
AI-agent disclosure section, which I filled honestly. The metadata CI checks
(`fixes-issue`, `missing-tests`, `label`, `size-label`) pass; the heavier
workflows wait on a maintainer's first-contributor CI approval.

## Review and Merge (Round 2)

sayakpaul reviewed and asked whether there should also be a test that exercises the
context path end-to-end, as reported in the issue — my original regression test
only covered the cache-invalidation mechanism at the registry level, not the real
`cache_context` → `enable_cache` → forward flow.

I added `test_cache_context_after_enable_cache_with_prior_context`, which runs the
issue's exact `FluxTransformer2DModel` reproduction through the public caching API.
I bite-tested it: with the fix reverted to upstream it fails with the original
`ValueError: No context is set`; with the fix it passes. I also fixed the
`check_code_quality` CI failure, which was a `doc-builder` docstring line-length
restyle (max_len 119), and replied to the review comment.

## Review (Round 3)

sayakpaul questioned whether the new end-to-end test reflects a practical flow —
entering `cache_context("cond")` before caching is enabled looks like a no-op. I
explained that pipelines call `cache_context()` unconditionally around every
transformer forward (e.g. `pipeline_qwenimage.py` does
`with self.transformer.cache_context("cond"):` on every denoise step regardless
of whether caching is on), so the pre-`enable_cache` context in the test mirrors
a real pipeline run: it is a caching no-op but still builds the parent
`HookRegistry`'s child-registry cache, which is exactly the stale-cache
precondition the bug needs. sayakpaul accepted the reasoning ("Yeah let's do
that"), greenlighting the test. Finalizing per that agreement.

## Learnings & Reflections

- Checking an issue is unclaimed needs more than an issue-number PR search:
  competing PRs often reference the issue by topic, and a merged-to-non-default
  branch PR can leave an issue open-but-fixed. List open PRs and scan by topic.
- A regression test should reproduce the *reported* failure, not just the internal
  mechanism. The maintainer was right that the end-to-end context path needed
  covering; a faithful test is also a better bite-test.
- `make quality` in diffusers is stricter than a bare `ruff check` — it also runs
  `doc-builder style` (docstring wrapping at 119) and `check_doc_toc`. Run the full
  `make quality` before pushing, and revert any unrelated files `make style`
  touches so the diff stays scoped.
