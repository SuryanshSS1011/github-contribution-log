# Contribution: Fix CodeLlama tokenizer dropping leading whitespace on decode

**Student:** Suryansh Sijwali
**Issue:** https://github.com/huggingface/transformers/issues/47487
**Pull Request:** https://github.com/huggingface/transformers/pull/47488
**Status:** Merged 2026-07-31 by `itazap` (merge commit `1b20eb168c`). Issue #47487 closed as completed.

---

## Why I Chose This Issue

I had already filed and bisected this one (refiled as #47487 after the original #46491 was stale-closed
without a fix), so the diagnostic work was mine and I had verified it was still live on `main`. It is a
pure tokenizer bug: no GPU, no model weights, reproducible with a two-line `encode`/`decode` round-trip,
and it sits in the SentencePiece tokenizer family I had been working in.

## Understanding the Issue

`decode(encode(s))` dropped a leading space whenever `s` began with whitespace. `"  hello"` came back as
`" hello"`, and indented code lost a level of indentation. The original PR attempt tried to recover the
space on the decode side, but the maintainer (itazap) pointed out that was the wrong layer, the fix
belonged in how the pipeline is constructed.

## Root Cause

`CodeLlamaTokenizer` built its plain-path pipeline with a `Metaspace(prepend_scheme="first")`
pre-tokenizer. The Rust `Metaspace` runs `replace(" ", "▁")` *before* its guarded prepend, so a real
leading space is turned into `▁` first, the prepend guard then sees a leading `▁` and skips, and the
user's space is fused into the synthetic prefix at encode time. `" hello"` and `"hello"` end up with
identical ids, so the leading space is already gone before decode runs. The ids also no longer matched
the pipeline the checkpoint ships in its `tokenizer.json`.

The same root cause broke the most common round-trip, `encode(text)` then
`decode(ids, skip_special_tokens=True)`: the BOS carried the synthetic prefix, so ordinary text came back
with a spurious leading space (`"def foo():"` decoding to `" def foo():"`).

## The Fix

Build the pipeline from the reference normalizer instead of the Metaspace pre-tokenizer:

- normalizer `Sequence([Prepend("▁"), Replace(" ", "▁")])` (the `Prepend` only when `add_prefix_space`)
- `pre_tokenizer = None`
- decoder keeps `Strip(content=" ", left=1)` gated on `add_prefix_space`, now correct because `Prepend`
  adds exactly one deterministic leading `▁`
- the `_decode` override is removed

This is the pipeline `LlamaConverter` emits for the legacy Llama path and the same normalizer the class
already installs in `set_infilling_processor`, so it matches what `codellama/CodeLlama-7b-hf` ships in
`tokenizer.json`. `CodeLlama` predates the Metaspace refactor (it is Llama-2 based), which is why the
normalizer, not the Metaspace pre-tokenizer, is the correct construction.

## Validation

Measured on `codellama/CodeLlama-7b-hf`: encode matches the shipped `tokenizer.json`, and leading
whitespace round-trips exactly (single leading space, tabs, newlines, multiple spaces, nested-indent
code). The common `encode` then `decode(skip_special_tokens=True)` path is fixed (4 of 5 sample strings
were affected before, all round-trip now). Infilling still works. Running the exact repro from the
separately-filed FIM issue #47602 shows that path round-trips correctly too.

## How It Landed

itazap reviewed, endorsed the approach, and pushed a commit on top that kept the fix and moved the tests
onto the real `codellama/CodeLlama-7b-hf` checkpoint (the `hf-internal-testing` fixture marked `<s>` as
`normalized=True`, which the published checkpoint does not). She merged it the same day.

## Follow-ups

`LlamaTokenizer` has the identical construction and the same latent bug, so the same fix applies; opened
as a follow-up PR. The FIM/infilling issue #47602 (filed by another contributor) is resolved by this
change.
