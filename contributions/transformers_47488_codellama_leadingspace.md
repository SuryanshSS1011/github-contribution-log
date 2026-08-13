# Fix the CodeLlama tokenizer dropping leading whitespace on decode

**Project:** [huggingface/transformers](https://github.com/huggingface/transformers)
**Issue:** [#47487](https://github.com/huggingface/transformers/issues/47487), filed by me · **Pull request:** [#47488](https://github.com/huggingface/transformers/pull/47488) · **Branch:** `fix/codellama-decode-leading-whitespace`
**Status:** Merged 2026-07-31 (`1b20eb168c`), then **reverted** 2026-08-10 ([#47861](https://github.com/huggingface/transformers/pull/47861)) in favour of a more optimal decoder-only fix ([#47862](https://github.com/huggingface/transformers/pull/47862)). See the update below.

---

## Update (2026-08-10): reverted and superseded

`ArthurZucker` reverted this in [#47861](https://github.com/huggingface/transformers/pull/47861) and landed a more optimal fix in [#47862](https://github.com/huggingface/transformers/pull/47862), a decoder-only change (`Strip(content=" ", left=1)` becomes `Replace(Regex(r"\A (?! )"), "")`) that solves the same bug without touching encoding. Two reasons the decoder-only approach is the tighter one are worth recording:

- **It matches the layer the bug lives in.** The case that matters for indentation, two-or-more leading spaces, encodes losslessly (`"  hello"` becomes `▁▁hello`), so it is a decoder bug, and #47862 fixes it right there by making the trailing `Strip` conditional (`\A`, single space only). A single leading space stays collapsed, which is documented SentencePiece behaviour that #47487 asks to keep. `qgallouedec` had in fact pointed straight at the trailing `Strip` when he confirmed the issue (quoted below), and staying on the decoder keeps encoding untouched, so no ids change.
- **It stays out of the encoding path.** Building the pipeline from a `Prepend + Replace` normalizer instead moves prefix handling into the added-vocabulary matching path, which `Metaspace` stays out of. That is what makes the decoder-only fix more robust across checkpoints: I verified that the normalizer path interacts with special tokens on some of them (a `normalized=True` special token like `<unk>` on `georgesung/llama2_7b_chat_uncensored` gets split, and `normalized=False` checkpoints double the metaspace after an added token, so `decode(encode("there are 2 cats"))` returns `there are 2  cats`), while the decoder-only fix has none of that because it never changes the ids.

Issue #47487 is still being fixed, now by #47862, and my follow-up Llama PR #47700 is closed in favour of it. The sections below are the record as it stood at merge time, except the Learnings section, which is revised to reflect the revert. Read the rest with this correction in mind.

---

## The issue

### Why I picked it

I found, bisected and filed this one myself. The original report, #46491, was stale-closed without a fix, so I refiled it as #47487 with an up-to-date reproduction verified on current `main`, which `qgallouedec` confirmed against commit `fdbd8b0`. Working an issue I diagnosed myself was the point, and my learning goal was the SentencePiece tokenizer pipeline, meaning how the normalizer, pre-tokenizer and decoder compose, since that is the layer where most "the model behaves oddly on my input" bugs actually live.

Practically it is also an ideal contribution shape, since it is a pure tokenizer bug with no GPU, no model weights required to see it, and a two-line `encode` and `decode` round-trip as the reproduction.

### What was wrong

`decode(encode(s))` dropped a leading space whenever `s` began with whitespace, so `"  hello"` came back as `" hello"` and indented code lost a level of indentation. The same root cause corrupted the most common round trip in the other direction, since `decode(ids, skip_special_tokens=True)` returned ordinary text with a spurious leading space, turning `"def foo():"` into `" def foo():"`. It matters most for CodeLlama specifically, because leading whitespace is semantics in Python and losing an indent level changes what the code means, and because the tokenizer's ids no longer matched the pipeline the checkpoint ships in its own `tokenizer.json`. I worked it because I had already localized it and confirmed it was still live on `main`.

### Before I started

Self-filed and confirmed by a maintainer. Before refiling I verified the bug still reproduced on current `main` rather than assuming the stale-closed issue was still accurate. `qgallouedec` confirmed on 2026-07-31, pointing at the exact suspect lines:

> thanks for the refile with an up-to-date repro, confirmed on current main (`fdbd8b0`): two-or-more leading spaces lose exactly one on round-trip, and the culprit is the trailing `decoders.Strip(content=" ", left=1)` [...] Sorry #46491 got stale-closed.

There was also a competing prior attempt in PR #46574, which I checked before starting, and it regresses tab, newline and emoji handling, so it was not a path to build on.

### Where it lives

- `src/transformers/models/code_llama/tokenization_code_llama.py`, holding the pipeline construction with the `Metaspace` pre-tokenizer, the `Strip` decoder and the `_decode` override, plus `set_infilling_processor`.
- `tests/models/code_llama/test_tokenization_code_llama.py`, the tokenizer test suite.
- The reference for the correct construction is `LlamaConverter`, the legacy Llama path, and the `tokenizer.json` shipped with `codellama/CodeLlama-7b-hf`.

### What counts as done

1. Leading whitespace round-trips exactly, covering a single space, multiple spaces, tabs, newlines and nested-indent code.
2. `encode` output matches the pipeline the published checkpoint ships in `tokenizer.json`.
3. The common `encode` then `decode(skip_special_tokens=True)` path stops adding a spurious leading space.
4. Infilling (FIM) still works.
5. No `_decode` override is left behind, since the fix belongs in pipeline construction.

---

## Diagnosis and plan

### Environment setup

I used transformers' `CONTRIBUTING.md` for the editable install and test invocation, and validated on Penn State's ROAR Collab cluster, because faithful validation needs the real `codellama/CodeLlama-7b-hf` tokenizer files rather than the small test fixture.

- **Challenge (the test fixture disagrees with the published checkpoint):** the `hf-internal-testing` fixture marks `<s>` as `normalized=True`, which the real `codellama/CodeLlama-7b-hf` checkpoint does not, so testing only against the fixture would validate the wrong pipeline. I resolved it by validating against the real checkpoint on ROAR, and the maintainer ultimately moved the tests onto the real checkpoint for the same reason.
- **Challenge (moving work from the cluster back to my Mac):** the ROAR clone is shallow, so `git bundle` refuses to produce a usable bundle. I resolved it by transferring with `git format-patch` and applying locally, which works fine against a shallow history.

### Steps to reproduce

1. Install transformers from `main`, CPU only and tokenizer files only, with no weights needed.
2. Load the CodeLlama tokenizer:
   ```python
   from transformers import AutoTokenizer
   tok = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
   ```
3. Round-trip a string with leading whitespace:
   ```python
   s = "  hello"
   tok.decode(tok.encode(s))
   ```
4. Separately, round-trip ordinary text through the common path:
   ```python
   tok.decode(tok.encode("def foo():"), skip_special_tokens=True)
   ```
5. Compare `tok.encode(s)` against the ids produced by the pipeline in the checkpoint's own `tokenizer.json`.

### Expected vs. actual

| Input | Expected | Actual |
|---|---|---|
| `"  hello"` round-trip | `"  hello"` | `" hello"`, so one leading space lost |
| indented code | indentation preserved | one indent level lost |
| `decode(encode("def foo():"), skip_special_tokens=True)` | `"def foo():"` | `" def foo():"`, a spurious leading space |
| `encode(" hello")` vs `encode("hello")` | different ids | identical ids |

That last row is the decisive one, because the information is destroyed at encode time, so no decode-side fix can recover it.

### Root cause

`CodeLlamaTokenizer` built its plain-path pipeline with a `Metaspace(prepend_scheme="first")` pre-tokenizer. The Rust `Metaspace` runs `replace(" ", "▁")` before its guarded prepend, so a real leading space is turned into `▁` first, the prepend guard then sees a leading `▁` and skips, and the user's space is fused into the synthetic prefix at encode time. That is why `" hello"` and `"hello"` produce identical ids, since the leading space is already gone before decode ever runs, and why the ids no longer match the pipeline the checkpoint ships in `tokenizer.json`.

The same root cause produces the spurious-space direction, because the BOS carries the synthetic prefix, so ordinary text decodes with a leading space under `skip_special_tokens=True`.

### The plan

**Understand:** The leading space is destroyed at encode time by the Metaspace ordering, so the fix must be in pipeline construction rather than in decode.

**Match:** This is the crux, and it is where the maintainer redirected me. `LlamaConverter`, the legacy Llama path, emits `normalizer = Sequence([Prepend("▁"), Replace(" ", "▁")])` with `pre_tokenizer = None`, and the same normalizer is already installed by this very class in `set_infilling_processor`. CodeLlama predates the Metaspace refactor since it is Llama-2 based, so the normalizer rather than the Metaspace pre-tokenizer is the correct construction, and it is what `codellama/CodeLlama-7b-hf` ships.

**Plan:**
1. Replace the plain path's `Metaspace` pre-tokenizer with `normalizer = Sequence([Prepend("▁"), Replace(" ", "▁")])`, with the `Prepend` only when `add_prefix_space`.
2. Set `pre_tokenizer = None`.
3. Keep the decoder's `Strip(content=" ", left=1)` gated on `add_prefix_space`, which is now correct because `Prepend` adds exactly one deterministic leading `▁`.
4. Remove the `_decode` override entirely.
5. Validate encode against the shipped `tokenizer.json` and round-trip the whitespace cases on the real checkpoint.

**Implement:** Branch `fix/codellama-decode-leading-whitespace`.

**Review:** My self-checklist was that no `_decode` override remains, that infilling is unaffected, that ids match the published checkpoint, and that behavior differences, however narrow, are named explicitly for the reviewer.

**Evaluate:** Real-checkpoint validation across the whitespace matrix, the common `skip_special_tokens` path, infilling, and the separately-reported FIM repro.

### Edge cases

- **Ordering of pipeline assignment:** The pipeline must be assigned after `super().__init__()`, otherwise the added-token trie is built against the `Prepend` normalizer and `normalized=True` special tokens get split mid-text, including `<s>`. I found this the hard way and flagged it explicitly for the reviewer.
- **One narrow behavior change**, where `"<s>hello"` decodes slightly differently under the new construction. I surfaced this in review rather than letting it be discovered.
- **The whitespace matrix**, covering a single leading space, multiple spaces, tabs, newlines and nested-indent code, all tested since leading whitespace is not one case.
- **FIM and infilling**, which share the class but not the plain path, verified separately including the repro from the independently-filed FIM issue #47602.

---

## Implementation

### Commits and files

| Commit | Date | Author | Message |
|---|---|---|---|
| `d01567f09` | 2026-07-29 | me | Build CodeLlama pipeline from the reference SentencePiece normalizer |
| `33a00a223` | 2026-07-31 | itazap | match tokenizer init to real checkpoint's `tokenizer.json`, code llama is llama2 |

The branch was force-pushed on 2026-07-29, since the original approach of a `_decode` override using a re-encode oracle to detect the synthetic prefix was replaced wholesale after the maintainer's redirect, so the merged history carries only the normalizer fix.

**Files modified:**

| File | Δ |
|---|---|
| `tests/models/code_llama/test_tokenization_code_llama.py` | +56 / −18 |
| `src/transformers/models/code_llama/tokenization_code_llama.py` | +16 / −14 |
| `docs/source/en/model_doc/code_llama.md` | +1 / −1 |

The fix is a net 2 lines of source, being a pipeline built from the reference normalizer plus a deleted `_decode` override.

### What was hard

**Being told the whole approach was at the wrong layer:** My first implementation added a `_decode` override that re-encoded the decoded text to detect whether a leading space was synthetic. It worked, and a community reviewer had already found an edge in it, where `skip_special_tokens=True` makes the re-encode comparison unsound when BOS or EOS are present. Then `itazap` cut through it:

> sorry we should not have to overwrite `_decode` here, the normalizer should be properly constructed in the `_tokenizer` above

That was right, and it reframed the bug, because I had been treating a decode-side symptom when `encode(" hello") == encode("hello")` already proved the information was destroyed earlier. Reading `LlamaConverter` and the class's own `set_infilling_processor` showed the correct construction was already present in two places in the codebase. The rewrite deleted more than it added and made the reviewer's edge case disappear entirely rather than needing a fix.

**Validating against the right artifact:** The `hf-internal-testing` fixture marks `<s>` as `normalized=True` while the published checkpoint does not, so fixture-only validation would have confirmed the wrong pipeline. Running against real `codellama/CodeLlama-7b-hf` tokenizer files on ROAR is what made the validation meaningful, and `itazap` independently reached the same conclusion and moved the tests onto the real checkpoint before merging.

### Testing

- **New and updated tests** in `tests/models/code_llama/test_tokenization_code_llama.py` at +56 and −18, following the file's existing round-trip test conventions and covering the whitespace matrix and the common `skip_special_tokens` path.
- **Before and after evidence, measured on the real `codellama/CodeLlama-7b-hf` checkpoint:**
  - `encode` output matches the pipeline the checkpoint ships in `tokenizer.json`.
  - Leading whitespace round-trips exactly across a single leading space, multiple spaces, tabs, newlines and nested-indent code, with 22 of 22 cases passing on the validation run.
  - The common `encode` then `decode(skip_special_tokens=True)` path is fixed, since 4 of 5 sample strings were affected before and all round-trip now.
  - Infilling still works, and running the exact repro from the separately-filed FIM issue #47602 shows that path round-trips correctly too.
- **Independent verification:** another contributor, `Yashash-Yallapragada`, tested the branch himself and reported plain `encode` and `decode` round-trips correct for 0, 1, 2 and 3-plus leading spaces, tabs and newlines.
- CI ran the maintainer-suggested `run-slow: code_llama` job before merge.

---

## Review and outcome

### The pull request

**[huggingface/transformers#47488](https://github.com/huggingface/transformers/pull/47488)**, opened 2026-07-22 against `huggingface/transformers:main`, referencing #47487 with a close keyword. The body opens with the user-visible failure of indentation silently lost in a code model, then the encode-time root cause with the `encode(" hello") == encode("hello")` evidence, then the pipeline construction and why it matches the shipped `tokenizer.json`, then the validation matrix. I rewrote the body when the approach changed on 2026-07-29 so it described what was actually being reviewed.

When the FIM question came up I @-mentioned `@ArthurZucker` and `@itazap` to get maintainer eyes on the scope split.

### Maintainer feedback

| Date | From | Feedback | My response |
|---|---|---|---|
| 2026-07-24 | Sanjays2402 (review) | In the original `_decode` approach, `skip_special_tokens=True` breaks the re-encode comparison whenever BOS or EOS are present, so a normal sequence keeps the synthetic space. | Verified and acknowledged the same day. It was superseded, because the approach change removed `_decode` entirely, and I said so on the thread so the reviewer knew his catch had been resolved rather than ignored. |
| 2026-07-26 | Yashash-Yallapragada | Tested the branch, found plain round-trips correct, but found a remaining gap in FIM and infilling. | Confirmed the FIM bug is independent of this PR, since raw tokenize shows it upstream of decode, asked maintainers for a scope call, and supported tracking it separately, which he did by opening #47602. |
| 2026-07-26 and 2026-07-28 | itazap (review) | "we should not have to overwrite `_decode` here, the normalizer should be properly constructed in the `_tokenzier` above" | Investigated the construction, found `LlamaConverter`'s normalizer and the class's own `set_infilling_processor` already using it, and rewrote the fix around it in `d01567f09`. Reported initial results before claiming it worked, then confirmed fully. |
| 2026-07-29 | Me | Own update, no maintainer feedback in this window. | Pushed the rewrite and flagged two things proactively, that the pipeline must be assigned after `super().__init__()` or the added-token trie splits `normalized=True` specials like `<s>`, and that there is one narrow behavior change in how `"<s>hello"` decodes. |
| 2026-07-31 | itazap | "code llama was based on llama2 so pre metaspace era. Additionally, CodeLlama-7b-hf is now in tests as opposed to the internal checkpoint which didn't have matching params [...] I made a push to support this, let me know what you think and if it covers all the edge cases you had." | Verified her push against my edge cases, covering leading whitespace down to a single space, `<s>` handling and the FIM prefix path, and all round-trip exactly. Confirmed nothing outstanding on my end. |
| 2026-07-31 | itazap | **Approved** with "Ready to merge! 🤝" | Merged as `1b20eb168c` the same day. |

### What I learned

**Technical:** I leaned on `encode(" hello") == encode("hello")` to argue the fix had to be at encode time, and that framing was too broad. The collision is real, but it only covers a single leading space, which is by-design SentencePiece behaviour that #47487 asks to keep. The case that actually mattered, two-or-more leading spaces and indentation, encoded losslessly (`"  hello"` becomes `▁▁hello`), so it was a pure decoder bug, which is exactly where `ArthurZucker` fixed it in #47862 by making the trailing `Strip` conditional. The information-flow argument still holds, but it has to be applied case by case rather than globally, and `qgallouedec` had pointed straight at the decoder `Strip` when he confirmed the issue. What holds up unchanged is that `Metaspace`'s `replace` runs before its guarded prepend, which is why a single real leading space fuses into the prefix.

**Process & collaboration:** Investigating a maintainer's redirect rather than defending working code is still the right instinct, and following `itazap`'s pointer did delete the `_decode` override and dissolve a reviewer's edge case. But the redirect moved the fix to the encoding layer, and `ArthurZucker`, who owns the tokenizer library, later reverted it in favour of a decoder-only fix. So the deeper lesson is that a redirect is strong input to weigh against the evidence rather than a final answer, and that two maintainers can genuinely disagree on the right layer. The smaller habits that held up are reporting initial results as initial rather than overclaiming, proactively flagging the subtleties a reviewer would otherwise have to find, and, when another contributor hit the adjacent FIM bug, confirming it was independent and supporting him filing it separately rather than absorbing it and stalling this PR.

**What I'd do differently:** I would separate the recoverable case from the unrecoverable one before choosing a layer, since the single-space collision and the two-or-more-space bug looked like one problem but lived at different layers. When changing a pipeline component I would test the paths that component sits in, so the normalizer change should have been exercised against the added-vocabulary path and not just the whitespace round-trip, which is where the regression that got it reverted was hiding. I would still read the reference construction first, `LlamaConverter` and the class's own `set_infilling_processor`, and validate against the real published checkpoint from the start rather than the fixture.

### Follow-ups

- `LlamaTokenizer` has the identical construction and the same latent bug, so the same fix applies, and I opened it as a follow-up PR.
- The FIM and infilling issue #47602, filed by another contributor, is resolved by this change.

### References

- Issues [#47487](https://github.com/huggingface/transformers/issues/47487), which is mine, and #46491, the stale-closed original, plus competing PR #46574, which I rejected because it regresses tab, newline and emoji handling.
- `src/transformers/models/code_llama/tokenization_code_llama.py`, including its own `set_infilling_processor`.
- `LlamaConverter`, the legacy Llama pipeline construction the fix mirrors.
- The `tokenizer.json` shipped with `codellama/CodeLlama-7b-hf`, the ground-truth pipeline.
