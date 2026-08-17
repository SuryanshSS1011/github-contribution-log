# Treat a BOS token id of 0 as set when concatenation is disabled

**Project:** [decoderesearch/SAELens](https://github.com/decoderesearch/SAELens)
**Issue:** none, self-sourced · **Pull request:** [#724](https://github.com/decoderesearch/SAELens/pull/724) · **Branch:** `fix/bos-token-id-zero`
**Status:** Open, awaiting review. Two red checks, neither attributable to this change (see Testing).

---

## The issue

### Why I picked it

This is my third SAELens contribution and the second self-sourced one, following the entry-credential fix in #716 and the device-placement fix in #722. The pattern in that repo is that its issue board is largely inactive while outside contributors land self-sourced PRs, so finding bugs myself is the way to keep contributing. My learning goal here was falsy-value bugs, which are the Python equivalent of the silent device bugs I chased in #722, in that they never raise and the wrong behavior looks like the configured behavior.

### What was wrong

`concat_and_batch_sequences` carries the batch-start token into the `disable_concat_sequences` path using a truthiness test rather than an `is not None` check, so a BOS token id of `0` reads as absent. Token id `0` is the BOS id of the entire GPT-NeoX family, including Pythia, which is the canonical model in the SAELens tutorials and tests. It matters because it reaches training rather than only pretokenization, so a SAE trained on Pythia with `disable_concat_sequences=True` gets activations with no BOS at all while the run's config records that BOS was prepended, and nothing warns. I picked it because BOS positions carry well-known outlier activations in SAE work, so their absence changes the distribution the SAE is trained on, and the run is not reproducible from its own metadata.

### How I found it

Self-sourced on 2026-08-13 by code reading, then confirmed with a minimal reproduction and a regression test that fails on current upstream `main` at `7c8df766` (release 6.49.1). What drew my eye was inconsistency rather than the logic itself. Every other token-id check in the same file uses `is not None`, including the three in `_add_tokens_to_batch` and the one immediately below this line, and this single line tests truthiness. Before opening the PR I searched the issue board and all 22 open PRs and confirmed none touch `tokenization_and_batching.py` or this behavior.

### Where it lives

- `sae_lens/tokenization_and_batching.py`, `concat_and_batch_sequences`, the one line that differs from every sibling check in the file.
- `sae_lens/training/activations_store.py`, `_iterate_tokenized_sequences`, which is how the bug reaches training.
- `sae_lens/config.py`, where `PretokenizeRunnerConfig.begin_batch_token` defaults to `"bos"` and `begin_sequence_token` defaults to `None`, which is exactly the argument shape the broken line exists to handle.
- `tests/training/test_tokenization_and_batching.py`, whose `disable_concat_sequences` tests all use id `998` or `999`.

### What counts as done

1. A requested BOS of `0` is actually prepended.
2. An explicitly configured `begin_sequence_token_id=0` is not silently overwritten.
3. Nothing changes for gpt2-style tokenizers, whose BOS id is 50256.
4. Both failure modes have a regression test that fails on `main`.

---

## Diagnosis and plan

### Environment setup

The repo's own `.venv` with python 3.12, torch 2.12.1 and transformers 5.12.1 on macOS CPU. No GPU or model download is needed, since the affected helper takes a plain token iterator.

- **Challenge (confirming the id is real rather than hypothetical):** the whole bug depends on `0` being an actual BOS id in the wild, so I checked the tokenizers directly rather than assuming.
  ```
  EleutherAI/pythia-70m-deduped    bos_token='<|endoftext|>'  bos_token_id=0
  EleutherAI/gpt-neox-20b          bos_token='<|endoftext|>'  bos_token_id=0
  gpt2                             bos_token='<|endoftext|>'  bos_token_id=50256
  ```
- Carried over from #722, I match the repo's pinned linter versions rather than whatever is installed, and run the type checker through the project's own configuration, since a bare invocation resolves the wrong interpreter and reports phantom errors.

### Steps to reproduce

1. Install SAELens 6.49.1.
2. Call the helper directly with a BOS id of 0 and concatenation disabled:
   ```python
   from sae_lens.tokenization_and_batching import concat_and_batch_sequences

   def run(bos_id):
       return [b.tolist() for b in concat_and_batch_sequences(
           tokens_iterator=iter([torch.arange(100, 110, dtype=torch.long)]),
           context_size=8,
           begin_batch_token_id=bos_id,
           begin_sequence_token_id=None,
           disable_concat_sequences=True,
       )]
   ```
3. Run it with `1`, with `50256`, and with `0`.
4. Separately, pass `begin_batch_token_id=999` together with an explicit `begin_sequence_token_id=0`.

### Expected vs. actual

**Actual:**

```
run(1)      # [[1, 100, 101, 102, 103, 104, 105, 106]]      BOS present
run(50256)  # [[50256, 100, 101, 102, 103, 104, 105, 106]]  BOS present
run(0)      # [[100, 101, 102, 103, 104, 105, 106, 107]]    BOS missing
```

and for the second failure mode, the first token is `999` rather than the explicitly requested `0`.

**Expected:** `run(0)` prepends `0` exactly as the other two prepend their ids, and an explicit `begin_sequence_token_id=0` is honored.

### Root cause

One line:

```python
if begin_batch_token_id and not begin_sequence_token_id:
    begin_sequence_token_id = begin_batch_token_id
```

Two distinct failures come out of it. A `begin_batch_token_id` of `0` is falsy, so the carry-over never runs, `begin_sequence_token_id` stays `None`, and no sequence gets a BOS. Separately, `not begin_sequence_token_id` is true for an explicitly configured `0`, so a caller who asks for token `0` gets `begin_batch_token_id` instead.

The reason it reaches training is that `ActivationsStore._iterate_tokenized_sequences` builds the arguments straight from the store's config, passing `begin_batch_token_id=(bos_token_id if self.prepend_bos else None)` with `begin_sequence_token_id=None`, and `prepend_bos` defaults to `True`.

### The plan

**Understand:** A truthiness test stands in for a presence test, and the one token id where those differ is the BOS id of a major model family.

**Match:** The rest of the file already does it correctly. Every other token-id check uses `is not None`, so the fix is to make this line consistent with its neighbours rather than to introduce a new convention.

**Plan:**
1. Change the condition to `begin_batch_token_id is not None and begin_sequence_token_id is None`.
2. Add a regression test for the dropped BOS.
3. Add a second for the overwritten explicit id, since they are separate failures from one line.
4. Confirm both fail on unpatched `main`.
5. State the behavior change for existing runs honestly in the PR.

**Implement:** Branch `fix/bos-token-id-zero`, single commit.

**Review:** My self-checklist was that nothing changes for non-zero ids, that both failure modes are covered separately, and that the consequence for anyone currently hitting this is stated rather than glossed.

**Evaluate:** Run both tests against `main` and against the fix.

### Edge cases

- **Why it was never caught.** The path is well covered, but every existing `disable_concat_sequences` test uses id `998` or `999`. The one value that breaks the line is the one value never tested, which is the general shape of falsy-value bugs.
- **This changes results for affected runs.** Anyone currently hitting this now gets the BOS they asked for, so their sequences shift by one token and drop the last token of each context. That is exactly what the non-zero-BOS path already does, so the change makes the two consistent rather than introducing new behavior, but it is not a no-op and I said so.
- **Blast radius.** Nothing changes for gpt2-style tokenizers, and `disable_concat_sequences` defaults to `False`, so the affected population is users who opted into that flag on a GPT-NeoX family model.
- **Both entry points.** The offline pretokenizer reaches the same line through `PretokenizeRunnerConfig`, whose defaults are precisely the argument shape the carry-over exists to handle, so the fix covers both paths rather than only the training one.

---

## Implementation

### Commits and files

| Commit | Date | Message |
|---|---|---|
| `6550f407e` | 2026-08-14 | fix: treat a BOS token id of 0 as set when concatenation is disabled |

**Files modified:**

| File | Δ |
|---|---|
| `tests/training/test_tokenization_and_batching.py` | +40 |
| `sae_lens/tokenization_and_batching.py` | +1 / −1 |

A one-character-class change to one condition, with forty lines of test proving it matters.

### What was hard

The fix is one line, so the work was establishing that it is worth anyone's attention. A truthiness-versus-`is not None` difference is the kind of thing a reviewer can reasonably wave off as style, and the only way to make it a bug report rather than a nitpick is to show the specific value that breaks and where it comes from.

That meant three things. Confirming `0` is the real BOS id for Pythia and GPT-NeoX rather than a contrived case. Tracing the call chain from `ActivationsStore` to show the line sits on the training path rather than only in an offline utility, including that `prepend_bos` defaults to `True`. And explaining why it matters at the level SAE researchers care about, which is that BOS positions carry outlier activations, so their absence changes the training distribution while the config claims otherwise.

The second difficulty was being honest that fixing it changes existing results. It is tempting to present a one-line correctness fix as free. It is not, since affected runs will now produce different sequences, and the PR says so plainly along with the reason that is the correct behavior.

### Testing

- **Two regression tests**, one per failure mode, both using id `0` and both failing on unpatched `main`:
  - `..._uses_begin_batch_token_id_of_zero_with_concat_sequences_disabled`, which fails as `[[4, 5, 6, 7, 8], ...] == [[0, 4, 5, 6, 7], ...]`.
  - `..._keeps_explicit_begin_sequence_token_id_of_zero`, which fails as `[[999, 4, 5, 6, 7], ...] == [[0, 4, 5, 6, 7], ...]`.
- They follow the existing tests in that file, which is why the gap is visible in the first place, since those use `998` and `999` in the same shape.
- `make check-ci` for format and linting, run with the repo's pinned versions.
- **CI has two red checks and I have not attributed either to this change.** `claude-review` fails on all four of the most recent open PRs in the repo, so it is repo-wide. `build (3.10)` fails here and on one other open PR while passing on two others, so I can neither call it pre-existing nor mine on the evidence I have.

---

## Review and outcome

### The pull request

**[decoderesearch/SAELens#724](https://github.com/decoderesearch/SAELens/pull/724)**, opened 2026-08-14 against `decoderesearch/SAELens:main`. No `Fixes` keyword, since it is self-sourced, which follows the same reasoning as #722, where the repo's norm is to open PRs directly and framing work as a bug fix rather than a proposal is what gets it merged.

The body opens with the inconsistency against the rest of the file, shows the offending condition, names token `0` as the GPT-NeoX BOS id, and then traces the path into training through `ActivationsStore` including the `prepend_bos` default. It gives the one-line diff, explains why the existing tests never caught it, and closes with the behavior change for affected runs. The repo's template sections for type of change, the checklist and the `make check-ci` confirmation are all completed.

An automated Copilot review restated the diagnosis correctly on the same day.

### Maintainer feedback

| Date | From | Feedback | My response |
|---|---|---|---|
| 2026-08-14 | Copilot review | Summarized it as a bug where a BOS token id of `0` was treated as unset under `disable_concat_sequences=True`, causing missing BOS prepends. | n/a, it matched the analysis. |

No human review yet.

### What I learned

**Technical:** Falsy-value bugs are found by looking for inconsistency rather than by looking for wrongness. The line is not obviously incorrect in isolation, and it only became a bug report once I noticed every other check in the same file uses `is not None`. That gives a cheap search pattern, which is to look for the one place in a file that tests a value differently from its neighbours, then ask which value distinguishes them. Here the answer is `0`, and `0` happens to be a real BOS id for a major model family.

**Process & collaboration:** For a one-line fix, essentially all the persuasion is in showing reach and consequence. Tracing the call chain into `ActivationsStore` and noting the `prepend_bos` default is what turns a style observation into a training-correctness bug, and stating that existing runs will produce different sequences is what lets a maintainer weigh it properly. I also checked the board and all 22 open PRs before opening, which costs a few minutes and avoids duplicating someone.

**What I'd do differently:** I would grep the whole repository for the same pattern rather than only reading the one file. A truthiness test on an integer id is a repeatable defect shape, and having found one instance I should have swept for others in the same pass, so any siblings could land together instead of becoming a second PR later.

### References

- `sae_lens/tokenization_and_batching.py`, and the `is not None` checks in `_add_tokens_to_batch` that make the inconsistency visible.
- `sae_lens/training/activations_store.py`, `_iterate_tokenized_sequences`, and the `prepend_bos` default.
- `sae_lens/config.py`, `PretokenizeRunnerConfig` defaults for `begin_batch_token` and `begin_sequence_token`.
- The GPT-NeoX, Pythia and gpt2 tokenizer configs, for the BOS ids.
- My earlier [#722](https://github.com/decoderesearch/SAELens/pull/722), the previous self-sourced fix in the same repo.
