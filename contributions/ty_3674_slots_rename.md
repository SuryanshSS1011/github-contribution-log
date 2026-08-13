# Update the `__slots__` string when renaming an attribute (ty)

**Project:** [astral-sh/ty](https://github.com/astral-sh/ty), developed in the [astral-sh/ruff](https://github.com/astral-sh/ruff) monorepo · **My fork:** https://github.com/SuryanshSS1011/ruff
**Issue:** [ty#3674](https://github.com/astral-sh/ty/issues/3674) · **Pull request:** [ruff#26438](https://github.com/astral-sh/ruff/pull/26438) · **Branch:** `ty-rename-slots`
**Status:** **Merged** 2026-07-06 (merge commit `f1da30d9`). Issue closed.

> Note on the split. ty's issues are filed in `astral-sh/ty`, but its source lives in the `ruff` monorepo under `crates/ty_ide`, so the PR is against `astral-sh/ruff`. I cross-linked both directions so the issue thread points at the PR.

---

## Phase I: Issue Selection

### Why I Chose This Issue

ty is Astral's new type checker and language server, written in Rust in the ruff monorepo. I wanted a language-server and IDE feature rather than another linter rule, because it is a different kind of correctness where the failure mode is that the refactor silently broke your code rather than that a diagnostic was wrong. My learning goal was how a rename refactor is actually implemented, meaning how references are discovered and how you keep a rename from over-matching.

The issue is well-scoped, since it is an LSP rename-symbol gap that Pylance already handles, it was discussed and clarified by maintainers in-thread, and it is self-contained to the rename and references code.

### Problem Summary

ty's rename symbol updates every use of an attribute but leaves the matching string in the class's `__slots__` untouched, so renaming `self.value` to `self.data` rewrites all the references and leaves `__slots__ = ("value",)` pointing at a name that no longer exists. That is worse than not renaming at all, because `__slots__` is the authoritative list of permitted instance attributes, so the refactor leaves the class broken and the breakage only shows up at runtime on attribute assignment. It matters because Pylance does this correctly and users reasonably expect parity when switching editors or servers. I chose it because it is a contained feature in a fast-moving, high-quality Rust codebase, with the expected behavior already pinned down by an existing implementation.

### Issue Vetting

Open and unassigned. The thread had already converged, since `MichaReiser` initially mistook it for ty#2464, `AlexWaygood` clarified what the reporter meant with a concrete example and a Pyright playground link showing the expected behavior, and the reporter `jonathandung` confirmed he had been relying on Pylance for exactly this.

I posted my implementation plan on the issue on 2026-06-09 before writing code, covering where I intended to hook in and the scope I proposed, and I explicitly waited for feedback rather than starting immediately. `jonathandung` responded the same day endorsing the same-class restriction and asking that a dict-shaped `__slots__` preserve values and rename only keys, which I folded into the scope. When I opened the PR I posted back to the issue thread with the link and the scope I had settled on, inviting maintainers to ask for it narrower or wider.

### Where It Lives

- `crates/ty_ide/src/references.rs`, the references visitor that walks every node looking for matches to the rename target.
- `crates/ty_ide/src/rename.rs`, where reference targets become text edits.

### Acceptance Criteria

1. Renaming an instance attribute also renames the matching string in the same class's `__slots__`.
2. Only the string's inner content is rewritten, so the quotes stay put.
3. Tuple, list, set and dict `__slots__` are handled, and for a dict only keys are renamed.
4. Two unrelated classes that both declare a `"value"` slot are not cross-renamed.
5. Nothing that merely looks like a slot, such as a parameter, a local or a slot on another class, is touched.

---

## Phase II: Reproduce & Plan

### Environment Setup

I used the same ruff monorepo checkout and toolchain as my earlier PLR2004 work, so the build and test loop was already established, and ty's crates build and test with the monorepo's standard `cargo` workflow.

- **Challenge (locating the right seam in an unfamiliar subsystem):** ty's IDE features are split across `ty_ide`, covering references, rename, goto and hover, and the semantic model, and it was not obvious whether slot renaming belonged in reference discovery or edit generation. I resolved it by reading how rename consumes references and confirming they share one node walk, which is what made the reference visitor the correct seam.
- Reproduction is manual and editor-driven, since it is an LSP rename, so the practical loop is ty's rename test harness rather than a running editor.

### Steps to Reproduce

1. Write a class whose attribute appears both as an assignment and in `__slots__`:
   ```python
   class Foo:
       __slots__ = ("value",)

       def __init__(self):
           self.value = 42
   ```
2. Open it with ty as the language server.
3. Invoke rename symbol on `self.value`, renaming it to `data`.
4. Inspect the resulting file.

### Expected vs. Actual

**Actual:** every `self.value` reference becomes `self.data`, and `__slots__` still reads `("value",)`. The class is now broken, because assigning `self.data` raises `AttributeError` at runtime since `data` is not a declared slot, and nothing in the editor indicates it.

**Expected, matching Pylance:** `__slots__` becomes `("data",)` in the same edit, with the quoting style preserved.

### Root Cause

This is not a bug in the rename logic but a gap in what counts as a reference. ty's references visitor recognizes identifiers bound to the rename target, and a slot name is a string literal, which is semantically an attribute name but syntactically just data, so the visitor never sees it. `__slots__` is the one place in Python where a string literal is load-bearing for attribute binding, which is precisely why Pylance special-cases it too. The fix belongs in `references.rs`, gated tightly enough that ordinary strings are unaffected.

### Plan (UMPIRE)

**Understand:** A rename must also rewrite string literals in `__slots__` that name the attribute being renamed, and only those.

**Match:** The references visitor already walks every node looking for matches, and rename reuses that same walk, so adding a narrowly-gated string-literal case gets slot renaming for free without a second traversal. The codebase's existing structure already supports it.

**Plan:**
1. In the references visitor's string-literal handling, check whether the string sits inside a `__slots__` assignment in a class body.
2. Additionally require that the attribute being renamed actually belongs to that class, by walking the ancestor scopes of the rename target's definitions to find the class that owns the `__slots__`.
3. In `rename.rs`, emit the edit for the string's inner content so quoting is preserved.
4. Support tuple, list, set and dict literals, and for dicts handle keys only.
5. Add rename tests covering the accepted scope and the look-alike negatives.

**Implement:** Branch `ty-rename-slots`.

**Review:** My self-checklist was that the scope matches what was agreed on the issue thread, that there are no cross-class renames, that quotes are preserved, and that the deliberately excluded cases, meaning subclass and superclass propagation and implicitly concatenated strings, are named in the PR body as follow-ups rather than silently omitted.

**Evaluate:** Rename tests for each accepted container shape, plus negatives for unrelated classes and non-attribute targets.

### Edge Cases Considered

Scope was deliberately a first cut, matching what was discussed on the issue, so it covers same class only with no subclass or superclass propagation, `__slots__` as a tuple, list, set or dict literal with dict keys only, and single-part string literals with no implicit concatenation. Saying so explicitly in the PR is what made the scope reviewable rather than looking like an oversight.

Beyond that, the interesting cases are all over-matching risks, including two unrelated classes each declaring a `"value"` slot, a parameter or local sharing a slot's name, and a slot declared on a different class in an enclosing scope. Review surfaced two more I had not covered, which were annotated slots and ellipsis-valued stub slots.

---

## Phase III: Build

### Implementation Progress

| Commit | Date | Author | Message |
|---|---|---|---|
| `1ad9272b4` | 2026-06-28 | me | [ty] Rename string slot in `__slots__` when renaming an attribute |
| `b7a8999ed` | 2026-07-02 | me | [ty] Restrict slot rename to instance attributes of the nearest enclosing class |
| `608b2908b` | 2026-07-05 | lerebear | [ty] Restrict annotated slots to assigned values |
| `3919d3923` | 2026-07-05 | lerebear | [ty] Recognize ellipsis-valued stub slots |
| `a788863d6` | 2026-07-05 | lerebear | [ty] Move slot rename imports to module scope |

**Files modified (+485 / −2):**

| File | Δ |
|---|---|
| `crates/ty_ide/src/references.rs` | +189 / −2 |
| `crates/ty_ide/src/rename.rs` | +296 |

Most of that volume is tests, since the rename harness is table-driven and each accepted or rejected shape is its own case.

### Challenges Faced

The hard part of a rename feature is not renaming the right thing but not renaming the look-alikes, and my first cut got that wrong in a way I did not catch. `lerebear`'s review found that the rename target was not restricted to an instance attribute, so renaming an unrelated parameter or local that happened to share a slot's name could rewrite the slot string, which silently corrupts a class while you think you are renaming a variable.

The fix was in `target_belongs_to_class`, where I find the nearest lexically enclosing class of the rename target and require it to be the class that declares the `__slots__`. That closed both the nested-class case, where an inner class renames an outer class's slot, and the not-an-attribute case with one condition rather than two special cases.

He also caught that annotated slots and ellipsis-valued stub slots such as `__slots__: tuple[str, ...] = ...`, which are common in `.pyi` files, were mishandled, so I restricted annotated slots to assigned values and added recognition of the ellipsis form.

### Testing

- **New rename tests** covering the accepted scope, meaning tuple, list, set and dict `__slots__` with dict keys only and values untouched, plus the same-class restriction where two unrelated classes declaring an identically named slot are not cross-renamed.
- **Negative tests added from review feedback**, covering a parameter sharing a slot's name, annotated slots and ellipsis-valued stub slots.
- **Patterns followed:** tests live in ty's existing table-driven rename harness in `rename.rs`, in the same form as the surrounding rename tests, rather than in a new test file or style.
- The monorepo's CI, which is the same strict Astral pipeline as ruff, was green at merge.

---

## Phase IV: Submit & Iterate

### Pull Request

**[astral-sh/ruff#26438](https://github.com/astral-sh/ruff/pull/26438)**, opened 2026-06-28 against `astral-sh/ruff:main`, cross-referencing `astral-sh/ty#3674`. The body opens with why a partial rename is worse than no rename, since it silently breaks the class, then the seam chosen and why, then the deliberate first-cut scope with the excluded cases named as follow-ups. I posted the PR link back on the ty issue the same day so the reporter and maintainers could find it:

> Opened a PR for this: astral-sh/ruff#26438. [...] I went ahead with the same-class + dict-keys-only scope from the discussion above. Happy to adjust if maintainers want it narrower or wider.

### Maintainer Feedback Log

| Date | From | Feedback | My response |
|---|---|---|---|
| 2026-06-09 | jonathandung (reporter, on the issue) | Endorsed limiting the rename to the class, and asked that a dict-shaped `__slots__` preserve values and rename only keys. | Folded into the scope before implementation, so dict handling renames keys only. |
| 2026-07-02 | lerebear | **Changes requested** with "It looks good in large part, but I've commented on a pair of bugs that should be addressed before this ships", covering (1) the target not being restricted to an instance attribute, so an unrelated parameter or local could rewrite a slot, and (2) annotated and ellipsis-valued slots being mishandled. | Pushed `b7a8999ed`, restricting the slot rename to instance attributes of the nearest lexically enclosing class, which closes both the nested-class and not-an-attribute cases, and replied on the thread with the reasoning. |
| 2026-07-05 | lerebear | Pushed three follow-up commits himself, restricting annotated slots to assigned values, recognizing ellipsis-valued stub slots, and moving the slot-rename imports to module scope. | Reviewed and confirmed. |
| 2026-07-06 | lerebear | **Approved** with "Thank you for the edits: the latest looks good. I made a few minor modifications in preparation for this to merge." | Merged as `f1da30d9`. |

### Learnings & Reflections

**Technical:** The references visitor was the right seam precisely because rename reuses it, so adding one narrowly-gated string-literal case gave slot renaming without a second traversal or a parallel code path. The deeper lesson is about the shape of the risk, because for a rename feature every bug is an over-match and the guard that matters is the one answering whether this string belongs to the thing you are renaming. Restricting to instance attributes of the nearest enclosing class turned out to answer several distinct over-match cases at once, which is a sign it was the right invariant rather than a patch.

**Process & collaboration:** Posting the implementation plan on the issue before coding and then waiting got the reporter's dict-keys refinement into the design for free, and meant the PR arrived matching what the thread had already agreed. Shipping a deliberately narrow first cut and saying so also worked, because naming subclass propagation and implicit string concatenation as known exclusions made the scope reviewable and left clean follow-ups, instead of leaving a reviewer to wonder whether the gaps were oversights.

**What I'd do differently:** I would have hunted the over-match cases adversarially before submitting, literally asking what else in a Python file is a string that looks like a slot, instead of relying on review to find them. Both of `lerebear`'s catches were reachable that way, since one was a parameter with the same name and the other a stub file's `= ...`. For any feature whose failure mode is touching something it should not, the negative tests should come first and be the ones I sweat.

### Resources Used

- Issue [ty#3674](https://github.com/astral-sh/ty/issues/3674), for the request, the Pyright playground link showing expected behavior, and the scope discussion.
- `crates/ty_ide/src/references.rs` and `crates/ty_ide/src/rename.rs`, the existing rename and references machinery.
