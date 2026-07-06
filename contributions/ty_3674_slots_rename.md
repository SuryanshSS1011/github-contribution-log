# Contribution 9: Update `__slots__` string when renaming an attribute (ty)

**Contribution Number:** 9
**Student:** Suryansh Sijwali
**Issue:** https://github.com/astral-sh/ty/issues/3674
**Pull Request:** https://github.com/astral-sh/ruff/pull/26438
**Status:** **Merged** on 2026-07-06 by lerebear (merge commit `f1da30d`)
after one review round.

---

## Why I Chose This Issue

ty is Astral's new type checker and language server (developed in the `ruff`
monorepo, in Rust). I wanted a language-server/IDE feature in a fast-moving,
high-quality Rust codebase, and this one was well-scoped: an LSP "rename symbol"
gap that Pylance already handles, maintainer-green-lit on the issue thread, and
self-contained to the rename/references code.

## Understanding the Issue

ty's "rename symbol" refactors an attribute everywhere it is used, but it did not
touch the matching string in the class's `__slots__`. So renaming `self.value`
would update every `self.value` reference yet leave `__slots__ = ("value",)`
pointing at the old name — which then breaks the class, since `__slots__` is the
authoritative list of allowed instance attributes. The issue asks ty to rename
the slot string too, matching Pylance's behavior.

## Solution Approach

The references visitor already walks every node looking for matches to the rename
target. I hooked into its string-literal handling: when it hits a string, it
checks whether the string sits inside a `__slots__` assignment in a class body and
whether the attribute being renamed actually belongs to that class; if both hold,
it emits an edit for the inner content of the string (leaving the quotes in
place). The same-class check walks the ancestor scopes of the rename target's
definitions to find the class that owns the `__slots__`, so two unrelated classes
that both declare a `"value"` slot don't get cross-renamed.

Scope was intentionally a first cut, matching what was discussed on the issue:
same class only (no subclass/superclass propagation); `__slots__` as a tuple,
list, set, or dict literal (for a dict, only keys are renamed); single-part string
literals only.

## Implementation Notes

- `crates/ty_ide/src/references.rs`: recognize a string literal that is an element
  of a `__slots__` assignment in the enclosing class, gated on the rename target
  being an instance attribute of that same class.
- `crates/ty_ide/src/rename.rs`: emit the rename edit for the slot string's inner
  content.

Diff: 2 files, +485 / -2.

## Testing Strategy

Added rename tests covering the accepted scope: tuple/list/set/dict `__slots__`,
the same-class restriction (unrelated classes with an identically named slot are
untouched), and the negative cases surfaced in review (a parameter named the same
as a slot, annotated slots, and ellipsis-valued stub slots).

## Review and Merge

lerebear reviewed and requested changes, catching two real defects in the first
cut:

1. The target wasn't restricted to an *instance attribute*, so renaming an
   unrelated parameter or local named the same as a slot could wrongly rewrite the
   slot string. I restricted the slot rename to instance attributes of the nearest
   enclosing class.
2. Annotated and ellipsis-valued slots (e.g. in stubs) weren't handled correctly;
   I restricted annotated slots to assigned values and added recognition of
   ellipsis-valued stub slots.

I also moved the slot-rename imports to module scope per the review. lerebear
approved and merged on 2026-07-06.

## Learnings & Reflections

- The references visitor is the right seam for this: because rename reuses the
  same node walk, adding a narrowly-gated string-literal case gets slot renaming
  "for free" without a second traversal.
- The reviewer's two catches were both about *over*-matching — the hard part of a
  rename feature isn't renaming the right thing, it's not renaming look-alikes
  (a parameter, a slot on a different class, an annotation). Restricting to
  instance attributes of the nearest enclosing class is what makes it safe.
- Shipping a deliberately narrow first cut (same class, literal `__slots__`,
  single-part strings) and saying so in the PR made the scope reviewable and left
  clean follow-ups for subclass propagation.

## Resources Used

- Issue #3674 (the request and Pylance-parity discussion)
- `crates/ty_ide/src/references.rs` and `rename.rs` (existing rename/references
  machinery)
