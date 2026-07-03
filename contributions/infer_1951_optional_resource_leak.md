# Contribution 8: Model `java.util.Optional` to catch resource leaks through it

**Contribution Number:** 8
**Student:** Suryansh Sijwali
**Issue:** https://github.com/facebook/infer/issues/1951
**Pull Request:** https://github.com/facebook/infer/pull/2068
**Status:** **Merged** on 2026-07-03 by dulmarod (merge commit `f123c84`).

---

## Why I Chose This Issue

Infer is Meta's static analyzer, and Pulse is its memory-and-resource lifetime
checker. I wanted a contribution in the static-analysis / program-analysis space,
and this one was a clean fit: a maintainer (davidpichardie) had already confirmed
the false negative on the issue, invited a PR, and pointed at the exact file and a
sibling model to copy (`PulseModelsJava.ml`, the `Boolean` model). That combination
— confirmed bug, prescribed approach, worked example — is what makes an OSS PR
actually land, so it was a strong pick despite the repo being OCaml with a heavy
build. Before starting I re-verified the issue was still open, unassigned, and had
no in-flight PR, and that `java.util.Optional` was genuinely unmodeled on current
`main`.

## Understanding the Issue

Pulse tracks a resource obligation from the constructor of a resource (e.g.
`new FileInputStream(...)` attaches a `JavaResource` attribute) and reports
`PULSE_RESOURCE_LEAK` if that resource becomes unreachable without being closed.
When the resource is wrapped in a `java.util.Optional` — `Optional.of(new
FileInputStream("file.txt"))` — Pulse has no model for `Optional`, so it can't
reason about the wrapped resource's reachability through the optional, and the
leak goes unreported. The issue's repro:

```java
void foo(boolean b) throws IOException {
    Optional<FileInputStream> stream;
    if (b) {
        stream = Optional.of(new FileInputStream("file.txt")); // leak, not reported
    }
}
```

## Reproduction Process

Built Infer from source and ran the exact repro through a **stock** binary at the
same commit: `infer run --pulse-only -- javac Main.java` reported `No issues
found` — the false negative reproduced. After applying the model, the same repro
reported `Main.java:9: error: PULSE_RESOURCE_LEAK`. Same binary, same commit, only
the model differed, so the fix is causally responsible rather than incidentally
correlated.

## Solution Approach

Model `Optional` as a value box that forwards its wrapped value, mirroring the
existing `Boolean` model. `Optional.of` / `ofNullable` are static factories, so
they behave like `Boolean.value_of`: allocate a fresh optional and store the
argument into a backing field. `Optional.get` loads the value back. The wrapped
resource keeps the `JavaResource` obligation from its own constructor, so
`optional.get().close()` reaches and releases it, while an `Optional` whose
contents are never closed still reports the leak — reporting exactly when the
resource is genuinely leaked, without a false positive when it is closed.

## Implementation Notes

- `PulseModelsJava.ml`: new `Optional` module (`init` / `of_` / `get`) using a
  `__infer_model_backing_optional` field, written in the same `DSL.Syntax` style
  as `Boolean`. Three matcher rows gated on
  `PatternMatch.Java.implements "java.util.Optional"` for `of`, `ofNullable`,
  and `get`.
- `ResourceLeaks.java`: `optionalOfNotClosedBad`,
  `optionalOfNullableNotClosedBad` (leak cases), and `optionalOfClosedOk`
  (closed via `get().close()`, must stay clean).
- `issues.exp`: the two new expected `PULSE_RESOURCE_LEAK` reports.

Diff: 3 files, +53 lines.

## Testing Strategy

- `make -C infer/tests/codetoanalyze/java/pulse test` passes — the two `Bad`
  methods report `PULSE_RESOURCE_LEAK`, `optionalOfClosedOk` reports nothing, and
  no other test in the pulse Java corpus regresses.
- Before/after verification against a stock build at the same commit (see
  Reproduction).
- `ocamlformat` at the repo-pinned version reported the model file format-clean;
  the test file's import was placed in alphabetical order to match the file's
  convention.

## Review and Merge

Merged by the maintainer dulmarod on 2026-07-03 (`f123c84`), imported through
Meta's internal review flow.

## Learnings & Reflections

- Matching an existing model exactly (`Boolean`) made the change small and
  reviewable, and let me reuse the DSL idioms rather than inventing a shape.
- Modeling a wrapper for resource-lifetime purposes is really about keeping the
  wrapped obligation reachable: because the `JavaResource` attribute already
  lives on the resource's own allocation, forwarding it through `get` is enough
  — no separate release/delegation machinery was needed for the low-false-positive
  behavior.
- Building Infer off a managed dev environment is the real cost: the OCaml
  toolchain must match `infer.opam.locked` exactly (ppxlib in particular — a
  newer version breaks the ppx-generated parsetree), and the build wants a couple
  of system libraries (e.g. GMP for zarith) that have to be provided when you
  lack root.

## Resources Used

- Issue #1951 (confirmed bug + prescribed approach)
- `infer/src/pulse/PulseModelsJava.ml` — the `Boolean` model as the structural
  template
- `infer/tests/codetoanalyze/java/pulse/ResourceLeaks.java` and `issues.exp` —
  the Pulse Java test conventions
