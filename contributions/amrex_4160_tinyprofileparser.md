# Contribution 1: document TinyProfileParser

**Contribution Number:** 1
**Student:** Suryansh Sijwali
**Issue:** https://github.com/AMReX-Codes/amrex/issues/4160
**Status:** Phase I Complete

---

## Why I Chose This Issue

AMReX is a block-structured AMR framework used heavily in HPC simulation codes
at DOE labs, and TinyProfiler is its built-in performance-measurement system —
the same class of perf-trace tool I'm working with on a separate research
project on cost-aware network forecasting (where measurement-design rigor is
load-bearing for the paper's claims). The `TinyProfileParser` script bridges
TinyProfiler's stdout dumps into Hatchet, the standard
performance-data-analysis tool from LLNL. The bridge exists in
`Tools/TinyProfileParser/profileparser.py` but is undocumented in the AMReX
Sphinx docs, so users have no way to discover it without reading the source.

I picked this as my first contribution because it lets me ship a useful patch
without first paying a heavy build-toolchain cost, while still being in a
domain (HPC profiling tooling) that's directly adjacent to my research
interests. It's also the only candidate I evaluated where, after a live
GitHub-state check, the issue had zero competing PRs and zero recent in-thread
activity — a clean, available scope.

---

## Understanding the Issue

### Problem Description

The AMReX repo ships a Python utility,
`Tools/TinyProfileParser/profileparser.py`, that parses the TinyProfiler stdout
output produced at the end of an AMReX simulation run and converts it into the
JSON schema that [Hatchet](https://github.com/hatchet/hatchet) consumes for
hierarchical performance-data analysis. The script has a usable docstring
header, but the AMReX Sphinx documentation site contains no mention of it, so
a user who wants Hatchet-compatible output has no path of discovery short of
grepping the `Tools/` directory.

### Expected Behavior

The AMReX profiling-tools docs should describe what `profileparser.py` is,
when to use it, how to invoke it, and what the resulting JSON is consumed by.
A user reading the profiling chapter should be able to go from "I have
TinyProfiler stdout" to "I have a Hatchet-compatible JSON file" without
leaving the docs site.

### Current Behavior

Searching `TinyProfiler` across the docs yields hits in
`AMReX_Profiling_Tools.rst`, `AMReX_Profiling_Tools_Chapter.rst`,
`External_Profiling_Tools.rst`, `GPU.rst`, and `RuntimeParameters.rst`, but
none mention the parser script. The script is reachable only by directly
browsing the `Tools/TinyProfileParser/` directory in the source tree.

### Affected Components

- `Docs/sphinx_documentation/source/AMReX_Profiling_Tools.rst` (most likely
  home for the new content, since it already documents TinyProfiler itself).
- Possibly a new file
  `Docs/sphinx_documentation/source/AMReX_TinyProfileParser.rst` plus an entry
  in `AMReX_Profiling_Tools_Chapter.rst`'s toctree, depending on maintainer
  preference (scope question in the claim comment).
- Reference: `Tools/TinyProfileParser/profileparser.py` itself (the
  documentation source-of-truth — header docstring already describes usage).

---

## Reproduction Process

### Environment Setup

No build toolchain is required to author the docs change itself — the diff is
Sphinx RST text. To verify rendered output locally:

- Python 3 with `sphinx` and the AMReX docs theme dependencies (the AMReX repo
  ships a `Docs/build_docs.sh` helper).
- Build the Sphinx site under `Docs/sphinx_documentation/` and open
  `build/html/AMReX_Profiling_Tools.html` to visually confirm the new
  section renders, cross-references resolve, and code-block formatting is
  correct.

### Steps to Reproduce

1. Open the rendered AMReX docs site (or
   `Docs/sphinx_documentation/source/AMReX_Profiling_Tools.rst` directly).
2. Search the page for `profileparser` or `Hatchet`.
3. Observed: zero matches. The script is undocumented.

### Reproduction Evidence

- **Commit showing reproduction:** N/A (this is a documentation-gap bug, not a
  runtime defect — "reproduction" is the absence of content).
- **Screenshots/logs:** To be added once the docs are built locally —
  side-by-side before/after of the AMReX_Profiling_Tools page.
- **My findings:** Documented in the "Understanding the Issue" section above.

---

## Solution Approach

### Analysis

The fix is purely additive: write a Sphinx RST section (or page) that
documents the parser script. The script's own header docstring already
describes the invocation pattern and the Hatchet downstream — I can lift
authoritative content from there rather than guessing.

### Proposed Solution

Add a "TinyProfileParser" section to `AMReX_Profiling_Tools.rst` (pending
maintainer scope confirmation — see claim comment). The section will cover:

1. **What it does** — parses TinyProfiler stdout into Hatchet-compatible JSON.
2. **When to use it** — workflow: run an AMReX sim with TinyProfiler enabled,
   capture stdout, run the parser, load the JSON in Hatchet for hierarchical
   analysis.
3. **Invocation** — `python profileparser.py path_to_stdout_file [write_dir]`,
   matching the script's own docstring.
4. **Output schema** — brief note on the JSON shape (a list of profile dicts)
   and the Hatchet link for downstream analysis.
5. **A worked example** — capture a small TinyProfiler stdout from one of the
   AMReX tutorial runs, run the parser, show the resulting JSON snippet.

### Implementation Plan

Using UMPIRE framework (adapted):

**Understand:** AMReX ships a TinyProfiler→Hatchet bridge script that is
discoverable only by reading source. The docs need a section pointing users
at it.

**Match:** The existing `AMReX_Profiling_Tools.rst` already documents
TinyProfiler usage. The new section follows the same RST conventions
(`.. _label:` anchors, `.. code-block:: bash` for command-line examples,
`.. note::` directives where appropriate). External profiling tools are
already pulled in via `External_Profiling_Tools.rst`, providing a structural
template for "external-consumer tool integrations."

**Plan:**
1. Post claim comment on issue #4160 confirming scope (subsection vs separate
   page).
2. Fork AMReX, create branch `docs/tiny-profile-parser`.
3. Read `Tools/TinyProfileParser/profileparser.py` end-to-end and confirm the
   invocation pattern and output JSON schema by running it on a small
   captured TinyProfiler stdout sample.
4. Draft the new RST section based on maintainer's preferred placement.
5. Build Sphinx docs locally; verify rendered output and cross-references.
6. Open PR referencing issue #4160 (mention only — no closing keyword unless
   the issue is explicitly a single-PR fix).

**Implement:** Branch link to be added after step 2.

**Review:** Self-review checklist before opening PR:
- [ ] Section follows existing RST conventions in
  `AMReX_Profiling_Tools.rst`.
- [ ] No broken cross-references (`make html` shows no warnings on touched
  files).
- [ ] Invocation example matches the script's actual behavior (verified by
  running on a sample).
- [ ] Hatchet link is the canonical one used elsewhere in AMReX docs.
- [ ] PR description references #4160 and summarizes the docs gap closed.

**Evaluate:** Local Sphinx build renders cleanly; the new section answers
"how do I get Hatchet-compatible output from TinyProfiler?" without leaving
the docs site.

---

## Testing Strategy

### Unit Tests

Not applicable — this is a docs change. No code is modified.

### Integration Tests

- [ ] Sphinx build (`make html` or `Docs/build_docs.sh`) completes without
  new warnings on the modified file.
- [ ] Rendered HTML displays the new section, with code blocks correctly
  formatted and the Hatchet link reachable.
- [ ] If a new page is added: it appears in the toctree of
  `AMReX_Profiling_Tools_Chapter.rst` and is reachable from the chapter
  index.

### Manual Testing

- [ ] Run `profileparser.py` on a small captured TinyProfiler stdout file
  to verify the documented invocation matches actual behavior.
- [ ] Open the rendered HTML in a browser and walk a first-time-reader path:
  "I have TinyProfiler output, what now?" → confirm the new section answers
  that.

---

## Implementation Notes

### Week 1 Progress

To be filled in as work proceeds.

### Code Changes

- **Files modified:** TBD (will list once branch exists).
- **Key commits:** TBD.
- **Approach decisions:** TBD.

---

## Pull Request

**PR Link:** Not yet opened.

**PR Description:** Draft below; will refine before submitting.

> Closes the docs gap noted in #4160. Adds a TinyProfileParser section to the
> AMReX profiling-tools documentation, describing the
> `Tools/TinyProfileParser/profileparser.py` utility, its invocation pattern,
> the JSON output schema it produces, and the Hatchet downstream consumer.
> Verified the documented invocation against the script's actual behavior on
> a captured TinyProfiler stdout sample, and confirmed the Sphinx build
> produces no new warnings on the touched files.

**Maintainer Feedback:** None yet.

**Status:** Phase I — issue identified, claim comment to be posted.

---

## Learnings & Reflections

To be filled in after submission and review cycle.

---

## Resources Used

- AMReX TinyProfileParser script:
  https://github.com/AMReX-Codes/amrex/blob/development/Tools/TinyProfileParser/profileparser.py
- Hatchet performance-analysis tool: https://github.com/hatchet/hatchet
- Existing AMReX profiling docs:
  - `Docs/sphinx_documentation/source/AMReX_Profiling_Tools.rst`
  - `Docs/sphinx_documentation/source/AMReX_Profiling_Tools_Chapter.rst`
  - `Docs/sphinx_documentation/source/External_Profiling_Tools.rst`
- Sphinx RST reference:
  https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html
