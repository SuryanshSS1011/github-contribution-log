# Isolate the MLX submodule build with ExternalProject

**Project:** [pytorch/executorch](https://github.com/pytorch/executorch) · **My fork:** https://github.com/SuryanshSS1011/executorch
**Issue:** [#20556](https://github.com/pytorch/executorch/issues/20556) · **Pull request:** [#20585](https://github.com/pytorch/executorch/pull/20585) · **Branch:** `fix/20556-mlx-build-isolation`
**Status:** **Merged** 2026-06-29 (merge commit `0cef6de29`). Issue closed.

---

## Phase I: Issue Selection

### Why I Chose This Issue

ExecuTorch is PyTorch's on-device inference runtime. I wanted a contribution in the ML-systems and build-infra space, and the MLX backend fit because I have an Apple Silicon machine to build and test it on, which for a CMake change is the whole game. My learning goal was dependency isolation in CMake, meaning what `add_subdirectory` really does to your target and option namespace, and what it costs to get out of it.

The issue was a refactor rather than a bug, filed by a maintainer (`metascroy`) with an unusually complete design that gave the exact ExternalProject CMake, the imported-target plumbing, three gotchas to watch for, and explicit acceptance criteria. A maintainer-authored issue with acceptance criteria written in is close to the ideal first contribution to a repo this size, because the design risk is zero and the work is real.

### Problem Summary

MLX was pulled into ExecuTorch's build with `add_subdirectory`, which drops MLX's entire CMake project into ExecuTorch's target and option namespace, so MLX's `FetchContent` of `nlohmann_json` collides with the copy ExecuTorch already provides. The existing workaround patched MLX's own `CMakeLists.txt` at configure time and was pinned to specific line numbers, so an MLX version bump could make the patch silently stop applying and bring the collision back with no clear signal. It matters because the build also left the pinned submodule dirty and leaked MLX's `MLX_BUILD_*` options into ExecuTorch's cache, so the failure mode is confusing rather than loud. I chose it because it is contained, verifiable end to end on hardware I own, and the maintainer had already specified what done means.

### Issue Vetting

Open, unassigned, with no in-flight PR touching `backends/mlx`. Before starting I confirmed the premise still held rather than trusting the issue text, and the fragile `mlx_json.patch` still existed and still dirtied the submodule during a build.

### Where It Lives

- `backends/mlx/CMakeLists.txt`, holding the `add_subdirectory` call, the `MLX_BUILD_*` options and the patch loop.
- `backends/mlx/patches/mlx_json.patch`, the line-number-pinned workaround to be deleted.
- `tools/cmake/executorch-config.cmake`, which already contains an imported `mlx` target and is therefore the pattern to match.
- Downstream consumers to leave untouched, which are the package config, the metallib copy helper in `Utils.cmake` and the wheel path in `setup.py`.

### Acceptance Criteria (from the issue)

1. MLX builds in its own CMake scope, so the json collision cannot occur.
2. `mlx_json.patch` and the patch loop are deleted.
3. The MLX submodule stays clean after a full build.
4. `libmlx.a` and `mlx.metallib` still land at the paths downstream consumers expect.

---

## Phase II: Reproduce & Plan

### Environment Setup

I used the backend's own build preset plus ExecuTorch's CI workflow files to see which jobs would exercise the change.

- macOS on Apple Silicon.
- **Challenge (Metal toolchain missing):** building the MLX backend needs Apple's Metal Toolchain, which is not installed with Xcode by default. I resolved it with `xcodebuild -downloadComponent MetalToolchain`.
- **Challenge (CLA gate):** the `meta-cla` bot blocked the PR immediately on open, since Meta requires a signed CLA before any PR can merge. I signed and confirmed within about 28 minutes of opening.
- **Challenge (label required for CI routing):** ExecuTorch wants a release-notes label, so I self-applied it with `@pytorchbot label "release notes: build"` rather than waiting for a maintainer.
- The build loop is `cmake --preset mlx-release`, which configures, builds and installs.

### Steps to Reproduce

1. Clone `pytorch/executorch` with submodules on an Apple Silicon Mac with the Metal Toolchain installed.
2. Run `cmake --preset mlx-release`.
3. While or after the build runs, inspect the pinned submodule:
   ```sh
   git -C backends/mlx/third-party/mlx status
   ```

### Expected vs. Actual

**Actual:** `CMakeLists.txt` shows as modified in the MLX submodule, because the build has applied `mlx_json.patch` in place. MLX's `MLX_BUILD_*` options are also present in ExecuTorch's CMake cache, since `add_subdirectory` shares the option namespace.

**Expected:** a build leaves the pinned submodule untouched and does not leak the dependency's options into the parent project's cache.

### Root Cause

`add_subdirectory` evaluates MLX's CMake project inside ExecuTorch's scope, so MLX's targets, options and `FetchContent` declarations all land in the same namespace as ExecuTorch's, which is why its `FetchContent` of `nlohmann_json` collides with ExecuTorch's own copy. The patch suppressed the symptom by editing MLX's `CMakeLists.txt` at configure time, but that fix is pinned to line numbers in a third-party file that ExecuTorch does not control, so it is one upstream refactor away from silently not applying. The cause is the shared scope rather than the json fetch.

### Plan (UMPIRE)

**Understand:** The dependency and the parent share a CMake scope, and every symptom, meaning the collision, the patch, the dirty submodule and the leaked options, follows from that.

**Match:** ExecuTorch already consumes MLX as an imported target in `tools/cmake/executorch-config.cmake`, so matching that existing shape rather than inventing a new one makes the change reviewable and keeps consumers consistent.

**Plan:**
1. Replace `add_subdirectory` with `ExternalProject_Add(mlx_external)` so MLX configures in its own scope.
2. Consume the result through an imported `mlx` target mirroring the one in `executorch-config.cmake`.
3. Keep the ExternalProject `BINARY_DIR` exactly where `add_subdirectory` put it, so `libmlx.a` and `mlx.metallib` land at unchanged paths.
4. Re-add the Metal, Foundation and QuartzCore frameworks on the imported target, which a static `libmlx.a` does not carry transitively.
5. Make `mlxdelegate` depend on `mlx_external` directly.
6. Swap `install(TARGETS mlx)` for `install(FILES ${_mlx_static_lib})`.
7. Point the metallib install and `MLX_METALLIB_PATH` at the ExternalProject output.
8. Delete `mlx_json.patch` and the patch loop.
9. Verify against all four acceptance criteria, including submodule cleanliness.

**Implement:** Branch `fix/20556-mlx-build-isolation`, one commit.

**Review:** My self-checklist was that no downstream consumer paths change, the imported target carries the frameworks, the build ordering is explicit, and the diff stays confined to `backends/mlx` plus the config file.

**Evaluate:** Run a full `mlx-release` build, check the artifact paths, check that the submodule `git status` is clean, compare the delegate binary size before and after, and build and run the tests with `-DEXECUTORCH_BUILD_TESTS=ON`.

### Edge Cases Considered

- **Build ordering:** `add_dependencies` on an imported target does not order the build, so `mlxdelegate` has to depend on `mlx_external` directly, otherwise the delegate can compile before MLX exists.
- **Transitive frameworks:** A static `libmlx.a` carries no link dependencies, so Metal, Foundation and QuartzCore must be re-added on the imported target or downstream links fail at the very end of a long build.
- **Install rules:** `install(TARGETS)` rejects an imported target, so the static lib has to be installed with `install(FILES ...)`.
- **Binary-size neutrality:** A build-system change should not change the shipped artifact, so I measured rather than assumed.

---

## Phase III: Build

### Implementation Progress

| Commit | Date | Author | Message |
|---|---|---|---|
| `351ee54ec` | 2026-06-28 | me | [MLX] Isolate submodule build with ExternalProject |
| `84d2a1d75` | 2026-06-29 | metascroy | nits (maintainer follow-up before merge) |

**Files modified:**

| File | Δ |
|---|---|
| `backends/mlx/CMakeLists.txt` | +85 / −84 |
| `tools/cmake/executorch-config.cmake` | +13 / −4 |
| `backends/mlx/patches/mlx_json.patch` | −29 (deleted) |

In detail, I replaced the `MLX_BUILD_*` options, the patch loop and `add_subdirectory` with `ExternalProject_Add(mlx_external)` plus an imported `mlx` target that re-adds the Metal, Foundation and QuartzCore frameworks, made `mlxdelegate` depend on `mlx_external` directly, changed `install(TARGETS mlx)` to `install(FILES ${_mlx_static_lib})`, and repointed the metallib install and `MLX_METALLIB_PATH` at the ExternalProject output.

### Challenges Faced

The genuine obstacle was that `ExternalProject` isolation breaks three things `add_subdirectory` gave you for free, and each one fails at a different and late point in a long build. Build ordering silently races because imported targets are not ordered by `add_dependencies`, linking fails at the very end because a static lib carries no transitive frameworks, and `install(TARGETS)` errors out only at install time. Each of these cost a full build cycle to discover. Fixing them meant reading the imported `mlx` target already in `executorch-config.cmake` and copying its framework list rather than deriving my own, since the repo had already solved the transitive-framework problem once.

The second decision that paid off was keeping `BINARY_DIR` where `add_subdirectory` had left it. That single constraint meant the package config, `Utils.cmake`'s metallib copy helper and `setup.py`'s wheel path all kept working untouched, which kept the diff to two files and gave the reviewer far less surface to check.

### Testing

CMake changes have no unit-test layer, so verification is the build itself, checked against the issue's acceptance criteria:

- `cmake --preset mlx-release` configures, builds and installs cleanly, and `libmlx.a`, `libmlxdelegate.a` and `mlx.metallib` all land in `cmake-out/lib/` at their previous paths.
- `git -C backends/mlx/third-party/mlx status` is clean after a full build, which is the core acceptance criterion and the one the old patch violated.
- With `-DEXECUTORCH_BUILD_TESTS=ON`, `op_test_runner`, `multi_thread_test_runner` and `mlx_mutable_state_test` all build and link against `mlx`, and `mlx_mutable_state_test` passes.
- **Before and after evidence:** `libmlxdelegate.a` is byte-identical at 3,893,456 bytes before and after, which confirms the change is purely about how MLX is pulled in.
- CI had one red job, `test-qnn-delegate-linux`, which is unrelated because the diff touches only the MLX backend and contains zero QNN files.

### Testing notes on process

ExecuTorch also ran an automated `@claude review this code` pass at the maintainer's request, which produced a checklist-style review of the ExternalProject wiring. Its nits were folded in by `metascroy` in the `84d2a1d75` commit before merge.

---

## Phase IV: Submit & Iterate

### Pull Request

**[pytorch/executorch#20585](https://github.com/pytorch/executorch/pull/20585)**, opened 2026-06-28 against `pytorch/executorch:main`, referencing the issue with a close keyword, labeled `release notes: build`, with the repo-required disclosure of AI assistance filled in honestly. The body opens with why the current patch-based approach is fragile, then the ExternalProject design, then the acceptance-criteria evidence covering a clean submodule, unchanged artifact paths and identical binary size.

### Maintainer Feedback Log

| Date | From | Feedback | My response |
|---|---|---|---|
| 2026-06-28 | meta-cla bot | CLA required before any merge. | Signed and confirmed within about 28 minutes. |
| 2026-06-29 | metascroy | "How did the `mlxdelegate.a` size change with this?" | Measured and answered with exact numbers, that it is unchanged at 3,893,456 bytes, which is expected since the delegate compiles from the same three sources (`MLXLoader.cpp`, `MLXBackend.cpp` and `mlx_mutable_state.cpp`) with the same flags and only how MLX is pulled in changed. |
| 2026-06-29 | metascroy | Is `executorch_target_copy_mlx_metallib` in `Utils.cmake` still needed? | Yes, because it reads `MLX_METALLIB_PATH`, which this PR preserves, to colocate the metallib next to binaries that statically link MLX. |
| 2026-06-29 | metascroy | "Looks great! Thanks for the contribution. I pushed an update to address some of the nits from Claude. If CI passes, I'll merge." | Confirmed the follow-up commit, and it merged the same day as `0cef6de29`. |

### Learnings & Reflections

**Technical:** There are three concrete CMake facts I did not know before and will not forget. Imported targets do not order the build, so you need an explicit dependency on the external target, static libraries carry no transitive link dependencies, so frameworks must be re-added by hand, and `install(TARGETS)` cannot install an imported target. More generally, `add_subdirectory` is not a neutral include, because it merges a third-party project into your namespace, and every symptom in this issue was downstream of that one choice.

**Process & collaboration:** Matching the repo's existing shape beat inventing a better one. There was already an imported `mlx` target in `executorch-config.cmake`, and copying its structure made the PR read as "this is what we already do, applied consistently" rather than "here is a new idea." The size question is the other lesson, because `metascroy` asked a factual question and the right answer was a measurement of 3,893,456 bytes on both sides plus the reason it should be unchanged, rather than a reassurance. That exchange took one round and ended in a merge.

**What I'd do differently:** I would handle the CLA and the release-notes label before opening the PR rather than after the bots asked, so the first thing a maintainer sees is a mergeable PR. I would also have included the binary-size measurement in the original PR body, since it was the first question asked and it was cheap to produce, and pre-empting it would have made the review a single approval instead of a round trip.

### Resources Used

- Issue #20556, for the maintainer's design and acceptance criteria.
- `tools/cmake/executorch-config.cmake`, for the existing imported `mlx` target pattern.
- CMake `ExternalProject` and imported-target documentation.
