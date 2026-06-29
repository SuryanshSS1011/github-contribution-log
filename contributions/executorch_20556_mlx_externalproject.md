# Contribution 6: Isolate the MLX submodule build with ExternalProject

**Contribution Number:** 6
**Student:** Suryansh Sijwali
**Issue:** https://github.com/pytorch/executorch/issues/20556
**Pull Request:** https://github.com/pytorch/executorch/pull/20585
**Status:** **Merged** on 2026-06-29 by metascroy (merge commit `0cef6de`)
after one review round.

---

## Why I Chose This Issue

ExecuTorch is PyTorch's on-device inference runtime. I wanted a contribution in
the ML-systems / build-infra space, and the MLX backend fit because I have an
Apple Silicon machine to build and test it on. The issue was a refactor rather
than a bug, filed by a maintainer (metascroy) with an unusually complete design:
the exact ExternalProject CMake, the imported-target plumbing, three gotchas to
watch, and explicit acceptance criteria. Before starting I confirmed the premise
still held — the fragile `mlx_json.patch` existed and dirtied the submodule
during the build.

## Understanding the Issue

MLX was pulled into the build with `add_subdirectory`, which drops MLX's whole
CMake project into ExecuTorch's target/option namespace. MLX fetches
`nlohmann_json`, which collides with the copy ExecuTorch already provides. The
existing workaround patched MLX's own `CMakeLists.txt` at configure time to
guard the fetch. That patch is pinned to specific line numbers, so an MLX
version bump touching that region makes it silently stop applying and the
collision returns with no clear signal. It also left the submodule dirty and
leaked MLX's `MLX_BUILD_*` options into ExecuTorch's cache.

## Reproduction Process

Built the MLX delegate with the existing setup (`cmake --preset mlx-release`)
and confirmed the patch was applied to the submodule mid-build:
`git -C backends/mlx/third-party/mlx status` showed `CMakeLists.txt` modified
(the json-guard patch). This is the fragility the issue describes — the build
mutates the pinned submodule in place.

## Solution Approach

Build MLX as an `ExternalProject` in its own CMake scope and consume it through
an imported `mlx` target. MLX then runs its `FetchContent` in a separate
namespace, so the collision can't happen and the patch is unnecessary. I kept
the ExternalProject `BINARY_DIR` where `add_subdirectory` had it, so `libmlx.a`
and `mlx.metallib` land at the same paths as before — which avoids touching the
downstream consumers (package config, the metallib copy helper in
`Utils.cmake`, and the wheel path in `setup.py`).

## Implementation Notes

- Replaced the `MLX_BUILD_*` options, patch loop, and `add_subdirectory` with
  `ExternalProject_Add(mlx_external)` plus an imported `mlx` target. The
  imported target re-adds the Metal/Foundation/QuartzCore frameworks a static
  `libmlx.a` doesn't carry (matching the imported `mlx` target already in
  `tools/cmake/executorch-config.cmake`). `mlxdelegate` depends on
  `mlx_external` directly, since `add_dependencies` on an imported target
  doesn't order the build.
- `install(TARGETS mlx)` → `install(FILES ${_mlx_static_lib})`; an imported
  target can't be installed with `install(TARGETS)`.
- Pointed the metallib install and `MLX_METALLIB_PATH` at the ExternalProject
  output.
- Deleted `backends/mlx/patches/mlx_json.patch` and the patch loop.

Diff: `backends/mlx/CMakeLists.txt` +80/-84, `patches/mlx_json.patch` deleted.

## Testing Strategy

On Apple Silicon (macOS, Metal Toolchain installed via
`xcodebuild -downloadComponent MetalToolchain`):

- `cmake --preset mlx-release` configures, builds, and installs. `libmlx.a`,
  `libmlxdelegate.a`, and `mlx.metallib` land in `cmake-out/lib/`, and the MLX
  submodule stays clean afterward (`git -C backends/mlx/third-party/mlx
  status`) — the core acceptance criterion.
- With `-DEXECUTORCH_BUILD_TESTS=ON`, `op_test_runner`,
  `multi_thread_test_runner`, and `mlx_mutable_state_test` build and link
  `mlx`; `mlx_mutable_state_test` passes.
- Confirmed `libmlxdelegate.a` is unchanged at 3,893,456 bytes before and after.

## Pull Request

Opened as PR #20585 against `main`, with a `release notes: build` label and a
disclosure of AI assistance in the description. The one red CI job
(`test-qnn-delegate-linux`) was unrelated — the diff touches only the MLX
backend and contains zero QNN files.

## Review and Merge

metascroy reviewed and asked two questions, both answered:
- Binary size: `libmlxdelegate.a` is unchanged (3,893,456 bytes), expected since
  the delegate compiles from the same sources with the same flags — only how MLX
  is pulled in changed.
- Whether `executorch_target_copy_mlx_metallib` in `Utils.cmake` is still
  needed: yes, it reads `MLX_METALLIB_PATH` (preserved by this PR) to colocate
  the metallib next to binaries that statically link MLX.

He pushed a follow-up `nits` commit and merged on 2026-06-29.

## Learnings & Reflections

- CMake `ExternalProject` for dependency isolation: imported targets don't order
  the build (need an explicit `add_dependencies` on the external target), static
  libs carry no transitive deps (re-add frameworks), and you can't
  `install(TARGETS)` an imported target.
- Matching a repo's existing pattern (the imported `mlx` target already in
  `executorch-config.cmake`) made the change easier to review than introducing a
  new shape.
- Keeping `BINARY_DIR` stable kept the diff small and left every downstream
  consumer untouched — less surface for the reviewer and for CI.

## Resources Used

- Issue #20556 (the design and acceptance criteria)
- `tools/cmake/executorch-config.cmake` (existing imported `mlx` target pattern)
- CMake `ExternalProject` and imported-target docs
