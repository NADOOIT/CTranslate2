# CTranslate2 Metal Integration Investigation Notes

## Objective

Successfully build CTranslate2 with Metal support and integrate it into the Faster-Whisper project such that `ctranslate2.list_supported_devices()` correctly reports `['cpu', 'mps']`.

## Current Status

Despite numerous attempts and confirmations, `ctranslate2.list_supported_devices()` consistently returns only `['cpu']` when called from the Python environment (`faster-whisper/venv`).

## Key Findings & Steps Taken

1.  **Initial Build:** Enabled `WITH_METAL=ON` in CMake.
2.  **Framework Linking:** Added `-framework Metal` and `-framework Foundation` to `CMAKE_CXX_FLAGS`, `CMAKE_OBJCXX_FLAGS`, and Python extension `extra_link_args`.
3.  **Build Method (`setup.py install`):** Initially used `python setup.py install`. Discovered this method was *not* rebuilding the underlying C++ library (`libctranslate2.dylib`) correctly, leading to outdated library files.
4.  **Build Method (`uv pip install .`):** Switched to `uv pip install . --no-cache-dir` from the `python/` directory. This correctly rebuilt and installed the C++ library and Python extension, confirmed by updated file timestamps.
5.  **Direct Metal Check (`ctypes`):** Created `metal_test.py` using `ctypes` to call `MTLCreateSystemDefaultDevice()` directly from Python. **This succeeded**, returning a valid device pointer. This confirms the Python environment *can* access Metal.
6.  **`@autoreleasepool` Test:** Temporarily removed `@autoreleasepool` from `metal::has_metal()` in `utils.mm`. This had no effect.
7.  **Standalone CMake Build:** Built the C++ library entirely outside Python packaging using direct CMake commands (correcting for `xcrun` path expansion). Manually copied the resulting `libctranslate2.dylib` into `site-packages`.
8.  **RPATH Issue:** Discovered the Python extension (`_ext.so`) had an incorrect RPATH (`@loader_path/../lib`) causing it to look for the dylib in the wrong place.
9.  **RPATH Fix:** Corrected the RPATH in `setup.py`'s `extra_link_args` to `@loader_path`.
10. **Rebuild with Correct RPATH:** Rebuilt and reinstalled using `uv pip install .` with the corrected RPATH.
11. **Debug Logging:** Added `printf` statements to `metal::has_metal()`. The output *was not visible* when running the Python check, even after confirming the library was rebuilt and RPATH was correct. This suggests the loaded library was *still* not the one with the printf statements.
12. **Cache Clearing:** Deleted `__pycache__` directories.
13. **Current State:** All known build/linking issues (RPATH, build methods) seem resolved, `ctypes` test works, but `list_supported_devices()` still fails, and C++ debug prints don't appear.

## Core Problem

The `metal::has_metal()` function within `libctranslate2.dylib` appears to return `false` when called via the Python extension (`_ext.so`), even though:
*   The library is built with Metal support.
*   The necessary frameworks are linked.
*   The RPATH is correct (`@loader_path`).
*   A direct `ctypes` call to `MTLCreateSystemDefaultDevice()` from the same Python environment succeeds.
*   The build/install process seems correct via `uv pip`.

There's a disconnect between the C++/Objective-C++ environment when run via the Python extension and the pure Python `ctypes` environment, or a persistent issue with loading the correct, freshly built dynamic library despite RPATH settings.

## Next Hypotheses & Investigation Steps

1.  **`dyld` Interference/Caching:** Is the dynamic linker (`dyld`) somehow caching an old version of the library or preventing the correct one from loading despite the RPATH? 
    *   **Action:** Use `dyld` environment variables (`DYLD_PRINT_LIBRARIES=1`) to trace library loading during Python execution.
    *   **Result:** `DYLD_PRINT_LIBRARIES` did *not* show `ctranslate2` libraries being loaded, possibly due to how Python extensions work.
    *   **Action:** Use `vmmap <PID>` on a running Python process after import.
    *   **Result:** `vmmap` *confirmed* that the correct `_ext.so` and `libctranslate2.dylib` from `site-packages` *are* being loaded. **Hypothesis 1 seems unlikely.**
2.  **Initialization Order within CTranslate2:** Does some other part of CTranslate2 initialize *before* `list_supported_devices` is called, potentially interfering with Metal setup specifically in the C++ context?
    *   **Action:** Examine `__init__.py` and `module.cc` for any C++ calls that happen automatically on import *before* our test call.
3.  **pybind11 & Objective-C++ Interaction:** Is there a subtle issue with how pybind11 wraps or manages the Objective-C++ code or memory related to the Metal framework?
    *   **Action:** Simplify the C++ check. Create a *minimal* C++ function (e.g., `check_metal_simple()`) in `module.cc` that *only* calls `ctranslate2::metal::has_metal()` and expose *only* that via pybind11. Build and test this minimal function.
    *   **Status:** Added direct binding `m.def("check_metal_directly", ...)` to `module.cc` and added symbol to `__init__.py` import list.
    *   **Result:** Still `AttributeError: module 'ctranslate2' has no attribute 'check_metal_directly'`. The binding is not appearing in the Python module despite code changes and rebuilds. **Hypothesis 3 seems incorrect/incomplete.**
4.  **Environment Variables during Build/Runtime:** Are there any environment variables set during the `uv pip install` build that differ from the runtime environment, affecting how Metal initializes?
    *   **Action:** Review build logs (if available from `uv`) and compare runtime environment (`env` command within the activated venv).
5.  **CMake Build Type/Flags:** Could `CMAKE_BUILD_TYPE=Release` be optimizing something away or changing behavior compared to a Debug build?
    *   **Status:** Changed build type to `Debug` in `setup.py`.
    *   **Result:** Build *failed* with `fatal error: 'ctranslate2/metal.h' file not found`. This indicates the Python extension build couldn't find headers from the core C++ library build, despite `include_dirs` seeming correct. Reverting to `Release` build.
    *   **Next Action:** Explicitly clean the CMake build directory (`python/build/lib.macosx-*/ctranslate2-build`) before reinstalling to rule out CMake caching issues.
