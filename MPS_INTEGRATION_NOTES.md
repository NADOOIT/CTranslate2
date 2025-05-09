# Notes on Integrating CTranslate2 with Metal (MPS) for faster-whisper

## Problem

When using this custom CTranslate2 build with `faster-whisper` and explicitly requesting the Metal backend via `device="mps"`:

```python
# In faster-whisper:
model = faster_whisper.WhisperModel(model_size, device="mps", compute_type="float16")
```

The following error occurs during model initialization:

```
Exception: unsupported device mps
```

This happens even though the Metal/MPS code has been compiled into the library.

## Root Cause

The core issue is that the **Python interface** of this CTranslate2 build does not correctly recognize the `"mps"` device string and/or does not properly trigger the initialization of the compiled Metal backend in the C++ core library when that string is passed.

While the Metal code exists (`src/metal/`), the bridge connecting the Python request (`device="mps"`) to the C++ Metal execution path is missing or incomplete.

## Required Changes in CTranslate2

To fix this and enable seamless integration with `faster-whisper` (or any other Python code requesting the MPS device), the following areas need investigation and likely modification:

1.  **Python Bindings (`python/ctranslate2_ext.cc` or similar C++ binding file):**
    *   **Device String Mapping:** The code that parses the `device` string argument passed from Python needs to be updated. It must recognize `"mps"` (and potentially `"metal"`) as a valid device identifier.
    *   **Backend Initialization:** When `"mps"` is detected, the binding code must call the appropriate C++ function within the CTranslate2 core library to initialize and select the Metal backend for computations. This involves ensuring the correct C++ `Device` enum value (e.g., `Device::kMPS` or `Device::kMETAL`, depending on its definition) is used.

2.  **CMake Configuration (`CMakeLists.txt`, `python/CMakeLists.txt`):**
    *   **Verify `WITH_METAL`:** Double-check that the `WITH_METAL` CMake option is correctly defined, enabled (`ON`), and propagated throughout the build process. Ensure all necessary Metal source files are compiled and linked **only** when `WITH_METAL` is `ON`.
    *   **Link Dependencies:** Confirm that the Python extension links correctly against the main CTranslate2 library that includes the Metal backend.

3.  **C++ Core Library (Less likely, but possible):**
    *   **Device Enum:** Ensure a `Device::kMPS` or `Device::kMETAL` enum value exists in the C++ core.
    *   **Backend Registration:** Verify that the Metal backend implementation is correctly registered within the CTranslate2 device framework if such a mechanism exists.

4.  **(Optional but Recommended) Add `list_supported_devices()`:**
    *   To improve compatibility with standard `faster-whisper` and potentially other libraries, consider adding the `list_supported_devices()` function to the Python bindings.
    *   This function should query the compiled C++ library (based on build flags like `WITH_METAL`) and return a list of supported device strings (e.g., `["cpu", "mps"]`). This would allow checks like the one originally in `faster-whisper/transcribe.py` to pass correctly.

## Next Steps

1.  Examine the Python binding code (`python/ctranslate2_ext.cc` is a likely candidate) to see how the `device` parameter is handled.
2.  Modify the bindings to recognize `"mps"` and trigger the C++ Metal backend.
3.  (Optional) Implement `list_supported_devices()`.
4.  Re-run the CMake configuration and build process for the CTranslate2 Python wheel.
5.  Reinstall the updated wheel in the `faster-whisper` environment and re-run the `test_transcription.py` script.
