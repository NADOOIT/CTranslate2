import sys

if sys.platform == "win32":
    import ctypes
    import glob
    import os

    import pkg_resources

    module_name = sys.modules[__name__].__name__
    package_dir = pkg_resources.resource_filename(module_name, "")

    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is not None:
        add_dll_directory(package_dir)

    for library in glob.glob(os.path.join(package_dir, "*.dll")):
        ctypes.CDLL(library)

import sys
import os
import glob
import ctypes
from pathlib import Path

if sys.platform == "darwin":
    # Ensure the .dylib is loaded from the same directory as the .so
    package_dir = Path(__file__).parent.absolute()
    for dylib_path in package_dir.glob("libctranslate2*.dylib"):
        try:
            ctypes.CDLL(str(dylib_path))
        except Exception as e:
            pass  # fallback: let import error happen if truly missing

try:
    from ctranslate2._ext import (
        AsyncGenerationResult,
        AsyncScoringResult,
        AsyncTranslationResult,
        DataType,
        Device,
        Encoder,
        EncoderForwardOutput,
        ExecutionStats,
        GenerationResult,
        GenerationStepResult,
        Generator,
        MpiInfo,
        ScoringResult,
        StorageView,
        TranslationResult,
        Translator,
        contains_model,
        get_cuda_device_count,
        get_supported_compute_types,
        list_supported_devices,
        set_random_seed,
        check_metal_directly,  # Added for debugging
    )
    from ctranslate2.extensions import register_extensions
    from ctranslate2.logging import get_log_level, set_log_level

    register_extensions()
    del register_extensions
except ImportError as e:
    # Allow using the Python package without the compiled extension.
    if "No module named" in str(e):
        pass
    else:
        raise

from ctranslate2 import converters, models, specs
from ctranslate2.version import __version__
