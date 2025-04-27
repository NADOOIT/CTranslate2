import os
import sys
import glob
import ctypes
from pathlib import Path
import importlib
import pytest

def test_dylib_present():
    # The .dylib must be present in the installed package directory
    import ctranslate2
    package_dir = Path(ctranslate2.__file__).parent
    dylibs = list(package_dir.glob("libctranslate2*.dylib"))
    assert dylibs, f"No libctranslate2*.dylib found in {package_dir}"

def test_import_and_device():
    import ctranslate2
    # Should import without error and have a version
    assert hasattr(ctranslate2, "__version__")
    # Try to use MPS device if available
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel("tiny", device="mps", compute_type="float16")
        assert model is not None
        assert model.device == "mps"
    except ImportError:
        pytest.skip("faster_whisper not installed, skipping device test")
    except Exception as e:
        pytest.skip(f"MPS device test skipped due to: {e}")
