import os
import platform
import subprocess
import sys
from pathlib import Path

import pybind11
from setuptools import Extension, setup, find_packages
from setuptools.command.build_ext import build_ext

VERSION = "4.5.1"  # Fixed version number matching the installed library

def build_cpp_lib():
    """Build and install the C++ library."""
    if platform.system() != "Darwin":
        raise RuntimeError("This package only supports macOS")
    
    # Get the root directory of the project
    root_dir = Path(__file__).parent.parent.absolute()
    
    # Run CMake configuration
    build_dir = root_dir / "build"
    build_dir.mkdir(exist_ok=True)
    
    cmake_args = [
        "-DCMAKE_BUILD_TYPE=Release",
        "-DWITH_METAL=ON",
        "-DWITH_MKL=OFF",
        "-DWITH_DNNL=OFF",
        "-DWITH_CUDA=OFF",
        "-DWITH_CUDNN=OFF",
        "-DBUILD_TESTS=OFF",
        "-DCMAKE_CXX_FLAGS=-std=c++17",
        "-DOpenMP_C_FLAGS=-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include",
        "-DOpenMP_CXX_FLAGS=-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include",
        "-DOpenMP_C_LIB_NAMES=omp",
        "-DOpenMP_CXX_LIB_NAMES=omp",
        "-DOpenMP_omp_LIBRARY=/opt/homebrew/opt/libomp/lib/libomp.dylib",
        f"-DCMAKE_INSTALL_PREFIX={sys.prefix}"
    ]
    
    subprocess.check_call(["cmake", "-S", str(root_dir), "-B", str(build_dir)] + cmake_args)
    
    # Build and install
    subprocess.check_call(["cmake", "--build", str(build_dir), "-j", str(os.cpu_count())])
    subprocess.check_call(["cmake", "--install", str(build_dir)])

import shutil
import glob

class CustomBuildExt(build_ext):
    """Custom build command that builds the C++ library first and copies the dylib into the package."""
    def run(self):
        build_cpp_lib()
        # Find and copy the libctranslate2*.dylib into ctranslate2/
        root_dir = Path(__file__).parent.parent.absolute()
        dylib_candidates = list(root_dir.glob('python/Agent/testenv/lib/libctranslate2*.dylib'))
        target_dir = Path(__file__).parent / 'ctranslate2'
        target_dir.mkdir(exist_ok=True)
        for dylib_path in dylib_candidates:
            shutil.copy2(dylib_path, target_dir)
        # Fix install_name of the .so to use @loader_path for the dylib
        if sys.platform == "darwin":
            try:
                so_files = list(target_dir.glob("_ext.cpython-*.so"))
                for so_path in so_files:
                    for dylib_path in target_dir.glob("libctranslate2*.dylib"):
                        import subprocess
                        subprocess.run([
                            "install_name_tool",
                            "-change",
                            f"@rpath/{dylib_path.name}",
                            f"@loader_path/{dylib_path.name}",
                            str(so_path)
                        ], check=True)
            except Exception as e:
                print(f"[WARNING] Failed to patch install_name for .so: {e}")
        super().run()

# Define the extension module
ext_module = Extension(
    "ctranslate2._ext",
    sources=[
        os.path.join("cpp", name)
        for name in [
            "module.cc",
            "encoder.cc",
            "execution_stats.cc",
            "generation_result.cc",
            "generator.cc",
            "logging.cc",
            "mpi.cc",
            "scoring_result.cc",
            "storage_view.cc",
            "translation_result.cc",
            "translator.cc",
            "wav2vec2.cc",
            "wav2vec2bert.cc",
            "whisper.cc",  # Added whisper.cc
        ]
    ],
    include_dirs=[
        pybind11.get_include(),
        f"{sys.prefix}/include",
    ],
    library_dirs=[f"{sys.prefix}/lib"],
    libraries=["ctranslate2"],
    extra_compile_args=["-std=c++17", "-mmacosx-version-min=10.14"],
    extra_link_args=[
        "-mmacosx-version-min=10.14",
        "-Wl,-rpath,@loader_path/../lib"
    ],
)

if platform.machine() == "arm64":
    os.environ["ARCHFLAGS"] = "-arch arm64"

def get_long_description():
    readme_path = Path(__file__).parent / "README.md"
    if not readme_path.exists():
        return ""
    with open(readme_path, encoding="utf-8") as f:
        return f.read()

setup(
    name="ctranslate2",
    version=VERSION,
    license="MIT",
    description="Fast inference engine for Transformer models",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="OpenNMT",
    author_email="guillaume.klein@systrangroup.com",
    url="https://github.com/OpenNMT/CTranslate2",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: GPU :: NVIDIA CUDA :: 11.0",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    project_urls={
        "Forum": "https://forum.opennmt.net",
        "Source": "https://github.com/OpenNMT/CTranslate2",
    },
    python_requires=">=3.7",
    packages=find_packages(),
    package_data={"ctranslate2": ["*.so", "*.dll", "*.dylib"]},
    include_package_data=True,
    setup_requires=[
        "pybind11>=2.6.0",
        "setuptools>=65",
    ],
    install_requires=[
        "numpy",
        "pyyaml>=5.3,<7",
    ],

    ext_modules=[ext_module],
    cmdclass={"build_ext": CustomBuildExt},
    entry_points={
        "console_scripts": [
            "ct2-fairseq-converter=ctranslate2.converters.fairseq:main",
            "ct2-marian-converter=ctranslate2.converters.marian:main",
            "ct2-openai-gpt2-converter=ctranslate2.converters.openai_gpt2:main",
            "ct2-opennmt-py-converter=ctranslate2.converters.opennmt_py:main",
            "ct2-opennmt-tf-converter=ctranslate2.converters.opennmt_tf:main",
            "ct2-opus-mt-converter=ctranslate2.converters.opus_mt:main",
            "ct2-transformers-converter=ctranslate2.converters.transformers:main",
        ],
    },
)
