import os
import platform
import subprocess
import sys

import pybind11
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

VERSION = "4.5.0"  # Fixed version number matching the installed library

include_dirs = [
    pybind11.get_include(),
    "/usr/local/include",  # System-installed CTranslate2 headers
]
library_dirs = ["/usr/local/lib"]  # System-installed CTranslate2 library

libraries = ["ctranslate2"]
extra_compile_args = []
extra_link_args = []

if platform.system() == "Darwin":
    extra_compile_args += [
        "-std=c++17",
        "-mmacosx-version-min=10.14",
        "-fvisibility=default",  # Make all symbols visible by default
        "-undefined", "dynamic_lookup",  # Allow undefined symbols to be looked up at runtime
    ]
    extra_link_args += [
        "-mmacosx-version-min=10.14",
        "-Wl,-rpath,/usr/local/lib",  # Add rpath to find the library
        "-Wl,-dead_strip_dylibs",  # Remove unused libraries
        "-Wl,-bind_at_load",  # Bind all symbols at load time
    ]
    if platform.machine() == "arm64":
        os.environ["ARCHFLAGS"] = "-arch arm64"

class CustomBuildExt(build_ext):
    """A custom build_ext command to add install_name_tool step."""
    def run(self):
        build_ext.run(self)
        if platform.system() == "Darwin":
            # Fix the library path in the extension
            ext_path = self.get_ext_fullpath(self.extensions[0].name)
            subprocess.check_call([
                "install_name_tool",
                "-change",
                "@rpath/libctranslate2.4.dylib",
                "/usr/local/lib/libctranslate2.4.dylib",
                ext_path
            ])

ctranslate2_module = Extension(
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
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language="c++",
)

setup(
    name="ctranslate2",
    version=VERSION,
    license="MIT",
    description="Fast inference engine for Transformer models",
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
    setup_requires=[
        "pybind11>=2.6.0",
        "setuptools>=65",
    ],
    install_requires=[
        "numpy",
        "pyyaml>=5.3,<7",
    ],
    packages=["ctranslate2"],
    ext_modules=[ctranslate2_module],
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
