#!/bin/zsh

source "$(dirname "$0")/build_utils.sh"

install_dependencies() {
    python -m pip install -U pip setuptools wheel || die "Pip upgrade failed"
    python -m pip install pybind11 numpy pyyaml || die "Dependency install failed"
}

build_cpp_core() {
    # Use project's build script
    ./build.sh || die "C++ build failed"
}

build_python_extension() {
    # Build Python wheel
    python3 -m pip wheel . -w dist || die "Python wheel build failed"
    python3 -m pip install dist/ctranslate2-*.whl || die "Python extension install failed"
}

verify_artifacts() {
    # Check for Python extension in the project's python directory
    [[ -f python/ctranslate2/_ext*.so ]] || die "Python extension missing"
    
    # Check for core library in the build directory
    [[ -f "$BUILD_DIR/src/libctranslate2.dylib" ]] || die "Core library missing"
    
    echo "🎉 Build successful!"
    echo "Python extension location: $(find python/ctranslate2 -name '_ext*.so')"
    echo "Core library location: $BUILD_DIR/src/libctranslate2.dylib"
}
    [[ -f "$VENV_DIR/lib/python3.9/site-packages/ctranslate2/_ext"*.so ]] || die "Python extension missing"
    [[ -f "$BUILD_DIR/src/libctranslate2.dylib" ]] || die "Core library missing"
    
    echo "🎉 Build successful!"
    echo "Activate venv with: source $VENV_DIR/bin/activate"
}
