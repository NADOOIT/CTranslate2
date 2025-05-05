#!/bin/zsh

# Logging and utility functions
log_info() { echo "[INFO] $*"; }
log_error() { echo "[ERROR] $*" >&2; }
die() { log_error "$*"; exit 1; }

# Parse command-line arguments
CLEAN_BUILD=false
export VERBOSE=1
# Add verbose compilation flags
export CXXFLAGS="$CXXFLAGS -v -fdiagnostics-show-template-tree -fdiagnostics-show-note-include-stack -fdiagnostics-show-template-tree -fno-elide-constructors -ftemplate-backtrace-limit=0 -Wall -Wextra -Wfatal-errors"
VERBOSE=true
set -x  # Enable verbose shell output
export VERBOSE=1
export CMAKE_VERBOSE_MAKEFILE=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            set -x
            shift
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        *)
            die "Unknown option: $1"
            ;;
    esac
done

# Fail on any error
set -e

# Check required tools
command -v cmake >/dev/null 2>&1 || die "CMake not found"
command -v python3 >/dev/null 2>&1 || die "Python 3 not found"
xcode-select -p >/dev/null 2>&1 || die "Xcode Command Line Tools not installed"

# Main build process
setup_third_party_deps() {
    log_info "Setting Up Third-Party Dependencies"
    
    # Get absolute project root path
    PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
    
    # Ensure third_party directory exists
    mkdir -p "$PROJECT_ROOT/third_party"
    
    # Download and extract GoogleTest
    log_info "Downloading GoogleTest"
    if [ ! -d "$PROJECT_ROOT/third_party/googletest" ]; then
        mkdir -p "$PROJECT_ROOT/third_party/googletest"
        # Download GoogleTest archive
        curl -L -o "$PROJECT_ROOT/third_party/googletest.tar.gz" "https://github.com/google/googletest/archive/refs/tags/v1.12.1.tar.gz"
        
        # Extract GoogleTest
        tar xzf "$PROJECT_ROOT/third_party/googletest.tar.gz" -C "$PROJECT_ROOT/third_party/googletest" --strip-components=1
        
        # Remove the archive
        rm "$PROJECT_ROOT/third_party/googletest.tar.gz"
        
        # Remove gmock and unnecessary files
        rm -rf "$PROJECT_ROOT/third_party/googletest/googlemock"
        rm -rf "$PROJECT_ROOT/third_party/googletest/googletest/test"
        rm -rf "$PROJECT_ROOT/third_party/googletest/googletest/samples"
    fi
    
    # Create a unified CMakeLists.txt for third_party
    log_info "Creating Unified Third-Party CMakeLists.txt"
    cat > "$PROJECT_ROOT/third_party/CMakeLists.txt" << 'EOL'
cmake_minimum_required(VERSION 3.10)
project(CTranslate2ThirdParty)

# Disable problematic GoogleTest features
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGTEST_DISABLE_PTHREADS=1")
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Minimal configuration to prevent build errors
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)

# Add GoogleTest as a direct subdirectory
add_subdirectory(googletest/googletest)

# Explicitly set up gtest and gtest_main targets
if(TARGET gtest)
  set_target_properties(gtest PROPERTIES EXCLUDE_FROM_ALL FALSE)
endif()
if(TARGET gtest_main)
  set_target_properties(gtest_main PROPERTIES EXCLUDE_FROM_ALL FALSE)
endif()

# Prevent building any gmock-related targets
set(UNWANTED_TARGETS 
  gmock gmock_main gmock-main gmock_test gmock_builder
  gtest_test
)
foreach(target ${UNWANTED_TARGETS})
  if(TARGET ${target})
    set_target_properties(${target} PROPERTIES EXCLUDE_FROM_ALL TRUE)
  endif()
endforeach()
EOL
    
    # Ensure tests directory structure
    mkdir -p "$PROJECT_ROOT/tests/metal/ops"
    
    # Create main tests CMakeLists.txt
    cat > "$PROJECT_ROOT/tests/CMakeLists.txt" << EOL
cmake_minimum_required(VERSION 3.10)
project(CTranslate2Tests)

# Add Third-Party Dependencies
add_subdirectory("$PROJECT_ROOT/third_party" third_party)

# Add your test sources here
EOL
    
    # Create metal/ops CMakeLists.txt
    cat > "$PROJECT_ROOT/tests/metal/ops/CMakeLists.txt" << EOL
cmake_minimum_required(VERSION 3.10)
project(CTranslate2MetalOpsTests)

# Add Third-Party Dependencies
add_subdirectory("$PROJECT_ROOT/third_party" third_party)

# Add your Metal Ops test sources here
EOL
    
    # Patch CMakeLists.txt to use absolute paths
    log_info "Patching CMakeLists.txt"
    find "$PROJECT_ROOT" -name CMakeLists.txt -print0 | xargs -0 sed -i '' "s|third_party/googletest|$PROJECT_ROOT/third_party/googletest|g"
    
    # Debug: List contents of third_party directory
    log_info "Third-party directory contents:"
    ls -l "$PROJECT_ROOT/third_party"
    
    # Debug: Check googletest directory
    if [ -d "$PROJECT_ROOT/third_party/googletest" ]; then
        log_info "GoogleTest directory present"
        ls -l "$PROJECT_ROOT/third_party/googletest"
    else
        log_error "GoogleTest directory missing"
    fi
}

main() {
    log_info "Starting CTranslate2 Build"
    
    # Validate Python executable
validate_python_executable() {
    local python_bin="$1"
    
    # Check if Python executable exists
    if [ ! -x "$python_bin" ]; then
        echo "[ERROR] Python executable not found: $python_bin"
        return 1
    fi
    
    # Check Python version
    local python_version
    python_version=$($python_bin -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    
    # Validate Python version
    if [[ "$python_version" != 3.9* ]] && [[ "$python_version" != 3.10* ]] && [[ "$python_version" != 3.11* ]] && [[ "$python_version" != 3.12* ]]; then
        echo "[ERROR] Unsupported Python version: $python_version. Requires Python 3.9-3.12"
        return 1
    fi
    
    # Check for required packages
    local missing_packages=()
    
    if ! $python_bin -c "import numpy" &> /dev/null; then
        missing_packages+=("numpy")
    fi
    
    if ! $python_bin -c "import pybind11" &> /dev/null; then
        missing_packages+=("pybind11")
    fi
    
    if [ ${#missing_packages[@]} -gt 0 ]; then
        echo "[INFO] Installing missing packages: ${missing_packages[*]}"
        $python_bin -m pip install "${missing_packages[@]}"
    fi
    
    echo "[INFO] Using Python $python_version from $python_bin"
    return 0
}

# Attempt to find a suitable Python executable
find_python_executable() {
    local python_candidates=(
        "python3.12"
        "python3.11"
        "python3.10"
        "python3.9"
        "python3"
        "python"
    )
    
    for candidate in "${python_candidates[@]}";
    do
        local python_bin
        python_bin=$(command -v "$candidate")
        
        if [ -n "$python_bin" ]; then
            echo "$python_bin"
            return 0
        fi
    done
    
    echo "[ERROR] No suitable Python executable found"
    return 1
}

# Find and set Python executable
PYTHON_BIN=$(find_python_executable)
if [ $? -ne 0 ]; then
    echo "[FATAL] Failed to find a suitable Python executable"
    exit 1
fi

# Ensure we use the found Python for all build steps
export PYTHON_BIN

# Install uv if not already available
if ! command -v uv &> /dev/null; then
    echo "[INFO] Installing uv..."
    $PYTHON_BIN -m pip install uv
fi

# Create virtual environment with uv
echo "🐍 Creating virtual environment..."
USING_PYTHON_VERSION=$($PYTHON_BIN -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
echo "Using CPython $USING_PYTHON_VERSION"

# Remove existing virtual environment if it exists
rm -rf python/testvenv

# Ensure uv is installed
if ! command -v uv &> /dev/null; then
    echo "[INFO] Installing uv..."
    $PYTHON_BIN -m pip install uv
fi

# Create new virtual environment using uv
uv venv python/testvenv
echo "Activate with: source python/testvenv/bin/activate"

# Set Python executable to the one in the virtual environment
VENV_PYTHON="python/testvenv/bin/python3"

# Bootstrap pip
$VENV_PYTHON -m ensurepip --upgrade

# Install core dependencies
$VENV_PYTHON -m pip install setuptools wheel

# Install test dependencies
$VENV_PYTHON -m pip install pytest numpy pybind11 pyyaml

# Verify installation
$VENV_PYTHON -m pip list

    # Clean build if requested
    if [ "$CLEAN_BUILD" = true ]; then
        log_info "Cleaning previous build"
        rm -rf build dist third_party
    fi
    
    # Initialize and update submodules
    log_info "Initializing Git Submodules"
    git submodule update --init --recursive || true
    
    # Set up third-party dependencies
    setup_third_party_deps
    
    # Install build dependencies
    log_info "Installing Build Dependencies"
    python3 -m pip install --upgrade pip setuptools wheel || die "Pip upgrade failed"
    python3 -m pip install cmake pybind11 numpy || die "Dependency install failed"
    
    # Prepare build directory
    BUILD_DIR="$PROJECT_ROOT/build"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    # Configure CMake
    log_info "Configuring CMake"
    # Set macOS deployment target to a newer version
    export MACOSX_DEPLOYMENT_TARGET=10.15

    # Detect SDK paths and compiler details
SDK_PATH="$(xcrun --show-sdk-path)"
SDK_VERSION="$(xcrun --show-sdk-version)"
CLANG_CXX_PATH="$(xcrun -f clang++)"
CLANG_C_PATH="$(xcrun -f clang)"
CLANG_VERSION="$($CLANG_CXX_PATH --version | head -n1)"

# Compiler and linker flags
# Detect standard library include path and compiler details
XCODE_TOOLCHAIN_PATH="$(xcrun -f clang++ | xargs dirname)/.."
STDLIB_INCLUDE_PATH="$XCODE_TOOLCHAIN_PATH/include/c++/v1"
COMPILER_BASE_PATH="$XCODE_TOOLCHAIN_PATH/usr/bin"

# Absolute paths to compilers
CLANG_C_PATH="/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang"
CLANG_CXX_PATH="/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++"

# Compiler flags
C_COMPILER_FLAGS="-isysroot $SDK_PATH -mmacosx-version-min=10.15 -arch arm64 -fPIC"
CXX_COMPILER_FLAGS="-isysroot $SDK_PATH -mmacosx-version-min=10.15 -arch arm64 -std=c++17 -stdlib=libc++ -Wno-deprecated-declarations -I$STDLIB_INCLUDE_PATH -fPIC"

# Debugging library paths
echo "Xcode Toolchain Path: $XCODE_TOOLCHAIN_PATH"
echo "Standard Library Include Path: $STDLIB_INCLUDE_PATH"
echo "Compiler Base Path: $COMPILER_BASE_PATH"

# Debugging compiler information
echo "SDK Path: $SDK_PATH"
echo "SDK Version: $SDK_VERSION"
echo "Clang Path: $CLANG_PATH"
echo "Clang Version: $CLANG_VERSION"
LINKER_FLAGS="-Wl,-syslibroot,$SDK_PATH -Wl,-undefined,dynamic_lookup -Wl,-dead_strip"

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DWITH_METAL=ON \
    -DCMAKE_C_FLAGS="$C_COMPILER_FLAGS" \
    -DCMAKE_CXX_FLAGS="$CXX_COMPILER_FLAGS" \
    -DCMAKE_SHARED_LINKER_FLAGS="$LINKER_FLAGS" \
    -DCMAKE_EXE_LINKER_FLAGS="$LINKER_FLAGS" \
    -DOPENMP_LIBRARIES="$(brew --prefix libomp)/lib/libomp.dylib" \
    -DOPENMP_INCLUDES="$(brew --prefix libomp)/include" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=10.15 \
    -DCMAKE_SYSTEM_NAME=Darwin \
    -DCMAKE_SYSTEM_PROCESSOR=arm64 \
    -DCMAKE_C_COMPILER="$CLANG_C_PATH" \
    -DCMAKE_CXX_COMPILER="$CLANG_CXX_PATH" \
    -DCMAKE_OSX_SYSROOT="$SDK_PATH" \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_CXX_STANDARD_REQUIRED=ON \
    -DCMAKE_CXX_EXTENSIONS=OFF \
    -DPYTHON_EXECUTABLE="$PYTHON_BIN" \
    -DCMAKE_PREFIX_PATH="/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain" \
    -DCMAKE_CXX_FLAGS_INIT="-stdlib=libc++" || die "CMake configuration failed"
    
    # Return to project root
    cd ..
    
    # Verify artifacts
    log_info "Verifying Build Artifacts"
    log_info "Python extension search path:"
    PYTHON_EXT_PATH=$(find . -name "_ext.cpython-*.so" -print | head -n 1)
    if [ -z "$PYTHON_EXT_PATH" ]; then
        die "Python extension not found"
    fi
    log_info "Found Python extension: $PYTHON_EXT_PATH"
    CORE_LIB_PATH=$(find . -name "libctranslate2.dylib" -print | head -n 1)
    if [ -z "$CORE_LIB_PATH" ]; then
        die "Core library not found"
    fi
    log_info "Found Core library: $CORE_LIB_PATH"
    
    log_info "🎉 Build Successful!"
}

# Run main function
main "$@"

# --------------------------
# Virtual environment setup
# --------------------------
echo "🐍 Creating virtual environment..."
uv venv python/testvenv
source python/testvenv/bin/activate
uv pip install numpy pytest
uv pip install -e .

# --------------------------
# Dependency installation
# --------------------------
echo "📦 Installing Python dependencies..."
uv pip install -U pip setuptools wheel || die "Pip upgrade failed"
uv pip install pybind11 numpy pyyaml || die "Dependency install failed"

# --------------------------
# C++ Core Build
# --------------------------
echo "🏭  Building C++ core..."
# Build directory already created in main function

cmake "$PROJECT_ROOT" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_SYSTEM_NAME=Darwin \
    -DCMAKE_SYSTEM_PROCESSOR=arm64 \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_INSTALL_PREFIX="$PROJECT_ROOT" \
    -DWITH_METAL=ON \
    -DWITH_TESTS=ON \
    -DWITH_MKL=OFF \
    -DWITH_CUDA=OFF \
    -DWITH_OPENMP=ON \
    -DCMAKE_C_COMPILER="$(xcrun -f clang)" \
    -DCMAKE_CXX_COMPILER="$(xcrun -f clang++)" \
    -DCMAKE_OSX_SYSROOT="$(xcrun --show-sdk-path)" \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_CXX_STANDARD_REQUIRED=ON \
    -DPYTHON_EXECUTABLE="$PYTHON_BIN" || die "CMake configuration failed"
make -j$(sysctl -n hw.ncpu) || die "C++ build failed"
make install || die "C++ install failed"
cd "$PROJECT_ROOT"

# --------------------------
# Python Extension Build
# --------------------------
echo "🐍 Building Python extension..."
cd python
PYTHONPATH=. MACOSX_DEPLOYMENT_TARGET=10.15 CFLAGS="$C_COMPILER_FLAGS -I/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/c++/v1 -D_LIBCPP_DISABLE_AVAILABILITY=1 -D_LIBCPP_DISABLE_VISIBILITY_ANNOTATIONS=1" CXXFLAGS="$CXX_COMPILER_FLAGS -I/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/c++/v1 -D_LIBCPP_DISABLE_AVAILABILITY=1 -D_LIBCPP_DISABLE_VISIBILITY_ANNOTATIONS=1" CC="/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang" CXX="/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++" ../python/testvenv/bin/python3 setup.py build_ext --inplace || {
    echo "Python build failed"
    echo "Python executable details:"
    ls -l "$PYTHON_BIN"
    echo "Setup script details:"
    ls -l setup.py
    die "Python extension build failed"
}

# --------------------------
# Validation
# --------------------------
# Main Build Process
main() {
    print_system_info
    validate_python "$PYTHON_BIN"
    setup_virtual_env
    install_dependencies
    build_cpp_core
    build_python_extension
    verify_artifacts
}

# Run main function
main "$@"
