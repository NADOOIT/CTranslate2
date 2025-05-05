# CTranslate2 Metal Development Guide

## Project Overview
CTranslate2 is a C++ library for efficient neural machine translation and language models, with Metal Performance Shaders (MPS) support for Apple Silicon.

## Build System Architecture

### Modular Scripts
- `build_mac.sh`: Main orchestration script
- `scripts/build_utils.sh`: Utility functions
- `scripts/python_check.sh`: Python environment validation
- `scripts/build_steps.sh`: Build and verification steps

## Prerequisites

### System Requirements
- macOS 14.0+ (Sonoma)
- Xcode 15+ with Command Line Tools
- Homebrew
- Python 3.9

### Required Tools
```sh
# Install dependencies
brew install python@3.9 cmake
xcode-select --install

# Optional: Install uv for Python package management
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Build Process

### Quick Start
```sh
# Clone the repository
git clone https://github.com/OpenNMT/CTranslate2.git
cd CTranslate2

# Run build script
./build_mac.sh

# Optional: Activate Python virtual environment
source python/testvenv/bin/activate
```

### Troubleshooting
- Ensure Xcode Command Line Tools are up to date
- Check Python version compatibility
- Verify Metal/MPS support on your Mac
- If build fails, check system logs and error messages

### Detailed Build Stages
1. **System Information Gathering**
   - Verify macOS version
   - Check build environment

2. **Python Validation**
   - Confirm Python 3.9 is available
   - Create virtual environment
   - Install dependencies

3. **C++ Core Build**
   - Configures CMake with Metal/MPS support
   - Builds third-party dependencies (GoogleTest, spdlog)
   - Compiles CTranslate2 core library
   - Generates shared libraries and CMake configuration files

4. **Dependency Management**
   - Uses `uv` for Python package management
   - Creates isolated virtual environments
   - Installs project-specific dependencies

5. **Artifact Verification**
   - Checks Python extension compatibility
   - Validates core library generation
   - Ensures build reproducibility
4. **Python Extension**
   - Build pybind11 bindings
   - Create installable package

## Troubleshooting

### Common Build Errors

#### Python Version Mismatch
- **Symptom:** Build fails with Python version error
- **Solution:** 
  ```sh
  # Verify Python version
  python3 --version
  
  # Use specific Python version
  /opt/homebrew/opt/python@3.9/bin/python3 -m venv venv
  ```

#### Metal Framework Issues
- **Symptom:** Compilation errors related to Metal
- **Solution:** 
  ```sh
  # Verify Xcode installation
  xcode-select -p
  
  # Reinstall Command Line Tools
  xcode-select --install
  ```

#### Dependency Installation
- **Symptom:** pip or build dependency failures
- **Solution:**
  ```sh
  # Upgrade pip and setuptools
  python3 -m pip install --upgrade pip setuptools
  
  # Install specific dependencies
  python3 -m pip install pybind11 numpy
  ```

## Debugging

### Verbose Build Logging
```sh
# Full build trace
./build_mac.sh --verbose 2>&1 | tee build.log

# Debug specific stage
env PYTHONVERBOSE=1 ./build_mac.sh
```

## Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run full test suite
5. Submit a pull request

## License
MIT License - See LICENSE file for details

## Automated Build Script (`build_mac.sh`)

### Purpose
This script automates the build process for CTranslate2 with Metal/MPS support on Apple Silicon, ensuring:
- Consistent environment setup
- Dependency management
- Reproducible builds
- Error checking at each stage

### Features
1. Toolchain verification
2. Clean environment setup
3. Build isolation
4. Artifact validation
5. Error recovery

### Prerequisites
```sh
# Install base dependencies
brew install python@3.9 cmake
```

### Usage
```sh
./build_mac.sh [--clean] [--verbose]
```

### Manual Steps (if needed)
```sh
# Force rebuild Metal shaders
./src/metal/compile_shaders.sh

# Debug Python bindings
python3 -m pybind11 --includes
```
