#!/bin/bash
set -e

# Universal2 build script for CTranslate2 Python wheel
# Usage: bash build_universal2_wheel.sh

# Use your universal binary Python (replace 'ub python3' with your actual command if different)
export ARCHFLAGS='-arch x86_64 -arch arm64'
export CMAKE_OSX_ARCHITECTURES='arm64;x86_64'
# Install build tool if needed
uv pip install --upgrade pip build

# Clean previous builds
echo "Cleaning previous build artifacts..."
rm -rf build/ dist/ ctranslate2.egg-info/

# Build sdist and universal2 wheel
echo "Building universal2 wheel..."
uv run -m build --sdist --wheel

echo "\nBuild complete. Check dist/ for the universal2 wheel."
ls -lh dist/

# Optional: Check architectures
for f in ctranslate2/_ext.cpython-*-darwin.so ctranslate2/libctranslate2*.dylib; do
  if [ -f "$f" ]; then
    echo "\nChecking architectures for $f:"
    lipo -info "$f"
  fi
done
