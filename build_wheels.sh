#!/bin/bash
set -e

# Ensure we're in the project root
cd "$(dirname "$0")"

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install build dependencies
pip install build wheel setuptools

# Clean previous builds
rm -rf python/dist

# Build wheel
cd python
python3 -m build --wheel

# Show the built wheels
ls dist/

# Optional: Clean up virtual environment
deactivate
