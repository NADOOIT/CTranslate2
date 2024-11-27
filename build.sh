#!/bin/bash

# Clean up previous build
rm -rf build
mkdir build

# Configure CMake
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Debug \
  -DWITH_METAL=ON \
  -DWITH_MKL=OFF \
  -DWITH_DNNL=OFF \
  -DWITH_CUDA=OFF \
  -DWITH_CUDNN=OFF \
  -DBUILD_TESTS=ON \
  -DCMAKE_CXX_FLAGS="-std=c++17" \
  -DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" \
  -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" \
  -DOpenMP_C_LIB_NAMES="omp" \
  -DOpenMP_CXX_LIB_NAMES="omp" \
  -DOpenMP_omp_LIBRARY="/opt/homebrew/opt/libomp/lib/libomp.dylib"

# Build
cmake --build build -j$(sysctl -n hw.ncpu)

# Run tests
cd build && ctest --output-on-failure
