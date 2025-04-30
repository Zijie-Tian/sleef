#!/bin/bash

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Run CMake and build
cmake ..
make -j$(nproc)

echo "Build complete. Run with:"
echo "./build/expf_f16_benchmark"
echo "or"
echo "make -C build run_benchmark" 