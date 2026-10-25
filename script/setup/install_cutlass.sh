#! /bin/bash

# conda install cutlass -c conda-forge -y
set -e  # Exit on error

version="v4.2.1"
# repo root: this script lives in <root>/script/setup/
workspaceFolder="$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")"
cd "$workspaceFolder"

if [ ! -d "third_party/cutlass" ]; then
    echo "Initializing CUTLASS submodule..."
    git submodule add -f https://github.com/NVIDIA/cutlass.git third_party/cutlass
else
    echo "Updating CUTLASS submodule..."
    git submodule update --init --recursive
fi

# Configure CUTLASS version
cd third_party/cutlass
git fetch --all --tags
git checkout $version

# Clean previous build
rm -rf build

# Set compiler paths and flags (from the active conda env: `conda activate cxx`)
: "${CONDA_PREFIX:?Run 'conda activate cxx' first}"
export CUDACXX=$CONDA_PREFIX/bin/nvcc
export CC=$CONDA_PREFIX/bin/gcc
export CXX=$CONDA_PREFIX/bin/g++

# Create build directory and configure CMake
mkdir -p build && cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX=$workspaceFolder/libs/cutlass \
    -DCUTLASS_NVCC_ARCHS=120 \
    -DCUTLASS_ENABLE_TESTS=ON \
    -DCUTLASS_UNITY_BUILD_ENABLED=ON

# Build and install
echo "Building CUTLASS..."
make default_target -j$(nproc)
make install


# make cutlass_profiler -j12

# https://github.com/NVIDIA/cutlass/blob/main/media/docs/profiler.md
# ./tools/profiler/cutlass_profiler --kernels=sgemm --m=4352 --n=4096 --k=4096
# ./tools/profiler/cutlass_profiler --operation=conv2d --n=8 --h=224 --w=224 --c=128 --k=128 --r=3 --s=3
# ./tools/profiler/cutlass_profiler --kernels=s1688fprop --n=8 --h=224 --w=224 --c=128 --k=128 --r=3 --s=3 --pad_h=1 --pad_w=1

# Build and run CUTLASS Unit Tests
# make test_unit -j
# make test_unit_gemm_warp -j
