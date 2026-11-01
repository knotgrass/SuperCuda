#! /bin/bash
# Build the CUTLASS tools (cutlass_profiler, examples) from the third_party/cutlass submodule into libs/cutlass.
# libs/cutlass/include is the CUTLASS include path of the repo (cutlass/, cute/ and cutlass/util/ in one tree),
# so run this once after a clone.
# Steps: build (tests included) -> run the unit tests -> install. The build tree stays, so a rerun is incremental.
# Usage: script/setup/install_cutlass.sh      (conda activate cxx first; JOBS=4 if the build runs out of RAM)

# conda install cutlass -c conda-forge -y
set -euo pipefail

version="v4.8.1"
# repo root: this script lives in <root>/script/setup/
workspaceFolder="$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")"
cd "$workspaceFolder"

buildDir="$workspaceFolder/third_party/cutlass/build"
installDir="$workspaceFolder/libs/cutlass"
# 8 jobs: nvcc on the unity test sources takes 2-4 GB each; 16 jobs ran out of RAM on 29 GB.
jobs="${JOBS:-8}"

# Set compiler paths and flags (from the active conda env: `conda activate cxx`)
: "${CONDA_PREFIX:?Run 'conda activate cxx' first}"
export CUDACXX=$CONDA_PREFIX/bin/nvcc
export CC=$CONDA_PREFIX/bin/gcc
export CXX=$CONDA_PREFIX/bin/g++

if [ ! -d "third_party/cutlass" ]; then
    echo "Initializing CUTLASS submodule..."
    git submodule add -f https://github.com/NVIDIA/cutlass.git third_party/cutlass
else
    echo "Updating CUTLASS submodule..."
    git submodule update --init --recursive
fi

# Configure CUTLASS version. The local clone has only the tags it fetched before.
if ! git -C third_party/cutlass rev-parse -q --verify "refs/tags/$version" >/dev/null; then
    git -C third_party/cutlass fetch origin tag "$version" --no-tags
fi
git -C third_party/cutlass checkout -q "$version"

# Configure. Tests are built and run here, but not installed. An existing build tree is reused.
cmake -S third_party/cutlass -B "$buildDir" -G "Unix Makefiles" -Wno-dev \
    -DCMAKE_INSTALL_PREFIX="$installDir" \
    -DCUTLASS_NVCC_ARCHS=120 \
    -DCUTLASS_ENABLE_TESTS=ON \
    -DCUTLASS_INSTALL_TESTS=OFF \
    -DCUTLASS_UNITY_BUILD_ENABLED=ON

echo "Building CUTLASS (library, profiler, examples, unit tests)..."
cmake --build "$buildDir" -j "$jobs"

# One test at a time: the tests share one GPU. -k runs every test group, then reports all failures.
echo "Running CUTLASS unit tests..."
if ! cmake --build "$buildDir" --target test_unit -j 1 -- -k; then
    echo "error: CUTLASS unit tests failed. $buildDir stays for inspection; $installDir is unchanged." >&2
    exit 1
fi

# Install from scratch: an install over an old one keeps its stale files (old test/ was 757 MB).
echo "Installing CUTLASS into $installDir..."
rm -rf "$installDir"
cmake --install "$buildDir"

# CUTLASS links with -Wl,-rpath,'$$ORIGIN/../lib' (CMakeLists.txt:805) and the installed binary keeps the
# literal "$$ORIGIN", so cutlass_profiler cannot find libcutlass.so. Rewrite its RPATH.
cmake -DF="$installDir/bin/cutlass_profiler" \
    -DR="\$ORIGIN/../lib:$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib" \
    -P <(printf '%s\n' 'file(RPATH_SET FILE "${F}" NEW_RPATH "${R}")')

# The build tree (test binaries, object files) stays in $buildDir: the next run rebuilds only what changed.
echo "CUTLASS $version installed in $installDir."

# https://github.com/NVIDIA/cutlass/blob/main/media/docs/profiler.md
# libs/cutlass/bin/cutlass_profiler --kernels=sgemm --m=4352 --n=4096 --k=4096
# libs/cutlass/bin/cutlass_profiler --operation=conv2d --n=8 --h=224 --w=224 --c=128 --k=128 --r=3 --s=3
# libs/cutlass/bin/cutlass_profiler --kernels=s1688fprop --n=8 --h=224 --w=224 --c=128 --k=128 --r=3 --s=3 --pad_h=1 --pad_w=1
