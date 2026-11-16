#! /bin/bash
# Install the CUDA toolkit into the conda env. PyTorch: script/setup/install_libtorch.sh.
# Run it from an activated env: `conda activate cxx && bash script/setup/cuda.sh`.
set -e

CONDAENV="cxx"    # edit conda env name
if [ "${CONDA_DEFAULT_ENV:-}" != "$CONDAENV" ]; then
    echo "Run 'conda activate $CONDAENV' first: pip and \$CONDA_PREFIX must point to that env." >&2
    exit 1
fi
# conda install -c conda-forge cudatoolkit-dev --force-reinstall --name "$CONDAENV" -y

# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#installing-previous-cuda-releases
# conda search cuda --override-channels -c conda-forge
# CUDA 13.4.3 is the newest CUDA that has a torch build (cu134). The `cuda` metapackage contains:
#   nvcc                cuda-nvcc 13.4.92
#   compute-sanitizer   cuda-sanitizer-api 13.4.92
#   ncu, ncu-ui         nsight-compute 2026.3.1.2 (the newest ncu for cuda-version 13.4)
# Use conda-forge only: gcc 15.3.0 (gxx.sh) and nsight-systems (nsys.sh) are not on pkgs/main.
# A dry-run (2026-11-16) with pkgs/main before conda-forge downgraded python 3.13.18 to 3.13.0
# and binutils_linux-64 2.44 to 2.28.1. With conda-forge only, the dry-run has no downgrade.
conda install cuda=13.4.3 --override-channels -c conda-forge --name "$CONDAENV" -y
nvcc --version
compute-sanitizer --version
ncu --version

pip install nvcc4jupyter -U --no-cache-dir --force-reinstall
pip install cuda-tile cupy-cuda13x # https://docs.nvidia.com/cuda/cutile-python/quickstart.html#prerequisites
# pip install -U "cuda-python[all]" # https://nvidia.github.io/cuda-python/cuda-bindings/latest/install.html
# conda install -c conda-forge cuda-python

# The `cuda` metapackage above already contains these; the explicit list stays for reference.
conda install --override-channels -c conda-forge cuda-nvcc cuda-cudart cuda-nvrtc -y
conda install --override-channels -c conda-forge libcublas libcublas-dev libcufft libcufft-dev libcurand libcurand-dev libcusparse libcusparse-dev libcusolver libcusolver-dev libnpp libnpp-dev -y
conda install nvidia::cudnn cuda-version=13 -y
conda install --override-channels -c conda-forge nccl -y

# JIT builds (torch.utils.cpp_extension) read these on `conda activate`.
# CUDA_HOME: use this env's nvcc, not /usr/local/cuda from ~/.zshrc.
# TORCH_CUDA_ARCH_LIST: build only for the local GPU (RTX 5060 Ti); use 8.6 on A10.
conda env config vars set CUDA_HOME=$CONDA_PREFIX TORCH_CUDA_ARCH_LIST="12.0"
echo "Run 'conda deactivate && conda activate $CONDAENV' to apply CUDA_HOME and TORCH_CUDA_ARCH_LIST."

# Nsight Compute UI libraries and the GPU counter permission: script/setup/ncu.sh.
# Nsight Systems (nsys, nsys-ui): script/setup/nsys.sh.
