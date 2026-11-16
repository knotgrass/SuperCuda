#! /bin/bash
# Install PyTorch (Python package, C++ headers and libs) into the conda env.
# Run it from an activated env: `conda activate cxx && bash script/setup/install_libtorch.sh`.
set -e

CONDAENV="cxx"    # edit conda env name
if [ "${CONDA_DEFAULT_ENV:-}" != "$CONDAENV" ]; then
    echo "Run 'conda activate $CONDAENV' first: pip must point to that env." >&2
    exit 1
fi

#cpuonly
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

#cuda
# torch 2.15.0 is the newest torch (2026-10-28). Its cu134 build matches CUDA 13.4 (cuda.sh).
# pip selects torchvision 0.29.2+cu134 from the same index.
pip install "torch==2.15.0" torchvision --index-url https://download.pytorch.org/whl/cu134
python -c "import torch; print(torch.__version__, torch.version.cuda)"
# https://pytorch.org/get-started/locally/

# Headers and shared libs of torch (with `conda activate $CONDAENV`):
#   headers: `${CONDA_PREFIX}/lib/python3.13/site-packages/torch/include`
#            `${CONDA_PREFIX}/lib/python3.13/site-packages/torch/include/torch/csrc/api/include`
#   libs:    `${CONDA_PREFIX}/lib/python3.13/site-packages/torch/lib`

# Compile the C++ frontend with g++, without CMake:
# https://discuss.pytorch.org/t/solved-build-the-c-frontend-using-g-or-any-other-compiler-no-cmake/35156/2

# ninja for cpp_extension
pip install ninja
