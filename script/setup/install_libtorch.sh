#! /bin/bash

CONDAENV="cxx"    # edit conda env name

#cpuonly
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

#cuda
# torch 2.14.1 is the newest; cu132 is its newest CUDA build. Keep this line in sync with cuda.sh.
pip install "torch==2.14.1" torchvision --index-url https://download.pytorch.org/whl/cu132
# https://pytorch.org/get-started/locally/

#! header and shared lib (with `conda activate $CONDAENV`)
# header file tại `${CONDA_PREFIX}/lib/python3.13/site-packages/torch/include`
# và              `${CONDA_PREFIX}/lib/python3.13/site-packages/torch/include/torch/csrc/api/include`
# lib tại         `${CONDA_PREFIX}/lib/python3.13/site-packages/torch/lib`

#note
#compile with g++ https://discuss.pytorch.org/t/solved-build-the-c-frontend-using-g-or-any-other-compiler-no-cmake/35156/2

# ninja for cpp_extension
pip install ninja
