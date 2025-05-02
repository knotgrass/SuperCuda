#! /bin/bash

CONDAENV="cxx"    # edit conda env name
# conda install -c conda-forge cudatoolkit-dev --force-reinstall --name "$CONDAENV" -y

# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#installing-previous-cuda-releases
# conda search cuda -c nvidia
conda install cuda -c nvidia/label/cuda-12.8.1 --name "$CONDAENV" -y
# pytorch 2.7.0
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install nvcc4jupyter -U --no-cache-dir --force-reinstall
pip install -U "cuda-python[all]" # https://nvidia.github.io/cuda-python/cuda-bindings/latest/install.html
