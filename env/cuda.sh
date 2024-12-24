#! /bin/bash

CONDAENV="cxx"    # edit conda env name
# conda install -c conda-forge cudatoolkit-dev --force-reinstall --name "$CONDAENV" -y

# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#installing-previous-cuda-releases
conda install cuda -c nvidia/label/cuda-12.4.1 --name "$CONDAENV" -y
# pytorch 2.5.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install nvcc4jupyter -U --no-cache-dir --force-reinstall
