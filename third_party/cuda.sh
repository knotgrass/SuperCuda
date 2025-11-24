#! /bin/bash

CONDAENV="cxx"    # edit conda env name
# conda install -c conda-forge cudatoolkit-dev --force-reinstall --name "$CONDAENV" -y

# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#installing-previous-cuda-releases
# conda search cuda -c nvidia
conda install cuda -c nvidia/label/cuda-13.0.2 --name "$CONDAENV" -y

# pytorch 2.9.0
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install nvcc4jupyter -U --no-cache-dir --force-reinstall
# ninja for cpp_extension
pip install ninja
# pip install -U "cuda-python[all]" # https://nvidia.github.io/cuda-python/cuda-bindings/latest/install.html
# conda install -c conda-forge cuda-python

conda install -c nvidia/label/cuda-13.0.2 cuda-nvcc cuda-cudart cuda-nvrtc -y
conda install -c nvidia/label/cuda-13.0.2 libcublas libcublas-dev libcufft libcufft-dev libcurand libcurand-dev libcusparse libcusparse-dev libcusolver libcusolver-dev libnpp libnpp-dev -y
conda install nvidia::cudnn cuda-version=13 -y
conda install -c nvidia/label/cuda-13.0.2 nccl -y

# nsight compute ui
sudo apt update
sudo apt install libxcb-cursor0 libxkbcommon-x11-0 libxcb-xinerama0
QT_PLUGIN_PATH="" QT_QPA_PLATFORM_PLUGIN_PATH="" QT_QPA_PLATFORM="xcb" ncu-ui
