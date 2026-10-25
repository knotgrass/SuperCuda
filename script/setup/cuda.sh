#! /bin/bash
# Install the CUDA toolkit and PyTorch into the conda env.
# Run it from an activated env: `conda activate cxx && bash script/setup/cuda.sh`.
set -e

CONDAENV="cxx"    # edit conda env name
if [ "${CONDA_DEFAULT_ENV:-}" != "$CONDAENV" ]; then
    echo "Run 'conda activate $CONDAENV' first: pip and \$CONDA_PREFIX must point to that env." >&2
    exit 1
fi
# conda install -c conda-forge cudatoolkit-dev --force-reinstall --name "$CONDAENV" -y

# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#installing-previous-cuda-releases
# conda search cuda -c nvidia
# CUDA 13.4.3 is on pkgs/main (Anaconda defaults). The newest nvidia/label/* is cuda-13.3.1.
conda install cuda=13.4.3 --override-channels -c pkgs/main --name "$CONDAENV" -y

# pytorch 2.14.1: newest torch; its newest CUDA build is cu132 (there is no cu134 wheel).
# nvcc 13.4 vs torch 13.2 is a minor mismatch: torch.utils.cpp_extension only warns.
pip install "torch==2.14.1" torchvision --index-url https://download.pytorch.org/whl/cu132
python -c "import torch; print(torch.__version__, torch.version.cuda)"
pip install nvcc4jupyter -U --no-cache-dir --force-reinstall
# ninja for cpp_extension
pip install ninja
pip install cuda-tile cupy-cuda13x # https://docs.nvidia.com/cuda/cutile-python/quickstart.html#prerequisites
# pip install -U "cuda-python[all]" # https://nvidia.github.io/cuda-python/cuda-bindings/latest/install.html
# conda install -c conda-forge cuda-python

# The `cuda` metapackage above already contains these; the explicit list stays for reference.
conda install --override-channels -c pkgs/main cuda-nvcc cuda-cudart cuda-nvrtc -y
conda install --override-channels -c pkgs/main libcublas libcublas-dev libcufft libcufft-dev libcurand libcurand-dev libcusparse libcusparse-dev libcusolver libcusolver-dev libnpp libnpp-dev -y
conda install nvidia::cudnn cuda-version=13 -y
conda install --override-channels -c pkgs/main nccl -y

# JIT builds (torch.utils.cpp_extension) read these on `conda activate`.
# CUDA_HOME: use this env's nvcc, not /usr/local/cuda from ~/.zshrc.
# TORCH_CUDA_ARCH_LIST: build only for the local GPU (RTX 5060 Ti); use 8.6 on A10.
conda env config vars set CUDA_HOME=$CONDA_PREFIX TORCH_CUDA_ARCH_LIST="12.0"
echo "Run 'conda deactivate && conda activate $CONDAENV' to apply CUDA_HOME and TORCH_CUDA_ARCH_LIST."

# nsight compute ui
# https://chatgpt.com/share/69243300-842c-8012-942c-377b45b2572e
sudo apt update -y
sudo apt install libxcb-cursor0 libxkbcommon-x11-0 libxcb-xinerama0 -y
QT_PLUGIN_PATH="" QT_QPA_PLATFORM_PLUGIN_PATH="" QT_QPA_PLATFORM="xcb" ncu-ui
