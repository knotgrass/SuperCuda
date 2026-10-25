#! /bin/bash
# Install nvcc (the CUDA compiler) into the conda env.
# Run it from an activated env: `conda activate cxx && bash script/setup/nvcc.sh`.
set -e

CONDAENV="cxx"    # edit conda env name
if [ "${CONDA_DEFAULT_ENV:-}" != "$CONDAENV" ]; then
    echo "Run 'conda activate $CONDAENV' first: \$CONDA_PREFIX must point to that env." >&2
    exit 1
fi

# cuda-nvcc 13.4.92 is the nvcc in cuda=13.4.3 (cuda.sh). It also installs gcc_linux-64 and gxx_linux-64.
conda install cuda-nvcc=13.4.92 --override-channels -c pkgs/main --name "$CONDAENV" -y
command -v nvcc
nvcc --version

# torch.utils.cpp_extension takes nvcc from CUDA_HOME, not from PATH. cuda.sh sets CUDA_HOME=$CONDA_PREFIX.
if [ "${CUDA_HOME:-}" != "$CONDA_PREFIX" ]; then
    echo "warning: CUDA_HOME=${CUDA_HOME:-<unset>} is not \$CONDA_PREFIX: JIT builds use that nvcc (see script/setup/cuda.sh)." >&2
fi
