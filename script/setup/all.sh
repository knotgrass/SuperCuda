#! /bin/bash
# Run all setup scripts in script/setup, in order. Each script also runs alone.
# Run it from an activated env: `conda activate cxx && bash script/setup/all.sh`.
# The scripts ask for the sudo password (apt) and for a reboot (ncu.sh).
set -e

CONDAENV="cxx"    # edit conda env name (also in each script)
if [ "${CONDA_DEFAULT_ENV:-}" != "$CONDAENV" ]; then
    echo "Run 'conda activate $CONDAENV' first: pip and \$CONDA_PREFIX must point to that env." >&2
    exit 1
fi

SETUP_DIR="$(dirname "$(realpath "$0")")"
# gxx.sh first: nvcc needs a host compiler. install_cutlass.sh last: its build needs nvcc and g++.
STEPS=(
    gxx.sh                 # gcc/g++ 13.4, gdb
    cuda.sh                # CUDA 13.4.3, torch, CUDA libraries, CUDA_HOME and TORCH_CUDA_ARCH_LIST
    nvcc.sh                # nvcc
    ncu.sh                 # Nsight Compute, GPU counter permission
    nsys.sh                # Nsight Systems
    compute_sanitizer.sh   # compute-sanitizer
    cudnn.sh               # cuDNN, cudnn-frontend
    install_libtorch.sh    # torch (C++ headers and libs)
    install_cutlass.sh     # CUTLASS into libs/cutlass
)
for step in "${STEPS[@]}"; do
    echo "==== script/setup/$step"
    bash "$SETUP_DIR/$step"
done

echo "Run 'conda deactivate && conda activate $CONDAENV' to apply CUDA_HOME and TORCH_CUDA_ARCH_LIST."
