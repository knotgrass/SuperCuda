#! /bin/bash
# Install Nsight Systems (nsys, nsys-ui) into the conda env.
# Run it from an activated env: `conda activate cxx && bash script/setup/nsys.sh`.
set -e

CONDAENV="cxx"    # edit conda env name
if [ "${CONDA_DEFAULT_ENV:-}" != "$CONDAENV" ]; then
    echo "Run 'conda activate $CONDAENV' first: \$CONDA_PREFIX must point to that env." >&2
    exit 1
fi

# nsight-systems is on conda-forge only. 2026.5.2.323 is its newest build (the nsys of CUDA 13.5).
# The package has no cuda-version dependency, so it installs next to cuda=13.4.3 (cuda.sh).
# nsight-systems has nsys-ui and depends on nsight-systems-cli, which has nsys.
# NVIDIA has a newer version (2026.6.1) as a .deb file only:
#   https://developer.download.nvidia.com/devtools/repos/ubuntu2604/amd64/nsight-systems-2026.6.1_2026.6.1.251-1_amd64.deb
# Run cuda.sh first: it moves the env to conda-forge. A dry-run (2026-11-16) of this install on the
# env before cuda.sh (CUDA 13.0.2 from the nvidia channel) downgraded python 3.13.18 to 3.13.2.
if ! grep -qs "conda-forge" "$CONDA_PREFIX"/conda-meta/cuda-version-*.json; then
    echo "cuda-version in $CONDA_PREFIX is not from conda-forge: run script/setup/cuda.sh first." >&2
    exit 1
fi
conda install nsight-systems=2026.5.2.323 --override-channels -c conda-forge --name "$CONDAENV" -y
command -v nsys
nsys --version

# CUDA trace works at any value. CPU sampling (--sample=cpu) needs kernel.perf_event_paranoid <= 2.
PARANOID="$(cat /proc/sys/kernel/perf_event_paranoid)"
if [ "$PARANOID" -gt 2 ]; then
    echo "note: kernel.perf_event_paranoid=$PARANOID, so nsys cannot sample the CPU. To allow it:"
    echo "  echo 'kernel.perf_event_paranoid=2' | sudo tee /etc/sysctl.d/99-nsys.conf && sudo sysctl --system"
fi
