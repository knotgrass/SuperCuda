#! /bin/bash
# Install Nsight Systems (nsys, nsys-ui) from the NVIDIA apt repo, and link it into the conda env.
# Run it from an activated env: `conda activate cxx && bash script/setup/nsys.sh`.
set -e

CONDAENV="cxx"    # edit conda env name
if [ "${CONDA_DEFAULT_ENV:-}" != "$CONDAENV" ]; then
    echo "Run 'conda activate $CONDAENV' first: \$CONDA_PREFIX must point to that env." >&2
    exit 1
fi

# 2026.3.2 is the Nsight Systems of CUDA 13.4: apt cuda-nsight-systems-13-4 depends on it and adds
# only a wrapper in /usr/local/cuda-13.4/bin, so install the plain package.
# Not from conda: nsight-systems is on conda-forge only. A dry-run (2026-10-25) showed that it
# downgrades python 3.13.14 to 3.13.2 and moves about 25 pkgs/main libraries to conda-forge.
NSYS_VERSION="2026.3.2"
sudo apt update -y
if ! apt-cache show "nsight-systems-$NSYS_VERSION" >/dev/null 2>&1; then
    echo "apt has no nsight-systems-$NSYS_VERSION. Add the NVIDIA CUDA repo (cuda-keyring) first:" >&2
    echo "  https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#network-repo-installation-for-ubuntu" >&2
    exit 1
fi
sudo apt install "nsight-systems-$NSYS_VERSION" -y

# PATH has /usr/local/cuda/bin (~/.zshrc) before /usr/local/bin, so `nsys` is the CUDA 13.3 one (2026.1.3).
# The links in $CONDA_PREFIX/bin come first on PATH while the env is active. Remove them with `rm`.
NSYS_DIR="/opt/nvidia/nsight-systems/$NSYS_VERSION"
ln -sf "$NSYS_DIR/bin/nsys" "$CONDA_PREFIX/bin/nsys"
ln -sf "$NSYS_DIR/bin/nsys-ui" "$CONDA_PREFIX/bin/nsys-ui"
command -v nsys
nsys --version

# CUDA trace works at any value. CPU sampling (--sample=cpu) needs kernel.perf_event_paranoid <= 2.
PARANOID="$(cat /proc/sys/kernel/perf_event_paranoid)"
if [ "$PARANOID" -gt 2 ]; then
    echo "note: kernel.perf_event_paranoid=$PARANOID, so nsys cannot sample the CPU. To allow it:"
    echo "  echo 'kernel.perf_event_paranoid=2' | sudo tee /etc/sysctl.d/99-nsys.conf && sudo sysctl --system"
fi
