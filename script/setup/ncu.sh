#! /bin/bash
# System setup for Nsight Compute: UI libraries, and GPU counter access for a normal user.
# ncu and ncu-ui come from the `cuda` metapackage of conda (script/setup/cuda.sh): there is no install here.
# Run it from an activated env: `conda activate cxx && bash script/setup/ncu.sh`.
set -e

CONDAENV="cxx"    # edit conda env name
if [ "${CONDA_DEFAULT_ENV:-}" != "$CONDAENV" ]; then
    echo "Run 'conda activate $CONDAENV' first: \$CONDA_PREFIX must point to that env." >&2
    exit 1
fi

if [ ! -x "$CONDA_PREFIX/bin/ncu" ]; then
    echo "No ncu in $CONDA_PREFIX/bin: run script/setup/cuda.sh first." >&2
    exit 1
fi
command -v ncu
ncu --version

# nsight compute ui
# https://chatgpt.com/share/69243300-842c-8012-942c-377b45b2572e
sudo apt update -y
sudo apt install libxcb-cursor0 libxkbcommon-x11-0 libxcb-xinerama0 -y
echo "Start the UI with: QT_PLUGIN_PATH=\"\" QT_QPA_PLATFORM_PLUGIN_PATH=\"\" QT_QPA_PLATFORM=\"xcb\" ncu-ui"

# ERR_NVGPUCTRPERM: the driver gives GPU performance counters to admin users only (see script/ncu.sh).
# https://developer.nvidia.com/ERR_NVGPUCTRPERM
if grep -q "RmProfilingAdminOnly: 1" /proc/driver/nvidia/params 2>/dev/null; then
    echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf
    sudo update-initramfs -u
    echo "Reboot to apply NVreg_RestrictProfilingToAdminUsers=0: then ncu runs without sudo."
fi
