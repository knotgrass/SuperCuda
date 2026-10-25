#! /bin/bash
# Install Nsight Compute (ncu, ncu-ui) into the conda env, and let a normal user read GPU counters.
# Run it from an activated env: `conda activate cxx && bash script/setup/ncu.sh`.
set -e

CONDAENV="cxx"    # edit conda env name
if [ "${CONDA_DEFAULT_ENV:-}" != "$CONDAENV" ]; then
    echo "Run 'conda activate $CONDAENV' first: \$CONDA_PREFIX must point to that env." >&2
    exit 1
fi

# nsight-compute 2026.3.1.2 is the ncu in cuda=13.4.3 (cuda.sh). It needs cuda-version 13.4.
conda install nsight-compute=2026.3.1.2 --override-channels -c pkgs/main --name "$CONDAENV" -y
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
