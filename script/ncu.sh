#!/usr/bin/env bash
# Profile one kernel launch with Nsight Compute (full metric set).
# Usage: script/ncu.sh <kernel_regex> <report_name> <command...>
#   script/ncu.sh naive_sgemm naive_1024 python benchmarks/profile_matmul.py --kernel naive
# Output: reports/<report_name>.ncu-repz (open it with ncu-ui). -f overwrites an old report.
#   ncu 2026.3 writes .ncu-repz; older versions write .ncu-rep.
# --launch-skip 3 skips the first 3 matching launches (warm-up); --launch-count 1 profiles one.
# Profile a release build (-lineinfo). A debug build (-G) turns off device optimizations.
#
# ERR_NVGPUCTRPERM: the driver gives GPU performance counters to admin users only
# (`grep RmProfilingAdminOnly /proc/driver/nvidia/params` shows 1). Fix once, with sudo and a reboot:
#   echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf
#   sudo update-initramfs -u && sudo reboot
# See https://developer.nvidia.com/ERR_NVGPUCTRPERM
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ $# -lt 3 ]]; then
    echo "usage: $0 <kernel_regex> <report_name> <command...>" >&2
    exit 2
fi
REGEX="$1"
NAME="$2"
shift 2

mkdir -p reports
# `--set full` does not store the FLOP counts and the bytes per memory level that script/roofline.py reads.
# That script prints the names of the metrics it needs: collect them too.
ROOFLINE_METRICS="$(python script/roofline.py --list-metrics)"
exec ncu --set full --metrics "$ROOFLINE_METRICS" -k "regex:$REGEX" --launch-skip 3 --launch-count 1 -f -o "reports/$NAME" "$@"
