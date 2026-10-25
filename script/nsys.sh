#!/usr/bin/env bash
# Profile a whole program with Nsight Systems: timeline of kernels, copies, and NVTX ranges.
# Usage: script/nsys.sh <report_name> <command...>
#   script/nsys.sh bench_reduction python benchmarks/bench_reduction.py
# Output: reports/<report_name>.nsys-rep (open it with nsys-ui).
#
# Replaces profile_perfommance.sh, which used:
#   time bin/vector_add.out      -> wall time only
#   nvprof bin/vector_add.out    -> nvprof does not support sm_80+ and is not in CUDA 13
# Use this script for the timeline, and script/ncu.sh for the metrics of one kernel.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <report_name> <command...>" >&2
    exit 2
fi
NAME="$1"
shift

mkdir -p reports
exec nsys profile --force-overwrite=true -o "reports/$NAME" "$@"
