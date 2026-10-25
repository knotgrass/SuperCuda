#!/usr/bin/env bash
# Build one standalone program (a .cu or .cpp file with main()) and run it.
# Usage: script/compile_n_run.sh <file.cu> [program args...]
#   script/compile_n_run.sh csrc/helloworld.cu
#   ARCH=sm_86 script/compile_n_run.sh csrc/helloworld.cu     (A10)
# Output: csrc/bin/<name>.out
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <file.cu> [program args...]" >&2
    exit 2
fi
SRC="$(realpath "$1")"
shift
cd "$ROOT"

OUT_DIR="$ROOT/csrc/bin"
mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/$(basename "${SRC%.*}").out"

nvcc -arch="${ARCH:-sm_120}" -std=c++20 -I"$ROOT/common" -I"$ROOT/third_party/cutlass/include" "$SRC" -o "$OUT"
echo "compile successful: ${OUT#"$ROOT"/}"
printf "start run program\n\n"
"$OUT" "$@"
