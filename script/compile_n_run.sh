#!/usr/bin/env bash
# Build one standalone program and run it. Output: csrc/bin/<name>.out
#
# A program is one file with main(), or one directory with main.cu + kernel files (one main() per directory):
#   script/compile_n_run.sh csrc/helloworld.cu      one file  -> csrc/bin/helloworld.out
#   script/compile_n_run.sh csrc/sgemm              directory -> csrc/bin/sgemm.out (every *.cu and *.cpp in it)
#   script/compile_n_run.sh csrc/sgemm/main.cu      same as the directory build: main.cu stands for its directory
#
# Usage: script/compile_n_run.sh <file.cu | dir> [program args...]
#   ARCH=sm_86 script/compile_n_run.sh csrc/sgemm                    (A10)
#   NVCC_FLAGS="-O3 -lineinfo" script/compile_n_run.sh csrc/sgemm    extra nvcc flags (default: none)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <file.cu | dir> [program args...]" >&2
    exit 2
fi
TARGET="$(realpath "$1")"
shift
cd "$ROOT"

# Sources and output name.
shopt -s nullglob
if [[ -d "$TARGET" ]]; then
    PROG_DIR="$TARGET"
elif [[ "$(basename "$TARGET")" == main.* ]]; then
    PROG_DIR="$(dirname "$TARGET")"
else
    PROG_DIR=""
fi
if [[ -n "$PROG_DIR" ]]; then
    SRCS=("$PROG_DIR"/*.cu "$PROG_DIR"/*.cpp)
    NAME="$(basename "$PROG_DIR")"
else
    SRCS=("$TARGET")
    NAME="$(basename "${TARGET%.*}")"
fi
if [[ ${#SRCS[@]} -eq 0 ]]; then
    echo "no .cu or .cpp file in $PROG_DIR" >&2
    exit 2
fi

OUT_DIR="$ROOT/csrc/bin"
mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/$NAME.out"

read -ra EXTRA_FLAGS <<< "${NVCC_FLAGS:-}"
echo "sources: ${SRCS[*]#"$ROOT"/}"
nvcc -arch="${ARCH:-sm_120}" -std=c++20 "${EXTRA_FLAGS[@]}" -I"$ROOT/include" -I"$ROOT/libs/cutlass/include" "${SRCS[@]}" -o "$OUT"
echo "compile successful: ${OUT#"$ROOT"/}"
printf "start run program\n\n"
"$OUT" "$@"
