#!/usr/bin/env bash
# Run a command under compute-sanitizer. (cuda-memcheck was removed in CUDA 12.)
# Usage: script/sanitizer.sh <tool> <command...>
#   script/sanitizer.sh memcheck python -m pytest tests/test_vector_add.py -q
#   script/sanitizer.sh racecheck csrc/bin/02_atomic.out
# Tools:
# • memcheck : memory errors (access violations, leaks, API errors)
# • synccheck: misuse of synchronization (invalid masks, conditions)
# • racecheck: data races (read-after-write, write-after-read hazards)
# • initcheck: evaluation of uninitialized values (global memory only)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <memcheck|synccheck|racecheck|initcheck> <command...>" >&2
    exit 2
fi
TOOL="$1"
shift

# The PyTorch caching allocator hides out-of-bounds access: it hands out slices of large
# cudaMalloc blocks, so a read past the end of a tensor lands in memory that is still allocated.
# With this variable, every tensor gets its own cudaMalloc and memcheck can see the error.
export PYTORCH_NO_CUDA_MEMORY_CACHING=1

exec compute-sanitizer --tool "$TOOL" "$@"
