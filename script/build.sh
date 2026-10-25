#!/usr/bin/env bash
# Build the supercuda Torch extension (JIT): import the package once.
# Usage: script/build.sh [release|debug]     (default: release)
# Output: build/<mode>/supercuda.so. Sources come from build.toml.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-release}"
case "$MODE" in
    release | debug) ;;
    *)
        echo "usage: $0 [release|debug]" >&2
        exit 2
        ;;
esac

if ! python -c "import torch" 2>/dev/null; then
    echo "error: python cannot import torch. Run 'conda activate cxx' first." >&2
    exit 1
fi
if [[ -n "${CUDA_HOME:-}" && "$CUDA_HOME" != "${CONDA_PREFIX:-}" ]]; then
    echo "warning: CUDA_HOME=$CUDA_HOME is not \$CONDA_PREFIX: torch builds with that nvcc (see script/setup/cuda.sh)." >&2
fi

SUPERCUDA_BUILD="$MODE" SUPERCUDA_VERBOSE=1 PYTHONPATH="$ROOT/torch-ext${PYTHONPATH:+:$PYTHONPATH}" \
    python -c "import supercuda; print('supercuda ($MODE) ready:', supercuda.ops)"
