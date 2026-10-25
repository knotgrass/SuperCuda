#!/usr/bin/env bash
# Write compile_commands.json at the repo root from the release build (for IntelliSense / clangd).
# Usage: script/compdb.sh     (run script/build.sh release first)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BUILD_DIR="$ROOT/build/release"
if [[ ! -f "$BUILD_DIR/build.ninja" ]]; then
    echo "error: $BUILD_DIR/build.ninja not found. Run script/build.sh release first." >&2
    exit 1
fi

# Only the compile rules from torch's build.ninja: the link step is not a translation unit.
(cd "$BUILD_DIR" && ninja -t compdb compile cuda_compile) >"$ROOT/compile_commands.json"
echo "wrote compile_commands.json:"
grep '"file"' "$ROOT/compile_commands.json"
