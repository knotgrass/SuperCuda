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

# cpptools reads this file inside VS Code, where PATH is the base env, not cxx. Make each command
# independent of PATH and of nvcc:
#   1. Compiler names (x86_64-conda-linux-gnu-c++, the -ccbin of nvcc) become absolute paths.
#      cpptools asks that compiler for the libstdc++ directories and the built-in macros.
#   2. nvcc adds the include directories of the CUDA toolkit itself, so build.ninja does not list
#      them. In a conda env they are in targets/x86_64-linux/include, and cpptools does not find
#      them there ('cannot open source file "cuda_runtime.h"'). `nvcc --dryrun` prints the flags.
python - "$ROOT/compile_commands.json" <<'PY'
import json, os, re, shlex, shutil, subprocess, sys

path = sys.argv[1]
with open(path) as f:
    entries = json.load(f)


def toolkit_includes(nvcc: str, source: str) -> list[str]:
    """Return the include flags that nvcc adds to every compile."""
    dryrun = subprocess.run([nvcc, "--dryrun", "-c", source, "-o", os.devnull],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True).stdout
    flags = []
    for line in dryrun.splitlines():
        if line.startswith(("#$ INCLUDES=", "#$ SYSTEM_INCLUDES=")):
            flags += shlex.split(line.split("=", 1)[1])
    # nvcc prints <env>/bin/../targets/...: remove the "bin/..".
    return ["-I" + os.path.normpath(f[2:]) if f.startswith("-I") else
            os.path.normpath(f) if f.startswith("/") else f
            for f in flags]


def absolute(compiler: str) -> str:
    """Return the absolute path of a compiler that the current PATH finds (the cxx env)."""
    found = shutil.which(compiler)
    if found is None:
        sys.exit(f"error: {compiler} is not on PATH. Run 'conda activate cxx' first.")
    return found


for entry in entries:
    compiler, args = entry["command"].split(" ", 1)
    compiler = absolute(compiler)
    if os.path.basename(compiler) == "nvcc":
        args = re.sub(r"-ccbin (\S+)", lambda m: "-ccbin " + absolute(m.group(1)), args)
        flags = toolkit_includes(compiler, entry["file"])
        if not flags:
            sys.exit(f"error: `{compiler} --dryrun` printed no INCLUDES line")
        args = shlex.join(flags) + " " + args
    entry["command"] = compiler + " " + args

with open(path, "w") as f:
    json.dump(entries, f, indent=2)
    f.write("\n")
PY

echo "wrote compile_commands.json:"
grep '"file"' "$ROOT/compile_commands.json"
