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

# Make the entries usable outside the cxx env. IntelliSense (cpptools) and clangd fail on two points:
# 1. nvcc adds the CUDA headers from nvcc.profile ("-I.../targets/x86_64-linux/include", "-isystem .../cccl").
#    The tools do not read nvcc.profile, so they do not find <cuda_runtime.h>. Write these flags into each nvcc entry.
# 2. torch writes the host compilers as bare names (x86_64-conda-linux-gnu-c++, -ccbin x86_64-conda-linux-gnu-cc).
#    VS Code does not have the cxx env on PATH. Replace each bare name with its absolute path.
python3 - "$ROOT/compile_commands.json" <<'PY'
import json, os, re, shlex, shutil, subprocess, sys, tempfile

path = sys.argv[1]
with open(path) as f:
    entries = json.load(f)

def absolute(name):
    return name if os.sep in name else (shutil.which(name) or name)

def nvcc_profile_flags(nvcc):
    # nvcc --dryrun prints the INCLUDES and SYSTEM_INCLUDES lines of nvcc.profile on stderr.
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "empty.cu")
        open(src, "w").close()
        log = subprocess.run([nvcc, "--dryrun", "-c", src, "-o", os.path.join(tmp, "empty.o")],
                             capture_output=True, text=True).stderr
    flags = []
    for key in ("INCLUDES", "SYSTEM_INCLUDES"):
        m = re.search(rf"^#\$ {key}=(.*)$", log, re.M)
        if m:
            flags += shlex.split(m.group(1))
    out = []
    for i, flag in enumerate(flags):
        if flag.startswith("-I") and len(flag) > 2:
            flag = "-I" + os.path.realpath(flag[2:])
        elif i > 0 and flags[i - 1] == "-isystem":
            flag = os.path.realpath(flag)
        out.append(flag)
    return out

cache = {}
for e in entries:
    args = shlex.split(e["command"])
    args[0] = absolute(args[0])
    for i in range(1, len(args)):
        if args[i - 1] == "-ccbin":
            args[i] = absolute(args[i])
    if os.path.basename(args[0]) == "nvcc":
        if args[0] not in cache:
            cache[args[0]] = nvcc_profile_flags(args[0])
        args += cache[args[0]]
    e["command"] = shlex.join(args)

with open(path, "w") as f:
    json.dump(entries, f, indent=2)
    f.write("\n")
PY

echo "wrote compile_commands.json:"
grep '"file"' "$ROOT/compile_commands.json"
