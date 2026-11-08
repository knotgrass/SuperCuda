"""Dev (JIT) loader for the supercuda Torch extension.

kernel-builder generates a module with this name for Hub builds. This file stands in for it:
on import it builds build/<mode>/supercuda.so from the sources listed in build.toml with
torch.utils.cpp_extension.load (ninja + nvcc), loads the .so, and exposes torch.ops.supercuda
as `ops`.

Environment:
    SUPERCUDA_BUILD    release (default, -O3 -lineinfo) or debug (-O0 -g -G)
    SUPERCUDA_VERBOSE  1 prints the compiler commands
    CUDA_HOME          must point to the nvcc of the conda env (CLAUDE.md, script/setup/cuda.sh)
    TORCH_CUDA_ARCH_LIST  e.g. 12.0 (8.6 on the A10); otherwise torch uses the GPU it finds
"""

import os
import tomllib
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

NAME = "supercuda"
ROOT = Path(__file__).resolve().parents[2]  # <repo>/torch-ext/supercuda/_ops.py -> <repo>
_SOURCE_SUFFIXES = (".cu", ".cpp", ".cc", ".c")


def _read_build_toml():
    """Return (sources, include_dirs) from build.toml: the [torch] section plus every [kernel.*]."""
    cfg = tomllib.loads((ROOT / "build.toml").read_text())
    name = cfg["general"]["name"]
    if name != NAME:
        raise RuntimeError(f"build.toml names the extension {name!r}, this loader is for {NAME!r}")

    sections = [cfg["torch"], *cfg.get("kernel", {}).values()]
    sources, includes = [], []
    for section in sections:
        for src in section["src"]:
            if src.endswith(_SOURCE_SUFFIXES):  # headers are listed for kernel-builder, not compiled
                sources.append(str(ROOT / src))
        for inc in section.get("include", []):
            path = str(ROOT / inc)
            if path not in includes:
                includes.append(path)
    return sources, includes


def _build():
    mode = os.environ.get("SUPERCUDA_BUILD", "release")
    if mode not in ("release", "debug"):
        raise RuntimeError(f"SUPERCUDA_BUILD must be release or debug, got {mode!r}")
    verbose = os.environ.get("SUPERCUDA_VERBOSE", "0") == "1"

    sources, includes = _read_build_toml()
    includes.append(str(ROOT / "torch-ext"))  # registration_dev.h, torch_binding.h

    common = ["-std=c++20", "-DCUDA_KERNEL"]  # kernel-builder defines CUDA_KERNEL for the cuda backend
    if mode == "release":
        cflags = common + ["-O3"]
        cuda_cflags = common + ["-O3", "-lineinfo"]
    else:
        cflags = common + ["-O0", "-g"]
        cuda_cflags = common + ["-O0", "-g", "-G"]

    build_dir = ROOT / "build" / mode
    build_dir.mkdir(parents=True, exist_ok=True)
    load(
        name=NAME,  # -> build/<mode>/supercuda.so, TORCH_EXTENSION_NAME=supercuda, PyInit_supercuda
        sources=sources,
        extra_include_paths=includes,
        extra_cflags=cflags,
        extra_cuda_cflags=cuda_cflags,
        build_directory=str(build_dir),
        verbose=verbose,
    )
    return getattr(torch.ops, NAME)


ops = _build()


def add_op_namespace_prefix(op_name: str) -> str:
    """Prefix an op name with this extension's namespace, e.g. for torch.library.register_fake."""
    return f"{NAME}::{op_name}"
