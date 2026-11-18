import os, tomllib
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

NAME = "gemm"
ROOT = Path(__file__).resolve().parents[2]
_SOURCE_SUFFIXES = (".cu", ".cpp")


def _read_build_toml():
    """Return (sources, include_dirs) from build.toml: [torch] plus every [kernel.*]."""
    cfg = tomllib.loads((ROOT / "build.toml").read_text())
    if cfg["general"]["name"] != NAME:
        raise RuntimeError(f"build.toml: name must be {NAME!r}")

    sources, includes = [], []
    for section in [cfg["torch"], *cfg["kernel"].values()]:
        for src in section["src"]:
            if src.endswith(_SOURCE_SUFFIXES):  # headers are not compiled
                sources.append(str(ROOT / src))
        for inc in section.get("include", []):
            includes.append(str(ROOT / inc))
    return sources, includes


def _build():
    mode = os.environ.get("SUPERCUDA_BUILD", "release")
    if mode not in ("release", "debug"):
        raise RuntimeError(f"SUPERCUDA_BUILD must be release or debug, got {mode!r}")

    verbose = os.environ.get("SUPERCUDA_VERBOSE", "0") == "1"
    sources, includes = _read_build_toml()

    common = ["-std=c++20", "-DCUDA_KERNEL"]
    if mode == "release":
        cflags = common + ["-O3"]
        cuda_cflags = common + ["-O3", "-lineinfo"]
    else:
        cflags = common + ["-O0", "-g"]
        cuda_cflags = common + ["-O0", "-g", "-G"]

    build_dir = ROOT / "build" / mode
    build_dir.mkdir(parents=True, exist_ok=True)
    load(
        name=NAME,
        sources=sources,
        extra_include_paths=includes,
        extra_cflags=cflags,
        extra_cuda_cflags=cuda_cflags,
        build_directory=str(build_dir),
        verbose=verbose,
        is_python_module=False,
    )
    return getattr(torch.ops, NAME)

ops = _build()
