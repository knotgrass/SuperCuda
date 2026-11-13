"""Time every kernel in matmul.KERNELS against cuBLAS (torch.addmm, TF32 off), with the ideal arithmetic intensity.

Usage:
    python benchmarks/benchmark_matmul.py                          # all kernels, default sizes
    python benchmarks/benchmark_matmul.py 4096 4096 4096           # one size: M N K
    python benchmarks/benchmark_matmul.py --kernels naive,coalescing
Run script/build.sh release first. Never time a debug (-G) build.

Columns:
    ms, GFLOP/s     median of triton.testing.do_bench (L2 flushed between runs).
                    FLOP = 2 M N K: the BLAS convention for GEMM (K multiplies + K adds per element of C).
                    It is the same for every kernel, so GFLOP/s compares kernels by time only.
                    Exact counts differ by a 1/K term: A @ B alone is 2MNK - MN (K - 1 adds per element);
                    alpha * A @ B + beta * C is 2MNK + 2MN. At K = 1024 that is 0.05 % or 0.1 %.
    % peak          GFLOP/s over the GPU's FP32 peak (PEAKS table below; 0 if the GPU is not listed)
    x cuBLAS        kernel time / cuBLAS time
    AI max          FLOP / minimum bytes = 2MNK / 4(MK + KN + 2MN): read A, B, C once, write C once.
                    No kernel can move fewer bytes, so this is the HIGHEST arithmetic intensity at this size.
                    If it is above the ridge point (peak / bandwidth), a kernel that reuses data from the
                    caches can be compute-bound. A kernel that re-reads data from DRAM has a lower intensity.
The bytes a kernel really moves, and the FP operations it really executes, come from ncu:
script/roofline.py reports/<name>.ncu-rep.
"""

import argparse
import sys
from pathlib import Path

import torch
import triton.testing

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "torch-ext"))
import matmul  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False

DEFAULT_SIZES = [(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096), (4096, 2048, 8192)]

# name substring -> (FP32 peak TFLOP/s, DRAM bandwidth GB/s), from NVIDIA spec sheets (no tensor cores).
PEAKS = {
    "RTX 5060 Ti": (23.7, 448.0),   # GB206: 36 SMs x 128 lanes x 2 x 2.57 GHz; GDDR7 128-bit 28 Gbps
    "A10": (31.2, 600.0),
}


def gpu_peaks(name):
    for key, val in PEAKS.items():
        if key in name:
            return val
    return None


def bench(M, N, K, kernels, peaks, alpha=0.5, beta=-0.1):
    A = torch.randn(M, K, device="cuda")
    B = torch.randn(K, N, device="cuda")
    C = torch.randn(M, N, device="cuda")

    flop = 2.0 * M * N * K                              # BLAS convention for GEMM (see the docstring)
    bytes_min = 4.0 * (M * K + K * N + 2 * M * N)       # read A, B, C once; write C once
    ai_max = flop / bytes_min                           # fewest bytes -> highest arithmetic intensity
    print(f"\n{M} x {N} x {K}   FLOP {flop:.3e}   AI max {ai_max:.0f} FLOP/B")

    def row(name, ms, ms_ref):
        gflops = flop / (ms * 1e-3) / 1e9
        pct = 100 * gflops / (peaks[0] * 1e3) if peaks else 0.0
        ratio = "" if ms_ref is None else f"{ms / ms_ref:8.1f} x cuBLAS"
        print(f"  {name:14s} {ms:10.3f} ms {gflops:10.1f} GFLOP/s {pct:6.1f} % peak {ratio}")

    ms_ref = triton.testing.do_bench(lambda: torch.addmm(C, A, B, beta=beta, alpha=alpha), return_mode="median")
    row("cuBLAS", ms_ref, None)
    for name in kernels:
        fn = matmul.KERNELS[name]
        ms = triton.testing.do_bench(lambda: fn(A, B, C, alpha, beta), return_mode="median")
        row(name, ms, ms_ref)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("size", nargs="*", type=int, metavar="N", help="M N K (default: a list of sizes)")
    parser.add_argument("--kernels", default=",".join(matmul.KERNELS),
                        help=f"comma-separated kernel names (default: all = {','.join(matmul.KERNELS)})")
    args = parser.parse_args()
    if len(args.size) not in (0, 3):
        parser.error("give M N K (three integers) or nothing")
    kernels = [k for k in args.kernels.split(",") if k]
    unknown = [k for k in kernels if k not in matmul.KERNELS]
    if unknown:
        parser.error(f"unknown kernel(s) {unknown}; available: {list(matmul.KERNELS)}")
    sizes = [tuple(args.size)] if args.size else DEFAULT_SIZES

    name = torch.cuda.get_device_name()
    peaks = gpu_peaks(name)
    if peaks:
        tflops, gbs = peaks
        print(f"{name}: FP32 peak {tflops} TFLOP/s, DRAM {gbs:.0f} GB/s, ridge point {tflops * 1e3 / gbs:.1f} FLOP/B")
    else:
        print(f"{name}: not in PEAKS, % peak shows 0. Add it to benchmarks/benchmark_matmul.py.")
    for M, N, K in sizes:
        bench(M, N, K, kernels, peaks)


if __name__ == "__main__":
    main()
