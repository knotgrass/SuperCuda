"""Time supercuda.sgemm against cuBLAS (torch.addmm, TF32 off), with the roofline lower bound per size.

Usage:
    python benchmarks/benchmark_sgemm.py                # default sizes
    python benchmarks/benchmark_sgemm.py 4096 4096 4096 # one size: M N K
Run script/build.sh release first. Never time a debug (-G) build.

Columns:
    naive / cuBLAS ms, GFLOP/s      median of triton.testing.do_bench (L2 flushed between runs)
    % peak                          GFLOP/s over the GPU's FP32 peak (PEAKS table below; 0 if unknown)
    AI min                          FLOP / minimum bytes = 2MNK / 4(MK + KN + 2MN): read A, B, C once, write C once.
                                    Above the ridge point (peak / bandwidth) the size is compute-bound in theory.
The real bytes a kernel moves come from ncu: script/roofline.py reports/<name>.ncu-rep.
"""

import sys
from pathlib import Path

import torch
import triton.testing

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "torch-ext"))
import supercuda  # noqa: E402

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


def bench(M, N, K, peaks, alpha=0.5, beta=-0.1):
    A = torch.randn(M, K, device="cuda")
    B = torch.randn(K, N, device="cuda")
    C = torch.randn(M, N, device="cuda")

    ms_naive = triton.testing.do_bench(lambda: supercuda.sgemm(A, B, C, alpha, beta), return_mode="median")
    ms_cublas = triton.testing.do_bench(lambda: torch.addmm(C, A, B, beta=beta, alpha=alpha), return_mode="median")

    flop = 2.0 * M * N * K
    bytes_min = 4.0 * (M * K + K * N + 2 * M * N)
    ai_min = flop / bytes_min
    gflops = lambda ms: flop / (ms * 1e-3) / 1e9  # noqa: E731
    pct = lambda ms: 100 * gflops(ms) / (peaks[0] * 1e3) if peaks else 0.0  # noqa: E731
    print(f"{M:5d} x {N:5d} x {K:5d} | naive {ms_naive:9.3f} ms {gflops(ms_naive):8.1f} GFLOP/s {pct(ms_naive):5.1f}%"
          f" | cuBLAS {ms_cublas:8.3f} ms {gflops(ms_cublas):8.1f} GFLOP/s {pct(ms_cublas):5.1f}%"
          f" | naive/cuBLAS {ms_naive / ms_cublas:6.1f}x | AI min {ai_min:6.0f}")


if __name__ == "__main__":
    sizes = [tuple(int(v) for v in sys.argv[1:4])] if len(sys.argv) == 4 else DEFAULT_SIZES
    name = torch.cuda.get_device_name()
    peaks = gpu_peaks(name)
    if peaks:
        tflops, gbs = peaks
        print(f"{name}: FP32 peak {tflops} TFLOP/s, DRAM {gbs:.0f} GB/s, ridge point {tflops * 1e3 / gbs:.1f} FLOP/B")
    else:
        print(f"{name}: not in PEAKS, % peak shows 0. Add it to benchmarks/benchmark_sgemm.py.")
    for M, N, K in sizes:
        bench(M, N, K, peaks)
