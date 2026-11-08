"""Time supercuda.sgemm against cuBLAS (torch.addmm, TF32 off).

Usage:
    python benchmarks/benchmark_sgemm.py                # default sizes
    python benchmarks/benchmark_sgemm.py 4096 4096 4096 # one size: M N K
Run script/build.sh release first. Never time a debug (-G) build.
"""

import sys
from pathlib import Path

import torch
import triton.testing

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "torch-ext"))
import supercuda  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False

DEFAULT_SIZES = [(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096), (4096, 2048, 8192)]


def bench(M, N, K, alpha=0.5, beta=-0.1):
    A = torch.randn(M, K, device="cuda")
    B = torch.randn(K, N, device="cuda")
    C = torch.randn(M, N, device="cuda")

    ms_naive = triton.testing.do_bench(lambda: supercuda.sgemm(A, B, C, alpha, beta), return_mode="median")
    ms_cublas = triton.testing.do_bench(lambda: torch.addmm(C, A, B, beta=beta, alpha=alpha), return_mode="median")

    flop = 2.0 * M * N * K
    gflops = lambda ms: flop / (ms * 1e-3) / 1e9  # noqa: E731
    print(f"{M:5d} x {N:5d} x {K:5d} | naive {ms_naive:9.3f} ms {gflops(ms_naive):8.1f} GFLOP/s"
          f" | cuBLAS {ms_cublas:8.3f} ms {gflops(ms_cublas):8.1f} GFLOP/s"
          f" | naive / cuBLAS {ms_naive / ms_cublas:6.1f}x")


if __name__ == "__main__":
    sizes = [tuple(int(v) for v in sys.argv[1:4])] if len(sys.argv) == 4 else DEFAULT_SIZES
    print(torch.cuda.get_device_name())
    for M, N, K in sizes:
        bench(M, N, K)
