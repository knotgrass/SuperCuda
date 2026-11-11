"""Launch one matmul kernel a few times for a profiler (ncu, nsys). No timing, no reference.

Usage:
    python benchmarks/profile_matmul.py                           # naive, 1024 x 1024 x 1024, 5 launches
    python benchmarks/profile_matmul.py --kernel coalescing
    python benchmarks/profile_matmul.py 4096 4096 4096 --iters 8  # M N K
With ncu, the first argument of script/ncu.sh is the name of the CUDA kernel (naive_sgemm), not the op:
    script/ncu.sh naive_sgemm naive_1024 python benchmarks/profile_matmul.py --kernel naive
script/ncu.sh skips the first 3 matching launches and profiles the 4th, so keep --iters >= 4.
Run script/build.sh release first. Never profile a debug (-G) build.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "torch-ext"))
import matmul  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("size", nargs="*", type=int, metavar="N", help="M N K (default: 1024 1024 1024)")
    parser.add_argument("--kernel", default=next(iter(matmul.KERNELS)), choices=list(matmul.KERNELS),
                        help="kernel to launch (default: the first one)")
    parser.add_argument("--iters", type=int, default=5, help="number of launches (default: 5)")
    args = parser.parse_args()
    if len(args.size) not in (0, 3):
        parser.error("give M N K (three integers) or nothing")
    M, N, K = args.size or (1024, 1024, 1024)
    kernel = matmul.KERNELS[args.kernel]

    A = torch.randn(M, K, device="cuda")
    B = torch.randn(K, N, device="cuda")
    C = torch.randn(M, N, device="cuda")
    for _ in range(args.iters):
        kernel(A, B, C, 0.5, -0.1)
    torch.cuda.synchronize()
    print(f"launched matmul.{args.kernel} x{args.iters} at {M}x{N}x{K} on {torch.cuda.get_device_name()}")


if __name__ == "__main__":
    main()
