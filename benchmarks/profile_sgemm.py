"""Launch supercuda.sgemm a few times for a profiler (ncu, nsys). No timing, no reference.

Usage:
    python benchmarks/profile_sgemm.py                 # 1024 x 1024 x 1024, 5 launches
    python benchmarks/profile_sgemm.py 4096 4096 4096  # M N K
    python benchmarks/profile_sgemm.py --iters 8
script/ncu.sh skips the first 3 matching launches and profiles the 4th, so keep --iters >= 4.
Run script/build.sh release first. Never profile a debug (-G) build.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "torch-ext"))
import supercuda  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("size", nargs="*", type=int, metavar="N", help="M N K (default: 1024 1024 1024)")
    parser.add_argument("--iters", type=int, default=5, help="number of launches (default: 5)")
    args = parser.parse_args()
    if len(args.size) not in (0, 3):
        parser.error("give M N K (three integers) or nothing")
    M, N, K = args.size or (1024, 1024, 1024)

    A = torch.randn(M, K, device="cuda")
    B = torch.randn(K, N, device="cuda")
    C = torch.randn(M, N, device="cuda")
    for _ in range(args.iters):
        supercuda.sgemm(A, B, C, 0.5, -0.1)
    torch.cuda.synchronize()
    print(f"launched naive_sgemm x{args.iters} at {M}x{N}x{K} on {torch.cuda.get_device_name()}")


if __name__ == "__main__":
    main()
