"""matmul: the SGEMM kernels of the CUDA learning series, as Torch ops.

    import matmul
    out = matmul.naive(A, B, C, alpha, beta)      # alpha * A @ B + beta * C

Importing this package builds the extension (see _ops.py). Every kernel has the same signature:
A: (M, K), B: (K, N), C: (M, N); float32, contiguous, on the same CUDA device. It returns a new
(M, N) tensor and does not modify C.
"""

import torch

from ._ops import ops


def naive(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
          alpha: float = 1.0, beta: float = 0.0) -> torch.Tensor:
    """One thread per element of C. Each thread reads a row of A and a column of B from global memory."""
    return ops.naive(A, B, C, alpha, beta)


# name -> function, in lesson order. tests/ and benchmarks/ loop over this dict:
# a kernel that is added here is tested and benchmarked with no other change.
KERNELS = {
    "naive": naive,
}

__all__ = ["ops", "KERNELS", *KERNELS]
