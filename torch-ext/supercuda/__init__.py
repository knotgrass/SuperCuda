"""supercuda: CUDA learning kernels as Torch ops. Importing this package builds the extension (see _ops.py)."""

import torch

from ._ops import ops


def sgemm(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
          alpha: float = 1.0, beta: float = 0.0) -> torch.Tensor:
    """alpha * A @ B + beta * C with the naive SGEMM kernel (one thread per element of C).

    A: (M, K), B: (K, N), C: (M, N); float32, contiguous, on the same CUDA device.
    Returns a new (M, N) tensor. C is not modified.
    """
    return ops.sgemm(A, B, C, alpha, beta)


__all__ = ["ops", "sgemm"]
