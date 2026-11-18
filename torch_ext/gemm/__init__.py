import torch
from ._ops import ops


def naive(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
          alpha: float, beta: float) -> torch.Tensor:
    return ops.naive(A, B, C, alpha, beta)

KERNELS = {
    "naive": naive,
}

__all__ = ["ops", "naive", "KERNELS"]
