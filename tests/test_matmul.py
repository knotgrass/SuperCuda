"""Tests for the matmul kernels: every kernel must return alpha * A @ B + beta * C.

Run:  python -m pytest tests -q              all tests
      python -m pytest tests -q -k naive     one kernel

Each pattern in this file is copied from a public test file:
  [relu]   huggingface/kernels            examples/kernels/relu/tests/test_relu.py
           random input -> kernel -> torch.testing.assert_close against PyTorch
  [vllm]   huggingface/kernels-community  activation/tests/kernels/test_activation.py   (from vLLM)
           lists of test values at the top, one parametrize decorator per list, a name selects the kernel
  [flash]  Dao-AILab/flash-attention      tests/test_flash_attn.py
           sizes that are not round numbers, a fixed seed, pytest.raises(match=...) for bad inputs
"""

import pytest
import torch

import matmul

torch.backends.cuda.matmul.allow_tf32 = False   # the reference A @ B must be true float32

# [vllm] Test values. Add a size or an (alpha, beta) pair here; a new kernel is added in matmul.KERNELS.
# The tests below do not change.
KERNELS = list(matmul.KERNELS)   # names: ["naive", "coalescing", ...]
SIZES = [                        # (M, N, K). The kernels work in 32 x 32 blocks.
    (1000, 1500, 777),           # [flash] no dimension is a multiple of 32: the edge blocks have unused threads
    (64, 96, 128),               # every dimension is a multiple of 32
    (33, 65, 1),                 # one element past a block edge; K = 1
    (1, 1, 1),                   # the smallest problem
    (0, 5, 3),                   # M = 0: empty output, nothing to launch
    (4, 6, 0),                   # K = 0: empty sum, out = beta * C
]
COEFFS = [(1.0, 0.0), (0.5, -0.1), (0.0, 1.0)]   # (alpha, beta): A @ B only; both terms; beta * C only

# A kernel and cuBLAS add the K products in a different order, so float32 results differ by rounding:
# about 1e-4 at K = 777. A real bug (wrong index, missing term) gives an error of 1 or more.
RTOL = 1e-3
ATOL = 1e-3


def random_inputs(M, N, K):
    torch.manual_seed(0)   # [flash] the same data on every run
    A = torch.randn(M, K, device="cuda")
    B = torch.randn(K, N, device="cuda")
    C = torch.randn(M, N, device="cuda")
    return A, B, C


# One test per combination. The ID reads like the arguments: test_matches_torch[naive-1000-1500-777-0.5--0.1]
@pytest.mark.parametrize("alpha, beta", COEFFS)
@pytest.mark.parametrize("M, N, K", SIZES)
@pytest.mark.parametrize("name", KERNELS)
def test_matches_torch(name, M, N, K, alpha, beta):
    """[relu] The result is close to PyTorch. A @ B runs cuBLAS: an independent implementation."""
    kernel = matmul.KERNELS[name]
    A, B, C = random_inputs(M, N, K)

    out = kernel(A, B, C, alpha, beta)

    expected = alpha * (A @ B) + beta * C
    torch.testing.assert_close(out, expected, rtol=RTOL, atol=ATOL)   # also checks shape, dtype, device


# The two tests below check run_sgemm (matmul_cuda/ops.cu). All kernels share it, so one kernel is enough.

def test_inputs_are_not_modified():
    """The op returns a new tensor: the kernel writes to a copy of C."""
    A, B, C = random_inputs(40, 50, 60)
    A_before, B_before, C_before = A.clone(), B.clone(), C.clone()

    matmul.naive(A, B, C, 0.5, -0.1)

    assert torch.equal(A, A_before)
    assert torch.equal(B, B_before)
    assert torch.equal(C, C_before)


def test_rejects_bad_inputs():
    """[flash] A bad input raises an error that names the problem. Without the check, the kernel reads out of bounds."""
    A, B, C = random_inputs(8, 6, 4)   # A: 8 x 4, B: 4 x 6, C: 8 x 6

    with pytest.raises(RuntimeError, match="float32"):
        matmul.naive(A.double(), B.double(), C.double())

    with pytest.raises(RuntimeError, match="contiguous"):
        matmul.naive(A[:, ::2], B, C)                            # every second column: not contiguous

    with pytest.raises(RuntimeError, match="B is"):
        matmul.naive(A, torch.randn(5, 6, device="cuda"), C)     # A has 4 columns, B has 5 rows

    with pytest.raises(RuntimeError, match="C must be"):
        matmul.naive(A, B, torch.randn(8, 7, device="cuda"))     # C must be 8 x 6
