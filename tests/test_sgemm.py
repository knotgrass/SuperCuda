"""supercuda.sgemm vs. PyTorch: alpha * (A @ B) + beta * C in float32, TF32 off."""

import pytest
import torch

import supercuda

torch.backends.cuda.matmul.allow_tf32 = False

# Sizes that are not multiples of the 32x32 block exercise the edge blocks; 1 and 0 the corner cases.
SIZES = [
    (1000, 1500, 777),
    (4096, 4096, 128),
    (32, 32, 32),
    (33, 65, 1),
    (1, 1, 1),
    (0, 5, 3),   # M == 0: empty output, no launch
    (5, 0, 3),   # N == 0
    (4, 6, 0),   # K == 0: out = beta * C
]


def make_inputs(M, N, K, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    A = torch.randn(M, K, device="cuda", generator=g)
    B = torch.randn(K, N, device="cuda", generator=g)
    C = torch.randn(M, N, device="cuda", generator=g)
    return A, B, C


@pytest.mark.parametrize("M,N,K", SIZES)
@pytest.mark.parametrize("alpha,beta", [(1.0, 0.0), (0.5, -0.1)])
def test_sgemm_matches_torch(M, N, K, alpha, beta):
    A, B, C = make_inputs(M, N, K)
    C_in = C.clone()

    out = supercuda.sgemm(A, B, C, alpha, beta)
    ref = alpha * (A @ B) + beta * C

    assert out.shape == (M, N)
    assert out.dtype == torch.float32
    assert out.device == A.device
    # The GPU sums K products in a different order than cuBLAS; 1e-3 is ~10x the expected
    # float32 rounding error for K ~ 1000 and far below any indexing bug.
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
    assert torch.equal(C, C_in), "sgemm must not modify C"


def test_sgemm_rejects_bad_inputs():
    A, B, C = make_inputs(8, 6, 4)

    with pytest.raises(RuntimeError, match="contiguous"):
        supercuda.sgemm(A.t().contiguous().t(), B, C)  # (8, 4) view with transposed strides
    with pytest.raises(RuntimeError, match="float32"):
        supercuda.sgemm(A.half(), B.half(), C.half())
    with pytest.raises(RuntimeError, match="B is"):
        supercuda.sgemm(A, torch.randn(5, 6, device="cuda"), C)
    with pytest.raises(RuntimeError, match="C must be"):
        supercuda.sgemm(A, B, torch.randn(8, 7, device="cuda"))
    with pytest.raises(RuntimeError, match="2-D"):
        supercuda.sgemm(A.unsqueeze(0), B, C)
    with pytest.raises((RuntimeError, NotImplementedError)):  # no CPU implementation is registered
        supercuda.sgemm(A.cpu(), B.cpu(), C.cpu())
