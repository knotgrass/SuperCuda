import pytest, torch, gemm


def test_gemm():
    M = 3
    N = 4
    K = 5
    A = torch.rand((M, K), dtype=torch.float32, device='cuda:0')
    B = torch.rand((K, N), dtype=torch.float32, device='cuda:0')
    C = torch.rand((M, N), dtype=torch.float32, device='cuda:0')

    alpha = 1.
    beta = 0.
    out = gemm.naive(A, B, C, alpha, beta)
    torch.testing.assert_close(out, alpha * (A@B) + beta * C, rtol=1e-3, atol=1e-3)

