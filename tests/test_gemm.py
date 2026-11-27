import random, pytest, torch, gemm

# Compare against a full FP32 reference: TF32 keeps only 10 mantissa bits of A and B.
torch.backends.cuda.matmul.allow_tf32 = False


def random_coeffs(rng: random.Random) -> tuple[float, float]:
    alpha = rng.uniform(-16.0, 16.0)
    beta = rng.uniform(-16.0, 16.0)
    return alpha, beta

def random_tensors_value(M: int, N: int, K: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)   # [flash] the same data on every run
    A = torch.randn(M, K, device="cuda")
    B = torch.randn(K, N, device="cuda")
    C = torch.randn(M, N, device="cuda")
    return A, B, C

Shape = tuple[int, int, int]   # (M, N, K)

def random_size(n_seeds: int = 8, max_size: int = 8192, block: int = 32) -> list[Shape]:
    """Return the test shapes (M, N, K) without duplicates, every dimension in [1, max_size].

    n_seeds:  each seed in range(n_seeds) gives one aligned shape and one odd shape.
    max_size: largest M, N, K.
    block:    block side of the naive kernel (csrc/gemm/naive.cu).

    edge:    a dimension of 1, and max_size x max_size x max_size. Random draws almost
             never give these.
    aligned: every dimension is a power of 2 from block to max_size (32, 64, ..., 8192
             with the defaults), so the grid covers C exactly.
    odd:     no dimension is a multiple of block, so the last blocks of the grid have
             threads outside C, and the test checks the bounds check of the kernel.
    """
    n_blocks = max_size // block

    def aligned_dim(rng: random.Random) -> int:
        return block * 2 ** rng.randint(0, n_blocks.bit_length() - 1)

    def odd_dim(rng: random.Random) -> int:
        # Subtract (not add) 1..block-1, so the dimension stays <= max_size.
        return block * rng.randint(1, n_blocks) - rng.randint(1, block - 1)

    shapes: list[Shape] = [    # edge
        (1, 1, 1),
        (1, max_size, max_size),           # decode: one token
        (block + 1, 1, 2 * block + 1),     # N = 1 (33 x 1 x 65 with the defaults)
        (max_size, max_size, max_size),    # largest case
    ]
    for seed in range(n_seeds):
        rng = random.Random(seed)
        shapes.append((aligned_dim(rng), aligned_dim(rng), aligned_dim(rng)))
        shapes.append((odd_dim(rng), odd_dim(rng), odd_dim(rng)))
    return list(dict.fromkeys(shapes))   # remove duplicates, keep the order


SHAPES = random_size(n_seeds=8, max_size=8192, block=32)


@pytest.mark.parametrize("M, N, K", SHAPES, ids=[f"{M}x{N}x{K}" for M, N, K in SHAPES])
def test_gemm(M: int, N: int, K: int):
    alpha, beta = random_coeffs(random.Random(f"{M}x{N}x{K}"))   # the shape sets the coefficients
    A, B, C = random_tensors_value(M, N, K)

    outp = gemm.naive(A, B, C, alpha, beta)
    expected = alpha * (A @ B) + beta * C

    # The K-term sum can cancel to ~0, so its rounding error needs an atol that grows with |alpha|
    # and with K (sequential sum: the error grows about linearly with K). For K <= 1024 it is 1e-3 * |alpha|.
    # The error from beta * C grows with |beta * C| of the same element: rtol covers it.
    atol = 1e-6 * max(1.0, abs(alpha)) * max(K, 1024)
    torch.testing.assert_close(
        outp, expected, rtol=1e-3, atol=atol,
        msg=lambda m: f"M={M} N={N} K={K} alpha={alpha:.4f} beta={beta:.4f}\n{m}",
    )
