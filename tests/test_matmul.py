"""Every kernel in matmul.KERNELS against PyTorch (cuBLAS): out = alpha * A @ B + beta * C, float32."""

import pytest
import torch

import matmul

# ------------------------------------------------------------------------------------------------
# Parameters. Each decorator is stored in a variable and reused.
# ------------------------------------------------------------------------------------------------
KERNELS = pytest.mark.parametrize("kernel", list(matmul.KERNELS.values()), ids=list(matmul.KERNELS))

SIZES = [  # (M, N, K). The kernels work in 32 x 32 blocks.
    (1000, 1500, 777),   # no dimension is a multiple of 32: edge blocks in x and y; M != N
    (64, 96, 128),       # every dimension is a multiple of 32: no edge block
    (33, 65, 1),         # one element past a block boundary; K = 1
    (31, 1, 5),          # one element short of a block; a single column
    (1, 1, 1),           # the smallest problem
    (0, 5, 3),           # M = 0: empty output, the launcher must not launch
    (5, 0, 3),           # N = 0
    (4, 6, 0),           # K = 0: empty sum, out = beta * C
    pytest.param((2048, 2048, 4096), marks=pytest.mark.slow),   # many blocks, long sums
]
SIZE = pytest.mark.parametrize("size", SIZES, ids=lambda s: "x".join(map(str, s)))

COEFFS = pytest.mark.parametrize(
    "alpha,beta",
    [(1.0, 0.0), (0.5, -0.1), (0.0, 1.0)],          # A @ B only; both terms; beta * C only
    ids=["a1_b0", "a0.5_b-0.1", "a0_b1"],
)

# ------------------------------------------------------------------------------------------------
# Tolerance. A kernel and cuBLAS add K products in a different order. For N(0, 1) inputs the
# float32 rounding error of the worst element is about 2 * 6e-8 * K (1e-4 at K = 777).
# 1e-6 * K leaves a margin of about 8x; an indexing bug gives errors of 1 or more.
# ------------------------------------------------------------------------------------------------
RTOL = 1e-3


def atol_for(K):
    return max(1e-3, 1e-6 * K)


def make_abc(M, N, K, seed=0):
    """Inputs with a fixed seed: every run of a test sees the same data."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    A = torch.randn(M, K, device="cuda", generator=g)
    B = torch.randn(K, N, device="cuda", generator=g)
    C = torch.randn(M, N, device="cuda", generator=g)
    return A, B, C


def reference(A, B, C, alpha, beta):
    """The independent answer: A @ B runs cuBLAS."""
    return alpha * (A @ B) + beta * C


# ------------------------------------------------------------------------------------------------
# 1. Results
# ------------------------------------------------------------------------------------------------
@COEFFS
@SIZE
@KERNELS
def test_matches_torch(kernel, size, alpha, beta):
    M, N, K = size
    A, B, C = make_abc(M, N, K)

    out = kernel(A, B, C, alpha, beta)

    torch.testing.assert_close(out, reference(A, B, C, alpha, beta), rtol=RTOL, atol=atol_for(K))


# ------------------------------------------------------------------------------------------------
# 2. Contract: what the op promises besides the numbers
# ------------------------------------------------------------------------------------------------
@KERNELS
def test_output_properties(kernel):
    A, B, C = make_abc(40, 50, 60)

    out = kernel(A, B, C, 0.5, -0.1)

    assert out.shape == (40, 50)
    assert out.dtype == torch.float32
    assert out.device == A.device
    assert out.is_contiguous()
    assert out.data_ptr() != C.data_ptr(), "out must be a new tensor, not C"


@KERNELS
def test_does_not_modify_inputs(kernel):
    A, B, C = make_abc(40, 50, 60)
    A0, B0, C0 = A.clone(), B.clone(), C.clone()

    kernel(A, B, C, 0.5, -0.1)

    assert torch.equal(A, A0), "A was modified"
    assert torch.equal(B, B0), "B was modified"
    assert torch.equal(C, C0), "C was modified: the kernel must write to a copy"


@KERNELS
def test_accepts_views_with_offset(kernel):
    """A contiguous slice of a larger tensor starts in the middle of its storage."""
    M, N, K = 40, 50, 60
    g = torch.Generator(device="cuda").manual_seed(1)
    A = torch.randn(M + 3, K, device="cuda", generator=g)[3:]
    B = torch.randn(K + 5, N, device="cuda", generator=g)[5:]
    C = torch.randn(M + 2, N, device="cuda", generator=g)[2:]
    assert A.is_contiguous() and A.storage_offset() > 0   # the precondition of this test

    out = kernel(A, B, C, 0.5, -0.1)

    torch.testing.assert_close(out, reference(A, B, C, 0.5, -0.1), rtol=RTOL, atol=atol_for(K))


# ------------------------------------------------------------------------------------------------
# 3. Error paths: one case per TORCH_CHECK in run_sgemm (matmul_cuda/ops.cu)
# ------------------------------------------------------------------------------------------------
BAD_INPUTS = [  # (how to break valid 8x4, 4x6, 8x6 inputs; text that the error must contain)
    pytest.param(lambda A, B, C: (A.t().contiguous().t(), B, C), "contiguous", id="non_contiguous_A"),
    pytest.param(lambda A, B, C: (A.half(), B.half(), C.half()), "float32", id="float16"),
    pytest.param(lambda A, B, C: (A.double(), B, C), "float32", id="mixed_dtype"),
    pytest.param(lambda A, B, C: (A, B[:-1], C), "B is", id="K_mismatch"),
    pytest.param(lambda A, B, C: (A, B, C[:, :-1].contiguous()), "C must be", id="C_shape"),
    pytest.param(lambda A, B, C: (A[None], B, C), "2-D", id="3d_A"),
    pytest.param(lambda A, B, C: (A, B.cpu(), C), "CUDA tensors", id="mixed_devices"),
    pytest.param(lambda A, B, C: (A.cpu(), B.cpu(), C.cpu()), "CPU", id="all_cpu"),   # no CPU op is registered
]


@pytest.mark.parametrize("break_inputs,message", BAD_INPUTS)
@KERNELS
def test_rejects_bad_inputs(kernel, break_inputs, message):
    A, B, C = make_abc(8, 6, 4)

    with pytest.raises(RuntimeError, match=message):
        kernel(*break_inputs(A, B, C))


# ------------------------------------------------------------------------------------------------
# 4. Launcher contract: use the given stream, never synchronize
# ------------------------------------------------------------------------------------------------
@KERNELS
def test_runs_on_a_side_stream(kernel):
    A, B, C = make_abc(256, 300, 77)
    expected = reference(A, B, C, 0.5, -0.1)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())     # the inputs are ready before `side` starts
    with torch.cuda.stream(side):
        out = kernel(A, B, C, 0.5, -0.1)              # the op reads the current stream: `side`
    torch.cuda.current_stream().wait_stream(side)     # the main stream waits for `out`

    torch.testing.assert_close(out, expected, rtol=RTOL, atol=atol_for(77))


@KERNELS
def test_cuda_graph_replay(kernel):
    """Capture fails if the launcher synchronizes or launches on a stream other than the current one."""
    A, B, C = make_abc(256, 300, 77)
    kernel(A, B, C, 0.5, -0.1)                        # warm-up outside the capture

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = kernel(A, B, C, 0.5, -0.1)              # recorded, not run

    A.copy_(torch.randn_like(A))                      # new data in the SAME buffer
    graph.replay()                                    # runs the recorded clone + kernel; fills `out`

    torch.testing.assert_close(out, reference(A, B, C, 0.5, -0.1), rtol=RTOL, atol=atol_for(77))


# ------------------------------------------------------------------------------------------------
# 5. Registration: the schema `-> Tensor` (no `(a!)`) says that no input is mutated or aliased
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(matmul.KERNELS))
def test_schema_matches_behaviour(name):
    A, B, C = make_abc(8, 6, 4)
    op = getattr(torch.ops.matmul, name)

    torch.library.opcheck(op, (A, B, C, 0.5, -0.1), test_utils="test_schema")
