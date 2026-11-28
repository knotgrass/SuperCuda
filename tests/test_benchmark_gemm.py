"""Check the % cuBLAS logic of benchmarks/benchmark_gemm.py.

Part A (no GPU, no torch): run the real bench_shape with scripted GPU times and compare
pct_cublas with numbers computed by hand.
Part B (GPU): use cuBLAS itself as the "kernel". One addmm call must give ~100 %,
two addmm calls must give ~50 %. This checks the timing path end to end.
"""

import importlib.util, math, sys, types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "benchmark_gemm.py"


def load_bench(monkeypatch, kernels: dict, fake_torch: bool):
    """Import benchmark_gemm.py as a new module. gemm is always a stub (no JIT build).
    With fake_torch, torch and triton are stubs too: no GPU and no torch install needed."""
    gemm = types.ModuleType("gemm")
    gemm.__file__ = str(ROOT / "torch_ext" / "gemm" / "__init__.py")
    gemm.KERNELS = kernels
    monkeypatch.setitem(sys.modules, "gemm", gemm)
    monkeypatch.setattr(sys, "path", list(sys.path))   # undo the sys.path.insert of the module
    if fake_torch:
        torch = types.ModuleType("torch")
        torch.Tensor = object                            # used in type hints at import time
        torch.backends = types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=types.SimpleNamespace()))
        torch.manual_seed = lambda seed: None
        torch.randn = lambda *shape, device=None: None
        torch.addmm = lambda C, A, B, beta, alpha: None
        monkeypatch.setitem(sys.modules, "torch", torch)
        monkeypatch.setitem(sys.modules, "triton", types.ModuleType("triton"))
    spec = importlib.util.spec_from_file_location("benchmark_gemm_under_test", BENCH)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    return bench


# ---------------------------------------------------------------- part A: scripted times
def flat(ms: float, n: int = 5) -> list[float]:
    return [ms] * n


def run_scripted(monkeypatch, times, kernels=("k",), failing=(), M=4096, N=4096, K=4096):
    """Run bench_shape. times[i] is the sample list that the i-th gpu_times call returns.
    Return (rows by kernel name, names of the functions that gpu_times got, in call order)."""
    stubs = {name: (lambda *args, name=name: name) for name in kernels}   # a "kernel" returns its name
    bench = load_bench(monkeypatch, stubs, fake_torch=True)
    script, timed = iter(times), []

    def fake_gpu_times(fn, flush, min_ms):
        timed.append(fn.__name__)
        return next(script)

    monkeypatch.setattr(bench, "gpu_times", fake_gpu_times)
    monkeypatch.setattr(bench, "smi", lambda query: "2602")
    monkeypatch.setattr(bench, "check", lambda out, ref, K: (out not in failing, 0.0))
    rows = bench.bench_shape(M, N, K, list(kernels), flush=None, min_ms=1.0, peak=None)
    assert next(script, None) is None, "bench_shape used fewer gpu_times calls than scripted"
    return {r["kernel"]: r for r in rows}, timed


def test_worked_example(monkeypatch, capsys):
    # cuBLAS before 8.0 ms, kernel 64.8 ms, cuBLAS after 8.2 ms.
    r, _ = run_scripted(monkeypatch, [flat(8.0), flat(64.8), flat(8.2)])
    # base = (8.0 + 8.2) / 2 = 8.1 ms   ->   pct = 100 * 8.1 / 64.8 = 12.5
    assert r["k"]["pct_cublas"] == pytest.approx(12.5)
    # The cuBLAS row uses t = before = 8.0 ms   ->   100 * 8.1 / 8.0 = 101.25
    assert r["cuBLAS"]["pct_cublas"] == pytest.approx(101.25)
    # drift = |8.2 - 8.0| / 8.0 = 0.025, below the 3 % limit: no warning
    assert r["k"]["cublas_drift"] == pytest.approx(0.025)
    assert "warning" not in capsys.readouterr().out


@pytest.mark.parametrize("slowdown", [0.5, 1, 2, 4, 8, 100])
def test_pct_is_100_over_slowdown(monkeypatch, slowdown):
    # A kernel that takes `slowdown` x the cuBLAS time gets 100 / slowdown %.
    r, _ = run_scripted(monkeypatch, [flat(3.0), flat(3.0 * slowdown), flat(3.0)])
    assert r["k"]["pct_cublas"] == pytest.approx(100 / slowdown)


def test_pct_equals_tflops_ratio(monkeypatch):
    # Same FLOP on both sides, so the time ratio is the throughput ratio.
    M, N, K = 1000, 3000, 2000
    before, t, after = 1.3, 5.9, 1.4
    r, _ = run_scripted(monkeypatch, [flat(before), flat(t), flat(after)], M=M, N=N, K=K)
    flop = 2 * M * N * K
    kernel_tflops = flop / (t * 1e-3) / 1e12
    cublas_tflops = flop / ((before + after) / 2 * 1e-3) / 1e12
    assert r["k"]["tflops"] == pytest.approx(kernel_tflops)
    assert r["k"]["pct_cublas"] == pytest.approx(100 * kernel_tflops / cublas_tflops)


def test_base_is_mean_of_before_and_after(monkeypatch):
    # cuBLAS goes from 10.0 to 10.2 ms. A kernel at 10.1 ms is exactly as fast as cuBLAS.
    r, _ = run_scripted(monkeypatch, [flat(10.0), flat(10.1), flat(10.2)])
    assert r["k"]["pct_cublas"] == pytest.approx(100.0)


def test_cublas_brackets_all_kernels(monkeypatch):
    # cuBLAS is timed first and last. Every kernel uses the same base.
    r, timed = run_scripted(monkeypatch, [flat(1.0), flat(2.0), flat(4.0), flat(1.0)], kernels=("a", "b"))
    assert timed == ["cublas", "call", "call", "cublas"]
    assert r["a"]["pct_cublas"] == pytest.approx(50.0)
    assert r["b"]["pct_cublas"] == pytest.approx(25.0)


def test_median_ignores_one_slow_sample(monkeypatch):
    # The mean of [4, 4, 4, 4, 400] is 83.2 ms; the median is 4 ms.
    r, _ = run_scripted(monkeypatch, [flat(2.0), [4.0, 4.0, 4.0, 4.0, 400.0], flat(2.0)])
    assert r["k"]["pct_cublas"] == pytest.approx(50.0)


def test_drift_above_3_percent_warns(monkeypatch, capsys):
    r, _ = run_scripted(monkeypatch, [flat(10.0), flat(20.0), flat(10.4)])
    assert r["k"]["cublas_drift"] == pytest.approx(0.04)
    assert "cuBLAS changed by 4.0 %" in capsys.readouterr().out


def test_wrong_kernel_gets_no_pct(monkeypatch):
    # check() fails: the kernel is not timed, and pct_cublas is NaN.
    r, timed = run_scripted(monkeypatch, [flat(5.0), flat(5.0)], kernels=("bad",), failing=("bad",))
    assert timed == ["cublas", "cublas"]
    assert math.isnan(r["bad"]["pct_cublas"])


# ---------------------------------------------------------------- part B: real GPU
def test_gpu_cublas_as_kernel(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")

    def addmm_x1(A, B, C, alpha, beta):
        return torch.addmm(C, A, B, beta=beta, alpha=alpha)

    def addmm_x2(A, B, C, alpha, beta):   # twice the work of cuBLAS
        torch.addmm(C, A, B, beta=beta, alpha=alpha)
        return torch.addmm(C, A, B, beta=beta, alpha=alpha)

    bench = load_bench(monkeypatch, {"x1": addmm_x1, "x2": addmm_x2}, fake_torch=False)
    flush = torch.empty(bench.L2_FLUSH_BYTES, dtype=torch.uint8, device="cuda")
    rows = bench.bench_shape(4096, 4096, 4096, ["x1", "x2"], flush, min_ms=300.0, peak=None)
    r = {row["kernel"]: row for row in rows}
    print(f"\nx1 {r['x1']['pct_cublas']:.1f} %   x2 {r['x2']['pct_cublas']:.1f} %   drift {r['x1']['cublas_drift']:.3f}")
    assert r["x1"]["pct_cublas"] == pytest.approx(100, rel=0.05)
    assert r["x2"]["pct_cublas"] == pytest.approx(50, rel=0.05)
