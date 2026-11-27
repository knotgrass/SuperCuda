"""Benchmark the kernels in gemm.KERNELS against cuBLAS (torch.addmm). FP32, TF32 off.

One sample = the GPU time of one op call: a CUDA event before the call and one after it,
on the current stream, with a cold L2 cache. The time includes the C.clone() in run_gemm;
torch.addmm also copies C into its output, so both sides do the same work.

Usage (repo root, env cxx, after script/build.sh release):
    python benchmarks/benchmark_gemm.py                          # suite "square", all kernels
    python benchmarks/benchmark_gemm.py --suite odd --kernels naive
    python benchmarks/benchmark_gemm.py --shape 1000 3000 2000   # one shape: M N K
    python benchmarks/benchmark_gemm.py --csv reports/$(git rev-parse --short HEAD).csv
"""

import csv, math, os, subprocess, sys
from pathlib import Path
from datetime import datetime
from time import perf_counter
from statistics import quantiles, median
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import torch, triton

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "torch_ext"))
import gemm  # noqa: E402

ROOT = Path(gemm.__file__).resolve().parents[2]   # the repo, for git

torch.backends.cuda.matmul.allow_tf32 = False      # cuBLAS in full FP32, like the kernels

ALPHA, BETA = 0.5, -0.1         # not 1 and 0: the kernel must do all of alpha * A @ B + beta * C
L2_FLUSH_BYTES = 256 * 2**20    # 8 x the 32 MiB L2 of the RTX 5060 Ti
DRIFT_LIMIT = 0.03              # cuBLAS before vs after the kernels: more than 3 % = the GPU changed
NOISE_LIMIT = 0.05              # (p80 - p20) / median above 5 % = a noisy measurement

# FP32 peak without tensor cores, from the spec sheet: SMs x 128 lanes x 2 FLOP x boost clock.
PEAK_TFLOPS = {"RTX 5060 Ti": 23.7, "A10": 31.2}

SUITES = {
    "square": [(n, n, n) for n in (256, 512, 1024, 2048, 4096, 8192)],
    "odd": [(1000, 1000, 1000), (1023, 1025, 1027), (3000, 2000, 1000)],
    "llm": [(m, 4096, 4096) for m in (1, 16, 128, 2048)] + [(2048, 14336, 4096), (2048, 4096, 14336)],
}


# ---------------------------------------------------------------- environment
def run(*cmd: str) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def smi(query: str) -> str:
    """One value from nvidia-smi for the current GPU ('' if nvidia-smi is missing)."""
    bus = torch.cuda.get_device_properties(torch.cuda.current_device()).pci_bus_id
    out = run("nvidia-smi", f"--query-gpu=pci.bus,{query}", "--format=csv,noheader,nounits")
    for line in out.splitlines():
        bus_hex, _, value = line.partition(", ")
        if int(bus_hex, 16) == bus:
            return value
    return ""


def environment() -> dict:
    dirty = run("git", "status", "--porcelain", "--", "csrc", "include", "torch_ext", "build.toml")
    return {
        "date": datetime.now().isoformat(timespec="seconds"),
        "gpu": torch.cuda.get_device_name(),
        "driver": smi("driver_version"),
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "commit": run("git", "rev-parse", "--short", "HEAD") + ("+dirty" if dirty else ""),
        "build": os.environ.get("SUPERCUDA_BUILD", "release"),
    }


def other_gpu_processes() -> list[str]:
    out = run("nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader")
    return [line for line in out.splitlines() if line and int(line.split(",")[0]) != os.getpid()]


# ---------------------------------------------------------------- measurement
def gpu_times(fn, flush: torch.Tensor, min_ms: float, min_samples: int = 10, max_samples: int = 1000) -> list[float]:
    """Run fn many times. Return the GPU time of each call in ms."""
    # Warm-up by time, not by count. The first call loads the kernel module and creates
    # the cuBLAS handle; the next calls raise the GPU clock from idle.
    fn()
    torch.cuda.synchronize()
    t0, calls = perf_counter(), 0
    while (perf_counter() - t0) * 1e3 < 100 or calls < 2:
        fn()
        torch.cuda.synchronize()
        calls += 1
    est_ms = (perf_counter() - t0) * 1e3 / calls      # includes host overhead: an upper bound
    n = min(max_samples, max(min_samples, math.ceil(min_ms / est_ms)))

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    for i in range(n):
        flush.zero_()          # evict A, B, C from L2; the CPU queues the call while the GPU writes
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()   # one sync for all samples
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def summarize(times: list[float]) -> dict:
    deciles = quantiles(times, n=10, method="inclusive")
    return {"median_ms": median(times), "p20_ms": deciles[1], "p80_ms": deciles[7], "n": len(times)}


def check(out: torch.Tensor, ref: torch.Tensor, K: int) -> tuple[bool, float]:
    """Same tolerance as tests/test_gemm.py."""
    atol = 1e-6 * max(1.0, abs(ALPHA)) * max(K, 1024)
    err = (out - ref).abs().max().item() if out.numel() else 0.0
    return torch.allclose(out, ref, rtol=1e-3, atol=atol), err


# ---------------------------------------------------------------- one shape
def bench_shape(M: int, N: int, K: int, kernels: list[str], flush, min_ms: float, peak: float | None) -> list[dict]:
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda")      # random data: zeros draw less power and run faster
    B = torch.randn(K, N, device="cuda")
    C = torch.randn(M, N, device="cuda")
    flop = 2.0 * M * N * K                     # BLAS convention: K multiplies + K adds per element of C

    def cublas():
        return torch.addmm(C, A, B, beta=BETA, alpha=ALPHA)

    ref = cublas()
    before = summarize(gpu_times(cublas, flush, min_ms))
    rows = [{"kernel": "cuBLAS", "ok": True, "err": 0.0, **before}]
    for name in kernels:
        fn = gemm.KERNELS[name]

        def call(fn=fn):
            return fn(A, B, C, ALPHA, BETA)

        ok, err = check(call(), ref, K)        # correctness gate: never time a wrong kernel
        stats = summarize(gpu_times(call, flush, min_ms)) if ok else {"median_ms": math.nan, "p20_ms": math.nan, "p80_ms": math.nan, "n": 0}
        rows.append({"kernel": name, "ok": ok, "err": err, **stats})
    after = summarize(gpu_times(cublas, flush, min_ms))

    drift = abs(after["median_ms"] - before["median_ms"]) / before["median_ms"]
    base_ms = (before["median_ms"] + after["median_ms"]) / 2
    sm_mhz = smi("clocks.sm")
    for r in rows:
        t = r["median_ms"]
        r.update(M=M, N=N, K=K, alpha=ALPHA, beta=BETA, sm_clock_mhz=sm_mhz, cublas_drift=round(drift, 4),
                 tflops=flop / (t * 1e-3) / 1e12,
                 pct_peak=100 * flop / (t * 1e-3) / 1e12 / peak if peak else math.nan,
                 pct_cublas=100 * base_ms / t,
                 spread=(r["p80_ms"] - r["p20_ms"]) / t)
    if drift > DRIFT_LIMIT:
        print(f"  warning: cuBLAS changed by {100 * drift:.1f} % during this shape (clock, heat, or another process)")
    return rows


def print_rows(rows: list[dict]) -> None:
    M, N, K = rows[0]["M"], rows[0]["N"], rows[0]["K"]
    print(f"\n{M} x {N} x {K}   SM clock {rows[0]['sm_clock_mhz']} MHz")
    for r in rows:
        if not r["ok"]:
            print(f"  {r['kernel']:12s}  FAIL: max abs error {r['err']:.3g}, not timed")
            continue
        noisy = "  noisy" if r["spread"] > NOISE_LIMIT else ""
        print(f"  {r['kernel']:12s} {r['median_ms']:10.4f} ms  [{r['p20_ms']:.4f}, {r['p80_ms']:.4f}]  n={r['n']:<5d}"
              f"{r['tflops']:7.2f} TFLOP/s {r['pct_peak']:6.1f} % peak {r['pct_cublas']:7.1f} % cuBLAS{noisy}")


# ---------------------------------------------------------------- main
def main() -> None:
    ap = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    ap.add_argument("--suite", choices=SUITES, default="square")
    ap.add_argument("--shape", type=int, nargs=3, metavar=("M", "N", "K"), help="one shape instead of a suite")
    ap.add_argument("--kernels", default=",".join(gemm.KERNELS), help="comma-separated names from gemm.KERNELS")
    ap.add_argument("--min-ms", type=float, default=500.0, help="measured GPU time per kernel and shape")
    ap.add_argument("--csv", default= "reports/naive.csv", type=Path, help="append one row per kernel and shape to this file")
    args = ap.parse_args()

    kernels = [k for k in args.kernels.split(",") if k]
    unknown = set(kernels) - set(gemm.KERNELS)
    if unknown:
        ap.error(f"unknown kernels {sorted(unknown)}; gemm.KERNELS has {list(gemm.KERNELS)}")
    shapes = [tuple(args.shape)] if args.shape else SUITES[args.suite]

    env = environment()
    peak = next((v for k, v in PEAK_TFLOPS.items() if k in env["gpu"]), None)
    print(" | ".join(f"{k} {v}" for k, v in env.items()))
    print(f"TF32 off | L2 flushed before each sample | FP32 peak {peak} TFLOP/s")
    others = other_gpu_processes()
    if others:
        print(f"warning: other processes use the GPU: {others}")
    if env["build"] != "release":
        print("warning: not a release build: the numbers mean nothing")

    flush = torch.empty(L2_FLUSH_BYTES, dtype=torch.uint8, device="cuda")
    all_rows = []
    for M, N, K in shapes:
        rows = bench_shape(M, N, K, kernels, flush, args.min_ms, peak)
        print_rows(rows)
        all_rows += [{**env, **r} for r in rows]

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        new = not args.csv.exists()
        with args.csv.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
            if new:
                w.writeheader()
            w.writerows(all_rows)
        print(f"\n{len(all_rows)} rows -> {args.csv}")


if __name__ == "__main__":
    main()
