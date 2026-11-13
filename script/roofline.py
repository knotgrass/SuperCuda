#!/usr/bin/env python
"""Roofline of one kernel from an Nsight Compute report: FLOP/s vs. arithmetic intensity at DRAM, L2 and L1.

Usage:
    python script/roofline.py reports/naive_1024.ncu-rep              # table on stdout
    python script/roofline.py reports/naive_1024.ncu-rep --png        # + reports/naive_1024_roofline.png
    python script/roofline.py reports/a.ncu-rep reports/b.ncu-rep --png   # several kernels on one chart
    python script/roofline.py reports/x.ncu-rep --dump                # raw metric rows (debug)

The report must come from `script/ncu.sh` (--set full collects the roofline metrics). Metrics are read with
`ncu --import <rep> --page raw --csv --print-units base`. Nothing is hard-coded per GPU:
  - FLOP       = 2 * ffma + fmul + fadd   (SASS thread instructions, predicated-on)
                 These are the FP operations the kernel EXECUTED. benchmarks/benchmark_matmul.py counts the
                 2MNK operations the algorithm needs. A kernel that does extra work per term (naive_sgemm
                 multiplies by alpha inside the loop: fmul + ffma = 3 per term) shows FLOP = 3MNK here.
                 Compare kernels with the benchmark's GFLOP/s; use this FLOP for the roofline position.
  - peak FP32  = SMs * FP32 lanes per SM * 2 * SM clock measured during the kernel (lanes: --fp32-lanes, 128 on
                 consumer Ampere/Ada/Blackwell, 64 on A100/H100)
  - peak BW    = achieved bytes/s / (% of peak that ncu reports), for DRAM, L2 and L1
The PNG needs matplotlib (`pip install matplotlib`); the table does not.
"""

import argparse
import csv
import io
import re
import subprocess
import sys
from pathlib import Path

# metric name -> key; several names are tried for the FLOP counters (older ncu versions use smsp__).
METRICS = {
    "time": ["gpu__time_duration.sum"],
    "ffma": ["smsp__sass_thread_inst_executed_op_ffma_pred_on.sum", "sm__sass_thread_inst_executed_op_ffma_pred_on.sum"],
    "fmul": ["smsp__sass_thread_inst_executed_op_fmul_pred_on.sum", "sm__sass_thread_inst_executed_op_fmul_pred_on.sum"],
    "fadd": ["smsp__sass_thread_inst_executed_op_fadd_pred_on.sum", "sm__sass_thread_inst_executed_op_fadd_pred_on.sum"],
    "dram_bytes": ["dram__bytes.sum"],
    "dram_pct": ["dram__throughput.avg.pct_of_peak_sustained_elapsed"],
    "l2_bytes": ["lts__t_bytes.sum"],
    "l2_pct": ["lts__throughput.avg.pct_of_peak_sustained_elapsed"],
    "l1_bytes": ["l1tex__t_bytes.sum"],
    "l1_pct": ["l1tex__throughput.avg.pct_of_peak_sustained_elapsed"],
    "sm_clock": ["sm__cycles_elapsed.avg.per_second"],
    "sm_count": ["device__attribute_multiprocessor_count"],
    "sm_pct": ["sm__throughput.avg.pct_of_peak_sustained_elapsed"],
    "fma_pipe_pct": ["sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active"],
}

_PREFIX = {"": 1.0, "k": 1e3, "m": 1e6, "g": 1e9, "t": 1e12}
_TIME = {"nsecond": 1e-9, "ns": 1e-9, "usecond": 1e-6, "us": 1e-6, "msecond": 1e-3, "ms": 1e-3, "second": 1.0, "s": 1.0}


def to_base(value: str, unit: str) -> float:
    """Convert an ncu CSV cell to SI base units (byte, second, Hz, %, count)."""
    v = float(value.replace(",", ""))
    u = unit.strip().lower()
    if u in ("", "%", "inst", "cycle", "byte", "sm", "block", "thread", "warp"):
        return v
    if u in _TIME:
        return v * _TIME[u]
    m = re.fullmatch(r"([kmgt]?)(byte|hz)", u)  # Kbyte, Mbyte, Ghz ...
    if m:
        return v * _PREFIX[m.group(1)]
    m = re.fullmatch(r"([kmgt]?)(cycle|byte|inst)/(nsecond|usecond|msecond|second|ns|us|ms|s)", u)  # cycle/nsecond -> Hz
    if m:
        return v * _PREFIX[m.group(1)] / _TIME[m.group(3)]
    raise ValueError(f"unknown unit {unit!r} for value {value!r}")


def read_report(path: Path):
    """Run ncu --import and return (kernel rows, header, units). Each row maps metric key -> float."""
    names = [n for alts in METRICS.values() for n in alts]
    cmd = ["ncu", "--import", str(path), "--page", "raw", "--csv", "--print-units", "base", "--metrics", ",".join(names)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"ncu failed ({out.returncode}):\n{out.stderr}")
    lines = [ln for ln in out.stdout.splitlines() if not ln.startswith("==")]
    rows = list(csv.reader(io.StringIO("\n".join(lines))))
    try:
        h = next(i for i, r in enumerate(rows) if "Kernel Name" in r)
    except StopIteration:
        sys.exit(f"no CSV header in ncu output:\n{out.stdout[:2000]}")
    header, units, data = rows[h], rows[h + 1], rows[h + 2:]
    col = {name: i for i, name in enumerate(header)}
    kernels = []
    for r in data:
        if len(r) != len(header):
            continue
        k = {"name": r[col["Kernel Name"]], "grid": r[col.get("Grid Size", 0)], "block": r[col.get("Block Size", 0)]}
        for key, alts in METRICS.items():
            for name in alts:
                i = col.get(name)
                if i is not None and r[i] not in ("", "n/a"):
                    k[key] = to_base(r[i], units[i])
                    break
        kernels.append(k)
    return kernels, header, units, data


def fmt(v, unit=""):
    for scale, p in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= scale:
            return f"{v / scale:7.2f} {p}{unit}"
    return f"{v:7.2f} {unit}"


def analyze(k, fp32_lanes):
    """Return a dict with FLOP, FLOP/s, FP32 peak, and per-level (bytes, AI, achieved BW, peak BW, roof, ratio)."""
    missing = [m for m in ("time", "ffma", "fmul", "fadd") if m not in k]
    if missing:
        sys.exit(f"metrics missing in the report: {missing}. Profile with script/ncu.sh (--set full).")
    t = k["time"]
    flop = 2 * k["ffma"] + k["fmul"] + k["fadd"]
    flops = flop / t
    peak = k["sm_count"] * fp32_lanes * 2 * k["sm_clock"] if "sm_count" in k and "sm_clock" in k else None
    levels = {}
    for lvl in ("dram", "l2", "l1"):
        if f"{lvl}_bytes" not in k:
            continue
        b = k[f"{lvl}_bytes"]
        bw = b / t
        pct = k.get(f"{lvl}_pct")
        peak_bw = bw / (pct / 100) if pct and pct > 0 else None
        ai = flop / b if b > 0 else float("inf")
        roof = None
        if peak_bw:
            roof = peak_bw * ai
            if peak:
                roof = min(roof, peak)
        levels[lvl] = {"bytes": b, "ai": ai, "bw": bw, "pct": pct, "peak_bw": peak_bw, "roof": roof,
                       "ratio": flops / roof if roof else None}
    return {"flop": flop, "flops": flops, "time": t, "peak": peak, "levels": levels}


def report(k, a, fp32_lanes):
    print(f"kernel : {k['name']}")
    print(f"launch : grid {k['grid']}  block {k['block']}")
    if a["peak"]:
        print(f"GPU    : {k['sm_count']:.0f} SMs x {fp32_lanes} FP32 lanes x 2 x {k['sm_clock'] / 1e9:.2f} GHz (clock during the kernel)"
              f"  ->  FP32 peak {fmt(a['peak'], 'FLOP/s')}")
    print(f"time   : {a['time'] * 1e3:9.3f} ms")
    print(f"FLOP   : {a['flop']:.4g} executed  (ffma {k['ffma']:.3g} x2, fmul {k['fmul']:.3g}, fadd {k['fadd']:.3g})")
    line = f"FLOP/s : {fmt(a['flops'], 'FLOP/s')}"
    if a["peak"]:
        line += f"  = {100 * a['flops'] / a['peak']:5.1f} % of FP32 peak"
    if "fma_pipe_pct" in k:
        line += f"   (FMA pipe active {k['fma_pipe_pct']:.1f} %)"
    print(line)
    print()
    print(f"{'level':6} {'bytes':>12} {'AI FLOP/B':>10} {'achieved BW':>14} {'peak BW (ncu %)':>20} {'roof at AI':>16} {'attained/roof':>14}")
    nearest = None
    for lvl, L in a["levels"].items():
        peak_s = f"{fmt(L['peak_bw'], 'B/s')} ({L['pct']:4.1f}%)" if L["peak_bw"] else "n/a"
        roof_s = fmt(L["roof"], "FLOP/s") if L["roof"] else "n/a"
        ratio_s = f"{100 * L['ratio']:6.1f} %" if L["ratio"] is not None else "n/a"
        print(f"{lvl.upper():6} {fmt(L['bytes'], 'B'):>12} {L['ai']:10.3f} {fmt(L['bw'], 'B/s'):>14} {peak_s:>20} {roof_s:>16} {ratio_s:>14}")
        if L["ratio"] is not None and (nearest is None or L["ratio"] > nearest[1]):
            nearest = (lvl, L["ratio"])
    print()
    if nearest:
        lvl, r = nearest
        if r >= 0.7:
            print(f"verdict: bound by the {lvl.upper()} roof ({100 * r:.0f} % of it). Raise the arithmetic intensity at {lvl.upper()}"
                  f" (reuse data from a faster level) or cut the bytes.")
        else:
            print(f"verdict: no roof reached (nearest: {lvl.upper()} at {100 * r:.0f} %). The kernel is latency- or"
                  f" occupancy-bound: look at Warp State Statistics and Occupancy in the ncu report.")
    if "sm_pct" in k:
        print(f"ncu SM throughput {k['sm_pct']:.1f} %  (compare with the FLOP/s % above: the SM can be busy with non-FP32 work)")


def plot(items, png: Path, fp32_lanes):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("--png needs matplotlib: pip install matplotlib")
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ai = np.logspace(-2, 4, 400)
    colors = {"dram": "tab:red", "l2": "tab:orange", "l1": "tab:green"}
    drawn_roofs = set()
    markers = ["o", "s", "^", "D", "v"]
    for idx, (k, a) in enumerate(items):
        peak = a["peak"]
        for lvl, L in a["levels"].items():
            if L["peak_bw"] and lvl not in drawn_roofs:  # roofs from the first report that has them
                roof = L["peak_bw"] * ai
                if peak:
                    roof = np.minimum(roof, peak)
                ax.plot(ai, roof, color=colors[lvl], lw=1.4, label=f"{lvl.upper()} roof: {L['peak_bw'] / 1e9:.0f} GB/s")
                drawn_roofs.add(lvl)
        if peak and "peak" not in drawn_roofs:
            ax.axhline(peak, color="k", lw=1.2, ls="--", label=f"FP32 peak {peak / 1e12:.1f} TFLOP/s")
            drawn_roofs.add("peak")
        short = re.sub(r"\(.*", "", k["name"])[:28]
        for lvl, L in a["levels"].items():
            ax.plot(L["ai"], a["flops"], marker=markers[idx % len(markers)], color=colors[lvl], ms=8, ls="none",
                    label=f"{short} @ {lvl.upper()} (AI {L['ai']:.3g})")
        ais = [L["ai"] for L in a["levels"].values()]
        if len(ais) > 1:
            ax.plot(ais, [a["flops"]] * len(ais), color="gray", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP / byte)")
    ax.set_ylabel("FLOP/s")
    ax.set_title("Roofline (one point per memory level; same FLOP/s, different bytes)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(png, dpi=140)
    print(f"wrote {png}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("reports", nargs="+", type=Path, help=".ncu-rep files from script/ncu.sh")
    p.add_argument("--png", action="store_true", help="also draw reports/<first report>_roofline.png")
    p.add_argument("--fp32-lanes", type=int, default=128, help="FP32 lanes per SM for the peak (default 128; A100/H100: 64)")
    p.add_argument("--dump", action="store_true", help="print the raw metric rows and exit")
    args = p.parse_args()

    items = []
    for rep in args.reports:
        kernels, header, units, data = read_report(rep)
        if args.dump:
            for name, unit, *vals in zip(header, units, *data):
                print(f"{name:70} {unit:14} {' | '.join(vals)}")
            continue
        if not kernels:
            sys.exit(f"{rep}: no kernel rows")
        for k in kernels:
            a = analyze(k, args.fp32_lanes)
            print(f"report : {rep}")
            report(k, a, args.fp32_lanes)
            print()
            items.append((k, a))
    if args.png and items:
        plot(items, args.reports[0].with_name(args.reports[0].stem + "_roofline.png"), args.fp32_lanes)


if __name__ == "__main__":
    main()
