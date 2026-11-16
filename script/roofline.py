#!/usr/bin/env python
"""Roofline of a kernel from an Nsight Compute report: what stops the kernel from running faster?

Usage:
    python script/roofline.py reports/naive_1024.ncu-repz                        # table on stdout
    python script/roofline.py reports/naive_1024.ncu-repz --png                  # + reports/naive_1024_roofline.png
    python script/roofline.py reports/a.ncu-repz reports/b.ncu-repz --png        # several kernels on one chart
    python script/roofline.py --list-metrics                                     # metric names, for script/ncu.sh

The report must come from script/ncu.sh. `ncu --set full` does not store the FLOP counts and the bytes per memory
level, so script/ncu.sh asks this script for the names (--list-metrics) and collects them too.
ncu 2026.3 writes .ncu-repz files; older versions write .ncu-rep. This script reads both.

The roofline model
    x = arithmetic intensity (AI) = FLOP executed / bytes moved     [FLOP/B]
    y = performance               = FLOP executed / kernel time     [FLOP/s]
    Two roofs limit y:
        compute roof = peak FP32 rate of the GPU      (a flat line)
        memory roof  = peak bandwidth * AI            (a rising line: more FLOP per byte -> more FLOP/s)
    A kernel moves bytes at three levels (DRAM, L2, L1). Each level has its own byte count, so the kernel has
    three AI values: three points at the same height, one under each memory roof.

Where the numbers come from (the ncu metric names are in METRICS)
    FLOP        = 2 * FFMA + FMUL + FADD thread instructions. These are the operations the kernel EXECUTED.
                  benchmarks/benchmark_matmul.py counts the 2MNK operations the algorithm NEEDS. naive_sgemm
                  multiplies by alpha inside the loop (FMUL + FFMA = 3 FLOP per term), so it shows 3MNK here.
                  Compare kernels with the GFLOP/s of the benchmark; use this FLOP for the roofline position.
    peak FP32   = 2 * (FFMA thread instructions per cycle at 100 %) * SM clock
    peak BW     = (bytes per cycle at 100 %) * clock of that level        BW = bandwidth, in bytes per second
    FLOP/s/roof = how close the kernel is to the roof of that level
    ncu busy    = the "throughput" that ncu reports for the level: the busiest part of that unit. It can be high
                  while FLOP/s/roof is low. Then the level is busy with requests, not with bytes.

The PNG needs matplotlib (`pip install matplotlib`); the table does not.
"""

import argparse
import csv
import io
import math
import subprocess
import sys
from pathlib import Path

# The values this script reads from a report: short name -> ncu metric name.
# ncu names: sm = streaming multiprocessor, dram = device memory, lts = L2 cache, l1tex = L1 cache.
METRICS = {
    # the kernel
    "time": "gpu__time_duration.sum",
    "ffma": "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",    # fused multiply-add: 2 FLOP each
    "fmul": "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "fadd": "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
    # the compute roof
    "ffma_per_cycle_max": "sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained",  # SMs x FP32 lanes
    "sm_hz": "sm__cycles_elapsed.avg.per_second",                     # SM clock during the kernel
    "sm_busy": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    # one memory roof per level: bytes moved, bytes per cycle at 100 %, clock, and how busy ncu says the level is
    "dram_bytes": "dram__bytes.sum",
    "dram_bytes_per_cycle_max": "dram__bytes.sum.peak_sustained",
    "dram_hz": "dram__cycles_elapsed.avg.per_second",
    "dram_busy": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "l2_bytes": "lts__t_bytes.sum",
    "l2_bytes_per_cycle_max": "lts__t_bytes.sum.peak_sustained",
    "l2_hz": "lts__cycles_elapsed.avg.per_second",
    "l2_busy": "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1_bytes": "l1tex__t_bytes.sum",
    "l1_bytes_per_cycle_max": "l1tex__t_bytes.sum.peak_sustained",
    "l1_hz": "l1tex__cycles_elapsed.avg.per_second",
    "l1_busy": "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
}
LEVELS = ["dram", "l2", "l1"]   # from the slowest memory to the fastest

# `ncu --print-units base` prints every value in one of these units. The factor converts it to SI (seconds).
UNIT_TO_SI = {"ns": 1e-9, "byte": 1, "hz": 1, "%": 1, "inst": 1, "byte/cycle": 1, "inst/cycle": 1}

# ncu says that a unit limits the kernel when the unit is more than 80 % busy.
BUSY_LIMIT = 80.0


def read_report(path):
    """Return one dict per kernel in the report: its name, grid, block, and every value of METRICS in SI units."""
    cmd = ["ncu", "--import", str(path), "--page", "raw", "--csv", "--print-units", "base",
           "--metrics", ",".join(METRICS.values())]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"{path}: `ncu --import` failed:\n{out.stderr or out.stdout}")

    # The CSV has a row of metric names, a row of units, then one row per profiled kernel.
    rows = [row for row in csv.reader(io.StringIO(out.stdout)) if row]
    first = next((i for i, row in enumerate(rows) if "Kernel Name" in row), None)
    if first is None:
        sys.exit(f"{path}: no CSV table in the ncu output:\n{out.stdout[:1000]}")
    names, units, kernel_rows = rows[first], rows[first + 1], rows[first + 2:]
    column = {name: i for i, name in enumerate(names)}

    kernels = []
    for row in kernel_rows:
        kernel = {"report": Path(path).stem,
                  "name": row[column["Kernel Name"]],
                  "grid": row[column["Grid Size"]],
                  "block": row[column["Block Size"]]}
        for key, metric in METRICS.items():
            i = column.get(metric)
            if i is None or row[i] in ("", "n/a"):
                sys.exit(f"{path}: the report has no metric {metric}.\n"
                         "Profile again with script/ncu.sh: it collects the metrics that this script needs.")
            if units[i] not in UNIT_TO_SI:
                sys.exit(f"{path}: unknown unit {units[i]!r} for {metric}. Add it to UNIT_TO_SI.")
            kernel[key] = float(row[i].replace(",", "")) * UNIT_TO_SI[units[i]]
        kernels.append(kernel)
    if not kernels:
        sys.exit(f"{path}: the report has no kernel")
    return kernels


def roofline(k):
    """Return the roofline numbers of one kernel `k` (a dict from read_report). See the module docstring."""
    flop = 2 * k["ffma"] + k["fmul"] + k["fadd"]
    flops = flop / k["time"]
    peak_flops = 2 * k["ffma_per_cycle_max"] * k["sm_hz"]

    levels = {}
    for level in LEVELS:
        nbytes = k[f"{level}_bytes"]
        peak_bw = k[f"{level}_bytes_per_cycle_max"] * k[f"{level}_hz"]
        ai = flop / nbytes if nbytes > 0 else math.inf     # a kernel can run with no byte from DRAM
        roof = min(peak_flops, peak_bw * ai)               # the lower of the compute roof and the memory roof
        levels[level] = {"bytes": nbytes, "ai": ai, "bw": nbytes / k["time"], "peak_bw": peak_bw,
                         "roof": roof, "attained": flops / roof, "busy": k[f"{level}_busy"]}
    return {"flop": flop, "flops": flops, "peak_flops": peak_flops, "levels": levels}


def verdict(k, r):
    """Say in one sentence what limits the kernel. `r` is the result of roofline(k)."""
    level = max(LEVELS, key=lambda lvl: r["levels"][lvl]["busy"])   # the busiest memory level
    L = r["levels"][level]
    if L["busy"] >= BUSY_LIMIT:
        text = f"memory-bound at {level.upper()}: ncu reports it {L['busy']:.0f} % busy."
        if 100 * L["attained"] < L["busy"] / 2:
            text += (f"\n         But FLOP/s is only {100 * L['attained']:.0f} % of the {level.upper()} roof: the level is busy with"
                     " requests, not with bytes.\n         Each request carries few useful bytes. Look for uncoalesced access"
                     " in `Memory Workload Analysis` of the report.")
        else:
            text += f" Raise the arithmetic intensity at {level.upper()}: reuse data from a faster level, or move fewer bytes."
        return text
    if k["sm_busy"] >= BUSY_LIMIT:
        return f"compute-bound: ncu reports the SMs {k['sm_busy']:.0f} % busy. Do fewer operations, or use tensor cores."
    return ("no unit is busy (memory levels and SMs are all below 80 %). The kernel waits: look at `Warp State Statistics`"
            " and `Occupancy` in the report.")


def si(value, unit=""):
    """Format a number with an SI prefix: si(2.62e11, 'FLOP/s') -> '262.0 GFLOP/s'."""
    for factor, prefix in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
        if value >= factor:
            return f"{value / factor:.1f} {prefix}{unit}"
    return f"{value:.1f} {unit}"


def print_report(k, r):
    print(f"report : {k['report']}")
    print(f"kernel : {k['name']}")
    print(f"launch : grid {k['grid']}  block {k['block']}")
    print(f"time   : {k['time'] * 1e3:.3f} ms")
    print(f"FLOP   : {r['flop']:.4g} executed = 2 x {k['ffma']:.4g} FFMA + {k['fmul']:.4g} FMUL + {k['fadd']:.4g} FADD")
    print(f"FLOP/s : {si(r['flops'], 'FLOP/s')} = {100 * r['flops'] / r['peak_flops']:.1f} % of the FP32 peak"
          f" ({si(r['peak_flops'], 'FLOP/s')} = 2 x {k['ffma_per_cycle_max']:.0f} FFMA per cycle x {k['sm_hz'] / 1e9:.2f} GHz)")
    print()
    print(f"{'level':<6}{'bytes':>10}{'AI FLOP/B':>12}{'BW':>14}{'peak BW':>14}{'roof at this AI':>20}{'FLOP/s/roof':>13}{'ncu busy':>10}")
    for level, L in r["levels"].items():
        print(f"{level.upper():<6}{si(L['bytes'], 'B'):>10}{L['ai']:>12.3f}{si(L['bw'], 'B/s'):>14}{si(L['peak_bw'], 'B/s'):>14}"
              f"{si(L['roof'], 'FLOP/s'):>20}{100 * L['attained']:>11.1f} %{L['busy']:>8.1f} %")
    print()
    print(f"verdict: {verdict(k, r)}")
    print()


# Chart colors: one color per memory level, the same in every chart. Text and grid stay gray.
SURFACE, INK, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
LEVEL_COLOR = {"dram": "#2a78d6", "l2": "#eb6834", "l1": "#1baf7a"}
MARKERS = ["o", "s", "^", "D", "v", "P"]   # one marker shape per kernel


def plot(results, png):
    """Draw the roofs and one point per kernel and memory level. `results` is a list of (k, roofline(k))."""
    try:
        import matplotlib
        matplotlib.use("Agg")   # draw to a file: no window
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.ticker import EngFormatter
    except ImportError:
        sys.exit("--png needs matplotlib: pip install matplotlib")

    # The roofs are properties of the GPU, so the first kernel gives the roofs for all of them.
    first = results[0][1]
    peak = first["peak_flops"]
    # Ridge point of a level: the AI where its memory roof meets the compute roof.
    ridges = [peak / first["levels"][level]["peak_bw"] for level in LEVELS]
    points = [L["ai"] for _, r in results for L in r["levels"].values() if math.isfinite(L["ai"])]
    ai = np.geomspace(min(points + ridges) / 5, max(points + ridges) * 5, 300)

    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for level in LEVELS:
        peak_bw = first["levels"][level]["peak_bw"]
        ax.plot(ai, np.minimum(peak, peak_bw * ai), color=LEVEL_COLOR[level], lw=2,
                label=f"{level.upper()} roof: {si(peak_bw, 'B/s')}")
    ax.axhline(peak, color=INK_MUTED, lw=1, ls="--", label=f"FP32 peak: {si(peak, 'FLOP/s')}")

    for i, (k, r) in enumerate(results):
        marker = MARKERS[i % len(MARKERS)]
        name = f"{k['report']}: {k['name'].split('(')[0]}"
        # A level with no bytes has an infinite AI: it has no point on the chart.
        drawn = {level: L["ai"] for level, L in r["levels"].items() if math.isfinite(L["ai"])}
        ax.plot(list(drawn.values()), [r["flops"]] * len(drawn), color=INK_MUTED, lw=1, ls=":", zorder=1)  # one kernel, one height
        for level, x in drawn.items():
            ax.plot(x, r["flops"], marker=marker, ms=10, ls="none", color=LEVEL_COLOR[level],
                    markeredgecolor=SURFACE, markeredgewidth=2)
        ax.plot([], [], marker=marker, ms=8, ls="none", color=INK_MUTED, label=name)   # legend: the shape is the kernel
        if drawn:
            ax.annotate(name, (min(drawn.values()), r["flops"]), xytext=(0, -16), textcoords="offset points",
                        fontsize=8, color=INK)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(ai[0], ai[-1])
    ax.set_ylim(min(r["flops"] for _, r in results) / 5, peak * 3)
    ax.yaxis.set_major_formatter(EngFormatter())
    ax.set_xlabel("arithmetic intensity (FLOP / byte)", color=INK_MUTED)
    ax.set_ylabel("FLOP/s", color=INK_MUTED)
    ax.set_title("Roofline: one point per memory level, color = level, shape = kernel", color=INK, fontsize=11, loc="left")
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9, labelcolor=INK)
    fig.savefig(png, dpi=140, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {png}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reports", nargs="*", type=Path, help="reports from script/ncu.sh (.ncu-repz or .ncu-rep)")
    parser.add_argument("--png", action="store_true", help="also draw <first report>_roofline.png next to the report")
    parser.add_argument("--list-metrics", action="store_true",
                        help="print the ncu metric names this script reads, separated by commas, and exit")
    args = parser.parse_args()

    if args.list_metrics:
        print(",".join(METRICS.values()))
        return
    if not args.reports:
        parser.error("give one report or more")

    results = []
    for path in args.reports:
        for k in read_report(path):
            r = roofline(k)
            print_report(k, r)
            results.append((k, r))
    if args.png:
        first = args.reports[0]
        plot(results, first.with_name(first.stem + "_roofline.png"))


if __name__ == "__main__":
    main()
