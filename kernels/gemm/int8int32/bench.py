#!/usr/bin/env python3
"""
Bench the INT8 GEMM in sim/HipKittens and plot TOPS vs problem size.

For each size in SIZES, this script edits `#define SIZE` in
sim/HipKittens/.../INT8_4wave/4_wave.cu, rebuilds (`make` with ROCM_PATH=/opt/rocm),
runs `./tk_kernel`, and parses the "Kernel time (best): ... TOPS: <x>" line.
Plot is written to int8_gemm_tops.png next to this script.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Match analysis/*/plot.py palette; HipKittens series uses colors[3].
COLORS = ["#8E69B8", "#E59952", "#68AC5A", "#7CB9BC", "#DE836B", "#55555A"]

SIZES = [1024, 2048, 4096, 8192, 16384]

# HipblasLT baseline TOPS for INT8 GEMM (M=N=K). Fill in measured values; leave
# None to skip that bar. The plot auto-detects which entries have data.
BASELINES = {
    "hipblaslt": {
        "1024":  200.699,
        "2048":  699.506,
        "4096":  1532.21,
        "8192":  2066.05,
        "16384": 2109.17,
    },
}

BASELINE_LABELS = {"hipblaslt": "HipblasLT"}
BASELINE_COLORS = {"hipblaslt": COLORS[1]}

# Resolve everything relative to this script so the bench is self-contained.
SCRIPT_DIR = Path(__file__).resolve().parent
KERNEL_DIR = SCRIPT_DIR / "INT8_4wave"
SRC = KERNEL_DIR / "4_wave.cu"
OUT_PNG = SCRIPT_DIR / "int8_gemm_tops.png"
ROCM_PATH = os.environ.get("ROCM_PATH", "/opt/rocm")

TOPS_RE = re.compile(r"Kernel time \(best\): ([\d.]+) ms,\s+TOPS: ([\d.]+)")


def set_size(size: int) -> None:
    text = SRC.read_text()
    new_text = re.sub(r"#define SIZE \d+", f"#define SIZE {size}", text)
    if new_text == text:
        raise RuntimeError(f"failed to update SIZE in {SRC}")
    SRC.write_text(new_text)


def build_and_run(size: int) -> tuple[float, float]:
    set_size(size)
    env = {**os.environ, "ROCM_PATH": ROCM_PATH}
    # Clean to force a recompile of 4_wave.o (its content is the only thing that changed).
    subprocess.run(["make", "clean"], cwd=KERNEL_DIR, env=env, check=True, capture_output=True)
    build = subprocess.run(["make"], cwd=KERNEL_DIR, env=env, capture_output=True, text=True)
    if build.returncode != 0:
        print(build.stdout, build.stderr, sep="\n", file=sys.stderr)
        raise RuntimeError(f"build failed for SIZE={size}")
    run = subprocess.run(["./tk_kernel"], cwd=KERNEL_DIR, capture_output=True, text=True, check=True)
    m = TOPS_RE.search(run.stdout)
    if not m:
        print(run.stdout, file=sys.stderr)
        raise RuntimeError(f"could not parse TOPS for SIZE={size}")
    return float(m.group(1)), float(m.group(2))


def main() -> None:
    print(f"{'size':>6}  {'best ms':>8}  {'TOPS':>8}")
    results: list[tuple[int, float, float]] = []
    for size in SIZES:
        ms, tops = build_and_run(size)
        results.append((size, ms, tops))
        print(f"{size:>6}  {ms:>8.3f}  {tops:>8.2f}")

    sizes = [r[0] for r in results]
    tops_vals = [r[2] for r in results]

    # Collect any baselines with at least one filled-in value; HipKittens is always last.
    series: list[tuple[str, str, list[float]]] = []  # (label, color, values_per_size)
    for name, table in BASELINES.items():
        vals = [table.get(str(s)) for s in sizes]
        if any(v is not None for v in vals):
            series.append((BASELINE_LABELS.get(name, name),
                           BASELINE_COLORS.get(name, COLORS[5]),
                           [v if v is not None else 0.0 for v in vals]))
    series.append(("HipKittens", COLORS[3], tops_vals))

    # Style matched to HipKittens/analysis/*/plot.py (bf16_gemm, fp8_gemm, ...).
    x = np.arange(len(sizes))
    n_series = len(series)
    width = 0.4
    max_tops = max(max(vals) for _, _, vals in series)

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_groups = []
    for i, (label, color, vals) in enumerate(series):
        # Center the group around each x: offset of (i - (n-1)/2) * width.
        offset = (i - (n_series - 1) / 2.0) * width
        bars = ax.bar(x + offset, vals, width, label=label, color=color)
        bar_groups.append(bars)
        for bar, value in zip(bars, vals):
            if value > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2.0, height + max_tops * 0.01,
                        f"{value:.0f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylim(0, max_tops * 1.15)
    ax.set_xlabel("Matrix Size (N×N)", fontsize=16)
    ax.set_ylabel("Performance (TOPS)", fontsize=16)
    ax.set_title("INT8 GEMM Performance MI355X", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sizes], fontsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.legend(fontsize=14)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"\nsaved {OUT_PNG}")


if __name__ == "__main__":
    main()
