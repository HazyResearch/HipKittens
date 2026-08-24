#!/usr/bin/env python3
"""Build and measure the GEMM ladder, and print the table. Runs on the box it measures.

Every round runs every arm once with the order rotated, so each step's delta is formed within a
round and drift between rounds cancels out. The null control is the top arm built twice under two
names, and the spread between the two bounds how small a delta can be and still mean something.
That bound applies to deltas, not to levels: the control is appended to the arm list and the
rotation is a cyclic shift, so it sits next to the top arm in all but one rotation and what it
measures is the noise between adjacent cells.

Levels depend on the protocol, which is a 512 MiB flush before every iteration, one event pair per
launch, and integer operands from `utils.init_operand`. Change any of the three and the whole table
moves together.

Each arm is timed by its own module's `bench()`, the protocol in `harness.h`. Every arm is checked
once against `torch.matmul` before any timing starts and a failure aborts the campaign there, and
it is the same module object in the same process that then gets timed.
"""

import argparse
import datetime
import importlib
import json
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

# torch has to be imported before any rung module. Both pull in LLVM's option registry, and loading
# a rung's .so first makes the second registration fatal: "Option 'spirv-expand-step' registered
# more than once". The rung modules are imported in `main`, well after this.
import torch

from utils import compare, gemm_reference, init_c, init_operand

# Worst to best. Each rung adds one feature to the rung below, so each step in the table is a paired
# delta against that rung's own parent. This is the same list `utils.RUNGS` gates for correctness.
LADDER = ["00_gemm_naive", "01_gemm_double_buf", "02_gemm_async", "03_gemm_128x128",
          "04_gemm_256x256", "05_gemm_deepk", "06_gemm_segment", "07_gemm_tdm",
          "08_gemm_split_bar", "09_gemm_wgc_multicast", "10_gemm_epilogue",
          "11_gemm_one_wave", "12_gemm_two_waves"]

HERE = Path(__file__).resolve().parent

# `torch.matmul` as an optional arm. It rotates and is timed like any rung, but is reported as a
# baseline rather than a numbered rung, so it has no step over the rung below.
TORCH_ARM = "torch"
WARMUP = 500        # the harness's own warmup, mirrored so the torch arm settles the same way
FLUSH_MB = 512

# 95% two-sided t, by degrees of freedom; 1.96 once n is large enough not to matter.
T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26,
       10: 2.23, 12: 2.18, 15: 2.13, 19: 2.09, 20: 2.09, 25: 2.06, 29: 2.05}

def build(arms, nullctl):
    """One module per arm, through the Makefile so the flags live in one place."""
    targets = [(a, f"{a}.cpp", []) for a in arms]
    # The null control is the top arm's own source under a second name, always recompiled: the
    # artifact carries no record of which rung built it, so make would otherwise keep a `nullctl`
    # built from a different rung.
    if nullctl:
        targets.append(("nullctl", f"{arms[-1]}.cpp", ["-B"]))
    for name, src, force in targets:
        r = subprocess.run(["make", *force, f"KERNEL={name}", f"SRC={src}"],
                           cwd=HERE, capture_output=True, text=True, timeout=3600)
        if r.returncode:
            sys.exit(f"build failed for {name}:\n{(r.stdout + r.stderr).strip()}")


def verify_arms(mods, operands, k):
    """Check every arm against `torch.matmul`, after the build and before any timing.

    One reference for the whole set rather than one fp32 matmul per arm. A rung that fails costs the
    seconds up to here rather than the whole campaign.
    """
    a, b, c = operands
    ref = gemm_reference(a, b)
    for name, mod in mods.items():
        c.fill_(float("nan"))       # a rung that declines to launch must fail, not inherit a pass
        mod.dispatch(a, b, c)
        st = compare(c, ref, k)
        print(f"  verify {name} bad={st['bad']}/{st['n']} "
              f"max_abs_err={st['max_abs_err']:.4f} nonfinite={st['nonfinite']}", file=sys.stderr)
        if st["bad"] or st["nonfinite"]:
            sys.exit(f"{name} failed verification; campaign aborted before timing")
    del ref                         # fp32, so three times an operand; hand it back before timing
    torch.cuda.empty_cache()


def bench_torch(operands, iters):
    """`torch.matmul` under the same protocol the rungs are measured with, so the two are comparable.

    The flush is enqueued ahead of the event that opens each window, so its cost lands between
    windows and every measured iteration starts with none of the previous one's operands cached.
    """
    a, b, _ = operands
    bt = b.t()
    flush = torch.empty(FLUSH_MB * 1024 * 1024 // 4, dtype=torch.float32, device=a.device)

    for _ in range(WARMUP):
        flush.fill_(0.0)
        torch.matmul(a, bt)
    torch.cuda.synchronize()

    beg = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        flush.fill_(0.0)
        beg[i].record()
        torch.matmul(a, bt)
        end[i].record()
    torch.cuda.synchronize()

    ms = [beg[i].elapsed_time(end[i]) for i in range(iters)]
    m, k = a.shape
    n = b.shape[0]
    mean = sum(ms) / iters
    return {"tflops": (2.0 * m * n * k) / (mean * 1e-3) / 1e12, "ms_per_iter": mean,
            "ms_min": min(ms), "ms_max": max(ms), "flush_mb": float(FLUSH_MB), "l2_mb": 0.0}


def run_arm(name, mods, operands, iters):
    """One timed run, kept whole rather than reduced to its mean.

    The record carries `ms_min` and `flush_mb` alongside the mean, which are what tell a slower
    kernel apart from a slower protocol. A failure is returned as a status string, never as a value.
    """
    try:
        if name == TORCH_ARM:
            return bench_torch(operands, iters), "OK"
        return dict(mods[name].bench(*operands, iters)), "OK"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def ci95(xs):
    n = len(xs)
    m = statistics.fmean(xs)
    if n < 2:
        return m, 0.0, m, m
    sd = statistics.stdev(xs)
    half = T95.get(n - 1, 1.96) * sd / (n ** 0.5)
    return m, sd, m - half, m + half


def main():
    started_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("rungs", nargs="*", default=LADDER,
                   help="worst to best; default is the whole ladder")
    p.add_argument("-r", "--rounds", type=int, default=10)
    # The harness always prepends its own 500 warmup iterations, so this sets the measured half only.
    p.add_argument("-i", "--iters", type=int, default=100, help="measured iterations per cell")
    p.add_argument("-c", "--cooldown", type=float, default=5.0,
                   help="seconds left idle after each cell; 0 runs them back to back")
    p.add_argument("-s", "--shape", default="8192 8192 8192")
    p.add_argument("--no-null", action="store_true",
                   help="skip the null control, and with it this campaign's resolution floor")
    p.add_argument("--torch", action="store_true",
                   help="time torch.matmul as a baseline arm, rotated alongside the rungs")
    p.add_argument("--json-out", type=Path,
                   help="write protocol metadata and every raw round to this JSON file")
    a = p.parse_args()

    shape = [int(x) for x in a.shape.split()]
    if len(shape) != 3:
        sys.exit(f"--shape wants three numbers, got {a.shape!r}")
    m, n, k = shape
    if not torch.cuda.is_available():
        sys.exit("no GPU visible to torch")

    arms = list(a.rungs)
    order = arms + ([] if a.no_null else ["nullctl"]) + ([TORCH_ARM] if a.torch else [])
    build(arms, not a.no_null)
    mods = {x: importlib.import_module(x) for x in order if x != TORCH_ARM}

    # One set of operands for the whole campaign, so every arm reads the same buffers at the same
    # addresses and no paired delta can pick up an allocation difference.
    torch.manual_seed(0)
    operands = (init_operand((m, k)), init_operand((n, k)), init_c(m, n))
    verify_arms(mods, operands, k)

    vals = {x: [] for x in order}
    diag = {x: [] for x in order}
    rounds = {}
    dropped = []
    for r in range(a.rounds):
        rot = order[r % len(order):] + order[:r % len(order)]
        rounds[r] = {}
        for arm in rot:
            v, st = run_arm(arm, mods, operands, a.iters)
            if v is None:
                dropped.append((r, arm, st))
            else:
                vals[arm].append(v["tflops"])
                diag[arm].append(v)
                rounds[r][arm] = v["tflops"]
            # A cell ends on a hot card and the next one's warmups are too short to settle on their
            # own, so run back to back a cell's level depends on which arm preceded it. Rotation
            # keeps that out of the paired deltas; idling keeps it out of the absolute numbers too.
            if a.cooldown > 0:
                torch.cuda.synchronize()
                time.sleep(a.cooldown)
        done = sum(len(x) for x in vals.values())
        print(f"  round {r + 1}/{a.rounds}  {done} cells", file=sys.stderr)

    print(f"\n{socket.gethostname()}  {a.shape}  iters={a.iters}  rounds={a.rounds}")
    print(f"correctness: all {len(mods)} arms verified bad=0 against torch.matmul at {a.shape} "
          f"before timing began; the cells below report timing only\n")
    print(f"{'#':>3}  {'rung':<22} {'TFLOP/s':>9} {'best':>9} {'sd':>7} {'n':>3}   "
          f"adds over the rung below")
    flop = 2.0 * m * n * k
    prev = None
    summaries = {}
    for i, arm in enumerate(arms):
        if not vals[arm]:
            print(f"{i:>3}  {arm:<22} {'-':>9} {'-':>9} {'-':>7} {0:>3}   NO CELLS")
            prev = arm
            continue
        mean, sd, _, _ = ci95(vals[arm])
        # The campaign's fastest single iteration. Cold operands or a drooped clock pull the mean
        # and `best` down together; a tenant arriving mid-campaign pulls down only the mean.
        ms_lo = min(c["ms_min"] for c in diag[arm])
        ms_hi = max(c["ms_max"] for c in diag[arm])
        best = flop / (ms_lo * 1e-3) / 1e12
        step = ""
        step_stats = None
        if prev:
            d = [100 * (rounds[r][arm] - rounds[r][prev]) / rounds[r][prev]
                 for r in rounds if arm in rounds[r] and prev in rounds[r]]
            if len(d) > 1:
                dm, dsd, lo, hi = ci95(d)
                step = f"{dm:+.2f}% [{lo:+.2f}, {hi:+.2f}] n={len(d)}"
                step_stats = {"mean_pct": dm, "sd_pct": dsd, "ci95_low_pct": lo,
                              "ci95_high_pct": hi, "n": len(d)}
        summaries[arm] = {"mean_tflops": mean, "sd_tflops": sd, "n": len(vals[arm]),
                          "best_iter_tflops": best, "ms_min": ms_lo, "ms_max": ms_hi,
                          "flush_mb": min(c["flush_mb"] for c in diag[arm]),
                          "step_over_parent": step_stats}
        print(f"{i:>3}  {arm:<22} {mean:>9.1f} {best:>9.1f} {100 * sd / mean:>6.2f}% "
              f"{len(vals[arm]):>3}   {step}")
        prev = arm

    # `cache_flusher::init` sizes itself against free VRAM and runs as a no-op if it cannot
    # allocate. Printing the size it settled on is what makes an unflushed cell show up as a missing
    # flush rather than as a fast kernel.
    flushes = sorted({round(c["flush_mb"]) for x in order for c in diag[x]})
    l2 = next((c["l2_mb"] for x in order for c in diag[x] if c["l2_mb"]), 0.0)
    if not flushes or flushes[0] == 0:
        print(f"\nWARNING: a cell ran with no cache flush (flush_mb {flushes}); its operands were "
              f"warm and its number does not belong in the same column as the rest")
    else:
        print(f"\nprotocol: {flushes[0] if len(flushes) == 1 else flushes} MB flush per iteration, "
              f"L2 {l2:.0f} MB, one event pair per launch, {WARMUP} warmup + {a.iters} measured, "
              f"{a.cooldown:g}s between cells, integer operands in [-3, 3]")

    if a.torch and vals[TORCH_ARM]:
        tm, tsd, _, _ = ci95(vals[TORCH_ARM])
        top, _, _, _ = ci95(vals[arms[-1]])
        print(f"\ntorch.matmul baseline: {tm:.1f} TFLOP/s sd {100 * tsd / tm:.2f}% "
              f"n={len(vals[TORCH_ARM])}  ->  {arms[-1]} is {top / tm:.1f}x it")

    null_stats = None
    if not a.no_null and vals["nullctl"]:
        d = [100 * (rounds[r]["nullctl"] - rounds[r][arms[-1]]) / rounds[r][arms[-1]]
             for r in rounds if "nullctl" in rounds[r] and arms[-1] in rounds[r]]
        if len(d) > 1:
            dm, dsd, _, _ = ci95(d)
            null_stats = {"mean_delta_pct": dm, "sd_pct": dsd, "n": len(d),
                          "resolution_pct": abs(dm) + 2 * dsd}
            print(f"\nnull control ({arms[-1]} built twice): {dm:+.3f}% sd {dsd:.3f} n={len(d)}"
                  f"  ->  resolution {abs(dm) + 2 * dsd:.2f}%")

    if dropped:
        print(f"\n{len(dropped)} cell(s) dropped:")
        for r, arm, st in dropped:
            print(f"  round {r + 1} {arm}: {st}")

    if a.json_out:
        payload = {
            "host": socket.gethostname(),
            "started_utc": started_utc,
            "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "shape": {"m": m, "n": n, "k": k},
            "protocol": {"warmups_per_cell": WARMUP, "measured_iters_per_cell": a.iters,
                         "rounds": a.rounds, "order_rotation": True,
                         "cache_flush_mb": FLUSH_MB, "seed": 0,
                         "event_pairs": "per-launch",
                         "cache_flush_mb_observed": flushes,
                         "cooldown_s": a.cooldown,
                         "operands": "integer uniform [-3, 3] cast to bf16",
                         "correctness_gate": "torch.matmul before timing"},
            "device": {"gcn_arch": torch.cuda.get_device_properties(0).gcnArchName,
                       "torch": torch.__version__},
            "arms": arms,
            "raw_rounds_tflops": rounds,
            "raw_cells": diag,
            "summaries": summaries,
            "null_control": null_stats,
            "dropped": [{"round": r, "arm": arm, "status": st} for r, arm, st in dropped],
        }
        a.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote raw results: {a.json_out}")


if __name__ == "__main__":
    main()
