#!/usr/bin/env python3
"""Build and measure the GEMM ladder, and print the table. Runs on the box it measures.

Every round runs every arm once with the order rotated, so each step's delta is formed within a
round and drift between rounds cancels. The null control is the top arm built twice under two
names; the spread between those two is the floor below which a delta means nothing.

Each arm is timed by its own module's `bench()`, which is the protocol in `harness.h`. Correctness
is a property of the binary, not of a timing sample, so every arm is checked once against
`torch.matmul` before any timing starts and a failure aborts the campaign there -- and it is the
same module object in the same process that then gets timed.
"""

import argparse
import importlib
import socket
import statistics
import subprocess
import sys
from pathlib import Path

# torch has to be imported before any rung module. Both pull in LLVM's option registry, and loading
# a rung's .so first makes the second registration fatal: "Option 'spirv-expand-step' registered
# more than once". The rung modules are imported in `main`, well after this.
import torch

from utils import compare, gemm_reference, init_c, init_uniform

# Worst to best, and every kernel on disk: the ladder is a single chain in which each rung adds one
# feature to the rung below, ending at the fastest kernel. So each step in the table below is a
# paired delta over that rung's own parent.
LADDER = ["gemm_naive", "gemm_double_buf", "gemm_async", "gemm_128x128", "gemm_256x256",
          "gemm_deepk", "gemm_segment", "gemm_tdm", "gemm_split_bar",
          "gemm_wgc_cluster", "gemm_epilogue_nomc", "gemm_one_wave_nomc",
          "gemm_wgc_multicast", "gemm_epilogue", "gemm_one_wave"]

HERE = Path(__file__).resolve().parent

# `torch.matmul` as an optional arm. It rotates and is timed like any rung so the comparison is
# in-session and interleaved, but it is reported as a baseline rather than a numbered rung: it has
# no step over the rung below because it is not on the ladder.
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
    # artifact carries no record of which rung built it, so make would find an existing `nullctl`
    # newer than a newly chosen top arm and keep a control built from something else.
    if nullctl:
        targets.append(("nullctl", f"{arms[-1]}.cpp", ["-B"]))
    for name, src, force in targets:
        r = subprocess.run(["make", *force, f"KERNEL={name}", f"SRC={src}"],
                           cwd=HERE, capture_output=True, text=True, timeout=3600)
        if r.returncode:
            sys.exit(f"build failed for {name}:\n{(r.stdout + r.stderr).strip()}")


def verify_arms(mods, operands, k):
    """Check every arm against `torch.matmul`, after the build and before any timing.

    One reference for the whole set, so the comparison is against one fp32 matmul rather than one
    per arm. A rung that fails costs the seconds up to here rather than the whole campaign.
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
    """`torch.matmul` under the protocol the rungs are measured with, so the two are comparable.

    The flush is enqueued ahead of the event that opens each window, so its cost lands between
    windows and every measured iteration starts from a cache holding none of the previous one's
    operands. Comparing a flushed kernel against an unflushed baseline compares cache states.
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

    ms = sum(beg[i].elapsed_time(end[i]) for i in range(iters)) / iters
    m, k = a.shape
    n = b.shape[0]
    return (2.0 * m * n * k) / (ms * 1e-3) / 1e12


def run_arm(name, mods, operands, iters):
    """One timed run. A failure is reported as itself, never as a value."""
    try:
        if name == TORCH_ARM:
            return bench_torch(operands, iters), "OK"
        return mods[name].bench(*operands, iters)["tflops"], "OK"
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
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("rungs", nargs="*", default=LADDER,
                   help="worst to best; default is the whole ladder")
    p.add_argument("-r", "--rounds", type=int, default=10)
    # The harness prepends its fixed 500 warmup iterations to every cell whatever this says, so
    # this is the second half of the 500/100 protocol only.
    p.add_argument("-i", "--iters", type=int, default=100, help="measured iterations per cell")
    p.add_argument("-s", "--shape", default="8192 8192 8192")
    p.add_argument("--no-null", action="store_true",
                   help="skip the null control, and with it this campaign's resolution floor")
    p.add_argument("--torch", action="store_true",
                   help="time torch.matmul as a baseline arm, rotated alongside the rungs")
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
    # addresses and a paired delta cannot pick up an allocation difference.
    torch.manual_seed(0)
    operands = (init_uniform((m, k)), init_uniform((n, k)), init_c(m, n))
    verify_arms(mods, operands, k)

    vals = {x: [] for x in order}
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
                vals[arm].append(v)
                rounds[r][arm] = v
        done = sum(len(x) for x in vals.values())
        print(f"  round {r + 1}/{a.rounds}  {done} cells", file=sys.stderr)

    print(f"\n{socket.gethostname()}  {a.shape}  iters={a.iters}  rounds={a.rounds}")
    print(f"correctness: all {len(mods)} arms verified bad=0 against torch.matmul at {a.shape} "
          f"before timing began; the cells below report timing only\n")
    print(f"{'#':>3}  {'rung':<17} {'TFLOP/s':>9} {'sd':>7} {'n':>3}   adds over the rung below")
    prev = None
    for i, arm in enumerate(arms, 1):
        if not vals[arm]:
            print(f"{i:>3}  {arm:<17} {'-':>9} {'-':>7} {0:>3}   NO CELLS")
            prev = arm
            continue
        mean, sd, _, _ = ci95(vals[arm])
        step = ""
        if prev:
            d = [100 * (rounds[r][arm] - rounds[r][prev]) / rounds[r][prev]
                 for r in rounds if arm in rounds[r] and prev in rounds[r]]
            if len(d) > 1:
                dm, dsd, lo, hi = ci95(d)
                step = f"{dm:+.2f}% [{lo:+.2f}, {hi:+.2f}] n={len(d)}"
        print(f"{i:>3}  {arm:<17} {mean:>9.1f} {100 * sd / mean:>6.2f}% {len(vals[arm]):>3}   "
              f"{step}")
        prev = arm

    if a.torch and vals[TORCH_ARM]:
        tm, tsd, _, _ = ci95(vals[TORCH_ARM])
        top, _, _, _ = ci95(vals[arms[-1]])
        print(f"\ntorch.matmul baseline: {tm:.1f} TFLOP/s sd {100 * tsd / tm:.2f}% "
              f"n={len(vals[TORCH_ARM])}  ->  {arms[-1]} is {top / tm:.1f}x it")

    if not a.no_null and vals["nullctl"]:
        d = [100 * (rounds[r]["nullctl"] - rounds[r][arms[-1]]) / rounds[r][arms[-1]]
             for r in rounds if "nullctl" in rounds[r] and arms[-1] in rounds[r]]
        if len(d) > 1:
            dm, dsd, _, _ = ci95(d)
            print(f"\nnull control ({arms[-1]} built twice): {dm:+.3f}% sd {dsd:.3f} n={len(d)}"
                  f"  ->  resolution {abs(dm) + 2 * dsd:.2f}%")

    if dropped:
        print(f"\n{len(dropped)} cell(s) dropped:")
        for r, arm, st in dropped:
            print(f"  round {r + 1} {arm}: {st}")


if __name__ == "__main__":
    main()
