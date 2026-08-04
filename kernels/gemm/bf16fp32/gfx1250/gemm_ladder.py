#!/usr/bin/env python3
"""Build and measure the GEMM ladder on a remote gfx1250 box, and print the table.

Every round runs every rung once with the arm order rotated, so each step's delta is formed
within a round and drift between rounds cancels. The null control is one rung built twice under
two names; the spread between those two is the floor below which a delta means nothing.
"""

import argparse
import re
import statistics
import subprocess
import sys

LADDER = ["gemm_naive", "gemm_double_buf", "gemm_async", "gemm_128x128", "gemm_256x256",
          "gemm_deepk", "gemm_segment", "gemm_tdm", "gemm_split_bar", "gemm_wgc_multicast"]

REMOTE = "HipKittens-upstream"
BIN = "/tmp/ladder"
HIPCC = ("hipcc -DKITTENS_UDNA1 --offload-arch=gfx1250 -std=c++20 -w -O3 -fopenmp -DHARNESS_MAIN "
         f"-I$HOME/{REMOTE}/include -I/opt/rocm/include/hip")

# 95% two-sided t, by degrees of freedom; 1.96 once n is large enough not to matter.
T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26,
       10: 2.23, 12: 2.18, 15: 2.13, 19: 2.09, 20: 2.09, 25: 2.06, 29: 2.05}


class Box:
    def __init__(self, host, user, key, gpu):
        self.pre = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
                    "-o", "StrictHostKeyChecking=no", "-o", "IdentitiesOnly=yes",
                    "-i", key, f"{user}@{host}"]
        self.env = (f"export PATH=/opt/rocm/bin:$PATH "
                    f"LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH HIP_VISIBLE_DEVICES={gpu}; ")
        self.host, self.user, self.key, self.gpu = host, user, key, gpu

    def run(self, cmd, timeout=900):
        r = subprocess.run(self.pre + [self.env + cmd], capture_output=True, text=True,
                           timeout=timeout)
        return r.returncode, r.stdout + r.stderr

    def sync(self, local_root):
        rsh = " ".join(self.pre[:-1])
        for sub in ("include/", "kernels/gemm/bf16fp32/gfx1250/"):
            r = subprocess.run(["rsync", "-az", "--exclude", "archive/", "-e", rsh,
                                f"{local_root}/{sub}", f"{self.user}@{self.host}:{REMOTE}/{sub}"],
                               capture_output=True, text=True)
            if r.returncode:
                sys.exit(f"sync failed: {r.stderr.strip()}")

    def vram_mb(self):
        rc, out = self.run(f"amd-smi metric -g {self.gpu} -m | awk '/USED_VRAM/{{print $2}}'", 120)
        m = re.search(r"^\s*(\d+)\s*$", out, re.M)
        return int(m.group(1)) if rc == 0 and m else None


def build(box, arms, nullctl):
    src = f"$HOME/{REMOTE}/kernels/gemm/bf16fp32/gfx1250"
    cmds = [f"mkdir -p {BIN}"]
    for a in arms:
        cmds.append(f"rm -f {BIN}/{a}; {HIPCC} {src}/{a}.cpp -o {BIN}/{a}")
    if nullctl:
        cmds.append(f"rm -f {BIN}/nullctl; {HIPCC} {src}/{arms[-1]}.cpp -o {BIN}/nullctl")
    box.run(" ; ".join(cmds), timeout=1800)
    rc, out = box.run(f"ls {BIN}")
    have = set(out.split())
    missing = [a for a in arms + (["nullctl"] if nullctl else []) if a not in have]
    if missing:
        sys.exit(f"did not build: {' '.join(missing)}")


def cell(box, arm, shape, iters, floor):
    """One timed, verified run. An absent number is reported as itself, never as a value."""
    vram = box.vram_mb()
    if vram is None:
        return None, "VRAM-UNREADABLE"
    if vram >= floor:
        return None, f"FOREIGN-VRAM-{vram}MB"
    rc, out = box.run(f"{BIN}/{arm} {shape} {iters} 1")
    if rc != 0:
        return None, f"EXIT-{rc}"
    tf = re.search(r"([\d.]+) GFLOP/s", out)
    bad = re.search(r"bad=(\d+)", out)
    if not tf:
        return None, "RAN-BUT-NO-GFLOPS-LINE"
    if not bad:
        return None, "RAN-BUT-NO-BAD-LINE"
    if bad.group(1) != "0":
        return None, f"VERIFY-FAIL-bad={bad.group(1)}"
    return float(tf.group(1)) / 1000.0, "OK"


def ci95(xs):
    n = len(xs)
    m = statistics.fmean(xs)
    if n < 2:
        return m, 0.0, m, m
    sd = statistics.stdev(xs)
    t = T95.get(n - 1, 1.96)
    half = t * sd / (n ** 0.5)
    return m, sd, m - half, m + half


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("rungs", nargs="*", default=LADDER, help="worst to best; default is the ladder")
    p.add_argument("--host", required=True)
    p.add_argument("--user", default="muhosama")
    p.add_argument("--key", default="~/.ssh/id_ecdsa")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("-r", "--rounds", type=int, default=10)
    p.add_argument("-i", "--iters", type=int, default=10)
    p.add_argument("-s", "--shape", default="8192 8192 8192")
    p.add_argument("--vram-floor", type=int, default=1024, help="MB; above this a cell is dropped")
    p.add_argument("--no-null", action="store_true")
    # Reuse binaries already on the box. On a contended GPU the sync+rebuild is dead time in
    # which another tenant can take the card, so skip it when the binaries are known current.
    p.add_argument("--no-build", action="store_true")
    p.add_argument("--root", default="/home/neoblizz/HipKittens-upstream")
    a = p.parse_args()

    arms = list(a.rungs)
    box = Box(a.host, a.user, a.key.replace("~", subprocess.os.path.expanduser("~")), a.gpu)
    if a.no_build:
        rc, out = box.run(f"ls {BIN}")
        have = set(out.split())
        stale = [x for x in arms + ([] if a.no_null else ["nullctl"]) if x not in have]
        if stale:
            sys.exit(f"--no-build but not on the box: {' '.join(stale)}")
    else:
        box.sync(a.root)
        build(box, arms, not a.no_null)

    order = arms + ([] if a.no_null else ["nullctl"])
    vals = {k: [] for k in order}
    rounds = {}
    bad_cells = []
    for r in range(a.rounds):
        rot = order[r % len(order):] + order[:r % len(order)]
        rounds[r] = {}
        for arm in rot:
            v, st = cell(box, arm, a.shape, a.iters, a.vram_floor)
            if v is None:
                bad_cells.append((r, arm, st))
            else:
                vals[arm].append(v)
                rounds[r][arm] = v
        done = sum(len(x) for x in vals.values())
        print(f"  round {r + 1}/{a.rounds}  {done} cells", file=sys.stderr)

    print(f"\n{a.host}  {a.shape}  iters={a.iters}  rounds={a.rounds}\n")
    print(f"{'#':>3}  {'rung':<17} {'TFLOP/s':>9} {'sd':>7} {'n':>3}   adds over the rung below")
    prev = None
    for i, arm in enumerate(arms, 1):
        if not vals[arm]:
            print(f"{i:>3}  {arm:<17} {'-':>9} {'-':>7} {0:>3}   NO CELLS")
            prev = arm
            continue
        m, sd, _, _ = ci95(vals[arm])
        step = ""
        if prev:
            d = [100 * (rounds[r][arm] - rounds[r][prev]) / rounds[r][prev]
                 for r in rounds if arm in rounds[r] and prev in rounds[r]]
            if len(d) > 1:
                dm, dsd, lo, hi = ci95(d)
                step = f"{dm:+.2f}% [{lo:+.2f}, {hi:+.2f}] n={len(d)}"
        print(f"{i:>3}  {arm:<17} {m:>9.1f} {100 * sd / m:>6.2f}% {len(vals[arm]):>3}   {step}")
        prev = arm

    if not a.no_null and vals["nullctl"]:
        d = [100 * (rounds[r]["nullctl"] - rounds[r][arms[-1]]) / rounds[r][arms[-1]]
             for r in rounds if "nullctl" in rounds[r] and arms[-1] in rounds[r]]
        if len(d) > 1:
            dm, dsd, _, _ = ci95(d)
            print(f"\nnull control ({arms[-1]} built twice): {dm:+.3f}% sd {dsd:.3f} n={len(d)}"
                  f"  ->  resolution {abs(dm) + 2 * dsd:.2f}%")

    if bad_cells:
        print(f"\n{len(bad_cells)} cell(s) dropped:")
        for r, arm, st in bad_cells:
            print(f"  round {r + 1} {arm}: {st}")


if __name__ == "__main__":
    main()
