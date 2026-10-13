#!/usr/bin/env python3
"""Which kernel each linear layer of a model should run on, at every M a step
can have -- measured, and written as the routing table vllm_gemm reads.

The rule this replaces was four lines transcribed from seven Ms. The GEMM's
speed is not monotone in M: the last 128-row block is computed whole however
little of it is kept, so between multiples of 128 the kernel wastes work in a
sawtooth, and at small grids two workgroups share a WGP. A rule written as
thresholds, or bucketed by powers of two, either routes the teeth or gives up
the gaps between them. So this measures the grid itself: every M up to 64,
every 16 rows to 1024, coarser above, where the wasted rows are at most an
eighth of the work.

At each (N, K) of the model and each M: the engine's own path (`F.linear`,
which is what vLLM's linear reaches on gfx1100), the generated GEMM, and the
decode GEMV where it takes the shape. Each is checked against fp32 before it
is timed; they interleave in one process, min over rounds, the weights
rotated through more than three times the Infinity Cache. A kernel takes an M
only if it beat the engine there by `--min-gain`; a tie stays the engine's.

    HIP_VISIBLE_DEVICES=3 python3 tools/hk-bench/route_tune.py \\
        --model Qwen/Qwen3-8B --out python/hk/tuned/routes_vllm_linear_gfx1100.json

The table is per arch and per model: a (N, K) it has not measured takes the
engine's path, so another model needs its own run (a few minutes) before
anything of it is routed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "python"))
sys.path.insert(0, str(HERE))

from vllm_gemm import MAX_SEQS, ROTATE_BYTES, linear_shapes, model_config  # noqa: E402

GRID = (list(range(1, 65)) + list(range(80, 1025, 16)) +
        list(range(1088, 2049, 64)) + list(range(2176, 4097, 128)) +
        list(range(4608, 8193, 512)) + [12288, 16384])


def runs(points, min_gain):
    """[[lo, hi, path], ...] over the measured Ms, consecutive equal winners
    merged, and the path at the largest M (what the table answers above it)."""
    out = []
    for m, t in points:
        eng = t["engine"]
        cand = {p: v for p, v in t.items() if p != "engine"}
        best = min(cand, key=cand.get) if cand else None
        path = best if best and cand[best] * (1 + min_gain) <= eng else "fallback"
        if out and out[-1][2] == path:
            out[-1][1] = m
        else:
            out.append([m, m, path])
    return out, (out[-1][2] if out else "fallback")


def dump(table: dict) -> str:
    """JSON with one line per run and per measured M, so a diff of two tables
    reads as a diff of decisions."""
    head = {k: v for k, v in table.items() if k != "shapes"}
    lines = ["{"] + [f" {json.dumps(k)}: {json.dumps(v)}," for k, v in head.items()]
    lines.append(' "shapes": {')
    keys = list(table["shapes"])
    for i, key in enumerate(keys):
        s = table["shapes"][key]
        lines.append(f'  {json.dumps(key)}: {{"name": {json.dumps(s["name"])}, '
                     f'"above": {json.dumps(s["above"])},')
        lines.append('   "runs": [')
        lines += [f"    {json.dumps(r)}," for r in s["runs"]]
        lines[-1] = lines[-1].rstrip(",")
        lines.append('   ], "points": [')
        lines += [f"    {json.dumps(p)}," for p in s["points"]]
        lines[-1] = lines[-1].rstrip(",")
        lines.append("   ]}" + ("," if i < len(keys) - 1 else ""))
    lines += [" }", "}"]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="Qwen/Qwen3-8B",
                    help="HF name, or a path to its config.json")
    ap.add_argument("--name", default="", help="what the table says it was "
                    "measured for (default: --model)")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--shapes", default="",
                    help="comma-separated subset of qkv,o,gate_up,down,lm_head")
    ap.add_argument("--m", default="", help="comma-separated Ms (default: the grid)")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--min-gain", type=float, default=0.03)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F

    import hk
    from hk.ops import decode_gemm as hk_decode
    from hk.ops import gemm as hk_gemm
    from hk.runtime.compile import toolchain_fingerprint

    dev = torch.device("cuda")
    arch = (torch.cuda.get_device_properties(dev).gcnArchName or "").split(":")[0]
    cfg = model_config(args.model)
    shapes = linear_shapes(cfg, args.tp)
    if args.shapes:
        keep = set(args.shapes.split(","))
        shapes = [s for s in shapes if s[0] in keep]
    grid = [int(x) for x in args.m.split(",")] if args.m else GRID

    ops = {}

    def gemm_op(m, n, k):
        s = hk_gemm.schedule_for(m, n, k)
        if k % s["k_step"] or n % s["block_n"]:
            return None
        key = hk_gemm._sched_name({**hk_gemm.DEFAULT, **s})
        if key not in ops:
            ops[key] = hk.torch_op(hk_gemm.TUNER._kernel_for(s))
        return ops[key]

    def decode_op(m, n, k):
        if not hk_decode.fits(m, n, k):
            return None
        s = hk_decode.schedule_for(m, k)
        key = ("decode",) + tuple(sorted(s.items()))
        if key not in ops:
            ops[key] = hk.torch_op(hk_decode.kernel_for(**s))
        return ops[key]

    table = dict(kind="vllm_linear", arch=arch, model=args.name or args.model,
                 tp=args.tp, device=torch.cuda.get_device_name(dev),
                 torch=torch.__version__, hip=torch.version.hip,
                 blas=str(torch.backends.cuda.preferred_blas_library()),
                 toolchain=toolchain_fingerprint(), min_gain=args.min_gain,
                 rounds=args.rounds, date=time.strftime("%Y-%m-%d"), shapes={})
    t0 = time.time()
    for sname, n, k, _ in shapes:
        w_bytes = n * k * 2
        copies = max(2, -(-ROTATE_BYTES // w_bytes))
        g = torch.Generator(device=dev).manual_seed(0)
        ws = [torch.randn(n, k, device=dev, dtype=torch.bfloat16, generator=g)
              * k ** -0.5 for _ in range(copies)]
        points = []
        for m in grid:
            if sname == "lm_head" and m > MAX_SEQS:
                continue
            x = torch.randn(m, k, device=dev, dtype=torch.bfloat16, generator=g)
            out = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
            ref = x.float() @ ws[0].float().t()
            scale = ref.abs().max().item()
            variants = {"engine": lambda w: F.linear(x, w)}
            op = gemm_op(m, n, k)
            if op is not None:
                variants["hk"] = lambda w, op=op: op(x, w, out) or out
            dop = decode_op(m, n, k)
            if dop is not None:
                variants["hk_decode"] = lambda w, dop=dop: dop(x, w, out) or out
            ok = {}
            for name, call in variants.items():
                y = call(ws[0]).float()
                err = (y - ref).abs().max().item() / max(scale, 1e-30)
                if name != "engine" and not err < 2e-2:
                    print(f"  {sname} m={m}: {name} FAILED the gate "
                          f"(rel err {err:.3g}); not a candidate", flush=True)
                    continue
                ok[name] = call
            best = {name: float("inf") for name in ok}
            order = list(ok)
            for r in range(args.rounds):
                for name in (order if r % 2 == 0 else order[::-1]):
                    call = ok[name]
                    call(ws[-1])
                    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                    s.record()
                    for w in ws:
                        call(w)
                    e.record()
                    torch.cuda.synchronize()
                    best[name] = min(best[name], s.elapsed_time(e) / copies)
            points.append([m, {p: round(v, 5) for p, v in best.items()}])
            del x, out, ref
        rs, above = runs(points, args.min_gain)
        table["shapes"][f"{n}x{k}"] = dict(name=sname, runs=rs, above=above,
                                           points=points)
        print(f"{sname} {n}x{k}: {len(points)} Ms, {time.time() - t0:.0f}s", flush=True)
        for lo, hi, path in rs:
            print(f"    {lo:>6}..{hi:<6} {path}", flush=True)
        del ws
        torch.cuda.empty_cache()

    Path(args.out).write_text(dump(table))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
