#!/usr/bin/env python3
"""The linear layers a vLLM decoder runs, timed against what vLLM calls for them.

GEMM is 57.6% of device time in the Qwen3-8B batch-16 / context-4096 profile
(vllm_profile.py). On gfx1100 every one of those GEMMs reaches
`torch.nn.functional.linear`: the installed vLLM's skinny-GEMM path is gated on
gfx9, so decode GEMVs go there too. The gate is not caution -- that build's
`wvSplitK` is compiled for gfx9 only, and called on gfx1100 it fails a
device-side assert and takes the process with it. The baseline is therefore
torch's default BLAS in the one layout `F.linear` uses -- x @ W.T with both
operands K-contiguous, the "NT" column of RDNA.md's table -- and nothing else.

The shapes come from the model's config.json, not from a list of round
numbers: qkv, o, gate_up and down per layer, plus the LM head. `--m` is the
token count of a step -- the batch size for decode, the summed prompt lengths
for a prefill step.

Each shape is timed the way a decode step reads it: one pass over as many
distinct weights as the model has layers, back to back, under one pair of
events. Not per call over a few rotating copies: that let the 96 MB Infinity
Cache serve most of a 100 MB matrix to upstream's wvSplitK, which then read
"at" 1917 GB/s, 2.2x the card's HBM peak, while hipBLASLt -- too slow to
notice -- measured the same either way. Every variant is checked against an
fp32 reference before it is timed, and variants interleave in one process.

    HIP_VISIBLE_DEVICES=0 python3 tools/hk-bench/vllm_gemm.py --model Qwen/Qwen3-8B
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

#: Measured ceilings on this card (RDNA.md): back-to-back WMMA, and HBM.
PEAK_TFLOPS = 100.5
PEAK_GBS = 864.0

#: The least weight traffic one timed pass may cover, for a model with too
#: few layers to overflow the Infinity Cache on its own. Three times its size.
ROTATE_BYTES = 3 * 96 * 2**20

#: vLLM computes logits for the last token of each sequence only, so the LM
#: head's M is a sequence count, never a prefill's token count.
MAX_SEQS = 256


def model_config(name: str) -> dict:
    """config.json out of the local HF cache -- the pods have no network."""
    if os.path.isfile(name):
        return json.loads(Path(name).read_text())
    root = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    pat = os.path.join(root, "hub", f"models--{name.replace('/', '--')}",
                       "snapshots", "*", "config.json")
    hits = sorted(glob.glob(pat))
    if not hits:
        raise SystemExit(f"no config.json for {name} under {pat}")
    return json.loads(Path(hits[-1]).read_text())


def linear_shapes(cfg: dict, tp: int = 1):
    """(name, N, K, calls per step) for every unquantized linear of a dense
    decoder, sharded the way vLLM shards them at tensor-parallel size `tp`."""
    hid = cfg["hidden_size"]
    heads = cfg["num_attention_heads"]
    kv = cfg.get("num_key_value_heads", heads)
    hd = cfg.get("head_dim", hid // heads)
    inter = cfg["intermediate_size"]
    layers = cfg["num_hidden_layers"]
    vocab = cfg["vocab_size"]
    return [
        ("qkv", (heads + 2 * kv) * hd // tp, hid, layers),
        ("o", hid, heads * hd // tp, layers),
        ("gate_up", 2 * inter // tp, hid, layers),
        ("down", hid, inter // tp, layers),
        ("lm_head", vocab // tp, hid, 1),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--m", default="1,4,16,32,1024,4096,16384")
    ap.add_argument("--shapes", default="",
                    help="comma-separated subset of qkv,o,gate_up,down,lm_head")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--upstream-skinny", default="",
                    help="directory holding a standalone build of upstream "
                         "vLLM's skinny GEMMs (its build.py registers "
                         "torch.ops.upstream_skinny); adds its wvSplitK "
                         "wherever upstream would dispatch it, M <= 5")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F

    import hk
    from hk.ops import gemm as hk_gemm

    dev = torch.device("cuda")
    cus = torch.cuda.get_device_properties(dev).multi_processor_count
    # The enum, not its string: preferred_blas_library() answers with
    # `_BlasBackend.Cublaslt`, which it does not accept back as a name.
    default_blas = torch.backends.cuda.preferred_blas_library()
    print(f"torch {torch.__version__}  hip {torch.version.hip}  "
          f"{torch.cuda.get_device_name(dev)}  {cus} CUs")
    print(f"F.linear default BLAS: {default_blas}\n")

    cfg = model_config(args.model)
    shapes = linear_shapes(cfg, args.tp)
    if args.shapes:
        keep = set(args.shapes.split(","))
        shapes = [s for s in shapes if s[0] in keep]
    ms = [int(x) for x in args.m.split(",")]

    hk_ops = {}

    def hk_op(m, n, k):
        """The torch.ops registration, not the pybind launcher: the op an
        engine would call, on torch's stream. None where the tiling refuses."""
        s = hk_gemm.schedule_for(m, n, k)
        if k % s["k_step"] or n % s["block_n"] or m < s["block_m"]:
            return None
        key = hk_gemm._sched_name({**hk_gemm.DEFAULT, **s})
        if key not in hk_ops:
            hk_ops[key] = hk.torch_op(hk_gemm.TUNER._kernel_for(s))
        return hk_ops[key]

    def set_blas(name):
        torch.backends.cuda.preferred_blas_library(name)

    # Upstream vLLM has since enabled its skinny GEMM on gfx11, for one to
    # five tokens. The installed engine cannot run it, but a newer one would,
    # so a decode kernel has to be measured against it too.
    wvsplitk = None
    if args.upstream_skinny:
        sys.path.insert(0, args.upstream_skinny)
        import build  # noqa: F401,PLC0415 -- registers the ops

        wvsplitk = torch.ops.upstream_skinny.wvSplitK

    results = []
    for sname, n, k, per_step in shapes:
        w_bytes = n * k * 2
        copies = max(per_step, -(-ROTATE_BYTES // w_bytes))
        g = torch.Generator(device=dev).manual_seed(0)
        ws = [torch.randn(n, k, device=dev, dtype=torch.bfloat16, generator=g)
              * k ** -0.5 for _ in range(copies)]
        for m in ms:
            if sname == "lm_head" and m > MAX_SEQS:
                continue
            x = torch.randn(m, k, device=dev, dtype=torch.bfloat16, generator=g)
            out = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
            ref = (x.float() @ ws[0].float().t())
            scale = ref.abs().max().item()

            # name -> (setup, call, teardown). setup/teardown run once per
            # block, outside the timed region, so a library switch is not
            # charged to the GEMM it selects.
            variants = {
                "F.linear": (None, lambda w: F.linear(x, w), None),
                "rocBLAS": (lambda: set_blas("cublas"),
                            lambda w: F.linear(x, w),
                            lambda: set_blas(default_blas)),
            }
            op = hk_op(m, n, k)
            if op is not None:
                variants["hk"] = (None, lambda w: op(x, w, out) or out, None)
            # Upstream's condition, in its names: tokens 0 < n <= 5 and
            # out-features m == 1 or m > 8; every shape here has m > 8.
            if wvsplitk is not None and m <= 5 and k % 8 == 0:
                variants["wvSplitK"] = (
                    None, lambda w: wvsplitk(w, x, None, cus), None)

            # The gate. A variant that is wrong is dropped, loudly, rather
            # than timed: a fast wrong GEMM is not a data point.
            ok = {}
            for name, (setup, call, teardown) in variants.items():
                try:
                    if setup:
                        setup()
                    y = call(ws[0]).float()
                    torch.cuda.synchronize()
                except Exception as ex:  # noqa: BLE001
                    print(f"  {sname} m={m}: {name} raised, dropped: "
                          f"{str(ex).splitlines()[0][:120]}")
                    continue
                finally:
                    if teardown:
                        teardown()
                err = (y - ref).abs().max().item() / max(scale, 1e-30)
                if scale == 0 or not err < 2e-2:
                    print(f"  {sname} m={m}: {name} FAILED the gate "
                          f"(rel err {err:.3g}, ref max {scale:.3g})")
                    continue
                ok[name] = (setup, call, teardown)

            best = {name: float("inf") for name in ok}
            order = list(ok)
            for r in range(args.rounds):
                for name in (order if r % 2 == 0 else order[::-1]):
                    setup, call, teardown = ok[name]
                    if setup:
                        setup()
                    for w in ws[-2:]:                       # settle
                        call(w)
                    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                    s.record()
                    for w in ws:
                        call(w)
                    e.record()
                    torch.cuda.synchronize()
                    best[name] = min(best[name], s.elapsed_time(e) / copies)
                    if teardown:
                        teardown()

            flops = 2.0 * m * n * k
            nbytes = (n * k + m * k + m * n) * 2
            base = best.get("F.linear")
            for name, t in best.items():
                results.append(dict(shape=sname, m=m, n=n, k=k,
                                    per_step=per_step, variant=name, ms=t,
                                    tflops=flops / (t * 1e9),
                                    gbs=nbytes / (t * 1e6),
                                    vs_flinear=(base / t) if base else None))
            del x, out, ref
        del ws
        torch.cuda.empty_cache()

    # Per-shape table.
    print(f"{'shape':>8}{'M':>7}{'N':>7}{'K':>7}  {'variant':<10}"
          f"{'ms':>9}{'TFLOP/s':>9}{'%peak':>7}{'GB/s':>7}{'%bw':>6}"
          f"{'vs F.lin':>9}")
    last = None
    for r in results:
        key = (r["shape"], r["m"])
        if last and key != last:
            print()
        last = key
        print(f"{r['shape']:>8}{r['m']:>7}{r['n']:>7}{r['k']:>7}  "
              f"{r['variant']:<10}{r['ms']:9.4f}{r['tflops']:9.1f}"
              f"{100 * r['tflops'] / PEAK_TFLOPS:6.0f}%{r['gbs']:7.0f}"
              f"{100 * r['gbs'] / PEAK_GBS:5.0f}%"
              f"{r['vs_flinear']:8.2f}x")

    # Per-step totals: what one forward of the whole model costs in GEMMs at
    # each M, with each variant where it applies and F.linear where it does
    # not -- which is exactly what a patch that routes by shape would run.
    print(f"\nGEMM ms per forward step ({cfg['num_hidden_layers']} layers; "
          f"lm_head only where M <= {MAX_SEQS}, i.e. decode)")
    upstream = any(r["variant"] == "wvSplitK" for r in results)
    print(f"{'M':>7}{'F.linear':>11}{'rocBLAS':>10}"
          + (f"{'upstream':>10}" if upstream else "")
          + f"{'best-of':>10}{'route hk':>10}{'saving':>8}")
    for m in ms:
        rows = [r for r in results if r["m"] == m]
        by = {}
        for r in rows:
            by.setdefault(r["shape"], {})[r["variant"]] = r["ms"] * r["per_step"]
        fl = sum(v["F.linear"] for v in by.values() if "F.linear" in v)
        rb = sum(v.get("rocBLAS", v["F.linear"]) for v in by.values())
        # What upstream vLLM runs: wvSplitK wherever it dispatches it,
        # unconditionally, and F.linear elsewhere.
        up = sum(v.get("wvSplitK", v["F.linear"]) for v in by.values())
        bo = sum(min(v.values()) for v in by.values())
        hk_route = sum(min(v["F.linear"], v.get("hk", float("inf")))
                       for v in by.values())
        print(f"{m:>7}{fl:11.3f}{rb:10.3f}"
              + (f"{up:10.3f}" if upstream else "")
              + f"{bo:10.3f}{hk_route:10.3f}{fl / hk_route:7.2f}x")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
