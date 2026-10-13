#!/usr/bin/env python3
"""End to end against the unmodified engine. Baseline 1.0x.

This is the number the IR exists to move: take a serving engine as it ships,
replace the kernels it has no good one for, and see whether the *request* got
faster. Not the kernel, not the decode step -- the request.

`vllm_sweep.py` deliberately cancels prefill to isolate a decode step, which
was the right tool for asking "is the paged kernel any good" and the wrong one
for asking "is the patch worth installing". A profile of this exact workload
(tools/hk-bench/vllm_profile.py, Qwen3-8B, batch 16, ctx 4096) says where the
time is:

    57.6%  hipBLASLt GEMM
    39.1%  Triton prefill attention
     2.3%  silu-and-mul, RMSNorm, RoPE, cache writes, everything else

-- so a patch that only touches decode attention is arguing over the last
percent, and the dense prefill kernel is sitting in front of 39%.

Arms alternate and every point reports the min of `--reps`, because two blocks
of runs on this chip are not comparable.

    python3 tools/hk-bench/vllm_e2e.py --context 4096 --batch 16 --tokens 72 \
        --arms hk,gemm,hk+gemm

`HK_EAGER=0` runs every arm with vLLM's CUDA graphs and torch.compile, which
is how it ships; the default is eager.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "python"))


def one_arm(args, backend: str) -> dict:
    """Run one arm in this process and return its timings.

    `hk` is the attention backend, `gemm` is the linear-layer patch over
    stock attention, `hk+gemm` is both; anything else names a vLLM backend.
    `gemm@NAME` routes the linear layers by the table `--routes` names NAME.
    """
    backend, _, table = backend.partition("@")
    if table:
        routes = dict(kv.split("=", 1) for kv in args.routes.split(",") if kv)
        os.environ["HK_VLLM_ROUTES"] = routes[table]
    attn, _, extra = backend.partition("+")
    gemm = attn == "gemm" or extra == "gemm"
    if attn == "hk":
        from hk.integration import vllm_backend

        slot = os.environ.get("HK_VLLM_BACKEND_SLOT", "TRITON_ATTN")
        print(vllm_backend.register(slot), flush=True)
        sel = slot
    elif attn in ("stock", "gemm"):
        # None, not a name. The baseline is the engine as it ships, which
        # means letting the platform choose -- naming TRITON_ATTN here would
        # be a different experiment that happens to agree today.
        sel = None
    else:
        sel = attn
    if gemm:
        from hk.integration import vllm_gemm

        print(vllm_gemm.apply(), flush=True)

    import torch  # noqa: F401
    from vllm import LLM, SamplingParams

    ctxs = [int(x) for x in args.context.split(",")]
    batches = [int(x) for x in args.batch.split(",")]
    llm = LLM(model=args.model, dtype="bfloat16", block_size=32,
              max_model_len=max(ctxs) + args.tokens + 64,
              max_num_seqs=max(batches),
              gpu_memory_utilization=float(os.environ.get("HK_GPU_UTIL", "0.90")),
              attention_backend=sel,
              enable_prefix_caching=False, enable_chunked_prefill=False,
              enforce_eager=os.environ.get("HK_EAGER", "1") == "1")
    tok = llm.get_tokenizer()
    out = {}
    for n_ctx in ctxs:
        for n_req in batches:
            reqs = [{"prompt_token_ids":
                     tok.encode(f"{i} " + "the quick brown fox " *
                                (n_ctx // 4))[:n_ctx]}
                    for i in range(n_req)]
            def params(n):
                return SamplingParams(temperature=0.0, max_tokens=n,
                                      ignore_eos=True)

            # Three lengths. One output token is the batch's prefill. The
            # difference between `tokens` and `short` is steady-state decode
            # at the full batch: with `short` >= the batch, both runs admit
            # every prompt -- including the steps where a new prompt rides
            # with the running requests' decode tokens -- identically, and
            # the extra steps are decode and nothing else.
            lengths = {"min": args.tokens, "short": args.short, "prefill": 1}
            warm = llm.generate(reqs, params(args.tokens))  # warm this shape
            got = {k: [] for k in lengths}
            for _ in range(args.reps):
                for k, n in lengths.items():
                    t0 = time.perf_counter()
                    llm.generate(reqs, params(n))
                    got[k].append(time.perf_counter() - t0)
            out[f"{n_ctx}x{n_req}"] = {
                **{k: min(v) for k, v in got.items()}, "all": got["min"],
                # Greedy, so a patch that changes the arithmetic beyond
                # rounding shows up as a different continuation.
                "tokens": list(warm[0].outputs[0].token_ids)}
            print(f"  ctx {n_ctx} batch {n_req}: {min(got['min']) * 1e3:9.1f}"
                  f" ms, {args.short} tokens {min(got['short']) * 1e3:9.1f} "
                  f"ms, prefill {min(got['prefill']) * 1e3:8.1f} ms",
                  flush=True)
    if gemm:
        # A patch that silently routed nothing would time exactly like stock.
        print(f"  linear routing: {vllm_gemm.PATCH.summary()}", flush=True)
        out["_gemm_stats"] = dict(vllm_gemm.STATS)
        out["_gemm_captured"] = dict(vllm_gemm.PATCH.captured)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--context", default="1024,4096")
    ap.add_argument("--batch", default="1,4,16")
    ap.add_argument("--tokens", type=int, default=72)
    ap.add_argument("--short", type=int, default=24,
                    help="the shorter run the decode slope is taken against; "
                         "must be at least the largest batch")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--arm", default="")
    ap.add_argument("--arms", default="stock,hk",
                    help="comma-separated: stock, hk, gemm, hk+gemm, or a "
                         "vLLM backend name; stock is always included. "
                         "gemm@NAME routes by the table --routes names NAME")
    ap.add_argument("--routes", default="",
                    help="NAME=routes.json,... for gemm@NAME arms")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if not max(int(b) for b in args.batch.split(",")) <= args.short < args.tokens:
        ap.error("need max(batch) <= --short < --tokens for the decode slope "
                 "to cancel the prefill steps")
    if args.arm:                       # child: one engine, one backend
        res = one_arm(args, args.arm)
        if args.out:
            Path(args.out).write_text(json.dumps(res, indent=2))
        return 0

    # Parent: two children, alternating, because two backends cannot share one
    # engine and a backend chosen at construction cannot be swapped.
    base = [sys.executable, str(HERE), "--model", args.model,
            "--context", args.context, "--batch", args.batch,
            "--tokens", str(args.tokens), "--short", str(args.short),
            "--reps", str(args.reps), "--routes", args.routes]
    arms = ["stock"] + [a for a in args.arms.split(",") if a and a != "stock"]
    files, failed = {}, {}
    for rnd in range(args.reps):
        # Rotate, so no arm always runs first or always follows the same one.
        order = arms[rnd % len(arms):] + arms[:rnd % len(arms)]
        for arm in order:
            f = f"/tmp/hk_e2e_{arm}_{rnd}.json"
            print(f"--- {arm} round {rnd}", flush=True)
            # An arm that dies -- a backend that cannot be captured, say --
            # is a result too; the others still get measured.
            r = subprocess.run(base + ["--arm", arm, "--out", f])
            if r.returncode:
                failed.setdefault(arm, []).append(r.returncode)
                print(f"--- {arm} round {rnd} FAILED, exit {r.returncode}",
                      flush=True)
                continue
            files.setdefault(arm, []).append(f)
    if "stock" not in files:
        print("stock failed every round; nothing to compare against")
        return 1

    best, toks = {}, {}
    for arm, fs in files.items():
        for f in fs:
            for k, v in json.loads(Path(f).read_text()).items():
                if k.startswith("_"):
                    continue
                b = best.setdefault(arm, {}).setdefault(
                    k, {"total": 1e9, "short": 1e9, "prefill": 1e9})
                b["total"] = min(b["total"], v["min"])
                b["short"] = min(b["short"], v["short"])
                b["prefill"] = min(b["prefill"], v["prefill"])
                toks.setdefault(arm, {}).setdefault(k, v["tokens"])
    for arm in best:
        for b in best[arm].values():
            b["decode"] = (b["total"] - b["short"]) / (args.tokens - args.short)

    print(f"\n{args.model}, {args.tokens} output tokens, enforce_eager="
          f"{os.environ.get('HK_EAGER', '1') == '1'}, baseline = stock vLLM "
          f"= 1.00x.\ntotal = the whole batch, prompt to last token; prefill "
          f"= the same batch generating one token; decode = (T({args.tokens})"
          f" - T({args.short})) / {args.tokens - args.short}, one step at the "
          f"full batch.")
    for arm, codes in failed.items():
        print(f"{arm}: {len(codes)} of {args.reps} rounds failed "
              f"(exit {', '.join(map(str, codes))})")
    shapes = sorted(best.get("stock", {}),
                    key=lambda x: (int(x.split("x")[0]), int(x.split("x")[1])))
    others = [a for a in arms[1:] if a in best]
    for what, unit in (("total", "ms"), ("prefill", "ms"),
                       ("decode", "ms/tok")):
        print(f"\n{what} ({unit})")
        print(f"{'ctx x batch':>12}{'stock':>10}"
              + "".join(f"{a:>12}{'':>8}" for a in others))
        for k in shapes:
            a = best["stock"][k][what]
            row = f"{k:>12}{a * 1e3:10.2f}"
            for arm in others:
                b = best[arm][k][what]
                row += f"{b * 1e3:12.2f}{a / b:7.3f}x"
            print(row)
    print("\nleading output tokens identical to stock (first request, greedy)")
    for k in shapes:
        ref = toks["stock"][k]
        same = []
        for arm in others:
            got = toks[arm][k]
            n = next((i for i, (x, y) in enumerate(zip(ref, got)) if x != y),
                     min(len(ref), len(got)))
            same.append(f"{arm} {n}/{len(ref)}")
        print(f"{k:>12}  " + ", ".join(same))
    if args.out:
        Path(args.out).write_text(json.dumps(best, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
