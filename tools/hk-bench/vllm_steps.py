#!/usr/bin/env python3
"""What does each engine step cost? Per-step wall time, by step composition.

A whole-request total cannot say which steps a patch helped: a batch of long
prompts runs one or two prefill steps, then steps that admit one prompt while
the running requests decode (M = prompt + r), then decode. Subtracting runs of
different lengths to isolate a kind of step is how a phantom cost got into an
earlier README. So this times every step directly -- a synchronize on each
side of execute_model + sample_tokens -- and records how many tokens and new
prompts it carried. Same LLM configuration as vllm_e2e.py.

    # one JSON line per run; alternate arms in separate processes
    python3 tools/hk-bench/vllm_steps.py --arm stock >> steps.jsonl
    python3 tools/hk-bench/vllm_steps.py --arm gemm  >> steps.jsonl
    python3 tools/hk-bench/vllm_steps.py --report steps.jsonl

Arms: `stock`; `gemm` (hk/integration/vllm_gemm.py); `fallback` (the patched
op routing nothing, i.e. vLLM's GEMM behind the same op boundary); `alternate`
(one process, even steps on the patch, odd steps on vLLM's GEMM).
`--time-linear` adds a synchronize around every linear call, so the GEMMs'
own time in place can be compared with the step.

The torch profiler is not a substitute here: in an eager step on this ROCm
build, its kernel durations summed to 4.4x the step's wall time -- they
include time spent queued behind earlier launches.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))


def run(args) -> None:
    import torch

    steps, cur, calls, on = [], {}, [], [False]
    if args.arm != "stock":
        from hk.integration import vllm_gemm

        if args.arm == "fallback":
            vllm_gemm.route = lambda *a, **k: False
            vllm_gemm.route_decode = lambda *a, **k: False
        elif args.arm == "alternate":
            orig = vllm_gemm.route
            vllm_gemm.route = lambda *a, **k: orig(*a, **k) and len(steps) % 2 == 0
        if args.time_linear:
            inner = vllm_gemm.linear

            def timed(x, weight, bias=None):
                if not on[0] or torch.cuda.is_current_stream_capturing():
                    return inner(x, weight, bias)
                torch.cuda.synchronize()
                t0, before = time.perf_counter(), dict(vllm_gemm.STATS)
                y = inner(x, weight, bias)
                torch.cuda.synchronize()
                path = next(k for k, v in vllm_gemm.STATS.items() if v != before[k])
                calls.append([len(steps), x.numel() // x.shape[-1], weight.shape[0],
                              x.shape[-1], path, (time.perf_counter() - t0) * 1e3])
                return y
            vllm_gemm.linear = timed
        print(vllm_gemm.apply(), file=sys.stderr, flush=True)

    from vllm import LLM, SamplingParams
    from vllm.v1.worker import gpu_model_runner as gmr

    orig_exec = gmr.GPUModelRunner.execute_model
    orig_sample = gmr.GPUModelRunner.sample_tokens

    def finish():
        torch.cuda.synchronize()
        if on[0]:
            steps.append([cur["tokens"], cur["new"], cur["reqs"],
                          (time.perf_counter() - cur["t0"]) * 1e3])

    def execute_model(self, so, *a, **k):
        torch.cuda.synchronize()
        cur.update(t0=time.perf_counter(), tokens=so.total_num_scheduled_tokens,
                   new=len(so.scheduled_new_reqs), reqs=len(so.num_scheduled_tokens))
        out = orig_exec(self, so, *a, **k)
        if out is not None:
            finish()
        return out

    def sample_tokens(self, *a, **k):
        out = orig_sample(self, *a, **k)
        finish()
        return out

    gmr.GPUModelRunner.execute_model = execute_model
    gmr.GPUModelRunner.sample_tokens = sample_tokens

    llm = LLM(model=args.model, dtype="bfloat16", block_size=32,
              max_model_len=args.context + args.tokens + 64, max_num_seqs=args.batch,
              gpu_memory_utilization=float(os.environ.get("HK_GPU_UTIL", "0.85")),
              enable_prefix_caching=False, enable_chunked_prefill=False,
              enforce_eager=os.environ.get("HK_EAGER", "1") == "1")
    tok = llm.get_tokenizer()
    reqs = [{"prompt_token_ids": tok.encode(f"{i} " + "the quick brown fox " *
                                            (args.context // 4))[:args.context]}
            for i in range(args.batch)]
    params = SamplingParams(temperature=0.0, max_tokens=args.tokens, ignore_eos=True)
    llm.generate(reqs, params)                              # warm
    on[0] = True
    t0 = time.perf_counter()
    llm.generate(reqs, params)
    print(json.dumps({"arm": args.arm, "total_ms": (time.perf_counter() - t0) * 1e3,
                      "steps": steps, "calls": calls}), flush=True)


def _kind(step):
    tokens, new, reqs = step[:3]
    if new and tokens == new * (tokens // new) and reqs == new:
        return f"prefill ({new} prompt{'s' if new > 1 else ''}, {tokens} tok)"
    if new:
        return "admit one prompt + decode"
    return "decode"


def report(paths) -> None:
    runs = [json.loads(line) for p in paths for line in open(p) if line.startswith("{")]
    by_arm = collections.defaultdict(list)
    for r in runs:
        by_arm[r["arm"]].append(r)
    print("per kind of step, ms per step (each step's min over runs):")
    for arm, rs in by_arm.items():
        comps = [[s[:3] for s in r["steps"]] for r in rs]
        if any(c != comps[0] for c in comps):
            print(f"  {arm}: runs scheduled differently; not combined")
            continue
        best = [min(r["steps"][i][3] for r in rs) for i in range(len(comps[0]))]
        kinds = collections.defaultdict(list)
        for comp, t in zip(comps[0], best):
            if comp[0]:
                kinds[_kind(comp)].append(t)
        total = min(r["total_ms"] for r in rs)
        print(f"  {arm:9} total {total:8.0f}  " + "  ".join(
            f"{k}: {sum(v) / len(v):.1f} x{len(v)}" for k, v in kinds.items()))
        if rs[0].get("calls"):
            per = collections.defaultdict(lambda: collections.defaultdict(float))
            seen = collections.defaultdict(set)
            for step, m, n, k, path, ms in rs[0]["calls"]:
                per[(_kind(comps[0][step]), path)][(n, k)] += ms
                seen[(_kind(comps[0][step]), path)].add(step)
            for (kind, path), shapes in per.items():
                n_steps = len(seen[(kind, path)])
                print(f"      linear in place, {kind}, {path}: "
                      f"{sum(shapes.values()) / n_steps:.1f} ms/step  (" + ", ".join(
                          f"N{n}xK{k} {v / n_steps:.1f}" for (n, k), v in sorted(shapes.items()))
                      + ")")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", choices=["stock", "gemm", "fallback", "alternate"])
    ap.add_argument("--time-linear", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--tokens", type=int, default=72)
    ap.add_argument("--report", nargs="+", metavar="JSONL")
    args = ap.parse_args()
    if args.report:
        report(args.report)
    elif args.arm:
        run(args)
    else:
        ap.error("--arm or --report")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
