"""Check that the template refactor did not change the base GEMM.

For each GEMM shape in one layer of --config, plus 8192^3, runs two kernels on the same bf16 inputs:
  tk_noop    gemm_base.cuh with NoOpEpilogue (make KERNEL=noop). All fused epilogues use this kernel.
  tk_kernel  upstream GEMM, ../256_256_64_32_with16x32.cpp (make in the parent directory).
C must be bitwise identical. Both are timed with a cold cache (common.Timer).

Timing: each kernel is one arm. One round times both arms with common.Timer (median of --iters cold
calls). Arm order flips each round to reduce order bias. ROUNDS = 5; an arm's time is the median of its
5 round results.
Status from noop/base: <= 1.03 PASS (exit 0); > 1.03 and <= 1.05 RERUN (exit 2: run the script in 3
fresh processes and compare the medians); > 1.05 or any differing bit FAIL (exit 1).
Build: `make KERNEL=noop` here and `make` in the parent.
"""
import importlib

import torch

import common

ROUNDS = 5


def shapes(cfg, Ms):
    """Unique layer (M, K, N) plus 8192^3."""
    out = []
    for M in Ms:
        for K, N in common.layer_gemms(cfg).values():
            if (M, K, N) not in out:
                out.append((M, K, N))
    if (8192, 8192, 8192) not in out:
        out.append((8192, 8192, 8192))
    return out


def main():
    args = common.parse_args("tk_noop vs upstream base GEMM (refactor neutrality)")
    common.require_epilogue_modules(["noop"])
    noop = importlib.import_module("tk_noop")
    base = common.load_extension(common.find_extension(common.BASE, "tk_kernel", f"make -C {common.BASE}"))
    torch.manual_seed(0)
    timer = common.Timer(args.warmup, args.iters)
    stream = torch.cuda.current_stream().cuda_stream
    rows = []
    for M, K, N in shapes(args.cfg, args.M):
        a = common.activation(M, K)
        bt = common.weight(K, N)                                    # [N, K]
        c_noop = torch.empty(M, N, dtype=common.BF16, device="cuda")
        c_base = torch.empty_like(c_noop)
        run_noop = lambda: noop.dispatch(a, bt, c_noop, stream)
        run_base = lambda: base.dispatch_micro(a, bt, c_base)
        run_noop()
        run_base()
        torch.cuda.synchronize()
        same = torch.equal(c_noop.view(torch.int16), c_base.view(torch.int16))  # compare raw bf16 bits
        noop_ms, base_ms = [], []
        for i in range(ROUNDS):
            arms = [(run_noop, noop_ms), (run_base, base_ms)]
            for fn, out in (arms if i % 2 == 0 else arms[::-1]):
                out.append(timer(fn).median)
        t_noop, t_base = sorted(noop_ms)[ROUNDS // 2], sorted(base_ms)[ROUNDS // 2]
        ratio = t_noop / t_base
        status = "FAIL" if not same or ratio > 1.05 else "RERUN" if ratio > 1.03 else "PASS"
        rows.append(dict(M=M, K=K, N=N, noop_ms=t_noop, base_ms=t_base, ratio=ratio,
                         bitwise="yes" if same else "NO", status=status))
        print(f"  {M}x{K}x{N}: noop/base {ratio:.3f} {status}", flush=True)
    columns = [("M", "M", "d"), ("K", "K", "d"), ("N", "N", "d"), ("noop ms", "noop_ms", ".3f"),
               ("base ms", "base_ms", ".3f"), ("noop/base", "ratio", ".3f"), ("bitwise", "bitwise", ""),
               ("status", "status", "")]
    common.print_table("tk_noop vs upstream base GEMM (cold cache)", columns, rows, args.markdown)
    if args.json:
        common.write_json(args.json, "ab_noop_vs_base", args.config, rows)
    raise SystemExit(common.exit_code(rows))


if __name__ == "__main__":
    main()
