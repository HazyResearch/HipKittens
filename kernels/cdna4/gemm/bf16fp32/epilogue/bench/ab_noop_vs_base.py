"""tk_noop vs the upstream base GEMM: C must be bitwise identical.

Arms alternate over ROUNDS rounds; time = median of round medians. noop <= 3% slower: PASS; 3-5%: RERUN
(3 fresh processes, median of medians); > 5%: FAIL. Build: `make KERNEL=noop` here and `make` in the parent.
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
        same = torch.equal(c_noop, c_base)
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
