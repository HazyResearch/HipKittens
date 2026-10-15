"""Each fused epilogue against PyTorch eager, on the layer GEMM where it belongs.

Arms (all cold cache):
  HK           one fused HipKittens call (hk.matmul, hk.inv_rms or hk.residual_rms)
  eager        torch.matmul (hipBLASLt) plus the epilogue as lean bf16 eager torch
  fusion-only  HK noop GEMM plus the same eager torch: the same GEMM as HK, so the ratio isolates fusion
HBM saved: one analytical formula per epilogue (saved= in make_case): the bytes of every [M, N] / [M, K]
intermediate the eager code writes and reads back, minus HK's own fp32 partial sums. It ignores cache, so
it is an upper bound. Every row is checked against fp32 torch (HK and eager outputs) before timing; a
failing row is not timed, and the script then exits 1.

Build first, in the epilogue directory: make KERNEL=<m> for every m in MODULES.
"""
import torch
import torch.nn.functional as F

import common
import hk

MODULES = ["noop", "silu", "scale", "residual_add", "rmsnorm_scale", "swiglu", "rmsnorm_swiglu",
           "rope", "rmsnorm_rope", "partialrms", "residual_rms", "rms_reduce"]

# epilogue -> the layer GEMMs it runs on
PLACEMENT = {
    "noop": ["Wq", "Wk", "Wo", "Wgu", "Wd"],
    "rope": ["Wq", "Wk"], "rmsnorm_rope": ["Wq", "Wk"],
    "scale": ["Wv"], "rmsnorm_scale": ["Wv"], "rmsnorm": ["Wv"],
    "residual_add": ["Wo", "Wd"], "residual_rms": ["Wo", "Wd"], "inv_rms": ["Wo", "Wd"],
    "silu": ["Wgu"], "swiglu": ["Wgu"], "rmsnorm_swiglu": ["Wgu"],
}


def rms_scale(t, r, g):
    """Eager RMSNorm with a given per-row inverse RMS r (fp32 [M]) and gain g (bf16): t * r * g in bf16."""
    return t * r.to(common.BF16)[:, None] * g


def swiglu(D):
    """silu(gate) * up, with D = [gate | up] along columns."""
    n = D.shape[1] // 2
    return F.silu(D[:, :n]) * D[:, n:]


def inv_rms_eager(D):
    """fp32 [M] 1 / rms per row with one bf16 read of D."""
    return torch.rsqrt(torch.linalg.vector_norm(D, dim=-1, dtype=torch.float32).square() / D.shape[1] + common.EPS)


def make_case(name, M, K, N, Dh):
    """One epilogue at [M, K] x [K, N]. eager = post(torch.matmul(pre(x), W)); fusion-only =
    post(hk noop(pre(x))); ref() is fp32. saved = HBM bytes HK avoids: every [M, N] / [M, K] intermediate
    the eager code writes and reads back (bf16, write + read), minus HK's own fp32 partial-sum round trip.
    Vector-sized tensors are ignored. The README lists the same formulas."""
    x, W = common.activation(M, K), common.weight(K, N)
    plain = hk.prepare(W)
    f = lambda t: t.float()
    xw = lambda: f(x) @ f(W)
    ident = lambda t: t
    MN, MK = M * N * 2, M * K * 2                 # bf16 bytes of an [M, N] / [M, K] tensor
    partials = 2 * (N // 64) * M * 4              # HK's fp32 partial sums, written then read by rms_reduce

    def case(hk_fn, pre, post, ref, saved, tol=(common.STAGE_TOL,)):
        return dict(x=x, W=W, plain=plain, hk=hk_fn, pre=pre, post=post, ref=ref, saved=saved, tol=list(tol))

    if name == "noop":
        return case(lambda: hk.matmul(x, plain), ident, ident, xw, saved=0)
    if name == "silu":                            # D
        return case(lambda: hk.matmul(x, plain, "silu"), ident, F.silu, lambda: F.silu(xw()), saved=2 * MN)
    if name == "scale":                           # D
        return case(lambda: hk.matmul(x, plain, "scale", alpha=0.5), ident, lambda D: D * 0.5,
                    lambda: xw() * 0.5, saved=2 * MN)
    if name == "residual_add":                    # D
        res = common.activation(M, N)
        return case(lambda: hk.matmul(x, plain, "residual_add", residual=res), ident, lambda D: D + res,
                    lambda: xw() + f(res), saved=2 * MN)
    if name == "rmsnorm_scale":                   # D, D*r
        r, gc = common.inv_rms(x), common.gamma(N)
        return case(lambda: hk.matmul(x, plain, "rmsnorm_scale", r=r, gamma=gc), ident,
                    lambda D: rms_scale(D, r, gc), lambda: xw() * r[:, None] * f(gc), saved=4 * MN)
    if name == "rmsnorm":                         # x*r, x*r*g
        r, g = common.inv_rms(x), common.gamma(K)
        folded = hk.prepare(W, gamma=g)
        return case(lambda: hk.matmul(x, folded, "rmsnorm", r=r), lambda t: rms_scale(t, r, g), ident,
                    lambda: (f(x) * r[:, None] * f(g)) @ f(W), saved=4 * MK)
    if name == "swiglu":                          # D, silu(gate) (half width)
        sw = hk.prepare(W, layout="swiglu")
        return case(lambda: hk.matmul(x, sw, "swiglu"), ident, swiglu, lambda: swiglu(xw()), saved=3 * MN)
    if name == "rmsnorm_swiglu":                  # rmsnorm + swiglu
        r, g = common.inv_rms(x), common.gamma(K)
        sw = hk.prepare(W, layout="swiglu", gamma=g)
        return case(lambda: hk.matmul(x, sw, "rmsnorm_swiglu", r=r), lambda t: rms_scale(t, r, g), swiglu,
                    lambda: swiglu((f(x) * r[:, None] * f(g)) @ f(W)), saved=4 * MK + 3 * MN)
    if name in ("rope", "rmsnorm_rope"):          # D, six half-width products and sums
        cs = common.rope_table(M, Dh).repeat(1, N // Dh)          # one head's table, tiled across heads
        table = hk.prepare_rope_table(cs)
        post = lambda D: common.rope(D, cs)
        if name == "rope":
            rw = hk.prepare(W, layout="rope")
            return case(lambda: hk.matmul(x, rw, "rope", cos_sin=table), ident, post,
                        lambda: common.rope(xw(), f(cs)), saved=8 * MN)
        r, g = common.inv_rms(x), common.gamma(K)
        rw = hk.prepare(W, layout="rope", gamma=g)
        return case(lambda: hk.matmul(x, rw, "rmsnorm_rope", r=r, cos_sin=table), lambda t: rms_scale(t, r, g),
                    post, lambda: common.rope((f(x) * r[:, None] * f(g)) @ f(W), f(cs)), saved=4 * MK + 8 * MN)
    if name == "inv_rms":                         # D
        return case(lambda: hk.inv_rms(x, plain), ident, inv_rms_eager, lambda: common.inv_rms(xw()),
                    saved=2 * MN - partials, tol=(common.VECTOR_TOL,))
    if name == "residual_rms":                    # D, second read of h
        res = common.activation(M, N)

        def post(D):
            h = D + res
            return h, inv_rms_eager(h)

        def ref():
            h = xw() + f(res)
            return h, common.inv_rms(h)

        return case(lambda: hk.residual_rms(x, plain, res), ident, post, ref, saved=3 * MN - partials,
                    tol=(common.STAGE_TOL, common.VECTOR_TOL))
    raise ValueError(f"unknown epilogue {name!r}")


def run_row(name, gemm, M, K, N, Dh, timer):
    c = make_case(name, M, K, N, Dh)
    x, W, plain = c["x"], c["W"], c["plain"]
    hk_fn = c["hk"]
    eager = lambda: c["post"](torch.matmul(c["pre"](x), W))
    fusion = lambda: c["post"](hk.matmul(c["pre"](x), plain))
    want = c["ref"]()
    ok_hk, rel = common.check(hk_fn(), want, c["tol"])
    ok_eager, _ = common.check(eager(), want, c["tol"])
    ok = ok_hk and ok_eager
    row = dict(epilogue=name, gemm=gemm, M=M, K=K, N=N, rel=rel, status="PASS" if ok else "FAIL")
    if not ok:
        return row
    row["saved_MB"] = c["saved"] / 1e6
    t_hk, t_eager, t_fusion = timer(hk_fn), timer(eager), timer(fusion)
    row.update(**common.ms_fields("hk", t_hk), **common.ms_fields("eager", t_eager),
               **common.ms_fields("fusion", t_fusion),
               eager_x=t_eager.median / t_hk.median, fusion_x=t_fusion.median / t_hk.median)
    return row


def main():
    args = common.parse_args("Fused epilogues vs PyTorch eager")
    common.require_epilogue_modules(MODULES)
    torch.manual_seed(0)
    timer = common.Timer(args.warmup, args.iters)
    gemms = common.layer_gemms(args.cfg)
    rows = []
    for M in args.M:
        for name, places in PLACEMENT.items():
            for gemm in places:
                K, N = gemms[gemm]
                row = run_row(name, gemm, M, K, N, args.cfg["Dh"], timer)
                rows.append(row)
                print(f"  M={M} {name} on {gemm}: {row['status']}", flush=True)
    columns = [("epilogue", "epilogue", ""), ("GEMM", "gemm", ""), ("M", "M", "d"), ("N", "N", "d"), ("K", "K", "d"),
               ("HK ms", "hk_ms", ".3f"), ("eager ms", "eager_ms", ".3f"), ("fusion-only ms", "fusion_ms", ".3f"),
               ("eager/HK", "eager_x", ".2f"), ("fusion-only/HK", "fusion_x", ".2f"),
               ("HBM saved MB", "saved_MB", ".1f"), ("rel err", "rel", ".1e"), ("status", "status", "")]
    common.print_table(f"Fused epilogues vs PyTorch eager ({args.config}, cold cache)", columns, rows, args.markdown)
    if args.json:
        common.write_json(args.json, "bench_epilogues", args.config, rows)
    raise SystemExit(common.exit_code(rows))


if __name__ == "__main__":
    main()
