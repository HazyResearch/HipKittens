"""One pre-norm GQA Transformer layer: HK vs PyTorch eager vs torch.compile (rules in README.md).

--phases  each phase timed alone on captured intermediates, plus a fusion-only control per GEMM phase.
--full    the whole layer (compiled as one function), and compiled full minus the sum of compiled phases.

HK launches per phase: qkv 4, attention 1, out_proj 2, gate_up 1, down 2. HK's producers emit r and h * gamma
for the next consumer; torch normalizes in the consumer. Per-phase times do not line up op-for-op.

Build: make KERNEL=<m> for every m in MODULES, and bench/build_attn.sh <H> <H_KV> <Dh> <M>...
"""
import dataclasses
import importlib

import torch
import torch._dynamo
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

import common
import hk

MODULES = ["rmsnorm_rope_qkv", "residual_rms_gamma", "rms_reduce", "rmsnorm_swiglu", "noop", "noop_qkv"]
HK_LAUNCHES = dict(qkv=4, attention=1, out_proj=2, gate_up=1, down=2)
COMPILE = common.COMPILE


def choose_sdpa(H, H_KV, Dh):
    """(backend name, native GQA) of the best fused SDPA path; disables the math backend globally."""
    q = torch.randn(1, H, 256, Dh, device="cuda", dtype=common.BF16)
    kv = torch.randn(1, H_KV, 256, Dh, device="cuda", dtype=common.BF16)
    for backend in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION):
        for native in (True, False):
            k = kv if native else kv.repeat_interleave(H // H_KV, dim=1)
            try:
                with sdpa_kernel(backend):
                    F.scaled_dot_product_attention(q, k, k, is_causal=True, enable_gqa=native)
                torch.cuda.synchronize()
            except RuntimeError:
                continue
            torch.backends.cuda.enable_flash_sdp(backend == SDPBackend.FLASH_ATTENTION)
            torch.backends.cuda.enable_mem_efficient_sdp(backend == SDPBackend.EFFICIENT_ATTENTION)
            torch.backends.cuda.enable_math_sdp(False)
            torch.backends.cuda.enable_cudnn_sdp(False)
            return backend.name, native
    raise RuntimeError("no fused SDPA backend is available; refusing the math fallback")


def load_gqa(cfg, M):
    """The gqa_causal build for this shape (unique module name per build)."""
    name = f"tk_attn_H{cfg['H']}_KV{cfg['H_KV']}_D{cfg['Dh']}_N{M}"
    hint = f"{common.BENCH / 'build_attn.sh'} {cfg['H']} {cfg['H_KV']} {cfg['Dh']} {M}"
    return common.load_extension(common.find_extension(common.BENCH / "attn_build", name, hint), name=name)


def noop_qkv(x, wq, wk, wv):
    """x @ [wq; wk; wv].t() in one noop launch."""
    nq, nkv = wq.shape[0], wk.shape[0]
    out = torch.empty(x.shape[0], nq + 2 * nkv, dtype=common.BF16, device=x.device)
    importlib.import_module("tk_noop_qkv").dispatch(x, wq, out, wk, wv, torch.cuda.current_stream().cuda_stream)
    return out


class Layer:
    """Shared weights and the five phases of each implementation."""

    def __init__(self, cfg, M, gqa, native_gqa):
        self.M, self.d, self.d_ff = M, cfg["d"], cfg["d_ff"]
        self.H, self.H_KV, self.Dh = cfg["H"], cfg["H_KV"], cfg["Dh"]
        d, kv = self.d, self.H_KV * self.Dh
        self.W = dict(q=common.weight(d, d), k=common.weight(d, kv), v=common.weight(d, kv), o=common.weight(d, d),
                      gu=common.weight(d, 2 * self.d_ff), dn=common.weight(self.d_ff, d))
        self.g_attn, self.g_mlp = common.gamma(d), common.gamma(d)
        self.x = common.activation(M, d)
        self.cs = common.rope_table(M, self.Dh)                     # [M, Dh], one head
        # HK inputs a previous layer's down phase would emit.
        self.r_attn = common.inv_rms(self.x)
        self.xg = (self.x.float() * self.g_attn.float()[None, :]).to(common.BF16)
        self.gqa, self.native_gqa = gqa, native_gqa

    # ---- HK phases ----
    def hk_qkv(self, xg, r):
        W = self.W
        return hk.qkv(xg, W["q"], W["k"], W["v"], r=r, cos_sin=self.cs)

    def hk_attention(self, q, k, v):
        M, H, H_KV, Dh = self.M, self.H, self.H_KV, self.Dh
        o = torch.empty(1, M, H, Dh, dtype=common.BF16, device="cuda")
        lse = torch.empty(1, H, 1, M, dtype=common.FP32, device="cuda")
        self.gqa.dispatch_micro(q.view(1, M, H, Dh), k.view(1, M, H_KV, Dh), v.view(1, M, H_KV, Dh), o, lse)
        return o.view(M, H * Dh)

    def hk_out_proj(self, o, x):
        return hk.residual_rms(o, self.W["o"], x, gamma=self.g_mlp)      # h, h * g_mlp, r_mlp

    def hk_gate_up(self, hg, r):
        return hk.matmul(hg, self.W["gu"], "rmsnorm_swiglu", r=r)

    def hk_down(self, g, h):
        # identical layers: the next attention norm uses g_attn
        return hk.residual_rms(g, self.W["dn"], h, gamma=self.g_attn)    # x_out, x_out * g_attn, r_next

    # ---- fusion-only control ----
    def eager_inv_rms(self, t):
        return torch.rsqrt(torch.linalg.vector_norm(t, dim=-1, dtype=torch.float32).square() / self.d + common.EPS)

    def row_scale(self, t, r):
        return t * r.to(common.BF16)[:, None]

    def f_qkv(self, xg, r):
        M, Dh, nq, nkv, W = self.M, self.Dh, self.d, self.H_KV * self.Dh, self.W
        c = noop_qkv(self.row_scale(xg, r), W["q"], W["k"], W["v"])
        cs = self.cs.view(M, 1, Dh)
        q = common.rope(c[:, :nq].view(M, self.H, Dh), cs).reshape(M, nq)
        k = common.rope(c[:, nq:nq + nkv].view(M, self.H_KV, Dh), cs).reshape(M, nkv)
        return q, k, c[:, nq + nkv:].contiguous()

    def f_out_proj(self, o, x):
        h = hk.matmul(o, self.W["o"]) + x
        return h, h * self.g_mlp, self.eager_inv_rms(h)            # h, h * g_mlp, r_mlp

    def f_gate_up(self, hg, r):
        gu = hk.matmul(self.row_scale(hg, r), self.W["gu"])
        return F.silu(gu[:, :self.d_ff]) * gu[:, self.d_ff:]

    def f_down(self, g, h):
        x_out = hk.matmul(g, self.W["dn"]) + h
        return x_out, x_out * self.g_attn, self.eager_inv_rms(x_out)  # x_out, x_out * g_attn, r_next

    # ---- torch phases (eager and compiled) ----
    def t_qkv(self, x):
        M, Dh, W = self.M, self.Dh, self.W
        xn = F.rms_norm(x, (self.d,), self.g_attn, common.EPS)
        cs = self.cs.view(M, 1, Dh)
        q = common.rope((xn @ W["q"].t()).view(M, self.H, Dh), cs)
        k = common.rope((xn @ W["k"].t()).view(M, self.H_KV, Dh), cs)
        v = (xn @ W["v"].t()).view(M, self.H_KV, Dh)
        return q, k, v

    def t_attention(self, q, k, v):
        qh, kh, vh = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))        # [1, heads, M, Dh]
        if not self.native_gqa:
            G = self.H // self.H_KV
            kh, vh = kh.repeat_interleave(G, dim=1), vh.repeat_interleave(G, dim=1)
        o = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True, enable_gqa=self.native_gqa)
        return o.squeeze(0).transpose(0, 1).reshape(self.M, self.d)

    def t_out_proj(self, o, x):
        return x + o @ self.W["o"].t()

    def t_gate_up(self, h):
        gu = F.rms_norm(h, (self.d,), self.g_mlp, common.EPS) @ self.W["gu"].t()
        return F.silu(gu[:, :self.d_ff]) * gu[:, self.d_ff:]

    def t_down(self, g, h):
        return h + g @ self.W["dn"].t()                             # x_out

    # ---- fp32 references ----
    def ref_qkv(self, x):
        M, Dh, W = self.M, self.Dh, self.W
        xn = x.float() * common.inv_rms(x)[:, None] * self.g_attn.float()
        cs = self.cs.float().view(M, 1, Dh)
        q = common.rope((xn @ W["q"].float().t()).view(M, self.H, Dh), cs).reshape(M, -1)
        k = common.rope((xn @ W["k"].float().t()).view(M, self.H_KV, Dh), cs).reshape(M, -1)
        return q, k, xn @ W["v"].float().t()

    def ref_attention(self, q, k, v):
        M, H, H_KV, Dh = self.M, self.H, self.H_KV, self.Dh
        G = H // H_KV
        qh = q.float().reshape(M, H, Dh).transpose(0, 1)                                   # [H, M, Dh]
        kh = k.float().reshape(M, H_KV, Dh).transpose(0, 1).repeat_interleave(G, dim=0)
        vh = v.float().reshape(M, H_KV, Dh).transpose(0, 1).repeat_interleave(G, dim=0)
        s = (qh @ kh.transpose(1, 2)) * Dh ** -0.5
        s.masked_fill_(torch.ones(M, M, dtype=torch.bool, device="cuda").triu(1), float("-inf"))
        return (torch.softmax(s, dim=-1) @ vh).transpose(0, 1).reshape(M, H * Dh)

    def ref_out_proj(self, o, x):
        h = x.float() + o.float() @ self.W["o"].float().t()
        return h, common.inv_rms(h)

    def ref_out_proj_hk(self, o, x):
        h, r = self.ref_out_proj(o, x)
        return h, h * self.g_mlp.float(), r                         # what HK's out_proj writes

    def ref_gate_up(self, h):
        hn = h.float() * common.inv_rms(h)[:, None] * self.g_mlp.float()
        gu = hn @ self.W["gu"].float().t()
        return F.silu(gu[:, :self.d_ff]) * gu[:, self.d_ff:]

    def ref_down(self, g, h):
        x_out = h.float() + g.float() @ self.W["dn"].float().t()
        return x_out, common.inv_rms(x_out)

    def ref_down_hk(self, g, h):
        x_out, r = self.ref_down(g, h)
        return x_out, x_out * self.g_attn.float(), r                # what HK's down writes

    # ---- captured intermediates ----
    def capture_hk(self):
        x, xg, r = self.x, self.xg, self.r_attn
        q, k, v = self.hk_qkv(xg, r)
        o = self.hk_attention(q, k, v)
        h, hg, r_mlp = self.hk_out_proj(o, x)
        g = self.hk_gate_up(hg, r_mlp)
        return dict(x=x, xg=xg, r=r, q=q, k=k, v=v, o=o, h=h, hg=hg, r_mlp=r_mlp, g=g)

    def capture_torch(self):
        x = self.x
        q, k, v = self.t_qkv(x)
        o = self.t_attention(q, k, v)
        h = self.t_out_proj(o, x)
        g = self.t_gate_up(h)
        return dict(x=x, q=q, k=k, v=v, o=o, h=h, g=g)

    # ---- full layer ----
    def hk_layer(self, x, xg, r):
        q, k, v = self.hk_qkv(xg, r)
        o = self.hk_attention(q, k, v)
        h, hg, r_mlp = self.hk_out_proj(o, x)
        g = self.hk_gate_up(hg, r_mlp)
        x_out, xg_out, r_next = self.hk_down(g, h)
        return x_out, r_next, xg_out                                # what the next layer reads

    def t_layer(self, x):
        q, k, v = self.t_qkv(x)
        o = self.t_attention(q, k, v)
        h = self.t_out_proj(o, x)
        g = self.t_gate_up(h)
        return self.t_down(g, h)                                    # x_out

    def ref_layer(self, x):
        q, k, v = self.ref_qkv(x)
        o = self.ref_attention(q, k, v)
        h, _ = self.ref_out_proj(o, x)
        return self.ref_down(self.ref_gate_up(h), h)                # fp32 x_out, r_next


@dataclasses.dataclass
class Phase:
    hk: object            # () -> HK outputs on HK-captured inputs
    torch_fn: object      # torch phase function
    torch_args: tuple     # its torch-captured inputs
    ref_hk: object        # () -> fp32 reference for the HK inputs
    ref_torch: object     # () -> fp32 reference for the torch inputs
    tol_hk: tuple
    tol_torch: tuple
    fusion: object = None # () -> fusion-only control on HK-captured inputs (None for attention)


def phases(L, a, b):
    S, V = common.STAGE_TOL, common.VECTOR_TOL
    return {
        "qkv": Phase(lambda: L.hk_qkv(a["xg"], a["r"]), L.t_qkv, (b["x"],),
                     lambda: L.ref_qkv(a["x"]), lambda: L.ref_qkv(b["x"]), (S, S, S), (S, S, S),
                     lambda: L.f_qkv(a["xg"], a["r"])),
        "attention": Phase(lambda: L.hk_attention(a["q"], a["k"], a["v"]), L.t_attention, (b["q"], b["k"], b["v"]),
                           lambda: L.ref_attention(a["q"], a["k"], a["v"]),
                           lambda: L.ref_attention(b["q"], b["k"], b["v"]), (S,), (S,)),
        "out_proj": Phase(lambda: L.hk_out_proj(a["o"], a["x"]), L.t_out_proj, (b["o"], b["x"]),
                          lambda: L.ref_out_proj_hk(a["o"], a["x"]), lambda: L.ref_out_proj(b["o"], b["x"])[:1],
                          (S, S, V), (S,), lambda: L.f_out_proj(a["o"], a["x"])),
        "gate_up": Phase(lambda: L.hk_gate_up(a["hg"], a["r_mlp"]), L.t_gate_up, (b["h"],),
                         lambda: L.ref_gate_up(a["h"]), lambda: L.ref_gate_up(b["h"]), (S,), (S,),
                         lambda: L.f_gate_up(a["hg"], a["r_mlp"])),
        "down": Phase(lambda: L.hk_down(a["g"], a["h"]), L.t_down, (b["g"], b["h"]),
                      lambda: L.ref_down_hk(a["g"], a["h"]), lambda: L.ref_down(b["g"], b["h"])[:1], (S, S, V), (S,),
                      lambda: L.f_down(a["g"], a["h"])),
    }


def run_phases(L, timer):
    a, b = L.capture_hk(), L.capture_torch()
    rows = []
    for name, p in phases(L, a, b).items():
        compiled = torch.compile(p.torch_fn, **COMPILE)
        eager = lambda: p.torch_fn(*p.torch_args)
        comp = lambda: compiled(*p.torch_args)
        ok_hk, rel = common.check(p.hk(), p.ref_hk(), p.tol_hk)
        ref_t = p.ref_torch()
        ok_eager, _ = common.check(eager(), ref_t, p.tol_torch)
        ok_comp, _ = common.check(comp(), ref_t, p.tol_torch)     # first call compiles
        ok_fusion = True
        if p.fusion is not None:
            ok_fusion, _ = common.check(p.fusion(), p.ref_hk(), common.control_tols(p.tol_hk))
        ok_hk = ok_hk and ok_fusion
        status = "PASS" if ok_hk and ok_eager and ok_comp else "TORCH_MISS" if ok_hk else "FAIL"
        row = dict(M=L.M, phase=name, hk_launches=HK_LAUNCHES[name], rel=rel, ok_eager=ok_eager, ok_compile=ok_comp,
                   status=status)
        if ok_hk:   # a PyTorch miss is recorded, not a reason to drop HK's timing
            t_hk, t_eager, t_comp = timer(p.hk), timer(eager), timer(comp)
            row.update(**common.ms_fields("hk", t_hk), **common.ms_fields("eager", t_eager),
                       **common.ms_fields("compile", t_comp),
                       eager_x=t_eager.median / t_hk.median, compile_x=t_comp.median / t_hk.median)
            if p.fusion is not None:
                t_fusion = timer(p.fusion)
                row.update(**common.ms_fields("fusion", t_fusion), fusion_x=t_fusion.median / t_hk.median)
        rows.append(row)
        print(f"  M={L.M} {name}: {row['status']}", flush=True)
    rows.append(sum_row(L.M, rows))
    return rows


TIMED = ("PASS", "TORCH_MISS")


def sum_row(M, rows):
    statuses = {r["status"] for r in rows}
    status = "FAIL" if "FAIL" in statuses else "TORCH_MISS" if "TORCH_MISS" in statuses else "PASS"
    row = dict(M=M, phase="sum", hk_launches=sum(r["hk_launches"] for r in rows), status=status)
    if status in TIMED:
        for arm in ("hk", "eager", "compile"):
            row[f"{arm}_ms"] = sum(r[f"{arm}_ms"] for r in rows)
        row["eager_x"] = row["eager_ms"] / row["hk_ms"]
        row["compile_x"] = row["compile_ms"] / row["hk_ms"]
    return row


PHASE_COLUMNS = [("M", "M", "d"), ("phase", "phase", ""), ("HK launches", "hk_launches", "d"),
                 ("HK ms", "hk_ms", ".3f"), ("eager ms", "eager_ms", ".3f"), ("compile ms", "compile_ms", ".3f"),
                 ("fusion-only ms", "fusion_ms", ".3f"), ("eager/HK", "eager_x", ".2f"),
                 ("compile/HK", "compile_x", ".2f"), ("fusion-only/HK", "fusion_x", ".2f"), ("rel err", "rel", ".1e"),
                 ("status", "status", "")]


def run_full(L, timer, phase_rows):
    hk_fn = lambda: L.hk_layer(L.x, L.xg, L.r_attn)
    eager = lambda: L.t_layer(L.x)
    compiled = torch.compile(L.t_layer, **COMPILE)
    comp = lambda: compiled(L.x)
    want = L.ref_layer(L.x)
    ok_hk, rel_hk = common.layer_check(hk_fn()[:2], want)            # x_out, r_next
    ok_eager, rel_eager = common.layer_check((eager(),), want[:1])  # torch: x_out only
    ok_comp, rel_comp = common.layer_check((comp(),), want[:1])     # first call compiles
    ok = ok_hk and ok_eager and ok_comp
    row = dict(M=L.M, rel_hk=rel_hk, rel_eager=rel_eager, rel_compile=rel_comp, status="PASS" if ok else "FAIL")
    if ok:
        t_hk, t_eager, t_comp = timer(hk_fn), timer(eager), timer(comp)
        row.update(**common.ms_fields("hk", t_hk), **common.ms_fields("eager", t_eager),
                   **common.ms_fields("compile", t_comp),
                   eager_x=t_eager.median / t_hk.median, compile_x=t_comp.median / t_hk.median)
        phase_sum = next(r for r in phase_rows if r["M"] == L.M and r["phase"] == "sum")
        if phase_sum["status"] in TIMED:
            row["compile_full_minus_phases_ms"] = t_comp.median - phase_sum["compile_ms"]
    print(f"  M={L.M} full layer: {row['status']}", flush=True)
    return row


FULL_COLUMNS = [("M", "M", "d"), ("HK ms", "hk_ms", ".3f"), ("eager ms", "eager_ms", ".3f"),
                ("compile ms", "compile_ms", ".3f"), ("eager/HK", "eager_x", ".2f"), ("compile/HK", "compile_x", ".2f"),
                ("compile full - phases ms", "compile_full_minus_phases_ms", "+.3f"),
                ("rel HK", "rel_hk", ".1e"), ("rel compile", "rel_compile", ".1e"), ("status", "status", "")]


def main():
    def flags(ap):
        ap.add_argument("--phases", action="store_true", help="per-phase table")
        ap.add_argument("--full", action="store_true", help="full-layer table")

    args = common.parse_args("Forward layer: HK vs eager vs torch.compile", add=flags)
    if not (args.phases or args.full):
        raise SystemExit("choose --phases and/or --full")
    common.require_epilogue_modules(MODULES)
    torch.manual_seed(0)
    torch._dynamo.config.cache_size_limit = 64
    cfg = args.cfg
    backend, native = choose_sdpa(cfg["H"], cfg["H_KV"], cfg["Dh"])
    print(f"SDPA baseline: {backend}, native GQA: {native}")
    timer = common.Timer(args.warmup, args.iters)
    phase_rows, full_rows = [], []
    for M in args.M:
        L = Layer(cfg, M, load_gqa(cfg, M), native)
        phase_rows += run_phases(L, timer)
        if args.full:
            full_rows.append(run_full(L, timer, phase_rows))
        del L
        torch.cuda.empty_cache()
    title = f"({args.config}, cold cache; SDPA {backend}, native GQA {native})"
    if args.phases:
        common.print_table(f"Layer phases {title}", PHASE_COLUMNS, phase_rows, args.markdown)
    if args.full:
        common.print_table(f"Full layer {title}", FULL_COLUMNS, full_rows, args.markdown)
    if args.json:
        common.write_json(args.json, "bench_layer", args.config, phase_rows + full_rows,
                          extra=dict(sdpa_backend=backend, native_gqa=native))
    raise SystemExit(common.exit_code(phase_rows + full_rows))


if __name__ == "__main__":
    main()
