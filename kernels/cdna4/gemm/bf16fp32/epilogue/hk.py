"""hk: Python API over the compiled GEMM-epilogue kernels (the tk_<name> modules).

Prepare static weights once, then call:

    w  = hk.prepare(W, layout="swiglu", gamma=g)        # W: natural [K, N] bf16
    cs = hk.prepare_rope_table(cos_sin)                  # natural [M, N] bf16 [cos, sin] table
    y  = hk.matmul(x, w, epilogue="rmsnorm_swiglu", r=r)
    v  = hk.matmul(x, hk.prepare(Wv, gamma=g), epilogue="rmsnorm", r=r)   # rmsnorm(x, g) @ Wv
    r  = hk.inv_rms(x, hk.prepare(W2))                   # [M] fp32 = 1 / rms(x @ W2)
    h, r = hk.residual_rms(x, hk.prepare(Wo), residual)  # h = x @ Wo + residual, r = 1 / rms(h)
    wqkv = hk.prepare_qkv(Wq, Wk, Wv, gamma=g)            # fused attention projections, once
    cs  = hk.prepare_qkv_rope_table(cos_sin_q, cos_sin_k) # V gets the identity rotation
    q, k, v = hk.qkv(x, wqkv, r=r, cos_sin=cs)            # one GEMM launch for all three
    hk.available()                                       # {epilogue: [argument names]}

Every call launches on torch.cuda.current_stream() and does not synchronize. Inputs are never
converted: a wrong dtype, shape, device or layout raises. A scalar `alpha` may be a Python number.
Shapes: M and N multiples of 256, K a multiple of 128. The tk_<name> modules must be built
(make KERNEL=<name>) and importable.
"""
import importlib
from dataclasses import dataclass

import torch

BF16, FP32 = torch.bfloat16, torch.float32


@dataclass(frozen=True)
class PreparedWeight:
    """A weight in kernel form: [N, K] bf16, columns permuted for `layout`, gamma optionally folded."""
    tensor: torch.Tensor
    layout: str
    gamma_folded: bool


@dataclass(frozen=True)
class RopeTable:
    """An [M, N] bf16 interleaved [cos, sin] table, columns permuted like a rope-layout weight."""
    tensor: torch.Tensor


@dataclass(frozen=True)
class QkvWeight:
    """[Wq | Wk | Wv] from hk.prepare_qkv: one rope-layout, gamma-folded weight plus the column split."""
    weight: PreparedWeight
    q_cols: int
    kv_cols: int


@dataclass(frozen=True)
class _Spec:
    module: str
    args: tuple = ()             # ((name, kind), ...); kind: scalar | row | col | tile | rope
    layout: str = "plain"
    gamma_folded: bool = False
    half_width: bool = False
    unit_gamma: bool = False     # the kernel takes an output-column gamma; pass ones (gamma is folded into w)


_EPILOGUES = {
    "noop":           _Spec("tk_noop"),
    "silu":           _Spec("tk_silu"),
    "scale":          _Spec("tk_scale", (("alpha", "scalar"),)),
    "residual_add":   _Spec("tk_residual_add", (("residual", "tile"),)),
    "rmsnorm_scale":  _Spec("tk_rmsnorm_scale", (("r", "row"), ("gamma", "col"))),
    "rmsnorm":        _Spec("tk_rmsnorm_scale", (("r", "row"),), gamma_folded=True, unit_gamma=True),
    "swiglu":         _Spec("tk_swiglu", layout="swiglu", half_width=True),
    "rmsnorm_swiglu": _Spec("tk_rmsnorm_swiglu", (("r", "row"),), layout="swiglu", gamma_folded=True, half_width=True),
    "rope":           _Spec("tk_rope", (("cos_sin", "rope"),), layout="rope"),
    "rmsnorm_rope":   _Spec("tk_rmsnorm_rope", (("r", "row"), ("cos_sin", "rope")), layout="rope", gamma_folded=True),
}

_ONES = {}


def _ones(n, device):
    """bf16 ones [n] on device, made once. Not cached while a CUDA graph is capturing: a tensor made during
    capture is only filled when the graph replays. The first eager fill is synchronized so a later call on
    another stream cannot read it early."""
    key = (n, device)
    if key not in _ONES:
        t = torch.ones(n, dtype=BF16, device=device)
        if torch.cuda.is_current_stream_capturing():
            return t
        torch.cuda.current_stream(device).synchronize()
        _ONES[key] = t
    return _ONES[key]


def available():
    """{epilogue name: [argument names]} accepted by hk.matmul (epilogue=None means noop)."""
    return {name: [a for a, _ in spec.args] for name, spec in _EPILOGUES.items()}


def _require(t, name, dtype, shape, device=None):
    """Raise unless t is a contiguous GPU tensor of `dtype` and `shape` (None matches any extent),
    on `device` when given."""
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(t).__name__}")
    shape_ok = t.dim() == len(shape) and all(s is None or s == d for s, d in zip(shape, t.shape))
    if t.dtype != dtype or not t.is_cuda or not t.is_contiguous() or not shape_ok:
        want = tuple("*" if s is None else s for s in shape)
        raise ValueError(f"{name} must be a contiguous {dtype} GPU tensor of shape {want}; got {t.dtype}, "
                         f"{'GPU' if t.is_cuda else 'CPU'}, {'contiguous' if t.is_contiguous() else 'non-contiguous'}, "
                         f"shape {tuple(t.shape)}")
    if device is not None and t.device != device:
        raise ValueError(f"{name} must be on {device} (the same device as x), got {t.device}")


def _pair_slots(pairs, device):
    """Column of each pair's first member after permutation: pair j -> (j/128)*256 + j%128."""
    j = torch.arange(pairs, device=device)
    return (j // 128) * 256 + j % 128


def _swiglu_perm(n, device):
    """perm[new_col] = old_col: gate j (old col j) -> slot(j), value j (old col n/2 + j) -> slot(j) + 128."""
    slot = _pair_slots(n // 2, device)
    perm = torch.empty(n, dtype=torch.long, device=device)
    perm[slot] = torch.arange(n // 2, device=device)
    perm[slot + 128] = torch.arange(n // 2, n, device=device)
    return perm


def _rope_perm(n, device):
    """perm[new_col] = old_col: pair k's even col 2k -> slot(k), odd col 2k+1 -> slot(k) + 128."""
    slot = _pair_slots(n // 2, device)
    k = torch.arange(n // 2, device=device)
    perm = torch.empty(n, dtype=torch.long, device=device)
    perm[slot] = 2 * k
    perm[slot + 128] = 2 * k + 1
    return perm


def prepare(W, layout="plain", gamma=None):
    """Turn a natural weight W [K, N] (bf16) into the kernel operand, once.

    gamma (bf16 [K]) is folded into W's rows (rmsnorm(x, gamma) @ W == r * (x @ (gamma * W))), for the
    rmsnorm_swiglu / rmsnorm_rope epilogues. layout "swiglu" or "rope" permutes the columns so each pair
    sits in one thread; "plain" leaves them. The result is transposed to [N, K]."""
    if layout not in ("plain", "swiglu", "rope"):
        raise ValueError(f"layout must be 'plain', 'swiglu' or 'rope', got {layout!r}")
    _require(W, "W", BF16, (None, None))
    K, N = W.shape
    if K % 128 or N % 256:
        raise ValueError(f"W must be [K, N] with K % 128 == 0 and N % 256 == 0, got {tuple(W.shape)}")
    if gamma is not None:
        _require(gamma, "gamma", BF16, (K,), W.device)
        W = (W.float() * gamma.float()[:, None]).to(BF16)
    if layout == "swiglu":
        W = W[:, _swiglu_perm(N, W.device)]
    elif layout == "rope":
        W = W[:, _rope_perm(N, W.device)]
    return PreparedWeight(W.t().contiguous(), layout, gamma is not None)


def prepare_rope_table(cos_sin):
    """Permute a natural interleaved [cos, sin] table (bf16 [M, N]) the way rope weights are permuted, once."""
    _require(cos_sin, "cos_sin", BF16, (None, None))
    if cos_sin.shape[1] % 256:
        raise ValueError(f"cos_sin must have N % 256 == 0, got {tuple(cos_sin.shape)}")
    return RopeTable(cos_sin[:, _rope_perm(cos_sin.shape[1], cos_sin.device)].contiguous())


def _argument(name, kind, value, M, N, device):
    if kind == "scalar":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return torch.full((1,), float(value), dtype=FP32, device=device)
        _require(value, name, FP32, (1,), device)
        return value
    if kind == "rope":
        if not isinstance(value, RopeTable):
            raise TypeError(f"{name} must come from hk.prepare_rope_table(...), got {type(value).__name__}")
        _require(value.tensor, name, BF16, (M, N), device)
        return value.tensor
    dtype, shape = {"row": (FP32, (M,)), "col": (BF16, (N,)), "tile": (BF16, (M, N))}[kind]
    _require(value, name, dtype, shape, device)
    return value


def _check_weight(w, layout, gamma_folded, what):
    if not isinstance(w, PreparedWeight):
        raise TypeError(f"w must come from hk.prepare(...), got {type(w).__name__}")
    if w.layout != layout:
        raise ValueError(f"{what} needs a {layout!r} weight, got {w.layout!r} (use hk.prepare(W, layout={layout!r}))")
    if w.gamma_folded != gamma_folded:
        need = "gamma folded in (hk.prepare(W, ..., gamma=g))" if gamma_folded else "no gamma folded in"
        raise ValueError(f"{what} needs a weight with {need}")


def matmul(x, w, epilogue=None, **args):
    """GEMM x @ W with a fused epilogue; returns a new bf16 output ([M, N], or [M, N/2] for swiglu variants).

    x: bf16 [M, K]. w: from hk.prepare with the layout / gamma the epilogue needs. args: the epilogue's
    inputs by name (see hk.available()). Launches on torch.cuda.current_stream()."""
    name = "noop" if epilogue is None else epilogue
    spec = _EPILOGUES.get(name)
    if spec is None:
        raise ValueError(f"unknown epilogue {epilogue!r}; available: {sorted(_EPILOGUES)}")
    _check_weight(w, spec.layout, spec.gamma_folded, f"epilogue {name!r}")
    expected = [a for a, _ in spec.args]
    if sorted(args) != sorted(expected):
        raise TypeError(f"epilogue {name!r} takes arguments {expected}, got {sorted(args)}")
    N, K = w.tensor.shape
    _require(x, "x", BF16, (None, K))
    _require(w.tensor, "w", BF16, (N, K), x.device)
    M = x.shape[0]
    values = [_argument(a, kind, args[a], M, N, x.device) for a, kind in spec.args]
    out = torch.empty(M, N // 2 if spec.half_width else N, dtype=BF16, device=x.device)
    with torch.cuda.device(x.device):                     # launch on x's GPU, on its current stream
        stream = torch.cuda.current_stream(x.device).cuda_stream
        if spec.unit_gamma:
            values.append(_ones(N, x.device))
        importlib.import_module(spec.module).dispatch(x, w.tensor, out, *values, stream)
    return out


def inv_rms(x, w):
    """r = 1 / rms(x @ W) per row, fp32 [M] (eps 1e-5), via partialrms + rms_reduce. Needs a plain weight
    without gamma. Does not return x @ W itself."""
    _check_weight(w, "plain", False, "inv_rms")
    N, K = w.tensor.shape
    _require(x, "x", BF16, (None, K))
    _require(w.tensor, "w", BF16, (N, K), x.device)
    M = x.shape[0]
    partials = torch.empty(N // 64, M, dtype=FP32, device=x.device)
    r = torch.empty(M, dtype=FP32, device=x.device)
    with torch.cuda.device(x.device):                     # launch on x's GPU, on its current stream
        stream = torch.cuda.current_stream(x.device).cuda_stream
        importlib.import_module("tk_partialrms").dispatch(x, w.tensor, partials, stream)
        importlib.import_module("tk_rms_reduce").dispatch(partials, r, stream)
    return r


def residual_rms(x, w, residual):
    """h = x @ W + residual (bf16 [M, N], the next residual stream) and r = 1 / rms(h) per row (fp32 [M],
    eps 1e-5, from h before its bf16 rounding), via the residual_rms GEMM + rms_reduce. Needs a plain
    weight without gamma; fold the norm's gamma into the next weight (hk.prepare(W, ..., gamma=g))."""
    _check_weight(w, "plain", False, "residual_rms")
    N, K = w.tensor.shape
    _require(x, "x", BF16, (None, K))
    _require(w.tensor, "w", BF16, (N, K), x.device)
    M = x.shape[0]
    _require(residual, "residual", BF16, (M, N), x.device)
    h = torch.empty(M, N, dtype=BF16, device=x.device)
    partials = torch.empty(N // 64, M, dtype=FP32, device=x.device)
    r = torch.empty(M, dtype=FP32, device=x.device)
    with torch.cuda.device(x.device):                     # launch on x's GPU, on its current stream
        stream = torch.cuda.current_stream(x.device).cuda_stream
        importlib.import_module("tk_residual_rms").dispatch(x, w.tensor, h, residual, partials, stream)
        importlib.import_module("tk_rms_reduce").dispatch(partials, r, stream)
    return h, r


def prepare_qkv(Wq, Wk, Wv, gamma):
    """Fuse the attention projection weights into one operand for hk.qkv, once. Wq [K, Nq], Wk and Wv [K, Nkv]:
    natural bf16. gamma (bf16 [K], the attention norm's gain) is folded in, as for "rmsnorm_rope". Nq and Nkv must
    each be multiples of 256, like any hk weight, so each part is a shape the separate calls also accept."""
    _require(Wq, "Wq", BF16, (None, None))
    K, Nq = Wq.shape
    _require(Wk, "Wk", BF16, (K, None), Wq.device)
    Nkv = Wk.shape[1]
    _require(Wv, "Wv", BF16, (K, Nkv), Wq.device)
    if Nq % 256 or Nkv % 256:
        raise ValueError(f"Wq and Wk/Wv need N % 256 == 0, got Nq={Nq}, Nkv={Nkv}")
    return QkvWeight(prepare(torch.cat([Wq, Wk, Wv], dim=1), layout="rope", gamma=gamma), Nq, Nkv)


def prepare_qkv_rope_table(cos_sin_q, cos_sin_k):
    """The [cos, sin] table for hk.qkv, once per sequence length. cos_sin_q [M, Nq] and cos_sin_k [M, Nkv] are
    natural interleaved bf16 tables, as for hk.prepare_rope_table. V's columns get cos = 1, sin = 0, so RoPE
    leaves V unrotated."""
    _require(cos_sin_q, "cos_sin_q", BF16, (None, None))
    _require(cos_sin_k, "cos_sin_k", BF16, (cos_sin_q.shape[0], None), cos_sin_q.device)
    identity = torch.zeros_like(cos_sin_k)
    identity[:, 0::2] = 1
    return prepare_rope_table(torch.cat([cos_sin_q, cos_sin_k, identity], dim=1))


def qkv(x, w, r, cos_sin):
    """q, k, v = RoPE(r * (x @ Wq)), RoPE(r * (x @ Wk)), r * (x @ Wv), gamma folded into all three, as ONE GEMM
    launch ("rmsnorm_rope" over [Wq | Wk | Wv]); then q, k and v are copied out as contiguous bf16 [M, Nq],
    [M, Nkv], [M, Nkv] (attention takes contiguous inputs). w from hk.prepare_qkv, cos_sin from
    hk.prepare_qkv_rope_table, r = 1 / rms(x) fp32 [M]. q and k equal separate "rmsnorm_rope" calls bitwise; v
    is within one bf16 step of the "rmsnorm" call (this store rounds to nearest, that one truncates)."""
    if not isinstance(w, QkvWeight):
        raise TypeError(f"w must come from hk.prepare_qkv(...), got {type(w).__name__}")
    out = matmul(x, w.weight, "rmsnorm_rope", r=r, cos_sin=cos_sin)
    nq, nkv = w.q_cols, w.kv_cols
    return out[:, :nq].contiguous(), out[:, nq:nq + nkv].contiguous(), out[:, nq + nkv:].contiguous()
