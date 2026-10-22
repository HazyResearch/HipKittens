"""hk: Python API over the compiled GEMM-epilogue kernels (the tk_<name> modules).

Weights are stored as [N, K] bf16, the layout of nn.Linear.weight; torch computes the same product as
x @ W.t(). Label a weight once, then call:

    fold = lambda W: hk.weight(hk.fold_gamma(W, g), gamma_folded=True)   # W: [N, K]; g: the norm's gain [K]
    y  = hk.matmul(x, fold(Wgu), epilogue="rmsnorm_swiglu", r=r)
    v  = hk.matmul(x, fold(Wv), epilogue="rmsnorm", r=r)                  # rmsnorm(x, g) @ Wv.t()
    q  = hk.matmul(x, fold(Wq), epilogue="rmsnorm_rope", r=r, cos_sin=cs)  # cs: one head's [M, Dh]
    r  = hk.inv_rms(x, hk.weight(W2))                    # [M] fp32 = 1 / rms(x @ W2.t())
    h, r = hk.residual_rms(x, hk.weight(Wo), residual)   # h = x @ Wo.t() + residual, r = 1 / rms(h)
    wqkv = hk.prepare_qkv(fold(Wq), fold(Wk), fold(Wv))  # concatenates [Wq; Wk; Wv], once
    q, k, v = hk.qkv(x, wqkv, r=r, cos_sin=cs)            # one GEMM launch for all three; V is not rotated
    hk.available()                                       # {epilogue: [argument names]}

Every call launches on torch.cuda.current_stream() and does not synchronize. Inputs are never
converted: a wrong dtype, shape or device raises. A scalar `alpha` may be a Python number.
Shapes: M and N multiples of 256, K a multiple of 128. The tk_<name> modules must be built
(make KERNEL=<name>) and importable.
"""
import importlib
from dataclasses import dataclass

import torch

BF16, FP32 = torch.bfloat16, torch.float32


@dataclass(frozen=True)
class Weight:
    """A weight as the kernels read it: [N, K] bf16, labelled with whether the RMSNorm gain is folded in."""
    tensor: torch.Tensor
    gamma_folded: bool


@dataclass(frozen=True)
class QkvWeight:
    """[Wq; Wk; Wv] from hk.prepare_qkv: one gamma-folded weight plus the output column split."""
    weight: Weight
    q_cols: int
    kv_cols: int


@dataclass(frozen=True)
class _Spec:
    module: str
    args: tuple = ()             # ((name, kind), ...); kind: scalar | row | col | tile | rope
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
    "swiglu":         _Spec("tk_swiglu", half_width=True),
    "rmsnorm_swiglu": _Spec("tk_rmsnorm_swiglu", (("r", "row"),), gamma_folded=True, half_width=True),
    "rope":           _Spec("tk_rope", (("cos_sin", "rope"),)),
    "rmsnorm_rope":   _Spec("tk_rmsnorm_rope", (("r", "row"), ("cos_sin", "rope")), gamma_folded=True),
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


def weight(W, gamma_folded=False):
    """Label W [N, K] (bf16, contiguous) for the kernels. No copy: hk reads W's memory as it is.

    gamma_folded says whether the RMSNorm gain is already folded into W (hk.fold_gamma); the rmsnorm_*
    epilogues require it, the others refuse it, so a weight is never silently used with the wrong scale."""
    _require(W, "W", BF16, (None, None))
    N, K = W.shape
    if K % 128 or N % 256:
        raise ValueError(f"W must be [N, K] with N % 256 == 0 and K % 128 == 0, got {tuple(W.shape)}")
    return Weight(W, bool(gamma_folded))


def fold_gamma(W, gamma):
    """W [N, K] bf16 with the RMSNorm gain gamma (bf16 [K]) folded into its columns: rmsnorm(x, gamma) @ W.t()
    == r * (x @ (W * gamma).t()). Computed in fp32 and rounded to bf16 once. Returns a new tensor; do this once
    per weight, and give the same folded tensor to every implementation you compare."""
    _require(W, "W", BF16, (None, None))
    _require(gamma, "gamma", BF16, (W.shape[1],), W.device)
    return (W.float() * gamma.float()[None, :]).to(BF16)


def _argument(name, kind, value, M, N, device):
    if kind == "scalar":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return torch.full((1,), float(value), dtype=FP32, device=device)
        _require(value, name, FP32, (1,), device)
        return value
    if kind == "rope":                                    # one head's natural interleaved table, shared by heads
        _require(value, name, BF16, (M, None), device)
        dh = value.shape[1]
        if dh == 0 or dh % 2 or N % dh:
            raise ValueError(f"{name} must be one head's [M, Dh] table with Dh even and N % Dh == 0, got "
                             f"{tuple(value.shape)} for N={N}")
        if value.data_ptr() % 4:                          # the kernel reads [cos, sin] as one bf16x2 word
            raise ValueError(f"{name} must start at a 4-byte aligned address")
        return value
    dtype, shape = {"row": (FP32, (M,)), "col": (BF16, (N,)), "tile": (BF16, (M, N))}[kind]
    _require(value, name, dtype, shape, device)
    return value


def _check_weight(w, gamma_folded, what):
    if not isinstance(w, Weight):
        raise TypeError(f"w must come from hk.weight(...), got {type(w).__name__}")
    if w.gamma_folded != gamma_folded:
        need = ("gamma folded in (hk.weight(hk.fold_gamma(W, g), gamma_folded=True))" if gamma_folded
                else "no gamma folded in")
        raise ValueError(f"{what} needs a weight with {need}")


def matmul(x, w, epilogue=None, **args):
    """GEMM x @ W.t() with a fused epilogue; returns a new bf16 output ([M, N], or [M, N/2] for swiglu variants).

    x: bf16 [M, K]. w: hk.weight of W [N, K], with gamma folded in when the epilogue needs it. args: the epilogue's
    inputs by name (see hk.available()). Launches on torch.cuda.current_stream()."""
    return _launch(x, w, epilogue, args)


def _launch(x, w, epilogue, args, rope_cols=None):
    """matmul's body. rope_cols (rope epilogues only): rotate columns [0, rope_cols), default all N."""
    name = "noop" if epilogue is None else epilogue
    spec = _EPILOGUES.get(name)
    if spec is None:
        raise ValueError(f"unknown epilogue {epilogue!r}; available: {sorted(_EPILOGUES)}")
    _check_weight(w, spec.gamma_folded, f"epilogue {name!r}")
    expected = [a for a, _ in spec.args]
    if sorted(args) != sorted(expected):
        raise TypeError(f"epilogue {name!r} takes arguments {expected}, got {sorted(args)}")
    N, K = w.tensor.shape
    _require(x, "x", BF16, (None, K))
    _require(w.tensor, "w", BF16, (N, K), x.device)
    M = x.shape[0]
    values = [_argument(a, kind, args[a], M, N, x.device) for a, kind in spec.args]
    if any(kind == "rope" for _, kind in spec.args):
        values.append(N if rope_cols is None else rope_cols)
    out = torch.empty(M, N // 2 if spec.half_width else N, dtype=BF16, device=x.device)
    with torch.cuda.device(x.device):                     # launch on x's GPU, on its current stream
        stream = torch.cuda.current_stream(x.device).cuda_stream
        if spec.unit_gamma:
            values.append(_ones(N, x.device))
        importlib.import_module(spec.module).dispatch(x, w.tensor, out, *values, stream)
    return out


def inv_rms(x, w):
    """r = 1 / rms(x @ W.t()) per row, fp32 [M] (eps 1e-5), via partialrms + rms_reduce. Needs a weight without
    gamma. Does not return x @ W.t() itself."""
    _check_weight(w, False, "inv_rms")
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
    """h = x @ W.t() + residual (bf16 [M, N], the next residual stream) and r = 1 / rms(h) per row (fp32 [M],
    eps 1e-5, from h before its bf16 rounding), via the residual_rms GEMM + rms_reduce. Needs a weight
    without gamma; fold the norm's gamma into the next weight (hk.fold_gamma)."""
    _check_weight(w, False, "residual_rms")
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


def prepare_qkv(wq, wk, wv):
    """Fuse the attention projection weights into one operand for hk.qkv, once: concatenates [Wq; Wk; Wv] along N
    (a copy). wq [Nq, K], wk and wv [Nkv, K]: hk.weight's, gamma folded in, as for "rmsnorm_rope". Nq and Nkv are
    multiples of 256 like any hk weight, so each part is a shape the separate calls also accept."""
    for name, w in (("Wq", wq), ("Wk", wk), ("Wv", wv)):
        _check_weight(w, True, f"prepare_qkv's {name}")
    Nq, K = wq.tensor.shape
    _require(wk.tensor, "Wk", BF16, (None, K), wq.tensor.device)
    Nkv = wk.tensor.shape[0]
    _require(wv.tensor, "Wv", BF16, (Nkv, K), wq.tensor.device)
    return QkvWeight(Weight(torch.cat([wq.tensor, wk.tensor, wv.tensor], dim=0), True), Nq, Nkv)


def qkv(x, w, r, cos_sin):
    """q, k, v = RoPE(r * (x @ Wq)), RoPE(r * (x @ Wk)), r * (x @ Wv), gamma folded into all three, as ONE GEMM
    launch ("rmsnorm_rope" over [Wq; Wk; Wv], rotating only the Q and K columns); then q, k and v are copied out
    as contiguous bf16 [M, Nq], [M, Nkv], [M, Nkv] (attention takes contiguous inputs). w from hk.prepare_qkv,
    cos_sin one head's natural [M, Dh] table, r = 1 / rms(x) fp32 [M]. q and k equal separate "rmsnorm_rope" calls
    bitwise; v is within one bf16 step of the "rmsnorm" call (this store rounds to nearest, that one truncates)."""
    if not isinstance(w, QkvWeight):
        raise TypeError(f"w must come from hk.prepare_qkv(...), got {type(w).__name__}")
    nq, nkv = w.q_cols, w.kv_cols
    if isinstance(cos_sin, torch.Tensor) and cos_sin.dim() == 2 and cos_sin.shape[1] and (
            nq % cos_sin.shape[1] or nkv % cos_sin.shape[1]):
        raise ValueError(f"cos_sin's head dim {cos_sin.shape[1]} must divide Nq={nq} and Nkv={nkv}, so every "
                         f"Q and K head starts at table column 0")
    out = _launch(x, w.weight, "rmsnorm_rope", dict(r=r, cos_sin=cos_sin), rope_cols=nq + nkv)
    return out[:, :nq].contiguous(), out[:, nq:nq + nkv].contiguous(), out[:, nq + nkv:].contiguous()
