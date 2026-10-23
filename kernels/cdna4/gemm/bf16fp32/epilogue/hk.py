"""hk: Python API over the compiled GEMM-epilogue kernels (the tk_<name> modules).

Weights are plain [N, K] bf16 tensors, the layout of nn.Linear.weight, read as they are: there is no weight
preparation. An RMSNorm is split across two GEMMs: the PRODUCER of its input (a residual_rms GEMM) also writes
x * gamma and the per-row partial sums that give r = 1 / rms(x); the CONSUMER multiplies x * gamma by its weight and
applies r in its epilogue. So rmsnorm(x, gamma) @ W.t() == r * ((x * gamma) @ W.t()).

    h, hg, r = hk.residual_rms(o, Wo, x, gamma=g_mlp)    # h = o @ Wo.t() + x, hg = h * g_mlp, r = 1 / rms(h)
    y  = hk.matmul(hg, Wgu, epilogue="rmsnorm_swiglu", r=r)
    v  = hk.matmul(xg, Wv, epilogue="rmsnorm", r=r)                    # xg = x * gamma, from its producer
    q  = hk.matmul(xg, Wq, epilogue="rmsnorm_rope", r=r, cos_sin=cs)   # cs: one head's [M, Dh]
    q, k, v = hk.qkv(xg, Wq, Wk, Wv, r=r, cos_sin=cs)    # one GEMM launch over the three weights; V is not rotated
    r  = hk.inv_rms(x, W2)                               # [M] fp32 = 1 / rms(x @ W2.t())
    h, r = hk.residual_rms(x, Wo, residual)              # without gamma: h and r only
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
class _Spec:
    module: str
    args: tuple = ()             # ((name, kind), ...); kind: scalar | row | col | tile | rope
    half_width: bool = False
    unit_gamma: bool = False     # the kernel takes an output-column gamma; pass ones (gamma came with the input)


_EPILOGUES = {
    "noop":           _Spec("tk_noop"),
    "silu":           _Spec("tk_silu"),
    "scale":          _Spec("tk_scale", (("alpha", "scalar"),)),
    "residual_add":   _Spec("tk_residual_add", (("residual", "tile"),)),
    "rmsnorm_scale":  _Spec("tk_rmsnorm_scale", (("r", "row"), ("gamma", "col"))),
    "rmsnorm":        _Spec("tk_rmsnorm_scale", (("r", "row"),), unit_gamma=True),
    "swiglu":         _Spec("tk_swiglu", half_width=True),
    "rmsnorm_swiglu": _Spec("tk_rmsnorm_swiglu", (("r", "row"),), half_width=True),
    "rope":           _Spec("tk_rope", (("cos_sin", "rope"),)),
    "rmsnorm_rope":   _Spec("tk_rmsnorm_rope", (("r", "row"), ("cos_sin", "rope"))),
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


def _weight(x, w, name="w"):
    """Check x [M, K] and the weight w [N, K] (bf16, contiguous, N % 256 == 0, K % 128 == 0); return (M, N, K)."""
    _require(w, name, BF16, (None, None))
    N, K = w.shape
    if K % 128 or N % 256:
        raise ValueError(f"{name} must be [N, K] with N % 256 == 0 and K % 128 == 0, got {tuple(w.shape)}")
    _require(x, "x", BF16, (None, K))
    _require(w, name, BF16, (N, K), x.device)
    return x.shape[0], N, K


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


def matmul(x, w, epilogue=None, **args):
    """GEMM x @ w.t() with a fused epilogue; returns a new bf16 output ([M, N], or [M, N/2] for swiglu variants).

    x: bf16 [M, K]; for the rmsnorm* epilogues, the RMSNorm input already multiplied by its gamma (as
    hk.residual_rms(..., gamma=g) writes it). w: bf16 [N, K]. args: the epilogue's inputs by name (see
    hk.available()). Launches on torch.cuda.current_stream()."""
    name = "noop" if epilogue is None else epilogue
    spec = _EPILOGUES.get(name)
    if spec is None:
        raise ValueError(f"unknown epilogue {epilogue!r}; available: {sorted(_EPILOGUES)}")
    expected = [a for a, _ in spec.args]
    if sorted(args) != sorted(expected):
        raise TypeError(f"epilogue {name!r} takes arguments {expected}, got {sorted(args)}")
    M, N, K = _weight(x, w)
    values = [_argument(a, kind, args[a], M, N, x.device) for a, kind in spec.args]
    if any(kind == "rope" for _, kind in spec.args):
        values.append(N)                                  # rotate every column
    out = torch.empty(M, N // 2 if spec.half_width else N, dtype=BF16, device=x.device)
    with torch.cuda.device(x.device):                     # launch on x's GPU, on its current stream
        stream = torch.cuda.current_stream(x.device).cuda_stream
        if spec.unit_gamma:
            values.append(_ones(N, x.device))
        importlib.import_module(spec.module).dispatch(x, w, out, *values, stream)
    return out


def inv_rms(x, w):
    """r = 1 / rms(x @ w.t()) per row, fp32 [M] (eps 1e-5), via partialrms + rms_reduce. Does not return
    x @ w.t() itself."""
    M, N, K = _weight(x, w)
    partials = torch.empty(N // 64, M, dtype=FP32, device=x.device)
    r = torch.empty(M, dtype=FP32, device=x.device)
    with torch.cuda.device(x.device):                     # launch on x's GPU, on its current stream
        stream = torch.cuda.current_stream(x.device).cuda_stream
        importlib.import_module("tk_partialrms").dispatch(x, w, partials, stream)
        importlib.import_module("tk_rms_reduce").dispatch(partials, r, stream)
    return r


def residual_rms(x, w, residual, gamma=None):
    """h = x @ w.t() + residual (bf16 [M, N], the next residual stream) and r = 1 / rms(h) per row (fp32 [M],
    eps 1e-5, from h before its bf16 rounding), via one GEMM + rms_reduce. Returns (h, r).

    With gamma (bf16 [N], the gain of the RMSNorm that h feeds), the same GEMM also writes hg = h * gamma, the
    input of the consuming GEMM, and returns (h, hg, r). r is always computed from h, not hg."""
    M, N, K = _weight(x, w)
    _require(residual, "residual", BF16, (M, N), x.device)
    if gamma is not None:
        _require(gamma, "gamma", BF16, (N,), x.device)
    h = torch.empty(M, N, dtype=BF16, device=x.device)
    partials = torch.empty(N // 64, M, dtype=FP32, device=x.device)
    r = torch.empty(M, dtype=FP32, device=x.device)
    with torch.cuda.device(x.device):                     # launch on x's GPU, on its current stream
        stream = torch.cuda.current_stream(x.device).cuda_stream
        if gamma is None:
            importlib.import_module("tk_residual_rms").dispatch(x, w, h, residual, partials, stream)
        else:
            hg = torch.empty_like(h)
            importlib.import_module("tk_residual_rms_gamma").dispatch(x, w, h, residual, partials, gamma, hg, stream)
        importlib.import_module("tk_rms_reduce").dispatch(partials, r, stream)
    return (h, r) if gamma is None else (h, hg, r)


def qkv(x, wq, wk, wv, r, cos_sin):
    """q, k, v = RoPE(r * (x @ wq.t())), RoPE(r * (x @ wk.t())), r * (x @ wv.t()) as ONE GEMM launch that reads the
    three weights where they are (no concatenated copy), rotating only the Q and K columns; then q, k and v are
    copied out as contiguous bf16 [M, Nq], [M, Nkv], [M, Nkv] (attention takes contiguous inputs). x is the
    attention RMSNorm's input times its gamma; wq [Nq, K], wk and wv [Nkv, K]; cos_sin one head's natural [M, Dh]
    table; r = 1 / rms(x) fp32 [M]. q and k equal separate "rmsnorm_rope" calls bitwise; v is within one bf16 step
    of the "rmsnorm" call (this store rounds to nearest, that one truncates)."""
    M, nq, K = _weight(x, wq, "wq")
    _, nkv, _ = _weight(x, wk, "wk")
    _require(wv, "wv", BF16, (nkv, K), x.device)
    _require(r, "r", FP32, (M,), x.device)
    _require(cos_sin, "cos_sin", BF16, (M, None), x.device)
    dh = cos_sin.shape[1]
    if dh == 0 or dh % 2 or nq % dh or nkv % dh:
        raise ValueError(f"cos_sin's head dim {dh} must be even and divide Nq={nq} and Nkv={nkv}, so every "
                         f"Q and K head starts at table column 0")
    if cos_sin.data_ptr() % 4:                            # the kernel reads [cos, sin] as one bf16x2 word
        raise ValueError("cos_sin must start at a 4-byte aligned address")
    out = torch.empty(M, nq + 2 * nkv, dtype=BF16, device=x.device)
    with torch.cuda.device(x.device):                     # launch on x's GPU, on its current stream
        stream = torch.cuda.current_stream(x.device).cuda_stream
        importlib.import_module("tk_rmsnorm_rope_qkv").dispatch(x, wq, out, r, cos_sin, nq + nkv, wk, wv, stream)
    return out[:, :nq].contiguous(), out[:, nq:nq + nkv].contiguous(), out[:, nq + nkv:].contiguous()
