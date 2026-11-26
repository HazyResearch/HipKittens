"""hk: Python API over the tk_<name> GEMM-epilogue kernels.

Weights are plain [N, K] bf16 (nn.Linear.weight layout). An RMSNorm is split across two GEMMs: the producer
writes x * gamma and r = 1 / rms(x); the consumer applies r: rmsnorm(x, gamma) @ W.t() == r * ((x * gamma) @ W.t()).

    h, hg, r = hk.residual_rms(o, Wo, x, gamma=g_mlp)    # h = o @ Wo.t() + x, hg = h * g_mlp, r = 1 / rms(h)
    y  = hk.matmul(hg, Wgu, epilogue="rmsnorm_swiglu", r=r)
    v  = hk.matmul(xg, Wv, epilogue="rmsnorm", r=r)                    # xg = x * gamma
    q  = hk.matmul(xg, Wq, epilogue="rmsnorm_rope", r=r, cos_sin=cs)   # cs: one head's [M, Dh]
    q, k, v = hk.qkv(xg, Wq, Wk, Wv, r=r, cos_sin=cs)    # one launch; V not rotated
    r  = hk.inv_rms(x, W2)                               # [M] fp32 = 1 / rms(x @ W2.t())
    h, _, r = hk.residual_rms(x, Wo, residual)           # without gamma: hg is None
    hk.available()                                       # {epilogue: [argument names]}

Calls launch on the inputs' GPU current stream without syncing. Bad dtype/shape/device: ValueError; bad argument
name: TypeError. `alpha` may be a Python number. M, N % 256 == 0, K % 128 == 0. Build with make KERNEL=<name>.
"""
import importlib
from dataclasses import dataclass
from typing import NamedTuple, Optional

import torch

BF16, FP32 = torch.bfloat16, torch.float32


@dataclass(frozen=True)
class _Spec:
    module: str
    args: tuple = ()             # ((name, kind), ...); kind: scalar | row | col | tile | rope
    half_width: bool = False
    unit_gamma: bool = False     # pass ones as the kernel's out_gamma


_EPILOGUES = {
    "noop":           _Spec("tk_noop"),
    "silu":           _Spec("tk_silu"),
    "scale":          _Spec("tk_scale", (("alpha", "scalar"),)),
    "residual_add":   _Spec("tk_residual_add", (("residual", "tile"),)),
    "rmsnorm_scale":  _Spec("tk_rmsnorm_scale", (("r", "row"), ("out_gamma", "col"))),   # output RMSNorm
    "rmsnorm":        _Spec("tk_rmsnorm_scale", (("r", "row"),), unit_gamma=True),
    "swiglu":         _Spec("tk_swiglu", half_width=True),
    "rmsnorm_swiglu": _Spec("tk_rmsnorm_swiglu", (("r", "row"),), half_width=True),
    "rope":           _Spec("tk_rope", (("cos_sin", "rope"),)),
    "rmsnorm_rope":   _Spec("tk_rmsnorm_rope", (("r", "row"), ("cos_sin", "rope"))),
}

_ONES = {}


def _ones(n, device):
    """Cached bf16 ones [n]. Not cached during graph capture (filled only on replay); first fill is synced."""
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
    """Raise unless t is a contiguous GPU `dtype` tensor of `shape` (None = any) on `device`."""
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
    """Check x [M, K] and w [N, K]; return (M, N, K)."""
    _require(w, name, BF16, (None, None))
    N, K = w.shape
    if K % 128 or N % 256:
        raise ValueError(f"{name} must be [N, K] with N % 256 == 0 and K % 128 == 0, got {tuple(w.shape)}")
    _require(x, "x", BF16, (None, K))
    if x.shape[0] % 256:
        raise ValueError(f"x must be [M, K] with M % 256 == 0, got {tuple(x.shape)}")
    _require(w, name, BF16, (N, K), x.device)
    return x.shape[0], N, K


def _argument(name, kind, value, M, N, device):
    if kind == "scalar":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return torch.full((1,), float(value), dtype=FP32, device=device)
        _require(value, name, FP32, (1,), device)
        return value
    if kind == "rope":                                    # one head's [M, Dh] table
        _require(value, name, BF16, (M, None), device)
        dh = value.shape[1]
        if dh == 0 or dh % 2 or N % dh:
            raise ValueError(f"{name} must be one head's [M, Dh] table with Dh even and N % Dh == 0, got "
                             f"{tuple(value.shape)} for N={N}")
        if value.data_ptr() % 4:                          # read as bf16x2
            raise ValueError(f"{name} must start at a 4-byte aligned address")
        return value
    dtype, shape = {"row": (FP32, (M,)), "col": (BF16, (N,)), "tile": (BF16, (M, N))}[kind]
    _require(value, name, dtype, shape, device)
    return value


def matmul(x, w, epilogue=None, **args):
    """x @ w.t() with a fused epilogue; returns bf16 [M, N] ([M, N/2] for swiglu variants).

    x: bf16 [M, K], w: bf16 [N, K]. For "rmsnorm*" except "rmsnorm_scale", x must already be times gamma (an
    unscaled x silently drops gamma). "rmsnorm_scale" normalizes the output: raw x, r = hk.inv_rms(x, w), out_gamma [N].
    args: the epilogue's inputs (hk.available())."""
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
        values.append(N)                                  # rope_cols
    out = torch.empty(M, N // 2 if spec.half_width else N, dtype=BF16, device=x.device)
    with torch.cuda.device(x.device):                     # x's GPU, current stream
        stream = torch.cuda.current_stream(x.device).cuda_stream
        if spec.unit_gamma:
            values.append(_ones(N, x.device))
        importlib.import_module(spec.module).dispatch(x, w, out, *values, stream)
    return out


def inv_rms(x, w):
    """r = 1 / rms(x @ w.t()) per row, fp32 [M], eps 1e-5."""
    M, N, K = _weight(x, w)
    partials = torch.empty(N // 64, M, dtype=FP32, device=x.device)
    r = torch.empty(M, dtype=FP32, device=x.device)
    with torch.cuda.device(x.device):                     # x's GPU, current stream
        stream = torch.cuda.current_stream(x.device).cuda_stream
        importlib.import_module("tk_partialrms").dispatch(x, w, partials, stream)
        importlib.import_module("tk_rms_reduce").dispatch(partials, r, stream)
    return r


class ResidualRMS(NamedTuple):
    """(h, hg, r); hg is None without gamma."""
    h: torch.Tensor                 # bf16 [M, N]
    hg: Optional[torch.Tensor]      # bf16 [M, N] h * gamma, or None
    r: torch.Tensor                 # fp32 [M], 1 / rms(h)


def residual_rms(x, w, residual, gamma=None):
    """h = x @ w.t() + residual (bf16 [M, N]), r = 1 / rms(h) (fp32 [M], eps 1e-5, from fp32 h).

    With gamma (bf16 [N]) also writes hg = h * gamma, else hg is None."""
    M, N, K = _weight(x, w)
    _require(residual, "residual", BF16, (M, N), x.device)
    if gamma is not None:
        _require(gamma, "gamma", BF16, (N,), x.device)
    h = torch.empty(M, N, dtype=BF16, device=x.device)
    partials = torch.empty(N // 64, M, dtype=FP32, device=x.device)
    r = torch.empty(M, dtype=FP32, device=x.device)
    with torch.cuda.device(x.device):                     # x's GPU, current stream
        stream = torch.cuda.current_stream(x.device).cuda_stream
        if gamma is None:
            importlib.import_module("tk_residual_rms").dispatch(x, w, h, residual, partials, stream)
        else:
            hg = torch.empty_like(h)
            importlib.import_module("tk_residual_rms_gamma").dispatch(x, w, h, residual, partials, gamma, hg, stream)
        importlib.import_module("tk_rms_reduce").dispatch(partials, r, stream)
    return ResidualRMS(h, None if gamma is None else hg, r)


def qkv(x, wq, wk, wv, r, cos_sin):
    """q, k, v = RoPE(r * x @ wq.t()), RoPE(r * x @ wk.t()), r * x @ wv.t() in one GEMM launch.

    x: input times gamma; wq [Nq, K], wk/wv [Nkv, K]; cos_sin one head's [M, Dh]; r fp32 [M]. Returns contiguous
    bf16 [M, Nq], [M, Nkv], [M, Nkv]. q, k match "rmsnorm_rope" bitwise; v is within one bf16 step of "rmsnorm"
    (rounds to nearest; that one truncates)."""
    M, nq, K = _weight(x, wq, "wq")
    _, nkv, _ = _weight(x, wk, "wk")
    _require(wv, "wv", BF16, (nkv, K), x.device)
    _require(r, "r", FP32, (M,), x.device)
    _require(cos_sin, "cos_sin", BF16, (M, None), x.device)
    dh = cos_sin.shape[1]
    if dh == 0 or dh % 2 or nq % dh or nkv % dh:
        raise ValueError(f"cos_sin's head dim {dh} must be even and divide Nq={nq} and Nkv={nkv}, so every "
                         f"Q and K head starts at table column 0")
    if cos_sin.data_ptr() % 4:                            # read as bf16x2
        raise ValueError("cos_sin must start at a 4-byte aligned address")
    out = torch.empty(M, nq + 2 * nkv, dtype=BF16, device=x.device)
    with torch.cuda.device(x.device):                     # x's GPU, current stream
        stream = torch.cuda.current_stream(x.device).cuda_stream
        importlib.import_module("tk_rmsnorm_rope_qkv").dispatch(x, wq, out, r, cos_sin, nq + nkv, wk, wv, stream)
    return out[:, :nq].contiguous(), out[:, nq:nq + nkv].contiguous(), out[:, nq + nkv:].contiguous()
