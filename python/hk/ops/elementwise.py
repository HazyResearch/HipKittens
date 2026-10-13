"""Elementwise kernels, written in the DSL.

These ship with the package, so they double as the worked examples: each one is
a kernel somebody could have written, not a special case the compiler knows
about.

The kernels are 2D. An elementwise op does not care about the shape, only about
the element count, so the wrappers flatten to (rows, cols) and let the pybind
layer left-pad that to the 4D global layout. What the kernels do *not* do yet is
handle a partial tile: the tail block would read past the end of the tensor,
which reads garbage rather than failing, so `_as_2d` refuses that shape instead
of masking it. Bounds-checked tails come with the reductions in Phase 2.
"""

from __future__ import annotations

from functools import lru_cache as _lru_cache

from typing import Any, Tuple

from ..ir.nodes import bf16, fp16, fp32
from ..lang import ops as _ops
from ..lang.host import cdiv
from ..lang.kernel import GL
from ..lang.kernel import kernel as _kernel

#: Tile the elementwise kernels work in. 16x64 is 4 base tiles: 32 VGPRs for a
#: bf16 tile on gfx1100, so three live tiles sit nowhere near the granule
#: boundary, and 64 columns is wide enough that the row-layout global load
#: issues 16-byte accesses rather than falling back to the elementwise path.
ROWS, COLS = 16, 64


def _grid(p):
    return (cdiv(p.o.cols, p.COLS), cdiv(p.o.rows, p.ROWS), 1)


def _binary_kernel(name: str, op: str, dtype):
    """Build a two-input elementwise kernel for one dtype.

    A factory rather than fifteen copied function bodies: the body is four lines
    and the only thing that varies is which map it calls, so copies would only
    be a place for them to drift.
    """

    def body(a, b, o, *, ROWS=ROWS, COLS=COLS):
        t = _ops.rt(dtype, ROWS, COLS)
        idx = _ops.tile_coord(0, 0, _ops.block_idx.y, _ops.block_idx.x)
        x = _ops.load(a, idx, t)
        y = _ops.load(b, idx, t)
        _ops.store(o, getattr(_ops, op)(x, y), idx)

    body.__name__ = name
    body.__annotations__ = {
        "a": GL[dtype], "b": GL[dtype], "o": GL[dtype],
    }
    return _kernel(body, warps=1, grid=_grid, name=name)


def _unary_kernel(name: str, op: str, dtype):
    def body(a, o, *, ROWS=ROWS, COLS=COLS):
        t = _ops.rt(dtype, ROWS, COLS)
        idx = _ops.tile_coord(0, 0, _ops.block_idx.y, _ops.block_idx.x)
        _ops.store(o, getattr(_ops, op)(_ops.load(a, idx, t)), idx)

    body.__name__ = name
    body.__annotations__ = {"a": GL[dtype], "o": GL[dtype]}
    return _kernel(body, warps=1, grid=_grid, name=name)


_DTYPES = {"bf16": bf16, "fp16": fp16, "fp32": fp32}
_BINARY = ("add", "sub", "mul", "max", "min")
_UNARY = ("exp", "exp2", "relu", "gelu", "abs", "neg")


def _supports(op: str, suffix: str) -> bool:
    """Whether include/rdna3 implements `op` for this element type.

    Asked rather than restated, so the shipped set cannot drift from what the
    tracer accepts -- gelu is specialised for float only, and instantiating it
    for bf16 fails at link time. `neg` is a multiply by -1, so it inherits
    mul's support.
    """
    m = _ops.UNARY.get(op) or _ops.BINARY.get(op) or _ops.BINARY["mul"]
    return suffix in m.dtypes


#: name -> Kernel, e.g. KERNELS["add_bf16"]. Nothing is compiled until called.
#: Not every (op, dtype) pair is present; use `_kernel_for` to get a diagnosis
#: rather than a KeyError.
KERNELS = {}
for _suffix, _dt in _DTYPES.items():
    for _op in _BINARY:
        if not _supports(_op, _suffix):
            continue
        _n = f"{_op}_{_suffix}"
        KERNELS[_n] = _binary_kernel(_n, _op, _dt)
    for _op in _UNARY:
        if not _supports(_op, _suffix):
            continue
        _n = f"{_op}_{_suffix}"
        KERNELS[_n] = _unary_kernel(_n, _op, _dt)
del _suffix, _dt, _op, _n


# ---------------------------------------------------------------- wrappers

_TORCH_DTYPES = {
    "torch.bfloat16": "bf16",
    "torch.float16": "fp16",
    "torch.float32": "fp32",
}


def _suffix_of(t) -> str:
    name = _TORCH_DTYPES.get(str(t.dtype))
    if name is None:
        raise TypeError(
            f"hk elementwise handles {sorted(set(_TORCH_DTYPES.values()))}, not {t.dtype}"
        )
    return name


@_lru_cache(maxsize=None)
def _tile_shape(shape: Tuple[int, ...], numel: int) -> Tuple[int, int]:
    """Pick the (rows, cols) 2D view to launch over.

    Memoized: `_apply` asks it once per operand plus once for the output, and
    the answer depends on nothing but the two arguments. lru_cache does not
    cache the ValueError below, so a shape that does not tile keeps raising.

    Preferring the tensor's own last axis keeps rows contiguous exactly as the
    caller laid them out; the flattened fallback is for the case where the last
    axis is not a multiple of the tile width, which for a hidden dimension it
    essentially never is.
    """
    last = shape[-1] if shape else 0
    if last and last % COLS == 0 and (numel // last) % ROWS == 0:
        return numel // last, last
    if numel % (ROWS * COLS) == 0:
        return numel // COLS, COLS
    raise ValueError(
        f"shape {tuple(shape)} ({numel} elements) does not tile: hk's elementwise "
        f"kernels need either a last dim that is a multiple of {COLS} with a "
        f"multiple of {ROWS} rows, or an element count divisible by "
        f"{ROWS * COLS}. A partial tile would read past the end of the tensor, "
        f"which returns garbage instead of failing, so it is refused."
    )


def _as_2d(t):
    if not t.is_contiguous():
        raise ValueError(
            "hk kernels take contiguous tensors; call .contiguous() first "
            "(the global layout is a base pointer plus strides, and a "
            "non-contiguous view does not have the strides it claims)."
        )
    r, c = _tile_shape(tuple(t.shape), t.numel())
    return t if t.dim() == 2 and t.shape[0] == r and t.shape[1] == c \
        else t.view(r, c)


@_lru_cache(maxsize=None)
def _kernel_for(op: str, suffix: str):
    k = KERNELS.get(f"{op}_{suffix}")
    if k is None:
        have = sorted(s for s in _DTYPES if f"{op}_{s}" in KERNELS)
        raise TypeError(
            f"hk has no {suffix} {op} kernel -- include/rdna3 implements it for "
            f"{', '.join(have)} only. Cast the tensor first."
        )
    return k


def _apply(op: str, *tensors: Any, out=None):
    a = tensors[0]
    for t in tensors[1:]:
        if t.shape != a.shape or t.dtype != a.dtype:
            raise ValueError(
                f"elementwise {op}: operands differ -- {tuple(a.shape)}/{a.dtype} vs "
                f"{tuple(t.shape)}/{t.dtype}. These kernels do not broadcast."
            )
    if out is None:
        import torch  # noqa: PLC0415  -- see norm._run
        out = torch.empty_like(a)
    elif out.shape != a.shape or out.dtype != a.dtype:
        raise ValueError(
            f"elementwise {op}: out is {tuple(out.shape)}/{out.dtype}, "
            f"expected {tuple(a.shape)}/{a.dtype}"
        )

    k = _kernel_for(op, _suffix_of(a))
    views = [_as_2d(t) for t in tensors] + [_as_2d(out)]
    k(*views)
    return out


def add(a, b, out=None):
    """Elementwise a + b."""
    return _apply("add", a, b, out=out)


def sub(a, b, out=None):
    """Elementwise a - b."""
    return _apply("sub", a, b, out=out)


def mul(a, b, out=None):
    """Elementwise a * b."""
    return _apply("mul", a, b, out=out)


def maximum(a, b, out=None):
    """Elementwise max(a, b)."""
    return _apply("max", a, b, out=out)


def minimum(a, b, out=None):
    """Elementwise min(a, b)."""
    return _apply("min", a, b, out=out)


def exp(a, out=None):
    """Elementwise exp(a)."""
    return _apply("exp", a, out=out)


def relu(a, out=None):
    """Elementwise max(a, 0)."""
    return _apply("relu", a, out=out)


def gelu(a, out=None):
    """Elementwise GELU, the tanh approximation -- base_ops::gelu is
    `x * (0.5 + 0.5 * fast_tanh(...))`, i.e. torch's approximate='tanh', not
    the erf form. fp32 only."""
    return _apply("gelu", a, out=out)


def neg(a, out=None):
    """Elementwise -a."""
    return _apply("neg", a, out=out)


def abs(a, out=None):  # noqa: A001 -- matches torch.abs, not the builtin
    """Elementwise |a|."""
    return _apply("abs", a, out=out)


__all__ = [
    "KERNELS", "ROWS", "COLS",
    "add", "sub", "mul", "maximum", "minimum",
    "exp", "relu", "gelu", "neg", "abs",
]
