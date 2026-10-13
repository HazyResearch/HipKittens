"""The hk kernel language.

    import hk

    @hk.kernel(warps=4, grid=lambda p: (1, hk.cdiv(p.x.rows, 16), 1))
    def rmsnorm(x: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16], *, EPS=1e-6):
        row = hk.arch.block_idx.y * 16
        sq = hk.zeros(hk.col_vec(hk.RT(hk.fp32, 16, 64)))
        for cb in range(hk.arch.warp_id(), hk.cdiv(x.cols, 64), 4):
            v = x.at[row, cb * 64].load(hk.RT(hk.fp32, 16, 64))
            sq += (v * v).sum(axis=1)
        ...

Types are values you build (`hk.RT`, `hk.ST`, `hk.col_vec`); device values
are objects with operators and methods (`Int`, `Tile`, `Vec`, a tensor's
`.at[...]` view); the machine and shared memory have namespaces
(`hk.arch`, `hk.lds`, `hk.math`, `hk.collective`, `hk.group`). A `for` over a
device range and an `if` on a device comparison compile to scf.for / scf.if.
"""

from __future__ import annotations

import builtins as _builtins

from ..ir.nodes import (  # noqa: F401
    DType, bf16, fp16, fp32, fp8e4m3, fp8e5m2, i8, i32,
    RegTileType as _RT, RegVecType as _RV, SharedTileType as _ST,
    SharedVecType as _SV,
)
from ..lang import host as _host
from . import arch, collective, lds, math  # noqa: F401
from .control import if_, range, range_constexpr, scope  # noqa: A004
from .group import Group, group
from .kernel import Constexpr, Kernel, TracedKernel, Tensor, func, kernel
from .values import (  # noqa: F401
    Bool, Coord, Global, Granules, Int, SharedTile, SharedVec, Tile, Value,
    Vec, View, as_vec, broadcast_col, broadcast_row, coord, elem_coord, full,
    mma, neg_infty, tile_coord, transpose, transpose_sep, zeros,
)
from .values import scalar_op as _scalar_op


def RT(dtype: DType, rows: int, cols: int, layout: str = "row") -> _RT:
    """A register tile type: kittens::rt<T, rows, cols, layout>."""
    return _RT(dtype, rows, cols, layout)


def ST(dtype: DType, rows: int, cols: int, count: int = 1) -> _ST:
    """An LDS tile type; count > 1 is a multi-buffer."""
    return _ST(dtype, rows, cols, count)


def SV(dtype: DType, length: int, count: int = 1) -> _SV:
    """An LDS vector type."""
    return _SV(dtype, length, count)


def col_vec(tile: _RT) -> _RV:
    """One value per *row* of `tile`: what `t.sum(axis=1)` gives."""
    return _RV(tile, "col")


def row_vec(tile: _RT) -> _RV:
    """One value per *column* of `tile`."""
    return _RV(tile, "row")


def _dyn(*xs) -> bool:
    return _builtins.any(isinstance(x, Value) for x in xs)


def cdiv(a, b):
    """Ceiling division: of ints at trace time, of device Ints in the kernel,
    of tensor extents in a grid lambda."""
    if isinstance(a, _host.HostExpr) or isinstance(b, _host.HostExpr):
        return _host.cdiv(a, b)
    if _dyn(a, b):
        return _scalar_op("cdiv", a, b)
    return -(-a // b)


def min(a, b):  # noqa: A001
    """Of ints, of device Ints, or elementwise of tiles and vectors."""
    if isinstance(a, (Tile, Vec)):
        return a.minimum(b)
    if _dyn(a, b):
        return _scalar_op("min", a, b)
    return _builtins.min(a, b)


def max(a, b):  # noqa: A001
    if isinstance(a, (Tile, Vec)):
        return a.maximum(b)
    if _dyn(a, b):
        return _scalar_op("max", a, b)
    return _builtins.max(a, b)


__all__ = [
    "kernel", "func", "Kernel", "TracedKernel", "Tensor", "Constexpr",
    "RT", "ST", "SV", "col_vec", "row_vec", "DType",
    "bf16", "fp16", "fp32", "i32", "i8", "fp8e4m3", "fp8e5m2",
    "arch", "lds", "math", "collective", "group", "Group",
    "range", "range_constexpr", "if_", "scope",
    "cdiv", "min", "max", "zeros", "full", "neg_infty", "transpose",
    "transpose_sep", "broadcast_row", "broadcast_col", "as_vec", "mma",
    "coord", "tile_coord", "elem_coord",
    "Int", "Bool", "Tile", "Vec", "Global", "View", "SharedTile", "SharedVec",
    "Granules", "Coord",
]
