"""hk.lds: shared memory -- allocation, and the wait that retires reads."""

from __future__ import annotations

from typing import Optional

from .._mlir.dialects import hk as hkd
from .module import mlir_type
from ..types import RegVecType, SharedTileType, SharedVecType
from . import builder as B
from .values import SharedTile, SharedVec, Tile, lds_loads

__all__ = ["alloc", "alloc_vec", "shared_vec", "wait_for", "lds_loads"]


def alloc(tile: SharedTileType, name: str = "smem",
          count: Optional[int] = None) -> SharedTile:
    """Reserve an LDS tile, or `count` of them in one size-aligned block (the
    two buffers of a pair must differ in one address bit for the XOR swap).
    Offsets and the 64 KB check come from the hk-lds-alloc pass."""
    if count is not None:
        tile = SharedTileType(tile.dtype, tile.rows, tile.cols, count)
    t = B.current()
    return SharedTile(hkd.AllocSharedOp(mlir_type(tile), loc=t.loc(name)).result,
                      tile)


def shared_vec(vt: RegVecType, count: int = 1) -> SharedVecType:
    """The LDS type a register vector round-trips through."""
    if not isinstance(vt, RegVecType):
        raise TypeError(f"shared_vec takes a register vector type, got {vt!r}")
    return SharedVecType(vt.dtype, vt.length, count)


def alloc_vec(ty: SharedVecType, name: str = "svec") -> SharedVec:
    if not isinstance(ty, SharedVecType):
        raise TypeError(f"alloc_vec takes a SharedVecType, got {ty!r}")
    t = B.current()
    return SharedVec(hkd.AllocSharedVecOp(mlir_type(ty), loc=t.loc(name)).result,
                     ty)


def wait_for(*tiles: Tile) -> None:
    """Retire the outstanding LDS reads of `tiles` and re-bind their
    registers. The lgkmcnt is derived by hk-lds-pipeline; there is no bare
    wait, because a wait that names nothing puts back no dependence."""
    if not tiles:
        raise TypeError("wait_for() needs the tiles it retires")
    for x in tiles:
        if not isinstance(x, Tile):
            raise TypeError(f"wait_for operand is {x!r}, expected a register tile")
    hkd.LdsWaitForOp([x.v for x in tiles], loc=B.current().loc())
