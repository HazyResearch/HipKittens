"""hk.collective: combining a reduction across the warps of a workgroup.

Every reduction on a tile is per-warp. A workgroup that splits a row across W
warps holds W partial answers, and nothing in the register file sees across a
wave; these two route them through LDS with the barriers in the right place.
"""

from __future__ import annotations

from typing import Optional

from ..types import RegTileType, RegVecType
from . import arch, lds
from .values import SharedVec, Vec, as_vec, broadcast_col, binary

#: combiner -> the reduction that implements it. Associative and commutative
#: only: the warps write concurrently and the read order is not fixed.
_REDUCE = {"add": "sum", "max": "max", "min": "min", "mul": "prod"}
COMBINERS = tuple(_REDUCE)


def cross_warp(vec: Vec, op: str, warps: int, *,
               scratch: Optional[SharedVec] = None, name: str = "xw") -> Vec:
    """One register vector per warp combined, the answer in every warp."""
    if op not in COMBINERS:
        raise ValueError(f"cross_warp combiner must be one of {COMBINERS}, got {op!r}")
    if not isinstance(vec, Vec):
        raise TypeError(f"cross_warp takes a register vector, got {vec!r}")
    if warps == 1:
        return vec
    if scratch is None:
        scratch = lds.alloc_vec(lds.shared_vec(vec.ty, warps), name=name)
    elif scratch.ty.count < warps:
        raise ValueError(f"scratch holds {scratch.ty.count} vectors, need {warps}")
    # [1] the scratch may still be being read by a previous combine; [2] the
    # stores must land before any warp reads. No third: after the loads every
    # warp holds the answer in registers.
    arch.barrier()
    scratch.store(vec, arch.warp_id())
    arch.barrier()
    acc = scratch.load(vec.ty, 0)
    for w in range(1, warps):
        binary(op, acc, scratch.load(vec.ty, w), into=acc)
    return acc


def fold_rows(vec: Vec, op: str, warps: int, *,
              scratch: Optional[SharedVec] = None, name: str = "fold") -> Vec:
    """Reduce a column vector's own entries, the answer in all of them -- for
    a row folded onto a tile's 16 rows. The result is uniform."""
    if op not in COMBINERS:
        raise ValueError(f"fold_rows combiner must be one of {COMBINERS}, got {op!r}")
    if not isinstance(vec, Vec) or vec.ty.kind != "col":
        raise TypeError(f"fold_rows takes a column vector, got {vec!r}")
    src = vec.ty.tile
    n = vec.ty.length
    square = RegTileType(src.dtype, n, n, src.layout)
    if scratch is None:
        scratch = lds.alloc_vec(lds.shared_vec(vec.ty, warps), name=name)
    slot = arch.warp_id() if warps > 1 else 0
    scratch.store(vec, slot)
    arch.barrier()
    across = scratch.load(RegVecType(square, "row"), slot)
    spread = broadcast_col(square, across)
    folded = getattr(spread, _REDUCE[op])(axis=1).value()
    return as_vec(folded, src, uniform=True)


__all__ = ["COMBINERS", "cross_warp", "fold_rows"]
