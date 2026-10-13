"""hk.arch: the machine -- where this workgroup and warp are, and the
instructions that order and schedule them."""

from __future__ import annotations

from .._mlir import ir
from .._mlir.dialects import hk as hkd
from ..types import CoordType, GlobalType, ScalarType, i32
from . import builder as B
from .values import Coord, Global, Int, Scalar, View, as_int


def _i32():
    return ir.IntegerType.get_signless(32)


class _BlockIdx:
    """`hk.arch.block_idx.x` / `.y` / `.z`: hoisted, read once per kernel."""

    def __getattr__(self, axis: str) -> Int:
        if axis not in ("x", "y", "z"):
            raise AttributeError(f"block_idx has no axis {axis!r}")
        t = B.current()

        def build():
            return hkd.BlockIdxOp(_i32(), getattr(hkd.Axis, axis),
                                  loc=t.loc(f"bid_{axis}")).result

        return Int(t.hoisted(("block_idx", axis), build), ScalarType(i32))

    def __call__(self):
        return self


block_idx = _BlockIdx()


def warp_id() -> Int:
    """This warp's index in the workgroup (wave-uniform)."""
    t = B.current()
    return Int(t.hoisted("warp_id", lambda: hkd.WarpIdOp(
        _i32(), loc=t.loc("warp_id")).result), ScalarType(i32))


def lane_id() -> Int:
    t = B.current()
    return Int(t.hoisted("lane_id", lambda: hkd.LaneIdOp(
        _i32(), loc=t.loc("lane_id")).result), ScalarType(i32))


def barrier(drain: bool = False) -> None:
    """`s_barrier`: orders execution across the workgroup, not memory.
    `drain=True` spends an lgkmcnt(0) in front of it."""
    hkd.BarrierOp(drain=bool(drain), loc=B.current().loc())


def sched_barrier() -> None:
    """Nothing crosses this line in the scheduler; it costs no instruction."""
    hkd.SchedBarrierOp(loc=B.current().loc())


def setprio(level: int) -> None:
    """`s_setprio`: this wave's issue priority, 0..3."""
    if not isinstance(level, int) or not 0 <= level <= 3:
        raise TypeError(f"setprio({level!r}): the priority field is 2 bits, 0..3")
    hkd.SetPrioOp(level, loc=B.current().loc())


def vm_wait(keep: int = 0) -> None:
    """`s_waitcnt vmcnt(keep)`: retire vector-memory loads, `keep` in flight."""
    if not isinstance(keep, int) or not 0 <= keep <= 63:
        raise TypeError(f"vm_wait(keep={keep!r}): vmcnt is 6 bits, 0..63")
    hkd.VmWaitOp(keep, loc=B.current().loc())


def wg_arrive(counter, expected: Scalar) -> Int:
    """Count this workgroup's arrival at `counter` (an element view of an
    int32 global) and return how many arrived before it. The last to arrive
    resets the counter. Contains barriers: every warp must reach it."""
    if not isinstance(counter, View):
        raise TypeError("wg_arrive takes a view of the counter: cnt.at[i]")
    if counter.idx.unit != "element":
        raise TypeError("wg_arrive takes an element coordinate: cnt.at[i]")
    if counter.tensor.dtype is not i32:
        raise TypeError(f"wg_arrive counts in an int32 tensor, not "
                        f"{counter.tensor.dtype.name}")
    t = B.current()
    return Int(hkd.WgArriveOp(_i32(), counter.tensor.v, counter.idx.v,
                              as_int(expected).v,
                              loc=t.loc("arrived")).result, ScalarType(i32))


__all__ = ["block_idx", "warp_id", "lane_id", "barrier", "sched_barrier",
           "setprio", "vm_wait", "wg_arrive"]
