"""`hk.group(W)`: what a whole workgroup does together -- the global<->LDS
copies, synchronous and staged through registers.

    g = hk.group(8)
    As = hk.lds.alloc(hk.ST(hk.bf16, 128, 64), count=2, name="As")
    buf = g.stage_buffer(As, name="buf_a")
    g.load(As[0], a[idx])           # issue and wait
    g.stage(buf, a[idx])            # issue into VGPRs
    hk.arch.vm_wait(0)
    g.commit(As[1], buf)            # VGPRs -> LDS
"""

from __future__ import annotations

from .._mlir.dialects import hk as hkd
from ..compiler.from_kernel_ir import mlir_type
from ..ir.nodes import StageBufferType
from . import builder as B
from .values import SharedTile, Stage, View


class Group:
    __slots__ = ("warps",)

    def __init__(self, warps: int):
        if not isinstance(warps, int) or warps <= 0 or warps > 32:
            raise TypeError(f"hk.group takes 1..32 warps, got {warps!r}")
        self.warps = warps

    @property
    def threads(self) -> int:
        return self.warps * 32

    def __repr__(self) -> str:
        return f"group<{self.warps}>"

    @staticmethod
    def _one(dst: SharedTile, what: str) -> None:
        if not isinstance(dst, SharedTile):
            raise TypeError(f"{what} is {dst!r}, expected an LDS tile")
        if dst.ty.count != 1:
            raise TypeError(f"{what} is {dst.ty}, a stack; index it: As[tic]")

    def load(self, dst: SharedTile, src: View) -> None:
        """Global -> LDS, issued and waited by the whole group."""
        self._one(dst, "group load destination")
        hkd.GroupLoadOp(dst.v, src.tensor.v, src.idx.v, self.threads,
                        loc=B.current().loc())

    def store(self, dst: View, src: SharedTile) -> None:
        """LDS -> global, by the whole group."""
        self._one(src, "group store source")
        hkd.GroupStoreOp(dst.tensor.v, src.v, dst.idx.v, self.threads,
                         loc=B.current().loc())

    def stage_buffer(self, dst: SharedTile, depth: int = 1,
                     name: str = "buf") -> Stage:
        """The register buffer for copies into `dst`, sized from it so the two
        cannot disagree. Declare it outside any loop."""
        ty = StageBufferType(dst.ty.element(), self.threads, depth)
        t = B.current()
        return Stage(hkd.StageBufferOp(mlir_type(ty), dst.v,
                                       loc=t.loc(name)).result, ty)

    def _check(self, buf: Stage, what: str) -> None:
        if not isinstance(buf, Stage):
            raise TypeError(f"group {what} buffer is {buf!r}")
        if buf.ty.threads != self.threads:
            raise TypeError(f"{self}.{what}() on a buffer sized for "
                            f"{buf.ty.threads} threads")

    def stage(self, buf: Stage, src: View, slot: int = 0) -> None:
        """Issue the global loads for one tile into `buf`. Does not wait."""
        self._check(buf, "stage")
        if not isinstance(slot, int) or not 0 <= slot < buf.ty.depth:
            raise TypeError(f"stage slot={slot!r}: an int in [0, {buf.ty.depth})")
        hkd.StageLoadOp(buf.v, src.tensor.v, src.idx.v, slot,
                        loc=B.current().loc())

    def commit(self, dst: SharedTile, buf: Stage, slot: int = 0,
               wait: bool = False) -> None:
        """Write a staged buffer to LDS. hk.arch.vm_wait must come first."""
        self._check(buf, "commit")
        self._one(dst, "commit destination")
        hkd.StageCommitOp(dst.v, buf.v, slot, wait=bool(wait),
                          loc=B.current().loc())


def group(warps: int) -> Group:
    return Group(warps)


__all__ = ["Group", "group"]
