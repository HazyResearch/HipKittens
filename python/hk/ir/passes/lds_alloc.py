"""Assign LDS offsets to shared tiles.

gfx11 has no global->LDS DMA, so there is no in-flight copy to track and
allocation is a straight bump allocator over the dynamic shared segment -- the
same thing kernels/rdna3 does by hand with a `shared_allocator`.

Tiles are laid out in declaration order and each is aligned to its own size when
that size is a power of two. That is not cosmetic: the double-buffer swap in the
attention kernel flips buffers by XOR-ing one bit into every LDS address rather
than re-deriving them, which requires the two buffers of a pair to differ in
exactly one bit. Allocating in declaration order with power-of-two alignment is
what makes that legal, and `lds_xor_ok` records whether it held.
"""

from __future__ import annotations

from ..nodes import KernelIR, SharedTileType, SharedVecType


def _align_up(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def lds_alloc(ir: KernelIR) -> None:
    offset = 0
    allocations = []

    for op in ir.body:
        if op.opcode not in ("alloc_shared", "alloc_shared_vec"):
            continue
        ty = op.results[0].type
        assert isinstance(ty, (SharedTileType, SharedVecType))
        size = ty.nbytes
        # Align to the tile size when that is a power of two, else to 16 B --
        # enough for the widest ds_read.
        align = size if size and (size & (size - 1)) == 0 else 16
        offset = _align_up(offset, align)
        op.attrs["offset"] = offset
        op.attrs["nbytes"] = size
        allocations.append((op.results[0].name, offset, size))
        offset += size

    ir.lds_bytes = offset
    ir.lds_map = allocations
    # kittens::wg_arrive declares one `__shared__ int` per instantiation;
    # 16 B is what the static segment rounds it to.
    ir.lds_static = 16 * len(ir.ops("wg_arrive"))
