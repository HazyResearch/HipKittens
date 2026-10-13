"""hk.wg_arrive: the cross-workgroup reduction primitive, without a GPU."""

from __future__ import annotations

import pytest

import hk
from hk import fp32, i32
from hk.ir.verify import VerifyError
from hk.lang import ops


def _kernel(name, lds_tile_rows=0, counter_dtype=i32):
    def body(x, cnt, y):
        if lds_tile_rows:
            buf = ops.alloc_shared(ops.st(fp32, lds_tile_rows, 64))
            ops.store_shared(buf, ops.zeros(ops.rt(fp32, lds_tile_rows, 64)))
            ops.barrier(drain=True)
        t = ops.load(x, ops.elem_coord(0, ops.block_idx.y, 0, 0), ops.rt(fp32, 16, 16))
        a = ops.wg_arrive(cnt, ops.elem_coord(0, 0, 0, ops.block_idx.x), 4)
        with hk.if_(ops.s_eq(a, 3)):
            ops.store(y, t, ops.elem_coord(0, 0, 0, 0))

    body.__name__ = name
    body.__annotations__ = {"x": hk.GL[fp32], "cnt": hk.GL[counter_dtype],
                            "y": hk.GL[fp32]}
    return hk.kernel(body, arch="gfx1100", warps=1, name=name,
                     grid=lambda p: (1, 4, 1))


def test_it_emits_the_library_call_and_returns_a_scalar():
    src = _kernel("arrive_emit").source()
    assert "kittens::wg_arrive(g.cnt, " in src


def test_its_static_lds_is_counted():
    ir = _kernel("arrive_count").trace()
    assert ir.lds_static == 16


def test_a_kernel_that_would_fail_to_launch_fails_to_build():
    # 256 x 64 fp32 is exactly the 64 KB a workgroup may take; the slot
    # wg_arrive declares on top of it is what the decode GEMM's first split
    # version launched with, into HSA_STATUS_ERROR_INVALID_ALLOCATION.
    with pytest.raises(VerifyError, match="static"):
        _kernel("arrive_over", lds_tile_rows=256).trace()


def test_the_counter_is_int32():
    with pytest.raises(TypeError, match="int32"):
        _kernel("arrive_f32", counter_dtype=fp32).trace()
