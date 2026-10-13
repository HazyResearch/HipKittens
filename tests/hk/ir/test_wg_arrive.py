"""hk.arch.wg_arrive: the cross-workgroup reduction primitive, without a GPU."""

from __future__ import annotations

import pytest

import hk
from hk import VerifyError, fp32, i32


def _kernel(name, lds_tile_rows=0, counter_dtype=i32):
    def body(x, cnt, y):
        if lds_tile_rows:
            buf = hk.lds.alloc(hk.ST(fp32, lds_tile_rows, 64))
            buf.store(hk.zeros(hk.RT(fp32, lds_tile_rows, 64)))
            hk.arch.barrier(drain=True)
        t = x[hk.elem_coord(0, hk.arch.block_idx.y, 0, 0)].load(hk.RT(fp32, 16, 16))
        a = hk.arch.wg_arrive(cnt[hk.elem_coord(0, 0, 0, hk.arch.block_idx.x)], 4)
        if a == 3:
            y[hk.elem_coord(0, 0, 0, 0)] = t

    body.__name__ = name
    body.__annotations__ = {"x": hk.Tensor[fp32], "cnt": hk.Tensor[counter_dtype],
                            "y": hk.Tensor[fp32]}
    return hk.kernel(body, arch="gfx1100", warps=1, name=name,
                     grid=lambda p: (1, 4, 1))


def test_it_emits_the_library_call_and_returns_a_scalar():
    src = _kernel("arrive_emit").source()
    assert "kittens::wg_arrive(g.cnt, " in src


def test_its_static_lds_is_counted():
    ir = _kernel("arrive_count").trace()
    assert "hk.lds_static = 16 : i64" in ir.dump()


def test_a_kernel_that_would_fail_to_launch_fails_to_build():
    # 256 x 64 fp32 is exactly the 64 KB a workgroup may take; the slot
    # wg_arrive declares on top of it is what the decode GEMM's first split
    # version launched with, into HSA_STATUS_ERROR_INVALID_ALLOCATION.
    with pytest.raises(VerifyError, match="static"):
        _kernel("arrive_over", lds_tile_rows=256).trace()


def test_the_counter_is_int32():
    with pytest.raises(TypeError, match="int32"):
        _kernel("arrive_f32", counter_dtype=fp32).trace()
