"""The decode GEMM's schedule selection and generated code, without a GPU."""

from __future__ import annotations

import pytest

from hk.ops import decode_gemm


@pytest.mark.parametrize("m,tokens", [(1, 1), (16, 1), (17, 2), (32, 2),
                                      (33, 4), (64, 4)])
def test_a_batch_is_padded_to_whole_token_tiles(m, tokens):
    assert decode_gemm.schedule_for(m, 4096)["tokens"] == tokens


@pytest.mark.parametrize("m", [0, 65])
def test_outside_four_token_tiles_there_is_no_schedule(m):
    assert decode_gemm.schedule_for(m, 4096) is None


def test_the_measured_schedule_where_k_allows_it():
    measured = decode_gemm.SCHEDULES[1]
    assert decode_gemm.schedule_for(1, 4096) == dict(measured, tokens=1)
    assert decode_gemm.schedule_for(1, 12288)["warps"] == measured["warps"]


def test_a_k_the_split_does_not_divide_gets_fewer_warps():
    # 3584 = 56 rounds of 64: 16 or 32 warps do not divide it, 8 do.
    s = decode_gemm.schedule_for(1, 3584)
    assert s["warps"] == 8 and 3584 % (16 * s["slices"] * s["warps"]) == 0
    # Halving below the batch of reads in flight leaves nothing to batch.
    s = decode_gemm.schedule_for(20, 256)
    assert s["warps"] == 8 and s["combine_depth"] == 0


def test_a_k_no_split_divides_is_refused():
    assert decode_gemm.schedule_for(1, 4000) is None
    assert not decode_gemm.fits(1, 4096, 4000)
    assert not decode_gemm.fits(1, 4100, 4096)


def test_resident_sizes_the_lds_so_exactly_that_many_fit():
    import re

    from hk.target import get_target

    pool = get_target("gfx1100").lds_pool_bytes
    for resident in (2, 3, 4):
        src = decode_gemm.decode_gemm_kernel(f"dg_r{resident}", warps=16,
                                             resident=resident).source()
        lds = int(re.search(r"// LDS\s*:\s*(\d+) B", src).group(1))
        assert pool // lds == resident


def test_a_cap_one_workgroup_cannot_hold_is_refused():
    with pytest.raises(ValueError, match="resident=1"):
        decode_gemm.decode_gemm_kernel("dg_r1", warps=16, resident=1)


def test_one_warp_has_no_combine_and_so_no_cap():
    s = decode_gemm.schedule_for(1, 64 * 3)       # only one warp divides it
    assert s["warps"] == 1 and s["resident"] == 0


def test_the_combine_waits_once_unless_told_otherwise():
    once = decode_gemm.decode_gemm_kernel("dg_once", warps=16).source()
    batched = decode_gemm.decode_gemm_kernel("dg_batched", warps=16,
                                             combine_depth=8).source()
    assert "sched_barrier" not in once
    assert batched.count("__builtin_amdgcn_sched_barrier(0);") == 1
