"""The RMSNorm the MLIR plan set as the target for the language, as written
there with two spellings changed: a vector type is named by the tile it
reduces (`hk.col_vec(T)`, not `hk.ColVec[fp32, 16]` -- the dialect ties a
vector's layout to its tile), and a load names the register tile it makes
(`x.at[r, c].load(T)`)."""

from __future__ import annotations

import re

import pytest

pytest.importorskip("hk._mlir")

import hk  # noqa: E402

T = hk.RT(hk.fp32, 16, 64)


def rmsnorm_kernel(warps: int = 4):
    @hk.kernel(arch="gfx1100", warps=warps,
               grid=lambda p: (1, hk.cdiv(p.x.rows, 16), p.x.batch))
    def rmsnorm(x: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16],
                WARPS: hk.Constexpr[int] = warps, EPS: hk.Constexpr[float] = 1e-6):
        row0, wid = hk.arch.block_idx().y * 16, hk.arch.warp_id()
        sq = hk.zeros(hk.col_vec(T))
        for cb in range(wid, hk.cdiv(x.cols, 64), WARPS):        # runtime loop -> scf.for
            v = x.at[row0, cb * 64].load(T)
            sq += (v * v).sum(axis=1)                             # row_sum, accumulates in place
        scale = hk.math.rsqrt(hk.collective.cross_warp(sq, "add", WARPS) / x.cols + EPS)
        for cb in range(wid, hk.cdiv(x.cols, 64), WARPS):
            o.at[row0, cb * 64] = (x.at[row0, cb * 64].load(T) * scale[:, None]).to(hk.bf16)

    return rmsnorm


def test_it_traces_into_two_runtime_loops_and_an_in_place_reduction():
    t = rmsnorm_kernel().trace()
    text = t.dump()
    assert text.count("scf.for %cb") == 2
    assert "hk.row_sum" in text and "{accumulate}" in text
    assert "hk.rsqrt" in text and "hk.mul_row" in text


def test_the_loops_are_the_cpp_loops():
    src = rmsnorm_kernel().source()
    loops = re.findall(r"for \(int (cb\d*) = warp_id; \1 < cdiv\d*; \1 \+= 4\)", src)
    assert loops == ["cb", "cb1"]


def test_a_positional_constexpr_specialises():
    assert "cb += 2)" in rmsnorm_kernel().source(WARPS=2)
    k = rmsnorm_kernel()
    assert {p.name for p in k.trace().params if p.is_const} == {"WARPS", "EPS"}


def test_the_broadcast_axis_is_checked():
    @hk.kernel(arch="gfx1100", grid=lambda p: (1, 1, 1))
    def wrong(x: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]):
        v = x.at[0, 0].load(T)
        o.at[0, 0] = (v * v.sum(axis=1)[None, :]).to(hk.bf16)

    with pytest.raises(TypeError, match="row vector, and this one is"):
        wrong.trace()


def test_a_vector_indexes_only_as_a_broadcast():
    @hk.kernel(arch="gfx1100", grid=lambda p: (1, 1, 1))
    def wrong(x: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]):
        v = x.at[0, 0].load(T)
        o.at[0, 0] = (v * v.sum(axis=1)[0]).to(hk.bf16)

    with pytest.raises(TypeError, match=r"only as v\[:, None\]"):
        wrong.trace()


@pytest.mark.hipcc
def test_it_compiles_without_spilling():
    for k in rmsnorm_kernel().build().kernels:
        assert k.scratch == 0 and not k.spills, k.line()
