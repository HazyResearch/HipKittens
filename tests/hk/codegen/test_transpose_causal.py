"""Transpose and causal masking: do they actually compile, and do the two
transposes really differ in cost?

The IR can only check the types. Whether `kittens::transpose` and
`kittens::transpose_sep` instantiate at all -- and what they cost -- is a
question for hipcc, which is this tier. The cost question matters because the
two have the same name in every kernel author's head and very different
register behaviour: `transpose` relabels, `transpose_sep` moves. The
handwritten attention epilogue transposes its output one fragment at a time
rather than whole-tile, and the note in that file says the whole-tile form was
worth 392 bytes/lane of scratch and 99 spilled VGPRs. So it is worth having a
test that would notice if the cheap one ever stopped being cheap.
"""

from __future__ import annotations

import pytest

import hk
from hk import bf16, fp32


@hk.kernel(arch="gfx1100", grid=lambda p: (1, 1, 1))
def causal_mask(s: hk.Tensor[fp32], o: hk.Tensor[fp32], *, N=32):
    idx = hk.tile_coord(0, 0, 0, 0)
    t = s[idx].load(hk.RT(fp32, N, N, "col"))
    # -inf, not 0: these entries go through exp2 before they are summed, and
    # exp2(0) is 1.
    o[idx] = t.make_causal(float("-inf"), transposed=True)


@hk.kernel(arch="gfx1100", grid=lambda p: (1, 1, 1))
def relabel(s: hk.Tensor[fp32], o: hk.Tensor[fp32], *, R=16, C=64):
    t = s[hk.tile_coord(0, 0, 0, 0)].load(hk.RT(fp32, R, C, "row"))
    # Mirrored shape *and* opposite layout -- the signature says so, and the
    # store has to agree or the coord template will not match.
    o[hk.tile_coord(0, 0, 0, 0)] = hk.transpose(t)


@hk.kernel(arch="gfx1100", grid=lambda p: (1, 1, 1))
def move(s: hk.Tensor[fp32], o: hk.Tensor[fp32], *, R=16, C=64):
    t = s[hk.tile_coord(0, 0, 0, 0)].load(hk.RT(fp32, R, C, "row"))
    o[hk.tile_coord(0, 0, 0, 0)] = hk.transpose_sep(t)


def _only(k):
    """The one device kernel in this build. `Build.kernels` is a list because
    one translation unit can hold several; these all hold exactly one."""
    ks = k.build().kernels
    assert len(ks) == 1, [x.name for x in ks]
    return ks[0]


@pytest.mark.parametrize("k", [causal_mask, relabel, move])
def test_they_compile_without_spilling(k):
    r = _only(k)
    assert r.scratch == 0, f"{k.name} spilled: {r}"
    assert r.vgpr_spill == 0


def test_each_one_emits_the_call_it_claims_to():
    """They are not interchangeable and the names are one token apart."""
    assert "kittens::transpose(" in relabel.source()
    assert "transpose_sep" not in relabel.source()
    assert "kittens::transpose_sep(" in move.source()


def test_relabelling_is_free_but_its_result_is_not():
    """The cost of `hk.transpose` is not in the transpose.

    Measured here: `relabel` wants 92 VGPRs and `move` wants 75, for the same
    16x64 -> 64x16 shape. The relabelling one is *dearer*, which is the
    opposite of what "moves no data" suggests, and the reason is the half of
    the signature that is easy to skim past: `transpose` mirrors the shape
    **and flips the layout**, so `relabel` ends up storing a `col` tile while
    `move` stores a `row` one. A col-layout store has the interleaved lane
    pattern that the compiler cannot merge -- the same effect that makes
    narrowing stores go bytewise on this chip -- and that is where the 17
    registers went.

    So the choice between them is not "which is cheaper" but "which layout
    does the consumer want". Feeding an mma that wants a col operand,
    `transpose` is free and `transpose_sep` would need a relabel afterwards.
    Feeding a global store, it is the other way round.

    The assertion is only that both stay spill-free at this size, because the
    exact counts are a compiler's business; the numbers above are a record of
    what was measured, not a contract.
    """
    cheap, dear = _only(relabel), _only(move)
    assert cheap.scratch == 0 and dear.scratch == 0
    assert cheap.vgpr_spill == 0 and dear.vgpr_spill == 0
