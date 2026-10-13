"""Phase 2: loops, reductions, in-place writes, fills, scalars.

Pure Python. What is asserted here is the structure the emitter depends on --
that a loop is a region rather than a flattened sequence, that an accumulating
reduction writes its own destination instead of declaring a new one, that a
fill's index is a runtime value. Each of those being wrong produces a kernel
that compiles and computes the wrong thing, which is the class of bug this
tier exists to catch before hipcc is ever invoked.
"""

from __future__ import annotations

import pytest

import hk
from hk import fp32, i8


@hk.kernel(arch="gfx1100", grid=lambda p: (1, hk.cdiv(p.x.rows, 16), 1))
def rowsum(x: hk.GL[fp32], o: hk.GL[fp32], *, ROWS=16, COLS=64):
    t = hk.rt(fp32, ROWS, COLS)
    acc = hk.zeros_vec(hk.col_vec(t))
    n = hk.s_cdiv(hk.cols(x), COLS)
    for cb in hk.range(n):
        idx = hk.elem_coord(0, 0, hk.s_mul(hk.block_idx.y, ROWS), hk.s_mul(cb, COLS))
        hk.row_sum(hk.load(x, idx, t), out=acc, accumulate=True)
    hk.store(o, acc, hk.elem_coord(0, 0, 0, hk.s_mul(hk.block_idx.y, ROWS)))


# -- regions ------------------------------------------------------------------


def test_a_loop_is_a_region_not_a_flattened_body():
    ir = rowsum.trace()
    (loop,) = ir.ops("for")
    assert loop.body, "hk.range produced no region"
    assert [op.opcode for op in ir.body].count("load_global") == 0
    assert [op.opcode for op in loop.walk()].count("load_global") == 1


def test_the_induction_variable_is_a_block_argument():
    """Scoped to the region. If it were an ordinary result the emitter would
    declare it outside the `for`, where the loop cannot assign it."""
    (loop,) = rowsum.trace().ops("for")
    (iv,) = loop.block_args
    assert iv not in [r for op in loop.walk() for r in op.results]


def test_walk_reaches_into_regions_and_ops_does_too():
    ir = rowsum.trace()
    assert len(ir.ops("row_sum")) == 1
    assert sum(1 for _ in ir.walk()) > len(ir.body)


# -- in-place -----------------------------------------------------------------


def test_an_accumulating_reduction_writes_the_accumulator_it_was_given():
    (red,) = rowsum.trace().ops("row_sum")
    assert red.attrs["inplace"] and red.attrs["accumulate"]
    assert red.dst is red.operands[0]
    assert not red.results, "an in-place op that also produces a result would "\
        "let the emitter declare a fresh destination inside the loop"


def test_the_accumulator_is_declared_once_outside_the_loop():
    """The whole point. A redeclaration inside the region shadows the carried
    value and the kernel keeps only the last iteration -- at full speed, with
    no diagnostic."""
    src = rowsum.source()
    body = src.split("for (int", 1)[1]
    decl = "_col zv;"  # named for hk.zeros_vec, the op that produced it
    assert src.count(decl) == 1 and decl not in body


def test_an_accumulating_reduction_passes_its_destination_as_the_third_argument():
    """kittens::row_sum(dst, src, src_accum). Dropping the third argument is
    an overwrite, which is a correct-looking kernel that loses every block but
    the last."""
    assert "kittens::row_sum(zv, ld, zv);" in rowsum.source()


def test_a_non_accumulating_reduction_declares_its_own_destination():
    @hk.kernel(arch="gfx1100", grid=lambda p: (1, 1, 1))
    def once(x: hk.GL[fp32], o: hk.GL[fp32]):
        t = hk.rt(fp32, 16, 64)
        i = hk.elem_coord(0, 0, 0, 0)
        hk.store(o, hk.row_max(hk.load(x, i, t)), hk.elem_coord(0, 0, 0, 0))

    (red,) = once.trace().ops("row_max")
    assert not red.attrs.get("inplace")
    assert "_col row_max;" in once.source()
    assert "kittens::row_max(row_max, ld);" in once.source()


# -- fills --------------------------------------------------------------------


def test_a_fill_carries_a_runtime_index_not_a_constant():
    """The tail idiom masks `blockIdx`-dependent many columns. A fill whose
    index had to be a literal could not express it at all."""

    @hk.kernel(arch="gfx1100", grid=lambda p: (1, 1, 1))
    def masked(x: hk.GL[fp32], o: hk.GL[fp32], *, COLS=64):
        t = hk.rt(fp32, 16, COLS)
        n = hk.cols(x)
        lo = hk.s_min(hk.s_mul(hk.block_idx.x, COLS), hk.s_sub(n, COLS))
        pad = hk.s_sub(hk.s_mul(hk.block_idx.x, COLS), lo)
        idx = hk.elem_coord(0, 0, 0, lo)
        hk.store(o, hk.left_fill(hk.load(x, idx, t), pad, 0.0), idx)

    ir = masked.trace()
    (fill,) = ir.ops("left_fill")
    assert fill.operands[2].producer.opcode == "s_sub"
    assert "kittens::left_fill(ld, ld, sub1, (float)0.0);" in masked.source()


def test_a_fill_writes_its_source_in_place():
    """conversions.cuh:530 is `if (col_idx <= 0) return;` -- an empty mask
    returns without touching dst. Since an empty mask is the *normal* case
    (every column block but the last), a fresh destination would hold
    uninitialized registers precisely when nothing needs masking, and the
    kernel would produce NaN on the shapes that do divide. It did. Filling in
    place makes the early return leave the loaded values alone."""

    @hk.kernel(arch="gfx1100", grid=lambda p: (1, 1, 1))
    def m(x: hk.GL[fp32], o: hk.GL[fp32]):
        t = hk.rt(fp32, 16, 64)
        i = hk.elem_coord(0, 0, 0, 0)
        v = hk.load(x, i, t)
        assert hk.left_fill(v, 3, 0.0) is v
        hk.store(o, v, i)

    (fill,) = m.trace().ops("left_fill")
    assert fill.attrs["inplace"] and fill.dst is fill.operands[1]
    src = m.source()
    assert src.count("rt_fp32_16x64_row ld;") == 1
    assert src.count("rt_fp32_16x64_row ") == 2, "a fill declared a second tile"


def test_an_explicit_out_on_a_fill_copies_first():
    """Allowed, but it costs the copy the early return would otherwise skip.
    Emitting the fill alone here would be the same uninitialized-dst bug."""

    @hk.kernel(arch="gfx1100", grid=lambda p: (1, 1, 1))
    def m(x: hk.GL[fp32], o: hk.GL[fp32]):
        t = hk.rt(fp32, 16, 64)
        i = hk.elem_coord(0, 0, 0, 0)
        v = hk.load(x, i, t)
        w = hk.zeros(t)
        hk.right_fill(v, 8, 1.0, out=w)
        hk.store(o, w, i)

    src = m.source()
    assert src.index("kittens::copy(z, ld);") < src.index("kittens::right_fill(z, ld,")


def test_neg_infty_fill_uses_the_named_constant_not_a_c_literal():
    """repr(float('-inf')) is `-inf`, which is not a C++ token: the generated
    file would not compile. It did not, once."""

    @hk.kernel(arch="gfx1100", grid=lambda p: (1, 1, 1))
    def m(o: hk.GL[fp32]):
        t = hk.rt(fp32, 16, 64)
        hk.store(o, hk.full(t, float("-inf")), hk.elem_coord(0, 0, 0, 0))

    src = m.source()
    assert "-inf" not in src
    assert "kittens::neg_infty(" in src


# -- scalars ------------------------------------------------------------------


def test_scalar_arithmetic_stays_on_the_host_side_of_the_type_system():
    ir = rowsum.trace()
    assert {op.opcode for op in ir.walk()} >= {"s_mul", "s_cdiv"}
    for op in ir.walk():
        if op.opcode.startswith("s_"):
            assert isinstance(op.result.type, hk.ir.ScalarType)


def test_a_hoisted_scalar_is_emitted_once():
    """blockIdx.y is the same in every use; one `const int` keeps the
    generated C++ readable enough to take to the disassembler."""
    assert rowsum.source().count("= blockIdx.y;") == 1


def test_there_is_no_rounding_division_on_a_tile():
    @hk.kernel(arch="gfx1100", grid=lambda p: (1, 1, 1))
    def bad(x: hk.GL[fp32], o: hk.GL[fp32]):
        i = hk.elem_coord(0, 0, 0, 0)
        hk.store(o, hk.load(x, i, hk.rt(fp32, 16, 64)) // 2, i)

    with pytest.raises(TypeError, match="rounding division"):
        bad.trace()


# -- dtypes -------------------------------------------------------------------


def test_int8_is_a_global_type_and_not_a_register_type():
    """gfx1100 has no int8 WMMA and no packed int8 ALU, so rt<int8> would be a
    template error at the end of a 30 s compile. Refuse it at trace time with
    the reason."""
    assert hk.GL[i8]
    with pytest.raises(TypeError, match="no i8 register tile"):
        hk.rt(i8, 16, 64)


def test_a_load_from_an_int8_global_lands_in_a_float_tile():
    """kittens::load runs base_types::convertor per element, so widening is
    free -- the quantized path never needs an integer tile."""
    src = hk.ops.quant.KERNELS["dequantize_fp32_w1"].source()
    assert "kittens::gl<int8, -1, -1, -1, -1> q;" in src
    assert "kittens::rt<int8" not in src
    assert "kittens::rt<float, 16, 64," in src


# -- EXACT --------------------------------------------------------------------


def test_the_three_ops_do_not_want_the_same_plan():
    """Why `plan` takes `kind`, against the sweep that put it there.

    Each row is a case from tools/hk-bench/norm_plan.py under burst timing.
    rms and layer want the row held unfolded wherever that is legal; softmax
    wants it folded at the same width, by 1.5x. One rule cannot say both, and
    `cols` alone cannot tell which is being asked.
    """
    from hk.ops.norm import plan

    # cols, kind, winner          measured (winner / what the other rule picks)
    cases = [
        (4096, "rms", (16, 4, 1)),      # 0.361 vs folded w4 tpw1  0.402
        (4096, "layer", (16, 4, 1)),    # 0.354 vs folded w4 tpw1  0.378
        (4096, "softmax", (4, 1, 16)),  # 0.191 vs held w16 tpw4   0.290
        # Too wide to hold unfolded: every kind folds, and the kind stops
        # mattering because there is only one plan left.
        (8192, "rms", (8, 1, 16)),      # 0.067
        (8192, "softmax", (8, 1, 16)),
        (16384, "rms", (16, 1, 16)),    # 0.364
        (16384, "softmax", (16, 1, 16)),# 0.717
        # Narrow enough that the folded view would be a two-warp workgroup
        # paying fold_rows' LDS round trip, so _FOLD_MIN_WARPS refuses it even
        # for softmax.
        (1024, "rms", (16, 1, 1)),      # 0.058
        (1024, "softmax", (16, 1, 1)),
        (2048, "rms", (16, 2, 1)),      # 0.349 vs folded w2 tpw0 0.339
        (2048, "softmax", (16, 2, 1)),
        # 80 blocks is five tiles a warp, one past MAX_TPW, so both kinds
        # fall to FOLD and land on the same plan: five folded blocks, one
        # warp each. Two constants are pinned by this one row -- MAX_TPW=4
        # (w16 tpw5 fold1 is 0.512 against this plan's 0.478) and _split's
        # one-warp-per-block rule (the old w4 tpw2 fold16 is 0.407 against
        # 0.368 for quantize here). See both.
        (5120, "rms", (5, 1, 16)),
        (5120, "softmax", (5, 1, 16)),
    ]
    for cols, kind, want in cases:
        assert plan(cols, kind) == want, (cols, kind)

    # rms is the default, because it is the shape of two of the three.
    assert plan(4096) == plan(4096, "rms") != plan(4096, "softmax")


def test_the_plan_is_not_simply_flat():
    """Streaming is the last resort, not the default.

    The first version of this function answered `TPW=0, FOLD=1` whenever the
    tensor looked cache-resident. That was fitted at 4096 columns and cost 40%
    at 1024, where the winner still holds the row. The regime question is gone
    now (see plan's docstring), but the tempting simplification it licensed --
    "when in doubt, stream" -- is the one worth pinning against.
    """
    from hk.ops.norm import MAX_TPW, plan

    for kind in ("rms", "layer", "softmax"):
        for cols in (1024, 2048, 4096, 8192, 16384):
            warps, tpw, _ = plan(cols, kind)
            assert tpw > 0, (cols, kind)
            assert tpw <= MAX_TPW
    # Streaming happens only where neither view can hold the row: 8199 is not
    # a multiple of 16, so no folded view exists, and 129 blocks is 9 a warp.
    assert plan(8199) == (16, 0, 1)


def test_quantize_and_norm_agree_about_the_regime_and_differ_about_fold():
    """Neither function asks whether the tensor is cache-resident any more.

    Both used to. The argument was that paying registers to hold the row is a
    straight loss when the re-read it saves would have been a cache hit, and
    both functions were corrected by the same measurement -- the held variant
    wins inside the cache too, and by more than it wins outside it:

        quantize 4096x4096   48 MB, resident   w16 tpw4 0.062 vs tpw0 0.094
        quantize 8192x4096   96 MB, astride    w16 tpw4 0.113 vs tpw0 0.161
        rms      4096x4096   64 MB, resident   w16 tpw4 0.063 vs tpw0 0.088
        rms      6144x4096   96 MB, astride    w16 tpw4 0.087 vs tpw0 0.117

    What they still disagree about is FOLD, and the disagreement is real:
    quantize folds only where the unfolded plan does not exist, and so does
    norm for rms and layer, but softmax folds first.
    """
    from hk.ops.norm import plan as norm_plan
    from hk.ops.quant import plan

    for cols in (1024, 4096, 16384):
        for ws in (None, 1 << 20, 50 << 20, 201 << 20, 1 << 40):
            assert plan(cols, ws) == plan(cols), (cols, ws)

    # Unfolded, and held: four tiles each across all sixteen warps.
    assert plan(4096) == (16, 4, 1) == norm_plan(4096, "rms")
    assert norm_plan(4096, "softmax") == (4, 1, 16)


def test_exact_still_holds_for_every_kind():
    """EXACT is about the width, not the op, and must survive the switch.

    All three kinds are asked, because EXACT is computed from (cols//fold,
    warps, tpw) and `kind` changes all three -- a flag that happened to be
    right for the rms plan is not thereby right for softmax's.
    """
    from hk.ops.norm import _exact, plan
    for cols, want in ((4096, True), (1024, True), (200, False)):
        for kind in ("rms", "layer", "softmax"):
            w, tpw, fold = plan(cols, kind)
            assert _exact(cols // fold, w, tpw) is want, \
                f"{cols} as {kind} -> {(w, tpw, fold)}"


def test_exact_is_what_plan_says_it_is():
    """The flag is a claim about the plan, so check it against the plan.

    `_exact` says no column block is ever backed up. That is true exactly when
    the row is a whole number of 64-wide blocks *and* the warp-slots land on
    them: WARPS*TPW of them for the persistent variant, WARPS per step for the
    streaming one. Both halves matter -- 8208 columns folds to a width that is
    not a multiple of 64, and 8199 does not fold at all.
    """
    from hk.ops.norm import _exact, plan

    for cols, want in ((4096, True), (16384, True), (2048, True), (256, True),
                       (200, False), (513, False), (8208, False),
                       (8199, False)):
        w, tpw, fold = plan(cols)
        assert _exact(cols // fold, w, tpw) is want, f"{cols} -> {(w, tpw, fold)}"


def test_an_exact_layernorm_neither_masks_nor_copies():
    """What EXACT is for, and the only place it pays more than a scalar op.

    LayerNorm's mean pass needs a tile it is allowed to scribble on, because
    masking writes in place and the tile it is handed is the one the write pass
    will store. When there is nothing to mask that copy has no reason to exist,
    and at 4096 columns -- which folds to four exact blocks -- it should not be
    in the source at all.
    """
    k = hk.ops.norm.KERNELS["layer_bf16_w4"]
    exact = k.source(TPW=1, FOLD=16, EXACT=1)
    ragged = k.source(TPW=1, FOLD=16, EXACT=0)

    assert "left_fill" not in exact
    assert "left_fill" in ragged
    # One `kittens::copy` either way is the bf16 narrowing in the store, which
    # is not the copy under discussion; the ragged variant has a second.
    assert exact.count("kittens::copy") == 1
    assert ragged.count("kittens::copy") == 2


def test_an_exact_kernel_does_not_clamp_its_column():
    """No back-up means no s_min against the end of the row. The clamp is one
    scalar op, but its presence is the observable half of the claim: if it is
    still there, some block can still be backed up and dropping the mask would
    be wrong."""
    k = hk.ops.norm.KERNELS["rms_bf16_w4"]
    # The clamp emits as a ternary. One of them is the *row* back-up, which
    # EXACT says nothing about; the column one is the second.
    assert k.source(TPW=1, FOLD=16, EXACT=1).count("? ") == 1
    assert k.source(TPW=1, FOLD=16, EXACT=0).count("? ") == 2


# -- the shipped Phase 2 kernels ----------------------------------------------


def test_every_phase2_kernel_traces():
    for mod in (hk.ops.norm, hk.ops.fused, hk.ops.quant):
        for name, k in mod.KERNELS.items():
            assert "hk.store" in k.trace().dump(), name


def test_specialize_is_memoized():
    """A fresh Kernel has empty trace/build/module caches, so a wrapper that
    specializes per call re-traces and re-resolves the module every launch.
    Measured, that was ~3.5 ms of Python in front of a 0.5 ms kernel."""
    a = rowsum.specialize(COLS=64)
    assert a is rowsum.specialize(COLS=64)
    assert a is not rowsum.specialize(COLS=128)
    assert a.trace() is a.trace()


def test_a_constexpr_can_be_passed_to_the_call_itself():
    """The cheaper surface: no object, and it hits the caches directly."""
    assert rowsum.trace(COLS=128).consts["COLS"] == 128
    assert rowsum.source(COLS=128) is not None
