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
from hk._mlir import ir
from hk.compiler import mlir as M


@hk.kernel(arch="gfx1100", grid=lambda p: (1, hk.cdiv(p.x.rows, 16), 1))
def rowsum(x: hk.Tensor[fp32], o: hk.Tensor[fp32], *, ROWS=16, COLS=64):
    t = hk.RT(fp32, ROWS, COLS)
    acc = hk.zeros(hk.col_vec(t))
    n = hk.cdiv(x.cols, COLS)
    for cb in hk.range(n):
        idx = hk.elem_coord(0, 0, hk.arch.block_idx.y * ROWS, cb * COLS)
        acc += x[idx].load(t).sum(axis=1)
    o[hk.elem_coord(0, 0, 0, hk.arch.block_idx.y * ROWS)] = acc


def _kernel(traced):
    """The kernel's func.func in the traced module."""
    return M.kernel_of(traced.module)


def _body(op):
    """The ops directly in `op`'s (first) region -- not the nested ones."""
    return list(op.regions[0].blocks[0].operations)


def _walk(op):
    """Every op nested in `op`'s regions at any depth, in program order."""
    out = []

    def visit(o):
        out.append(o)
        return ir.WalkResult.ADVANCE

    with M.context():
        op.operation.walk(visit, ir.WalkOrder.PRE_ORDER)
    return out[1:]  # pre-order: the first is `op` itself


def _ops(op, name):
    return [o for o in _walk(op) if o.name == name]


# -- regions ------------------------------------------------------------------


def test_a_loop_is_a_region_not_a_flattened_body():
    k = _kernel(rowsum.trace())
    (loop,) = _ops(k, "scf.for")
    assert [op for op in _body(loop) if op.name != "scf.yield"], \
        "hk.range produced no region"
    assert [op.name for op in _body(k)].count("hk.load") == 0
    assert [op.name for op in _walk(loop)].count("hk.load") == 1


def test_the_induction_variable_is_a_block_argument():
    """Scoped to the region. If it were an ordinary result the emitter would
    declare it outside the `for`, where the loop cannot assign it."""
    (loop,) = _ops(_kernel(rowsum.trace()), "scf.for")
    (iv,) = loop.regions[0].blocks[0].arguments
    assert isinstance(iv, ir.BlockArgument)
    assert iv not in [r for op in _walk(loop) for r in op.results]
    # ... and so the C++ declares it in the `for` header and nowhere else.
    src = rowsum.source()
    assert "for (int cb = 0;" in src and src.count("int cb") == 1


def test_walk_reaches_into_regions_and_ops_does_too():
    k = _kernel(rowsum.trace())
    assert len(_ops(k, "hk.row_sum")) == 1
    assert len(_walk(k)) > len(_body(k))


# -- in-place -----------------------------------------------------------------


def test_an_accumulating_reduction_writes_the_accumulator_it_was_given():
    k = _kernel(rowsum.trace())
    (red,) = _ops(k, "hk.row_sum")
    (acc,) = _ops(k, "hk.zero")
    assert red.opview.dst is not None and "accumulate" in red.attributes
    assert red.opview.dst == acc.result
    assert not list(red.results), "an in-place op that also produces a result "\
        "would let the emitter declare a fresh destination inside the loop"


def test_the_accumulator_is_declared_once_outside_the_loop():
    """The whole point. A redeclaration inside the region shadows the carried
    value and the kernel keeps only the last iteration -- at full speed, with
    no diagnostic."""
    src = rowsum.source()
    body = src.split("for (int", 1)[1]
    decl = "_col zv;"  # named for hk.zeros of a vector, the op that produced it
    assert src.count(decl) == 1 and decl not in body


def test_an_accumulating_reduction_passes_its_destination_as_the_third_argument():
    """kittens::row_sum(dst, src, src_accum). Dropping the third argument is
    an overwrite, which is a correct-looking kernel that loses every block but
    the last."""
    assert "kittens::row_sum(zv, ld, zv);" in rowsum.source()


def test_a_non_accumulating_reduction_declares_its_own_destination():
    @hk.kernel(arch="gfx1100", grid=lambda p: (1, 1, 1))
    def once(x: hk.Tensor[fp32], o: hk.Tensor[fp32]):
        t = hk.RT(fp32, 16, 64)
        i = hk.elem_coord(0, 0, 0, 0)
        o[hk.elem_coord(0, 0, 0, 0)] = x[i].load(t).max(axis=1)

    (red,) = _ops(_kernel(once.trace()), "hk.row_max")
    assert red.opview.dst is None and len(red.results) == 1
    assert "_col row_max;" in once.source()
    assert "kittens::row_max(row_max, ld);" in once.source()


# -- fills --------------------------------------------------------------------


def test_a_fill_carries_a_runtime_index_not_a_constant():
    """The tail idiom masks `blockIdx`-dependent many columns. A fill whose
    index had to be a literal could not express it at all."""

    @hk.kernel(arch="gfx1100", grid=lambda p: (1, 1, 1))
    def masked(x: hk.Tensor[fp32], o: hk.Tensor[fp32], *, COLS=64):
        t = hk.RT(fp32, 16, COLS)
        n = x.cols
        lo = hk.min(hk.arch.block_idx.x * COLS, n - COLS)
        pad = hk.arch.block_idx.x * COLS - lo
        idx = hk.elem_coord(0, 0, 0, lo)
        o[idx] = x[idx].load(t).left_fill_(pad, 0.0)

    (fill,) = _ops(_kernel(masked.trace()), "hk.left_fill")
    assert fill.opview.idx.owner.name == "arith.subi"
    assert "kittens::left_fill(ld, ld, sub1, (float)0.0);" in masked.source()


def test_a_fill_writes_its_source_in_place():
    """conversions.cuh:530 is `if (col_idx <= 0) return;` -- an empty mask
    returns without touching dst. Since an empty mask is the *normal* case
    (every column block but the last), a fresh destination would hold
    uninitialized registers precisely when nothing needs masking, and the
    kernel would produce NaN on the shapes that do divide. It did. Filling in
    place makes the early return leave the loaded values alone."""

    @hk.kernel(arch="gfx1100", grid=lambda p: (1, 1, 1))
    def m(x: hk.Tensor[fp32], o: hk.Tensor[fp32]):
        t = hk.RT(fp32, 16, 64)
        i = hk.elem_coord(0, 0, 0, 0)
        v = x[i].load(t)
        assert v.left_fill_(3, 0.0) is v
        o[i] = v

    (fill,) = _ops(_kernel(m.trace()), "hk.left_fill")
    assert fill.opview.dst is not None and fill.opview.dst == fill.opview.src
    src = m.source()
    assert src.count("rt_fp32_16x64_row ld;") == 1
    assert src.count("rt_fp32_16x64_row ") == 2, "a fill declared a second tile"


def test_an_explicit_out_on_a_fill_copies_first():
    """Allowed, but it costs the copy the early return would otherwise skip.
    Emitting the fill alone here would be the same uninitialized-dst bug."""

    @hk.kernel(arch="gfx1100", grid=lambda p: (1, 1, 1))
    def m(x: hk.Tensor[fp32], o: hk.Tensor[fp32]):
        t = hk.RT(fp32, 16, 64)
        i = hk.elem_coord(0, 0, 0, 0)
        v = x[i].load(t)
        w = hk.zeros(t)
        v.right_fill_(8, 1.0, into=w)
        o[i] = w

    src = m.source()
    assert src.index("kittens::copy(z, ld);") < src.index("kittens::right_fill(z, ld,")


def test_neg_infty_fill_uses_the_named_constant_not_a_c_literal():
    """repr(float('-inf')) is `-inf`, which is not a C++ token: the generated
    file would not compile. It did not, once."""

    @hk.kernel(arch="gfx1100", grid=lambda p: (1, 1, 1))
    def m(o: hk.Tensor[fp32]):
        t = hk.RT(fp32, 16, 64)
        o[hk.elem_coord(0, 0, 0, 0)] = hk.full(t, float("-inf"))

    src = m.source()
    assert "-inf" not in src
    assert "kittens::neg_infty(" in src


# -- scalars ------------------------------------------------------------------


def test_scalar_arithmetic_stays_on_the_host_side_of_the_type_system():
    ops = _walk(_kernel(rowsum.trace()))
    assert {op.name for op in ops} >= {"arith.muli", "arith.ceildivsi"}
    for op in ops:
        if op.name.startswith("arith."):
            assert [str(r.type) for r in op.results] == ["i32"], op


def test_a_hoisted_scalar_is_emitted_once():
    """blockIdx.y is the same in every use; one `const int` keeps the
    generated C++ readable enough to take to the disassembler."""
    assert rowsum.source().count("= blockIdx.y;") == 1


def test_there_is_no_rounding_division_on_a_tile():
    @hk.kernel(arch="gfx1100", grid=lambda p: (1, 1, 1))
    def bad(x: hk.Tensor[fp32], o: hk.Tensor[fp32]):
        i = hk.elem_coord(0, 0, 0, 0)
        o[i] = x[i].load(hk.RT(fp32, 16, 64)) // 2

    with pytest.raises(TypeError, match="rounding division"):
        bad.trace()


# -- dtypes -------------------------------------------------------------------


def test_int8_is_a_global_type_and_not_a_register_type():
    """gfx1100 has no int8 WMMA and no packed int8 ALU, so rt<int8> would be a
    template error at the end of a 30 s compile. Refuse it at trace time with
    the reason."""
    assert hk.Tensor[i8]
    with pytest.raises(TypeError, match="no i8 register tile"):
        hk.RT(i8, 16, 64)


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
    """No back-up means no hk.min against the end of the row. The clamp is one
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
