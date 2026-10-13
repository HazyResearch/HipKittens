"""Runtime branches -- `if` on a device condition, `hk.if_`, `hk.scope` -- and
the rules that keep a branch from hanging.

Branches are not a neutral construct on this target. Three separate things can
go wrong and each is checked here:

  * `if c:` on a bare index reads like "if c is in range" and means "if c is
    not zero", and those differ exactly at c == 0 -- one warp, wrong answer,
    no diagnostic;
  * a workgroup operation inside a branch is a hang, because `s_barrier` waits
    for a wave count and a warp that skipped it never arrives;
  * a branch is also a register-allocation lever in both directions, which is
    why `scope()` exists at all and why it has to survive the optimizer.

A plain `if` in a kernel body is rewritten (hk.dsl.rewrite): on a device
condition it is an scf.if, the same region `with hk.if_(cond):` builds. Both
spellings are checked where they could disagree -- the condition.
"""

import operator

import pytest

import hk
from hk import VerifyError, fp32


def _k(body, warps=4):
    return hk.kernel(body, arch="gfx1100", warps=warps,
                     grid=lambda p: (1, 1, 1), name=body.__name__)


# ----------  the condition  ----------

def test_rejects_a_bare_index():
    def body(x: hk.Tensor[fp32]):
        if hk.arch.block_idx.x:
            pass
    with pytest.raises(TypeError, match="not a comparison"):
        _k(body).trace()

    def with_if_(x: hk.Tensor[fp32]):
        with hk.if_(hk.arch.block_idx.x):
            pass
    with pytest.raises(TypeError, match="not a comparison"):
        _k(with_if_).trace()


def test_rejects_a_python_bool():
    """`hk.if_` makes a runtime branch or nothing. A plain `if` on a Python
    value is a different thing on purpose: it is decided at trace time and
    leaves no branch in the kernel."""
    def body(x: hk.Tensor[fp32]):
        with hk.if_(True):
            pass
    with pytest.raises(TypeError):
        _k(body).trace()

    def plain_if(x: hk.Tensor[fp32]):
        if True:
            x[hk.tile_coord()] = hk.zeros(hk.RT(fp32, 16, 16, "row"))
    assert "scf.if" not in _k(plain_if).trace().dump()


def test_rejects_a_tile():
    def body(x: hk.Tensor[fp32]):
        if hk.zeros(hk.RT(fp32, 16, 16, "row")):
            pass
    with pytest.raises(TypeError):
        _k(body).trace()

    def with_if_(x: hk.Tensor[fp32]):
        with hk.if_(hk.zeros(hk.RT(fp32, 16, 16, "row"))):
            pass
    with pytest.raises(TypeError):
        _k(with_if_).trace()


@pytest.mark.parametrize("pred,cmpi", [
    (operator.lt, "slt"), (operator.le, "sle"), (operator.gt, "sgt"),
    (operator.ge, "sge"), (operator.eq, "eq"), (operator.ne, "ne"),
], ids=["lt", "le", "gt", "ge", "eq", "ne"])
def test_accepts_every_comparison(pred, cmpi):
    def body(x: hk.Tensor[fp32]):
        if pred(hk.arch.block_idx.x, 4):
            x[hk.tile_coord()] = hk.zeros(hk.RT(fp32, 16, 16, "row"))
    dump = _k(body).trace().dump()
    assert f"arith.cmpi {cmpi}," in dump and "scf.if" in dump, dump


# ----------  what may not be inside one  ----------

def test_barrier_inside_a_branch_is_rejected():
    def body(x: hk.Tensor[fp32]):
        if hk.arch.block_idx.x > 0:
            hk.arch.barrier()
    with pytest.raises(VerifyError, match="workgroup operation"):
        _k(body).trace()


def test_rejected_however_deeply_nested():
    """The walk carries the flag down, so a barrier two regions inside a
    branch is the same hang as one directly inside it."""
    def body(x: hk.Tensor[fp32]):
        if hk.arch.block_idx.x > 0:
            for _i in hk.range(4):
                hk.arch.barrier()
    with pytest.raises(VerifyError, match="workgroup operation"):
        _k(body).trace()


def test_a_barrier_in_a_loop_is_fine():
    """Divergence is not the problem; skipping a rendezvous is. Every warp
    reaches a loop body the same number of times."""
    def body(x: hk.Tensor[fp32]):
        for _i in hk.range(4):
            hk.arch.barrier()
    _k(body).trace()


def test_divergent_math_is_fine():
    """The attention kernel depends on this: warps with nothing above the
    diagonal skip their math, and that is what exec masks are for."""
    def body(x: hk.Tensor[fp32]):
        if hk.arch.warp_id() > 1:
            x[hk.tile_coord()] = hk.zeros(hk.RT(fp32, 16, 16, "row"))
    _k(body).trace()


# ----------  codegen  ----------

def test_branch_lowers_to_a_plain_if():
    def body(x: hk.Tensor[fp32]):
        if hk.arch.block_idx.x > 0:
            x[hk.tile_coord()] = hk.zeros(hk.RT(fp32, 16, 16, "row"))
    src = _k(body).source()
    assert "if (" in src
    assert "+v" not in src          # not opaque: no asm clobber


def test_opaque_branch_survives_the_optimizer():
    """`asm volatile("" : "+v"(cond))`. Without it a condition that folds to
    a constant takes the branch with it, and the branch is the point."""
    def body(x: hk.Tensor[fp32]):
        with hk.if_(hk.arch.block_idx.x > 0, opaque=True):
            x[hk.tile_coord()] = hk.zeros(hk.RT(fp32, 16, 16, "row"))
    src = _k(body).source()
    assert 'asm volatile("" : "+v"' in src


def test_scope_emits_an_always_taken_branch():
    """One VGPR and a never-taken s_cbranch, bought to stop the allocator
    holding a kernel's staging and its math live at once."""
    def body(x: hk.Tensor[fp32]):
        with hk.scope():
            x[hk.tile_coord()] = hk.zeros(hk.RT(fp32, 16, 16, "row"))
    src = _k(body).source()
    assert 'asm volatile("" : "+v"' in src
    assert "if (" in src


def test_scope_is_not_a_place_for_a_barrier_either():
    def body(x: hk.Tensor[fp32]):
        with hk.scope():
            hk.arch.barrier()
    with pytest.raises(VerifyError, match="workgroup operation"):
        _k(body).trace()


# ----------  warp id  ----------

def test_warp_id_is_wave_uniform():
    """`threadIdx.x >> 5` lives in a VGPR because the compiler cannot see that
    a wave is 32 consecutive threads. Anything derived from it inherits the
    VGPR, so a comparison on it becomes a divergent v_cmpx with exec save and
    restore rather than one s_cbranch, and every value the branch needs is
    pinned in a vector register across it. Measured at 90 spilled VGPRs in the
    causal attention kernel, and none with the readfirstlane."""
    def body(x: hk.Tensor[fp32]):
        if hk.arch.warp_id() > 1:
            x[hk.tile_coord()] = hk.zeros(hk.RT(fp32, 16, 16, "row"))
    src = _k(body).source()
    assert "kittens::warpid_uniform()" in src
    assert "kittens::warpid()" not in src


# ----------  the wave-level predicate  ----------

def test_any_ne_is_a_condition():
    """It is a comparison, so `if` takes it -- the only one whose operands
    are register vectors and whose answer is wave-level rather than per-lane."""
    def body(x: hk.Tensor[fp32]):
        t = hk.RT(fp32, 16, 16, "col")
        a, b = hk.col_vec(t), hk.col_vec(t)
        if hk.zeros(a).any_ne(hk.zeros(b)):
            x[hk.tile_coord()] = hk.zeros(hk.RT(fp32, 16, 16, "row"))
    _k(body).trace()


def test_any_ne_rejects_a_tile():
    """A tile has no wave-uniform reading: lanes hold different elements, so
    "did any entry change" would be a divergent answer dressed as a scalar.
    any_ne is a method of a vector, so a tile does not have it at all, and a
    vector refuses one as the other operand."""
    def body(x: hk.Tensor[fp32]):
        t = hk.RT(fp32, 16, 16, "col")
        hk.zeros(t).any_ne(hk.zeros(t))
    with pytest.raises(AttributeError, match="any_ne"):
        _k(body).trace()

    def vec_and_tile(x: hk.Tensor[fp32]):
        t = hk.RT(fp32, 16, 16, "col")
        hk.zeros(hk.col_vec(t)).any_ne(hk.zeros(t))
    with pytest.raises(TypeError, match="any_ne"):
        _k(vec_and_tile).trace()


def test_any_ne_rejects_mismatched_lengths():
    def body(x: hk.Tensor[fp32]):
        a = hk.col_vec(hk.RT(fp32, 16, 16, "col"))
        b = hk.col_vec(hk.RT(fp32, 32, 16, "col"))
        hk.zeros(a).any_ne(hk.zeros(b))
    with pytest.raises(TypeError, match="same shape"):
        _k(body).trace()


def test_vectors_off_different_tiles_may_still_match():
    """A col_vec spans the row axis, so the tile's column count is not part of
    it. Two vectors off differently shaped tiles are the same type when they
    are the same length, and the online softmax relies on that: m and l come
    off the score tile and are compared against each other across a loop whose
    kv extent changes with the shape."""
    assert (hk.col_vec(hk.RT(fp32, 16, 16, "col"))
            == hk.col_vec(hk.RT(fp32, 16, 32, "col")))
    assert (hk.col_vec(hk.RT(fp32, 16, 16, "col"))
            != hk.col_vec(hk.RT(fp32, 32, 16, "col")))

    def body(x: hk.Tensor[fp32]):
        a = hk.col_vec(hk.RT(fp32, 16, 16, "col"))
        b = hk.col_vec(hk.RT(fp32, 16, 32, "col"))
        if hk.zeros(a).any_ne(hk.zeros(b)):
            x[hk.tile_coord()] = hk.zeros(hk.RT(fp32, 16, 16, "row"))
    _k(body).trace()


def test_any_ne_lowers_to_an_unrolled_or_tree_and_one_any():
    def body(x: hk.Tensor[fp32]):
        t = hk.RT(fp32, 16, 16, "col")
        a, b = hk.col_vec(t), hk.col_vec(t)
        if hk.zeros(a).any_ne(hk.zeros(b)):
            x[hk.tile_coord()] = hk.zeros(hk.RT(fp32, 16, 16, "row"))
    src = _k(body).source()
    assert "__any(" in src
    assert "outer_dim" in src and "inner_dim" in src
    assert src.count("#pragma unroll") >= 2
