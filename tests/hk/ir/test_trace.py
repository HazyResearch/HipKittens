"""Tracing: does the Python body produce the IR we expect.

The IR is the hk MLIR module `k.trace()` holds; facts about it are read off
the kernel func's ops or its dump. No compiler, no GPU.
"""

from __future__ import annotations

import re
from typing import List, NamedTuple

import pytest

import hk
from hk import bf16, fp32
from hk.compiler import mlir as M


@hk.kernel(
    arch="gfx1100",
    grid=lambda p: (hk.cdiv(p.o.cols, p.COLS), hk.cdiv(p.o.rows, p.ROWS), 1),
)
def add2(a: hk.Tensor[bf16], b: hk.Tensor[bf16], o: hk.Tensor[bf16], *, ROWS=16, COLS=64):
    t = hk.RT(bf16, ROWS, COLS)
    idx = hk.tile_coord(0, 0, hk.arch.block_idx.y, hk.arch.block_idx.x)
    o[idx] = a[idx].load(t) + b[idx].load(t)


class Op(NamedTuple):
    name: str
    operands: List[str]  # types, as the dialect spells them
    results: List[str]


def _ops(traced) -> List[Op]:
    """The kernel func's ops, nested regions included, in program order."""
    out: List[Op] = []

    def visit(block):
        for op in block.operations:
            out.append(Op(op.name, [str(v.type) for v in op.operands],
                          [str(r.type) for r in op.results]))
            for region in op.regions:
                for b in region.blocks:
                    visit(b)

    with M.context():
        visit(M.kernel_of(traced.module).regions[0].blocks[0])
    return out


def _kernel_text(traced) -> str:
    """The kernel func of the dump, without the grid func that follows it."""
    dump = traced.dump()
    return dump[:dump.index(f"func.func @{traced.name}__grid")]


def test_params_split_into_tensors_and_constexprs():
    ir = add2.trace()
    assert [p.name for p in ir.tensors()] == ["a", "b", "o"]
    assert {p.name: p.const_value for p in ir.params if p.is_const} == {"ROWS": 16, "COLS": 64}


def test_ops_recorded_in_order():
    ir = add2.trace()
    names = [op.name for op in _ops(ir)
             if op.name in ("hk.load", "hk.add", "hk.store")]
    assert names == ["hk.load", "hk.load", "hk.add", "hk.store"]


def test_operator_sugar_is_the_same_op_as_the_function():
    """`a + b` has to be exactly hk.math.add, or the two surfaces diverge
    silently."""

    @hk.kernel(arch="gfx1100", grid=lambda p: (1, 1, 1))
    def explicit(a: hk.Tensor[bf16], o: hk.Tensor[bf16]):
        t = hk.RT(bf16, 16, 16)
        i = hk.tile_coord()
        x = a[i].load(t)
        o[i] = hk.math.add(x, x)

    @hk.kernel(arch="gfx1100", grid=lambda p: (1, 1, 1))
    def sugared(a: hk.Tensor[bf16], o: hk.Tensor[bf16]):
        t = hk.RT(bf16, 16, 16)
        i = hk.tile_coord()
        x = a[i].load(t)
        o[i] = x + x

    assert [op.name for op in _ops(explicit.trace())] == [
        op.name for op in _ops(sugared.trace())
    ]


def test_constexpr_specialisation_produces_a_different_ir():
    wide = add2.trace(COLS=128)
    assert wide.consts["COLS"] == 128
    (load,) = [op for op in _ops(wide) if op.name == "hk.load"][:1]
    assert load.results[0] == "!hk.rt<bf16, 16x128, row>"
    # and the default is untouched
    assert add2.trace().consts["COLS"] == 64


def test_trace_is_cached_per_constexpr_set():
    assert add2.trace() is add2.trace()
    assert add2.trace() is not add2.trace(COLS=128)


def test_integer_literals_are_emitted_once():
    """A coordinate built from literals must not emit one const op per use."""
    ir = add2.trace()
    assert [op.name for op in _ops(ir)].count("arith.constant") == 1


def test_value_names_are_unique_and_deterministic():
    ir = add2.trace()
    names = re.findall(r"^\s*(%[\w$.-]+) = ", _kernel_text(ir), re.M)
    assert len(names) == sum(len(op.results) for op in _ops(ir))
    assert len(set(names)) == len(names)
    assert ir.dump() == add2.trace().dump()
    # and a fresh trace of the same body names everything the same way
    again = hk.kernel(add2.fn, arch="gfx1100", grid=add2.grid_fn).trace()
    assert again is not ir and again.dump() == ir.dump()


def test_dump_distinguishes_the_two_loads():
    dump = add2.trace().dump()
    assert "%ld1 = hk.load" in dump, dump


# -- rejections ---------------------------------------------------------------


def test_tile_extents_must_be_multiples_of_the_wmma_fragment():
    with pytest.raises(ValueError, match="multiple of 16"):
        hk.RT(bf16, 16, 24)


def test_unannotated_tensor_parameter_is_rejected():
    with pytest.raises(TypeError, match="hk.Tensor"):

        @hk.kernel(arch="gfx1100", grid=lambda p: (1, 1, 1))
        def bad(a, o: hk.Tensor[bf16]):
            pass


def test_constexpr_without_a_default_is_rejected():
    with pytest.raises(TypeError, match="needs a default"):

        @hk.kernel(arch="gfx1100", grid=lambda p: (1, 1, 1))
        def bad(o: hk.Tensor[bf16], *, D):
            pass


def test_grid_expression_naming_an_unknown_parameter_fails_at_trace_time():
    @hk.kernel(arch="gfx1100", grid=lambda p: (p.nope.rows, 1, 1))
    def bad(o: hk.Tensor[bf16]):
        i = hk.tile_coord()
        o[i] = hk.zeros(hk.RT(bf16, 16, 16))

    with pytest.raises(AttributeError, match="not a parameter"):
        bad.trace()


def test_elementwise_ops_do_not_broadcast():
    @hk.kernel(arch="gfx1100", grid=lambda p: (1, 1, 1))
    def bad(a: hk.Tensor[bf16], o: hk.Tensor[bf16]):
        i = hk.tile_coord()
        x = a[i].load(hk.RT(bf16, 16, 16))
        y = a[i].load(hk.RT(bf16, 32, 16))
        o[i] = x + y

    with pytest.raises(TypeError, match="do not broadcast"):
        bad.trace()


def test_tile_and_element_coordinates_are_different_types():
    assert hk.RT(bf16, 16, 16) != hk.RT(fp32, 16, 16)
    ir_unit = hk.types.CoordType("tile")
    assert ir_unit != hk.types.CoordType("element")
    assert M.type_("!hk.coord<tile>") != M.type_("!hk.coord<element>")
