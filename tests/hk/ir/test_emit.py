"""Codegen, without a compiler.

What is asserted here is the shape of the generated C++ -- that it declares
what it uses, that tile coordinates are converted at the use site, that the
scaffold is separable. Whether it *compiles* is tests/hk/codegen's job; that
tier needs hipcc and this one does not, so the fast loop stays fast.

The C++ is written by hk-translate from the traced MLIR module.

Deliberately not a golden-file diff of the whole output: a golden file makes
every formatting change a test change, and the properties below are the ones
that being wrong would produce a wrong kernel rather than an ugly one.
"""

from __future__ import annotations

import pytest

import hk
from hk import bf16, fp32


@hk.kernel(
    arch="gfx1100",
    grid=lambda p: (hk.cdiv(p.o.cols, 64), hk.cdiv(p.o.rows, 16), 1),
)
def add2(a: hk.Tensor[bf16], b: hk.Tensor[bf16], o: hk.Tensor[bf16], *,
         ROWS=16, COLS=64):
    t = hk.RT(bf16, ROWS, COLS)
    idx = hk.tile_coord(0, 0, hk.arch.block_idx.y, hk.arch.block_idx.x)
    o[idx] = a[idx].load(t) + b[idx].load(t)


@pytest.fixture(scope="module")
def src():
    return add2.source()


def test_every_tensor_parameter_becomes_a_globals_member(src):
    for name in ("a", "b", "o"):
        assert f"kittens::gl<bf16, -1, -1, -1, -1> {name};" in src


def test_constexprs_are_baked_in_not_passed(src):
    """A constexpr that reached the kernel as an argument would not be a tile
    extent any more, so the whole thing would fail to instantiate."""
    assert "int ROWS" not in src and "int COLS" not in src
    assert "rt<bf16, 16, 64," in src


def test_the_tile_type_is_named_once_and_reused(src):
    assert src.count("using rt_bf16_16x64_row =") == 1
    assert src.count("kittens::rt<bf16, 16, 64,") == 1


def test_tile_coordinates_are_converted_at_each_use(src):
    """`coord<>` and `coord<RT>` index in different units. The IR carries the
    raw int4 and the emitter has to re-attach the tile type at every use; a
    missed one indexes in elements and reads the wrong memory silently."""
    assert src.count("kittens::coord<rt_bf16_16x64_row>(idx)") == 3
    assert "const int4 idx = make_int4(" in src


def test_launch_bounds_match_the_declared_warp_count(src):
    assert "__launch_bounds__(32, 1)" in src
    assert "dim3 block() const { return dim3(32); }" in src


def test_no_lds_means_no_allocator(src):
    assert "__shm" not in src and "shared_allocator" not in src


def test_names_are_shared_with_the_ir_dump(src):
    """The generated C++ is what you take to the disassembler when a kernel
    spills; if its names did not match the dump's you could not follow it."""
    for line in add2.trace().dump().splitlines():
        line = line.strip()
        if line.startswith("%ld1"):
            assert " ld1;" in src
            break
    else:
        pytest.fail("expected a %ld1 value in the dump")


def test_the_kernel_body_is_independent_of_the_module_scaffold():
    """Phase 6 swaps pybind for TORCH_LIBRARY. If the two were entangled the
    torch build would be a second emitter, and the second one would drift."""
    bare = add2.source(scaffold="bare")
    assert "PYBIND11_MODULE" not in bare
    body = "void add2_kernel(const globals g) {"
    assert body in bare and body in add2.source(scaffold="pybind")


def test_unknown_scaffold_is_refused():
    with pytest.raises(ValueError, match="unknown scaffold"):
        add2.source(scaffold="nope")


# -- op coverage --------------------------------------------------------------


def test_scalar_operand_is_cast_to_the_tile_dtype():
    """kittens' bin_map takes `const typename T::dtype &`, so an untyped
    literal binds to the wrong overload or does not bind at all."""

    @hk.kernel(arch="gfx1100", grid=lambda p: (1, 1, 1))
    def scaled(a: hk.Tensor[bf16], o: hk.Tensor[bf16]):
        i = hk.tile_coord()
        o[i] = a[i].load(hk.RT(bf16, 16, 16)) * 2.0

    assert "kittens::mul(mul, ld, (bf16)2.0);" in scaled.source()


def test_neg_lowers_to_a_multiply():
    """There is no kittens::neg -- only neg_infty, which is a fill. Emitting
    `neg` would not compile, so the tracer has to lower it."""

    @hk.kernel(arch="gfx1100", grid=lambda p: (1, 1, 1))
    def negate(a: hk.Tensor[fp32], o: hk.Tensor[fp32]):
        i = hk.tile_coord()
        o[i] = -a[i].load(hk.RT(fp32, 16, 16))

    src = negate.source()
    assert "kittens::neg(" not in src
    assert "kittens::mul(mul, ld, (float)-1.0);" in src


def test_an_op_without_a_dtype_specialisation_is_refused_at_trace_time():
    @hk.kernel(arch="gfx1100", grid=lambda p: (1, 1, 1))
    def bad(a: hk.Tensor[bf16], o: hk.Tensor[bf16]):
        i = hk.tile_coord()
        o[i] = a[i].load(hk.RT(bf16, 16, 16)).gelu()

    with pytest.raises(TypeError, match="no bf16 implementation"):
        bad.trace()


def test_shipped_kernels_only_exist_where_the_library_supports_them():
    from hk.ops.elementwise import KERNELS

    assert "gelu_fp32" in KERNELS
    assert "gelu_bf16" not in KERNELS and "gelu_fp16" not in KERNELS
    assert "neg_bf16" in KERNELS  # a multiply, so every dtype has it


# ------------------------------------------------------------- load_scalar
#
# The primitive a paged kernel is built on. A page table turns "which block"
# into a value fetched from memory, which `block_idx` and arithmetic on it
# cannot express at all.


def _scalar_kernel(coord=None, name="scalar_probe"):
    mk = coord or hk.elem_coord

    # hk.i32 rather than a bare `i32`: PEP 563 makes these annotations strings
    # resolved against this module's globals, not against this function's.
    def body(table: hk.Tensor[hk.i32], x: hk.Tensor[bf16], o: hk.Tensor[bf16]):
        blk = table[mk(0, 0, 0, hk.arch.block_idx.x)].load_scalar()
        t = hk.RT(bf16, 16, 64)
        o[hk.elem_coord(0, 0, 0, 0)] = x[hk.elem_coord(blk, 0, 0, 0)].load(t)

    return hk.kernel(body, arch="gfx1100", warps=1, name=name,
                     grid=lambda p: (1, 1, 1))


def test_load_scalar_emits_the_library_call():
    src = _scalar_kernel().source("bare")
    # A plain int, so it composes with Int arithmetic and elem_coord like
    # block_idx does.
    assert "const int scalar = kittens::load_scalar(g.table," in src


def test_the_loaded_value_reaches_an_address():
    # The whole point of the op: the page number indexes a load.
    src = _scalar_kernel().source("bare")
    coords = [l for l in src.splitlines() if "kittens::coord<>" in l]
    assert any("scalar" in l for l in coords), coords


def test_load_scalar_refuses_a_tile_coordinate():
    with pytest.raises(TypeError, match="element coordinate"):
        _scalar_kernel(coord=hk.tile_coord, name="p_tile").source("bare")


def test_load_scalar_refuses_a_float_global():
    def body(x: hk.Tensor[bf16], o: hk.Tensor[bf16]):
        x[hk.elem_coord(0, 0, 0, 0)].load_scalar()
        o[hk.elem_coord(0, 0, 0, 0)] = hk.zeros(hk.RT(bf16, 16, 64))

    with pytest.raises(TypeError, match="must be i32"):
        hk.kernel(body, arch="gfx1100", warps=1, name="p_float",
                  grid=lambda p: (1, 1, 1)).source("bare")
