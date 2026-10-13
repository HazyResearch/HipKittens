"""Granule sets, the fragment ops on them, and the diagonal fills.

A granule set exists for one reason: a shared tile's LDS address is affine in
the fragment coordinate, so the whole (row, col) dependence folds into
`ds_read_b128`'s 16-bit immediate and what is left is `swizzle_bytes/16`
lane-dependent registers for the tile, however many fragments are read out of
it. The alternative -- `subtile` plus a shared load -- gives every fragment its
own loop-invariant address, and the compiler hoists all of them into the
preheader and spills them. Measured at 312 B/lane that way in the attention
kernel and 0 this way.

So the tests here are mostly about what the type *refuses*. A granule set built
on a tile whose addressing is not affine in the fragment coordinate would be
silently wrong -- the reads would land on the wrong bytes, the kernel would
still compile, and `hip_resources.py` would still say ScratchSize 0.
"""

import pytest

import hk
from hk import bf16, fp16, fp32


def _trace(body, **kw):
    k = hk.kernel(body, arch="gfx1100", warps=1,
                  grid=lambda p: (1, 1, 1), name=body.__name__)
    return k.trace(**kw)


def _src(body):
    return hk.kernel(body, arch="gfx1100", warps=1,
                     grid=lambda p: (1, 1, 1), name=body.__name__).source()


# ----------  the type  ----------

@pytest.mark.parametrize("cols, swizzle, n", [
    (128, 128, 8),   # width 8, divisible by 4
    (64, 128, 8),    # width 4
    (32, 64, 4),     # width 2
])
def test_swizzle_follows_the_cpp(cols, swizzle, n):
    """The granule count is st.cuh's swizzle rule, not a parameter."""
    def body(x: hk.Tensor[bf16]):
        gr = hk.lds.alloc(hk.ST(bf16, 32, cols)).granules()
        assert gr.type.swizzle_bytes == swizzle
        assert gr.type.n == n
    _trace(body)


def test_rejects_32_byte_swizzle():
    """width 1: 16 rows reach the bit the swizzle keys on, so `idx` stops
    being affine in the row and the decomposition does not hold."""
    def body(x: hk.Tensor[bf16]):
        hk.lds.alloc(hk.ST(bf16, 128, 16)).granules()
    with pytest.raises(Exception, match="swizzle_bytes is 32"):
        _trace(body)


def test_rejects_4_byte_dtype():
    def body(x: hk.Tensor[fp32]):
        hk.lds.alloc(hk.ST(fp32, 32, 128)).granules()
    with pytest.raises(Exception):
        _trace(body)


def test_accepts_fp16():
    """The derivation is about the element *width*, not the format."""
    def body(x: hk.Tensor[fp16]):
        gr = hk.lds.alloc(hk.ST(fp16, 32, 128)).granules()
        assert gr.type.n == 8
    _trace(body)


# ----------  load_frag  ----------

def test_load_frag_emits_one_call_per_fragment():
    def body(x: hk.Tensor[bf16]):
        gr = hk.lds.alloc(hk.ST(bf16, 32, 128)).granules()
        t = hk.RT(bf16, 16, 16, "row")
        a = gr.load_frag(0, 3, t)
        b = gr.load_frag(1, 3, t)
        hk.lds.wait_for(a, b)
    src = _src(body)
    assert "lds_read_frag<0, 48>" in src
    assert "lds_read_frag<16, 48>" in src


def test_load_frag_rejects_col_layout():
    """Only row-layout 16-bit operands reach ds_read_b128; col degrades to
    sixteen ds_read_u16 and the granule arithmetic does not describe it."""
    def body(x: hk.Tensor[bf16]):
        gr = hk.lds.alloc(hk.ST(bf16, 32, 128)).granules()
        gr.load_frag(0, 0, hk.RT(bf16, 16, 16, "col"))
    with pytest.raises(Exception):
        _trace(body)


def test_load_frag_rejects_wide_tile():
    def body(x: hk.Tensor[bf16]):
        gr = hk.lds.alloc(hk.ST(bf16, 32, 128)).granules()
        gr.load_frag(0, 0, hk.RT(bf16, 16, 32, "row"))
    with pytest.raises(Exception):
        _trace(body)


def test_load_frag_rejects_runtime_row():
    """The row is the `ds_read_b128` immediate, so it has to be a constant."""
    def body(x: hk.Tensor[bf16]):
        gr = hk.lds.alloc(hk.ST(bf16, 32, 128)).granules()
        gr.load_frag(hk.arch.block_idx.x, 0, hk.RT(bf16, 16, 16, "row"))
    with pytest.raises(Exception):
        _trace(body)


def test_load_frag_rejects_dtype_mismatch():
    def body(x: hk.Tensor[bf16]):
        gr = hk.lds.alloc(hk.ST(bf16, 32, 128)).granules()
        gr.load_frag(0, 0, hk.RT(fp16, 16, 16, "row"))
    with pytest.raises(Exception):
        _trace(body)


# ----------  store_frag  ----------

def test_store_frag_runtime_row_scales_by_swizzle():
    """A runtime row cannot be an immediate, so it becomes a byte offset --
    `rows * swizzle_bytes` per 16 rows, which is the affine term."""
    def body(x: hk.Tensor[bf16]):
        gr = hk.lds.alloc(hk.ST(bf16, 128, 32)).granules()  # swizzle 64
        v = hk.zeros(hk.RT(bf16, 16, 16, "row"))
        gr.store_frag(v, hk.arch.block_idx.x, 1)
    src = _src(body)
    assert "lds_write_frag<0, 16>" in src
    assert "1024 *" in src          # 16 * 64


def test_store_frag_constant_row_is_an_immediate():
    def body(x: hk.Tensor[bf16]):
        gr = hk.lds.alloc(hk.ST(bf16, 128, 32)).granules()
        v = hk.zeros(hk.RT(bf16, 16, 16, "row"))
        gr.store_frag(v, 2, 1)
    src = _src(body)
    assert "lds_write_frag<32, 16>" in src
    assert "0u" in src


# ----------  the diagonal fills  ----------

@pytest.mark.parametrize("op, cpp", [("triu", "triu"), ("tril", "tril")])
def test_diagonal_fills_lower(op, cpp):
    def body(x: hk.Tensor[fp32]):
        t = hk.zeros(hk.RT(fp32, 16, 16, "col"))
        getattr(t, f"{op}_")(hk.arch.block_idx.x, float("-inf"))
    assert f"kittens::{cpp}(" in _src(body)


def test_diagonal_fill_takes_a_runtime_diagonal():
    """Unlike `make_causal_t`, which is the d == 0 case and is only enough
    when every extent is 16-aligned."""
    def body(x: hk.Tensor[fp32]):
        t = hk.zeros(hk.RT(fp32, 16, 16, "col"))
        t.triu_(hk.arch.block_idx.x - 16, float("-inf"))
    _trace(body)
