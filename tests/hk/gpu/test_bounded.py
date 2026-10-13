"""Row-predicated global load and store: `hk.load(..., bounded=True)` and
`hk.store(..., bounded=True)`.

What has to hold, and the case that reaches it:

  * rows past the tensor's last read as zero -- a short batch, M < 16;
  * rows past the tensor's last are not written -- the output carved out of a
    larger buffer, whose rows after it must keep their contents;
  * the 16-byte buffer loads have no alignment fallback, on the strength of
    gfx11 running with unaligned access enabled -- a tensor that starts 2 bytes
    past a 16-byte boundary;
  * the element-by-element path (a bf16 tensor into an fp32 tile) is bounded
    the same way.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import hk  # noqa: E402
from hk import bf16, fp32  # noqa: E402
from hk.lang import ops  # noqa: E402


def _copy_kernel(name, tile_dtype):
    ty = ops.rt(tile_dtype, 16, 64, "row")

    def body(x: hk.GL[bf16], y: hk.GL[bf16]):
        col = ops.block_idx.x * 64
        t = ops.load(x, ops.elem_coord(0, 0, 0, col), ty, bounded=True)
        ops.store(y, t, ops.elem_coord(0, 0, 0, col), bounded=True)

    body.__name__ = name
    return hk.kernel(body, arch="gfx1100", warps=1, name=name,
                     grid=lambda p: (p.x.cols // 64, 1, 1))


COPY = {"bf16": _copy_kernel("bounded_copy_bf16", bf16),
        "fp32": _copy_kernel("bounded_copy_fp32", fp32)}


@pytest.mark.parametrize("tile", ["bf16", "fp32"])
@pytest.mark.parametrize("m", [1, 5, 15, 16])
def test_a_short_tensor_round_trips_and_nothing_past_it_is_written(tile, m):
    torch.manual_seed(m)
    x = torch.randn(m, 256, device="cuda", dtype=torch.bfloat16)
    room = torch.full((16, 256), 3.0, device="cuda", dtype=torch.bfloat16)
    COPY[tile](x, room[:m])
    torch.cuda.synchronize()
    assert torch.equal(room[:m], x)
    assert torch.equal(room[m:], torch.full_like(room[m:], 3.0))


def test_rows_past_the_end_are_read_as_zero_not_from_memory():
    # The bytes after x are nonzero; a load that read them would carry them
    # into the WMMA. Store the whole 16-row tile into a 16-row output to see
    # what the rows past x were loaded as.
    k = _zero_probe()
    buf = torch.full((16 * 64,), 5.0, device="cuda", dtype=torch.bfloat16)
    x = buf[:3 * 64].view(3, 64)
    x.copy_(torch.randn(3, 64, device="cuda", dtype=torch.bfloat16))
    y = torch.full((16, 64), 9.0, device="cuda", dtype=torch.bfloat16)
    k(x, y)
    torch.cuda.synchronize()
    assert torch.equal(y[:3], x)
    assert torch.equal(y[3:], torch.zeros_like(y[3:]))


def _zero_probe():
    ty = ops.rt(bf16, 16, 64, "row")

    def body(x: hk.GL[bf16], y: hk.GL[bf16]):
        t = ops.load(x, ops.elem_coord(0, 0, 0, 0), ty, bounded=True)
        ops.store(y, t, ops.elem_coord(0, 0, 0, 0))

    body.__name__ = "bounded_zero_probe"
    return hk.kernel(body, arch="gfx1100", warps=1, name="bounded_zero_probe",
                     grid=lambda p: (1, 1, 1))


@pytest.mark.parametrize("tile", ["bf16", "fp32"])
def test_a_tensor_two_bytes_off_alignment(tile):
    torch.manual_seed(1)
    m = 7
    big = torch.randn(m * 256 + 8, device="cuda", dtype=torch.bfloat16)
    x = big[1:1 + m * 256].view(m, 256)            # starts 2 bytes past 16
    assert x.data_ptr() % 16 == 2
    out = torch.zeros(m, 256, device="cuda", dtype=torch.bfloat16)
    COPY[tile](x, out)
    torch.cuda.synchronize()
    assert torch.equal(out, x)
