"""Globals over strided views: a slice of a tensor's last axis, read and
written in place, through both entry points -- the pybind launcher
(pyutils/hk_bind.cuh) and the torch op scaffold (codegen/scaffold.py).

The view that matters is the one an engine hands over: the two halves of a
fused (M, 2I) gate/up output, x[:, :I] and x[:, I:], each with I columns and a
row stride of 2I. A view the global cannot describe must be refused, not read
as if it were one it can.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import hk  # noqa: E402
from hk import bf16  # noqa: E402


def _copy_kernel():
    ty = hk.RT(bf16, 16, 64, "row")

    def strided_copy(x: hk.Tensor[bf16], y: hk.Tensor[bf16]):
        idx = hk.elem_coord(0, 0, hk.arch.block_idx.y * 16,
                            hk.arch.block_idx.x * 64)
        y[idx] = x[idx].load(ty)

    return hk.kernel(strided_copy, arch="gfx1100", warps=1, name="strided_copy",
                     grid=lambda p: (p.x.cols // 64, p.x.rows // 16, 1))


COPY = _copy_kernel()


@pytest.fixture(scope="module")
def op():
    return hk.torch_op(COPY)


@pytest.mark.parametrize("half", [0, 1])
def test_each_half_of_a_fused_output_is_read_in_place(half):
    torch.manual_seed(half)
    full = torch.randn(32, 2 * 256, device="cuda", dtype=torch.bfloat16)
    x = full[:, half * 256:(half + 1) * 256]
    assert not x.is_contiguous()
    out = torch.zeros(32, 256, device="cuda", dtype=torch.bfloat16)
    COPY(x, out)
    torch.cuda.synchronize()
    assert torch.equal(out, x)


def test_a_strided_destination_is_written_in_place_and_nowhere_else():
    torch.manual_seed(2)
    x = torch.randn(32, 256, device="cuda", dtype=torch.bfloat16)
    full = torch.full((32, 512), 7.0, device="cuda", dtype=torch.bfloat16)
    COPY(x, full[:, 256:])
    torch.cuda.synchronize()
    assert torch.equal(full[:, 256:], x)
    assert torch.equal(full[:, :256], torch.full_like(full[:, :256], 7.0))


def test_the_torch_op_takes_the_same_views(op):
    torch.manual_seed(3)
    full = torch.randn(16, 3 * 128, device="cuda", dtype=torch.bfloat16)
    x = full[:, 128:256]
    out = torch.empty(16, 128, device="cuda", dtype=torch.bfloat16)
    op(x, out)
    torch.cuda.synchronize()
    assert torch.equal(out, x)


@pytest.mark.parametrize("view", ["transposed", "every_other_column"])
def test_a_view_the_global_cannot_describe_is_refused(view, op):
    full = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    x = full.t()[:64] if view == "transposed" else full[:, ::2]
    out = torch.empty(x.shape, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="slice of"):
        COPY(x, out)
    with pytest.raises(RuntimeError, match="slice of"):
        op(x, out)
