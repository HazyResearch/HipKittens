"""A runtime `while` on the device: the trip count comes from the tensor's
shape, so the loop the kernel runs is the only thing that can get it right."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import hk  # noqa: E402

pytestmark = pytest.mark.gpu

T_IN = hk.RT(hk.bf16, 16, 16, "row")
T_ACC = hk.RT(hk.fp32, 16, 16, "row")


def block_sum(step: int = 16):
    """o = the sum of x's 16x16 column blocks, walked with a runtime while."""

    def body(x: hk.Tensor[hk.bf16], o: hk.Tensor[hk.fp32]):
        acc = hk.zeros(T_ACC)
        k = 0
        while k < x.cols:
            acc += x[hk.elem_coord(0, 0, 0, k)].load(T_IN).to(hk.fp32)
            k += step
        o[hk.elem_coord(0, 0, 0, 0)] = acc

    return hk.kernel(body, arch="gfx1100", warps=1, name=f"while_sum_s{step}",
                     grid=lambda p: (1, 1, 1))


@pytest.mark.parametrize("blocks", [1, 2, 5])
def test_it_sums_every_block(blocks):
    x = torch.randn(1, 1, 16, 16 * blocks, device="cuda", dtype=torch.bfloat16)
    o = torch.empty(1, 1, 16, 16, device="cuda", dtype=torch.float32)
    block_sum()(x, o)
    want = x.float().reshape(16, blocks, 16).sum(1)
    torch.testing.assert_close(o[0, 0], want, rtol=1e-5, atol=1e-5)


def test_a_step_past_the_end_stops():
    # cols = 48 with a step of 32: blocks at 0 and 32, then 64 >= 48 ends it.
    x = torch.randn(1, 1, 16, 48, device="cuda", dtype=torch.bfloat16)
    o = torch.empty(1, 1, 16, 16, device="cuda", dtype=torch.float32)
    block_sum(step=32)(x, o)
    want = x[0, 0, :, 0:16].float() + x[0, 0, :, 32:48].float()
    torch.testing.assert_close(o[0, 0], want, rtol=1e-5, atol=1e-5)
