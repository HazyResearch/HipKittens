"""Numerics for the decode GEMM: does hk.ops.decode_gemm compute x @ w.T.

The places an index can be wrong, and the shapes that reach them:

  * the interleaved K split -- warp w takes chunks w, w+W, w+2W, ... -- is
    wrong in a way every warp count hides except the ones that differ, so the
    schedule is checked at more than one `warps`;
  * the LDS combine adds W partials, so a missing or doubled one is off by a
    whole warp's share of K, not by rounding;
  * the transpose on the way out: C^T is accumulated, and a store that forgot
    the transpose writes a square tile's worth of plausible numbers into the
    wrong places -- a non-square N x M is what tells them apart;
  * padding: M below a whole token tile reads rows that are not the caller's.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hk.ops import decode_gemm  # noqa: E402

RTOL, ATOL = 2e-2, 1e-2


def _run(m, n, k, seed=0, **sched):
    torch.manual_seed(seed)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * k ** -0.5
    got = decode_gemm.matmul(x, w, **sched)
    assert got.shape == (m, n) and got.dtype == torch.bfloat16
    torch.testing.assert_close(got.float(), x.float() @ w.float().T,
                               rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("m", [1, 5, 16, 17, 32, 33, 64])
@pytest.mark.parametrize("n,k", [(6144, 4096), (4096, 12288), (16, 256)])
def test_every_batch_width(m, n, k):
    _run(m, n, k)


@pytest.mark.parametrize("slices,warps", [(4, 1), (4, 4), (4, 8), (2, 8),
                                          (8, 4), (2, 16)])
def test_every_split(slices, warps):
    _run(7, 4096, 4096, slices=slices, warps=warps)


@pytest.mark.parametrize("resident,depth", [(0, 0), (2, 0), (3, 8), (4, 4)])
def test_the_combine_is_the_same_however_much_lds_it_holds(resident, depth):
    # A residency cap only enlarges the LDS the combine owns; the partials
    # land in the first `warps` slots whatever the size. Batched reads and the
    # slots past the warps are both places an off-by-one would hide.
    _run(9, 4096, 12288, warps=16, resident=resident, combine_depth=depth)


def test_a_strided_activation_is_copied_not_misread():
    torch.manual_seed(0)
    x = torch.randn(8, 2 * 4096, device="cuda", dtype=torch.bfloat16)[:, ::2]
    w = torch.randn(512, 4096, device="cuda", dtype=torch.bfloat16) * 4096 ** -0.5
    torch.testing.assert_close(decode_gemm.matmul(x, w).float(),
                               x.float() @ w.float().T, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("m,n,k", [(0, 4096, 4096), (65, 4096, 4096),
                                   (4, 4100, 4096), (4, 4096, 4000)])
def test_it_refuses_what_it_cannot_tile(m, n, k):
    x = torch.zeros(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.zeros(n, k, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        decode_gemm.matmul(x, w)
