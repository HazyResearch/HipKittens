"""Numerics for the DSL GEMM: does the generated kernel compute A @ B.T.

Needs a Radeon and torch. The resource half of the Phase 3 gate -- ScratchSize
0, no spill, the same occupancy as the handwritten kernel -- is checked at
build time and so runs on the no-GPU tier. This file is the other half, and it
is the one that can catch what the gates cannot: a kernel that reads the wrong
subtile spills nothing, occupies the same registers, and is wrong.

Why the shapes below are what they are. The schedule has three places an index
can be wrong in a way no single shape exposes:

  * the L2 swizzle (WGM) only does anything once there is more than one
    block-row, so M has to exceed 128 and the shapes have to be non-square;
  * the predicated M remainder only runs when M is not a multiple of 128,
    and when it does the last block's loads past M must read zero and its
    stores past M must not happen;
  * the K loop's prologue and epilogue are separate code from its body, so a
    shape with a single K-tile exercises neither rotation nor staging.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import hk  # noqa: E402

#: bf16 inputs with fp32 accumulate over K terms. The error floor is the input
#: rounding, not the accumulator: each product is exact to fp32 and the sum is
#: fp32, so what is left is the 8-bit mantissa on A and B. Comparing against a
#: torch matmul in the same dtype rather than against fp64 keeps this about the
#: kernel instead of about bf16.
RTOL, ATOL = 2e-2, 1e-2


def _ref(a, b):
    """A @ B.T in fp32, which is what the kernel accumulates in."""
    return (a.float() @ b.float().T)


def _run(m, n, k, seed=0):
    torch.manual_seed(seed)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    got = hk.ops.matmul(a, b)
    assert got.shape == (m, n) and got.dtype == torch.bfloat16
    torch.testing.assert_close(got.float(), _ref(a, b), rtol=RTOL, atol=ATOL)
    return got


# -- the gate ----------------------------------------------------------------


def test_square_tiles_exactly():
    _run(512, 512, 512)


@pytest.mark.parametrize("m,n,k", [
    (128, 128, 64),       # one block, one K-tile: prologue straight to epilogue
    (128, 128, 128),      # two K-tiles: the loop body runs exactly once
    (256, 128, 256),
    (128, 256, 256),
    (1024, 512, 256),     # several block-rows, so the WGM swizzle is live
    (512, 1024, 192),     # K = 3 tiles, odd count
])
def test_shapes_that_tile(m, n, k):
    _run(m, n, k)


@pytest.mark.parametrize("m", [129, 200, 255, 384 + 17])
def test_ragged_m_is_predicated(m):
    """M need not be a multiple of 128.

    The last block runs past M: its A rows past M are staged as zeros by the
    buffer range and its bounded store writes only rows below M. The epilogue's
    row coordinate has to be in elements for that bound to land on the right
    row, which is the kind of thing that is right in the prologue and wrong
    here.
    """
    _run(m, 256, 128)


@pytest.mark.parametrize("m", [1, 5, 16, 17, 64, 127])
def test_m_below_one_block(m):
    """Below 128 rows there is one partial block -- or, at 16 and under, the
    thin tiling's one block -- and every row past M is predicated off."""
    _run(m, 256, 192)


def test_nothing_is_written_past_m():
    """The rows after the output's last are someone else's memory. Carve the
    output out of a larger buffer and check the rows after it are untouched."""
    torch.manual_seed(7)
    a = torch.randn(200, 128, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
    room = torch.full((256, 256), 7.0, device="cuda", dtype=torch.bfloat16)
    hk.ops.matmul(a, b, out=room[:200])
    torch.testing.assert_close(room[:200].float(), _ref(a, b), rtol=RTOL, atol=ATOL)
    assert torch.equal(room[200:], torch.full_like(room[200:], 7.0))


def test_k_is_not_summed_twice():
    """A ones-by-ones matmul is K everywhere. Catches a K loop that runs the
    wrong number of tiles, which a random-input comparison hides inside the
    tolerance when K is large."""
    for k in (64, 128, 512):
        a = torch.ones(128, k, device="cuda", dtype=torch.bfloat16)
        b = torch.ones(128, k, device="cuda", dtype=torch.bfloat16)
        got = hk.ops.matmul(a, b).float()
        assert torch.equal(got, torch.full_like(got, float(k))), k


def test_it_is_a_times_b_transposed():
    """An asymmetric pair, so transposing or swapping the operands fails.

    A @ B.T with distinct non-symmetric factors has no accidental symmetry to
    hide behind; the square random test above would pass for A.T @ B on a
    symmetric input.
    """
    torch.manual_seed(3)
    a = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(256, 64, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(hk.ops.matmul(a, b).float(), _ref(a, b),
                               rtol=RTOL, atol=ATOL)


# -- what it refuses ---------------------------------------------------------


@pytest.mark.parametrize("m,n,k,match", [
    (128, 128, 96, "multiple of 64"),     # K does not tile
    (128, 192, 128, "multiple of 128"),   # N does not tile
])
def test_untileable_shapes_are_refused(m, n, k, match):
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=match):
        hk.ops.matmul(a, b)


def test_b_is_pre_transposed_and_says_so():
    a = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)   # (K, N)
    with pytest.raises(ValueError, match="pre-transposed"):
        hk.ops.matmul(a, b)
