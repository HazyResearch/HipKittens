"""tk_rmsnorm_rope: c = RoPE(r[:, None] * (a @ b.T)), interleaved pairs, natural column order (bf16),
r = [M] fp32. Same store and rule as tk_rope (assert_rope_close). Here the rule's fp32 term matters: x and y
are r-scaled, so x*cos and y*sin carry fp32 rounding that is all that is left when they nearly cancel."""
import pytest
import torch

import hk
from conftest import BAD_SHAPES, DEV, SHAPES, gemm_exact, hk_shape_error, inputs, module, pow2_gamma, stream, weight
from test_rope import assert_rope_close, cos_sin_table


def r_like(M, seed=6):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return torch.rand(M, generator=g, device=DEV) * 1.5 + 0.5


def run(a, b, r, cos_sin, gamma):
    module("rmsnorm_rope")
    return hk.matmul(a, weight(b, gamma=gamma), epilogue="rmsnorm_rope", r=r, cos_sin=cos_sin)


def folded(b, gamma):
    return (b.float() * gamma.float()[None, :]).to(torch.bfloat16)   # gamma scales b's K axis


@pytest.mark.parametrize("dh", ["N", 128])                # one head spanning N, or 128-wide heads sharing a table
@pytest.mark.parametrize("kind", ["random", "large"])
@pytest.mark.parametrize("M,N,K", SHAPES)
def test_shapes(M, N, K, kind, dh):
    a, b = inputs(M, N, K, kind)
    r, cs, gamma = r_like(M), cos_sin_table(M, N if dh == "N" else dh), pow2_gamma(K)
    want = gemm_exact(a, folded(b, gamma)) * r[:, None]                                           # r first
    assert_rope_close(run(a, b, r, cs, gamma), want, cs.repeat(1, N // cs.shape[1]))


def test_zeros():
    a, b = inputs(256, 256, 128, "zeros")
    assert torch.count_nonzero(run(a, b, r_like(256), cos_sin_table(256, 256), pow2_gamma(128))) == 0


def test_known_answer():
    # b = I, r = 0.5, 90-degree rotation: out[2k] = 0.5 * a[2k+1], out[2k+1] = -0.5 * a[2k]. Exact.
    a = ((torch.arange(256, device=DEV) % 7) - 3).float().repeat(512, 1).to(torch.bfloat16)
    eye = torch.eye(256, dtype=torch.bfloat16, device=DEV)
    cs = torch.zeros(512, 256, device=DEV)
    cs[:, 1::2] = 1.0
    out = run(a, eye, torch.full((512,), 0.5, device=DEV), cs.to(torch.bfloat16), torch.ones(256, dtype=torch.bfloat16, device=DEV))
    want = torch.empty_like(a)
    want[:, 0::2], want[:, 1::2] = 0.5 * a[:, 1::2], -0.5 * a[:, 0::2]
    assert torch.equal(out, want)
    with pytest.raises(ValueError):                      # a weight without gamma folded is refused
        hk.matmul(a, weight(eye), epilogue="rmsnorm_rope", r=torch.full((512,), 0.5, device=DEV),
                  cos_sin=cs.to(torch.bfloat16))


@pytest.mark.parametrize("M,N,K", BAD_SHAPES)
def test_bad_shapes(M, N, K):
    a, b = inputs(M, N, K, "random")
    exc, msg = hk_shape_error(M, N, K)
    with pytest.raises(exc, match=msg):
        run(a, b, r_like(M), cos_sin_table(M, N), pow2_gamma(K))


def test_rejects_wrong_dtype():
    a, b = inputs(256, 256, 128, "random")
    with pytest.raises(ValueError, match="must be"):
        run(a, b, r_like(256).bfloat16(), cos_sin_table(256, 256), pow2_gamma(128))


def test_rejects_misaligned_output():
    a, b = inputs(256, 256, 128, "random")
    misaligned = torch.empty(256 * 256 + 1, dtype=torch.bfloat16, device=DEV)[1:].view(256, 256)
    with pytest.raises(RuntimeError, match="4-byte alignment"):     # hk always allocates aligned output
        module("rmsnorm_rope").dispatch(a, b, misaligned, r_like(256), cos_sin_table(256, 256), 256, stream())
