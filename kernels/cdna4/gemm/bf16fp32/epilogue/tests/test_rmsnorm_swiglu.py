"""tk_rmsnorm_swiglu: c = silu(gate) * value with [gate | value] = r[:, None] * (a @ b.T), [M, N] -> [M, N/2]
(bf16, r = [M] fp32). The caller permutes b's rows as for SwiGLU (and folds the norm's gamma into b,
which this test does not need). Fast SiLU allows one bf16 step."""
import pytest
import torch

import hk
from conftest import (BAD_SHAPES, DEV, SHAPES, assert_within_bf16_step, gemm_exact, hk_shape_error, inputs, module,
                      pow2_gamma, silu_kernel, weight)

SILU_HALF = 0.31122966560092725   # silu(0.5)


def r_like(M, seed=4):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return torch.rand(M, generator=g, device=DEV) * 1.5 + 0.5


def run(a, b, r, gamma):
    module("rmsnorm_swiglu")
    return hk.matmul(a, weight(b, layout="swiglu", gamma=gamma), epilogue="rmsnorm_swiglu", r=r)


def reference(a, b, r, gamma):
    folded = (b.float() * gamma.float()[None, :]).to(torch.bfloat16)   # gamma scales b's K axis
    h = gemm_exact(a, folded) * r[:, None]              # r first, as the kernel does
    d = h.shape[1] // 2
    return silu_kernel(h[:, :d]) * h[:, d:]


@pytest.mark.parametrize("kind", ["random", "large"])
@pytest.mark.parametrize("M,N,K", SHAPES)
def test_shapes(M, N, K, kind):
    a, b = inputs(M, N, K, kind)
    r, gamma = r_like(M), pow2_gamma(K)
    assert_within_bf16_step(run(a, b, r, gamma), reference(a, b, r, gamma))


def test_zeros():
    a, b = inputs(256, 256, 128, "zeros")
    assert torch.count_nonzero(run(a, b, r_like(256), pow2_gamma(128))) == 0


def test_known_answer():
    # b = I and r = 0.5: gate = 0.5 * 1, value = 0.5 * 2, so c = silu(0.5) * 1.
    a = torch.cat([torch.ones(512, 128), 2 * torch.ones(512, 128)], dim=1).to(torch.bfloat16).to(DEV)
    eye = torch.eye(256, dtype=torch.bfloat16, device=DEV)
    r = torch.full((512,), 0.5, device=DEV)
    ones = torch.ones(256, dtype=torch.bfloat16, device=DEV)
    assert_within_bf16_step(run(a, eye, r, ones), torch.full((512, 128), SILU_HALF, device=DEV))
    with pytest.raises(ValueError):                      # a weight without gamma folded is refused
        hk.matmul(a, weight(eye, layout="swiglu"), epilogue="rmsnorm_swiglu", r=r)


@pytest.mark.parametrize("M,N,K", BAD_SHAPES)
def test_bad_shapes(M, N, K):
    a, b = inputs(M, N, K, "random")
    exc, msg = hk_shape_error(M, N, K)
    with pytest.raises(exc, match=msg):
        run(a, b, r_like(M), pow2_gamma(K))


def test_rejects_wrong_dtype():
    a, b = inputs(256, 256, 128, "random")
    with pytest.raises(ValueError, match="must be"):
        run(a, b, r_like(256).bfloat16(), pow2_gamma(128))
