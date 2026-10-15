"""hk.residual_rms (tk_residual_rms + tk_rms_reduce): h = a @ b.T + residual (bf16 [M, N]) and
r = 1 / sqrt(mean(h^2) + 1e-5) (fp32 [M]), with r taken from h in fp32 before it is rounded to bf16.
h is one IEEE fp32 add stored chopped, so it must be bit-identical; r may differ from fp64 by the fp32
rounding of 64 squares per group, G = N/64 group sums, the division and rsqrt: (64 + G + 4) * 2^-24."""
import pytest
import torch

import hk
from conftest import (BAD_SHAPES, DEV, SHAPES, assert_bitexact, assert_rel_close, assert_within_bf16_step,
                      gemm_exact, hk_shape_error, inputs, module, pow2_gamma, weight)
from test_residual_add import residual_like
from test_rmsnorm_swiglu import reference as rmsnorm_swiglu_reference

EPS = 1e-5


def run(a, b, residual):
    module("residual_rms"), module("rms_reduce")
    return hk.residual_rms(a, weight(b), residual)


def reference(a, b, residual):
    h = gemm_exact(a, b) + residual.float()                       # fp32, as the kernel holds it
    return h, torch.rsqrt(h.double().square().mean(1) + EPS)


def rel(N):
    return (64 + N // 64 + 4) * 2.0 ** -24


@pytest.mark.parametrize("kind", ["random", "large"])
@pytest.mark.parametrize("M,N,K", SHAPES)
def test_shapes(M, N, K, kind):
    a, b = inputs(M, N, K, kind)
    residual = residual_like(M, N, 1.0 if kind == "random" else 1e3)
    h, r = run(a, b, residual)
    h_ref, r_ref = reference(a, b, residual)
    assert_bitexact(h, h_ref)
    assert_rel_close(r.double(), r_ref, rel(N))


def test_zeros():
    a, b = inputs(256, 256, 128, "zeros")
    h, r = run(a, b, torch.zeros(256, 256, dtype=torch.bfloat16, device=DEV))
    assert torch.count_nonzero(h) == 0
    assert_rel_close(r.double(), torch.full((256,), EPS ** -0.5, dtype=torch.float64, device=DEV), rel(256))


def test_known_answer():
    """a @ I = ones; residual row m holds m % 4, so h row m is 1 + m % 4 and r[m] = 1 / sqrt((1 + m % 4)^2 + eps).
    A residual read with rows and columns swapped would make h vary along columns instead."""
    a = torch.ones(512, 256, dtype=torch.bfloat16, device=DEV)
    eye = torch.eye(256, dtype=torch.bfloat16, device=DEV)
    m4 = (torch.arange(512, device=DEV) % 4).float()
    residual = m4[:, None].expand(512, 256).to(torch.bfloat16).contiguous()
    h, r = run(a, eye, residual)
    assert torch.equal(h.float(), (1 + m4)[:, None].expand(512, 256))
    assert_rel_close(r.double(), torch.rsqrt((1 + m4.double()) ** 2 + EPS), rel(256))


def test_chain_rmsnorm_swiglu():
    """residual_rms -> rmsnorm_swiglu: out = swiglu(rmsnorm(a @ b.T + residual, gamma) @ W_gate_up).
    Small integer inputs keep h exact in bf16 and the second GEMM exact in fp32, so the only allowed
    difference is fast SiLU's one bf16 step."""
    a, b = inputs(256, 256, 128, "random")                         # a scaled by 2^-4
    g = torch.Generator(device=DEV).manual_seed(9)
    residual = (torch.randint(-2, 3, (256, 256), generator=g, device=DEV).float() * 2.0 ** -4).to(torch.bfloat16)
    h, r = run(a, b, residual)
    assert torch.equal(h.float(), gemm_exact(a, b) + residual.float())   # h exact: no rounding in the chain input
    _, b2 = inputs(256, 1024, 256, "random", seed=1)               # W_gate_up as [2*d_ff, d_model]
    gamma = pow2_gamma(256)
    module("rmsnorm_swiglu")
    out = hk.matmul(h, weight(b2, layout="swiglu", gamma=gamma), epilogue="rmsnorm_swiglu", r=r)
    assert_within_bf16_step(out, rmsnorm_swiglu_reference(h, b2, r, gamma))


@pytest.mark.parametrize("M,N,K", BAD_SHAPES)
def test_bad_shapes(M, N, K):
    a, b = inputs(M, N, K, "random")
    exc, msg = hk_shape_error(M, N, K)
    with pytest.raises(exc, match=msg):
        run(a, b, residual_like(M, N, 1.0))


def test_rejects_wrong_dtype():
    a, b = inputs(256, 256, 128, "random")
    with pytest.raises(ValueError, match="residual must be"):
        run(a, b, residual_like(256, 256, 1.0).float())


def test_rejects_wrong_residual_shape():
    a, b = inputs(256, 256, 128, "random")
    with pytest.raises(ValueError, match="residual must be"):
        run(a, b, residual_like(256, 512, 1.0))


def test_rejects_gamma_folded_weight():
    a, b = inputs(256, 256, 128, "random")
    with pytest.raises(ValueError, match="no gamma folded"):
        hk.residual_rms(a, weight(b, gamma=pow2_gamma(128)), residual_like(256, 256, 1.0))
