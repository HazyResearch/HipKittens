"""hk.residual_rms (tk_residual_rms + tk_rms_reduce): h = a @ b.T + residual (bf16 [M, N]) and
r = 1 / sqrt(mean(h^2) + 1e-5) (fp32 [M]), with r taken from h in fp32 before it is rounded to bf16.
h is one IEEE fp32 add stored chopped, so it must be bit-identical; r may differ from fp64 by the fp32
rounding of 64 squares per group, G = N/64 group sums, the division and rsqrt: (64 + G + 4) * 2^-24."""
import pytest
import torch

import hk
from conftest import (BAD_SHAPES, DEV, SHAPES, assert_bitexact, assert_norm_close, assert_rel_close, gemm_exact,
                      general_gamma, hk_shape_error, inputs, module, silu_kernel, to_bf16_chopped, weight)
from test_residual_add import residual_like

EPS = 1e-5


def run(a, b, residual):
    module("residual_rms"), module("rms_reduce")
    out = hk.residual_rms(a, weight(b), residual)
    assert out.hg is None
    return out.h, out.r


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


def test_chain_gamma_into_rmsnorm_swiglu():
    """The layer's MLP input: residual_rms(gamma=g) writes hg = h * g, and rmsnorm_swiglu multiplies hg by the PLAIN
    gate_up weight and applies r. Must match swiglu(rmsnorm(h, g) @ W.t()) computed from h, g and an fp64 r."""
    module("residual_rms_gamma"), module("rms_reduce"), module("rmsnorm_swiglu")
    a, b = inputs(256, 256, 128, "random")
    res = residual_like(256, 256, 1.0)
    g = general_gamma(256, seed=13)
    out = hk.residual_rms(a, weight(b), res, gamma=g)
    _, w = inputs(256, 1024, 256, "random", seed=1)                # W_gate_up [2*d_ff, d_model]
    y = hk.matmul(out.hg, w, epilogue="rmsnorm_swiglu", r=out.r)
    h_ref, r_ref = reference(a, b, res)
    gu = (h_ref.double() * r_ref[:, None] * g.double()[None, :]) @ w.double().T
    assert_norm_close(y, silu_kernel(gu[:, :512].float()).double() * gu[:, 512:], 1e-2)


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


@pytest.mark.parametrize("M,N,K", SHAPES)
def test_gamma_output(M, N, K):
    """With gamma, the same GEMM also writes hg = h * gamma from h in fp32, before h's bf16 rounding; h and r are
    bitwise the gamma-less call's, so r stays the norm of h, not of hg. A non-power-of-two gamma makes rounding h
    first give different bits, so the bit-exact check pins the order."""
    module("residual_rms_gamma")
    a, b = inputs(M, N, K, "random")
    res = residual_like(M, N, 1.0)
    gamma = general_gamma(N, seed=11)
    h0, r0 = run(a, b, res)
    h, hg, r = hk.residual_rms(a, weight(b), res, gamma=gamma)
    assert torch.equal(h, h0) and torch.equal(r, r0)
    h_ref, _ = reference(a, b, res)
    assert_bitexact(hg, h_ref * gamma.float()[None, :])
    rounded_first = to_bf16_chopped(to_bf16_chopped(h_ref).float() * gamma.float()[None, :])
    assert (rounded_first != hg).any()                            # the data can tell the two orders apart


def test_rejects_wrong_gamma():
    module("residual_rms_gamma")
    a, b = inputs(256, 256, 128, "random")
    with pytest.raises(ValueError, match="gamma must be"):
        hk.residual_rms(a, weight(b), residual_like(256, 256, 1.0), gamma=general_gamma(128))
