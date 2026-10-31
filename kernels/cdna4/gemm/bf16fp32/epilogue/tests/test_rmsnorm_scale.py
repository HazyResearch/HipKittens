"""tk_rmsnorm_scale: c = (a @ b.T) * r[:, None] * out_gamma[None, :] (bf16), r fp32 [M], out_gamma bf16 [N];
bit-exact. "rmsnorm" is the same kernel with out_gamma = ones."""
import pytest
import torch

import hk
from conftest import (BAD_SHAPES, DEV, SHAPES, assert_bitexact, gemm_exact, hk_shape_error, inputs, module,
                      pow2_gamma, weight)


def vectors(M, N, seed=2):
    """fp32 r in [0.5, 2), bf16 gamma in [0.25, 1.75)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    r = torch.rand(M, generator=g, device=DEV) * 1.5 + 0.5
    gamma = (torch.rand(N, generator=g, device=DEV) * 1.5 + 0.25).to(torch.bfloat16)
    return r, gamma


def run(a, b, r, gamma):
    module("rmsnorm_scale")
    return hk.matmul(a, weight(b), epilogue="rmsnorm_scale", r=r, out_gamma=gamma)


def reference(a, b, r, gamma):
    return gemm_exact(a, b) * r[:, None] * gamma.float()[None, :]   # r first, then gamma


@pytest.mark.parametrize("kind", ["random", "large"])
@pytest.mark.parametrize("M,N,K", SHAPES)
def test_shapes(M, N, K, kind):
    a, b = inputs(M, N, K, kind)
    r, gamma = vectors(M, N)
    assert_bitexact(run(a, b, r, gamma), reference(a, b, r, gamma))


def test_zeros():
    a, b = inputs(256, 256, 128, "zeros")
    assert torch.count_nonzero(run(a, b, *vectors(256, 256))) == 0


def test_known_answer():
    # c[m, n] = r[m] * gamma[n]
    a = torch.ones(512, 256, dtype=torch.bfloat16, device=DEV)
    eye = torch.eye(256, dtype=torch.bfloat16, device=DEV)
    r = torch.tensor([1.0, 2.0], device=DEV).repeat(256)                           # per row
    gamma = torch.tensor([4.0, 8.0], device=DEV).repeat(128).to(torch.bfloat16)    # per column
    c = run(a, eye, r, gamma)
    assert torch.equal(c, (r[:, None] * gamma.float()[None, :]).to(torch.bfloat16))
    with pytest.raises(ValueError):                      # swapped
        hk.matmul(a, weight(eye), epilogue="rmsnorm_scale", r=gamma, out_gamma=r)
    with pytest.raises(TypeError, match="takes arguments"):            # `gamma` is not an argument
        hk.matmul(a, weight(eye), epilogue="rmsnorm_scale", r=r, gamma=gamma)


@pytest.mark.parametrize("M,N,K", BAD_SHAPES)
def test_bad_shapes(M, N, K):
    a, b = inputs(M, N, K, "random")
    exc, msg = hk_shape_error(M, N, K)
    with pytest.raises(exc, match=msg):
        run(a, b, *vectors(M, N))


def test_rejects_wrong_dtype():
    a, b = inputs(256, 256, 128, "random")
    r, gamma = vectors(256, 256)
    with pytest.raises(ValueError, match="must be"):
        run(a, b, r.bfloat16(), gamma)


@pytest.mark.parametrize("M,N,K", SHAPES)
def test_rmsnorm_input_gamma(M, N, K):
    """"rmsnorm" with gamma on the input side, bit-exact."""
    a, b = inputs(M, N, K, "random")
    r, _ = vectors(M, N)
    gamma = pow2_gamma(K)
    module("rmsnorm_scale")
    c = hk.matmul(a, weight(b, gamma=gamma), epilogue="rmsnorm", r=r)
    folded = (b.float() * gamma.float()[None, :]).to(torch.bfloat16)   # gamma over K
    assert_bitexact(c, gemm_exact(a, folded) * r[:, None])            # times ones


def test_rmsnorm_takes_only_r():
    a, b = inputs(256, 256, 128, "random")
    r, gamma = vectors(256, 256)
    with pytest.raises(TypeError, match="takes arguments"):            # gamma comes with the input
        hk.matmul(a, weight(b), epilogue="rmsnorm", r=r, gamma=gamma)
