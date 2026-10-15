"""tk_residual_add: c = a @ b.T + residual (bf16, residual = [M, N] bf16). The add is one IEEE fp32 add,
so c must be bit-identical to the reference stored the way the kernel stores bf16 (to_bf16_chopped)."""
import pytest
import torch

import hk
from conftest import BAD_SHAPES, DEV, SHAPES, assert_bitexact, gemm_exact, hk_shape_error, inputs, module, weight


def residual_like(M, N, scale, seed=3):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return (torch.randn(M, N, generator=g, device=DEV) * scale).to(torch.bfloat16)


def run(a, b, residual):
    module("residual_add")
    return hk.matmul(a, weight(b), epilogue="residual_add", residual=residual)


def reference(a, b, residual):
    return gemm_exact(a, b) + residual.float()


@pytest.mark.parametrize("kind", ["random", "large"])
@pytest.mark.parametrize("M,N,K", SHAPES)
def test_shapes(M, N, K, kind):
    a, b = inputs(M, N, K, kind)
    residual = residual_like(M, N, 1.0 if kind == "random" else 1e3)
    assert_bitexact(run(a, b, residual), reference(a, b, residual))


def test_zeros():
    a, b = inputs(256, 256, 128, "zeros")
    residual = torch.zeros(256, 256, dtype=torch.bfloat16, device=DEV)
    assert torch.count_nonzero(run(a, b, residual)) == 0


def test_known_answer():
    a = torch.ones(512, 256, dtype=torch.bfloat16, device=DEV)
    eye = torch.eye(256, dtype=torch.bfloat16, device=DEV)
    residual = torch.full((512, 256), 2.5, dtype=torch.bfloat16, device=DEV)
    c = run(a, eye, residual)
    assert torch.equal(c, torch.full_like(c, 3.5))       # 1 + 2.5
    with pytest.raises(TypeError):                       # residual passed under the wrong name
        hk.matmul(a, weight(eye), epilogue="residual_add", skip=residual)


@pytest.mark.parametrize("M,N,K", BAD_SHAPES)
def test_bad_shapes(M, N, K):
    a, b = inputs(M, N, K, "random")
    exc, msg = hk_shape_error(M, N, K)
    with pytest.raises(exc, match=msg):
        run(a, b, residual_like(M, N, 1.0))


def test_rejects_wrong_dtype():
    a, b = inputs(256, 256, 128, "random")
    with pytest.raises(ValueError, match="must be"):
        run(a, b, residual_like(256, 256, 1.0).float())
