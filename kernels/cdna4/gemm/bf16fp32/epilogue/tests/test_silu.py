"""tk_silu: c = silu(a @ b.T) (bf16). The reference is silu_kernel (0 where the hardware reciprocal
flushes) stored as the kernel stores bf16. The fast SiLU differs from torch's by <= 2e-6 relative, so c
may differ from it by at most one bf16 step."""
import pytest
import torch

import hk
from conftest import (BAD_SHAPES, DEV, SHAPES, assert_within_bf16_step, gemm_exact, hk_shape_error, inputs, module,
                      silu_kernel, weight)

SILU_1 = 0.7310585786300049          # silu(1) = 1 / (1 + e^-1)


def run(a, b):
    module("silu")
    return hk.matmul(a, weight(b), epilogue="silu")


def reference(a, b):
    return silu_kernel(gemm_exact(a, b))


@pytest.mark.parametrize("kind", ["random", "large"])
@pytest.mark.parametrize("M,N,K", SHAPES)
def test_shapes(M, N, K, kind):
    a, b = inputs(M, N, K, kind)
    assert_within_bf16_step(run(a, b), reference(a, b))


def test_zeros():
    a, b = inputs(256, 256, 128, "zeros")
    assert torch.count_nonzero(run(a, b)) == 0


def test_known_answer():
    a = torch.ones(512, 256, dtype=torch.bfloat16, device=DEV)
    eye = torch.eye(256, dtype=torch.bfloat16, device=DEV)
    assert_within_bf16_step(run(a, eye), torch.full((512, 256), SILU_1, device=DEV))
    with pytest.raises(TypeError):                            # an unprepared weight is refused
        hk.matmul(a, eye, epilogue="silu")


@pytest.mark.parametrize("M,N,K", BAD_SHAPES)
def test_bad_shapes(M, N, K):
    a, b = inputs(M, N, K, "random")
    exc, msg = hk_shape_error(M, N, K)
    with pytest.raises(exc, match=msg):
        run(a, b)


def test_rejects_wrong_dtype():
    a, b = inputs(256, 256, 128, "random")
    with pytest.raises(ValueError, match="must be"):
        run(a, b.float())
