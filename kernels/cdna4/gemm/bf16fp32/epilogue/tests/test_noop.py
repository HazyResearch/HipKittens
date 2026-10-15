"""tk_noop: c = a @ b.T (bf16). The GEMM is exact, so c must equal the exact product stored the way
the kernel stores bf16 (to_bf16_chopped)."""
import pytest
import torch

import hk
from conftest import BAD_SHAPES, DEV, SHAPES, assert_bitexact, gemm_exact, hk_shape_error, inputs, module, weight


def run(a, b):
    module("noop")
    return hk.matmul(a, weight(b))


def reference(a, b):
    return gemm_exact(a, b)


@pytest.mark.parametrize("kind", ["random", "large"])
@pytest.mark.parametrize("M,N,K", SHAPES)
def test_shapes(M, N, K, kind):
    a, b = inputs(M, N, K, kind)
    assert_bitexact(run(a, b), reference(a, b))


def test_zeros():
    a, b = inputs(256, 256, 128, "zeros")
    assert torch.count_nonzero(run(a, b)) == 0


def test_known_answer():
    a, _ = inputs(512, 256, 256, "random")
    eye = torch.eye(256, dtype=torch.bfloat16, device=DEV)
    assert torch.equal(run(a, eye), a)                       # a @ I.T == a
    with pytest.raises(TypeError):                            # an unprepared weight is refused
        hk.matmul(a, eye)


@pytest.mark.parametrize("M,N,K", BAD_SHAPES)
def test_bad_shapes(M, N, K):
    a, b = inputs(M, N, K, "random")
    exc, msg = hk_shape_error(M, N, K)
    with pytest.raises(exc, match=msg):
        run(a, b)


def test_rejects_wrong_dtype():
    a, b = inputs(256, 256, 128, "random")
    with pytest.raises(ValueError, match="must be"):
        run(a.float(), b)


@pytest.mark.parametrize("M,N,K", SHAPES)
def test_matches_base_gemm(M, N, K):
    a, b = inputs(M, N, K, "random", seed=1)
    theirs = torch.empty(M, N, dtype=torch.bfloat16, device=DEV)
    module("base").dispatch_micro(a, b, theirs)
    torch.cuda.synchronize()
    assert torch.equal(run(a, b), theirs)
