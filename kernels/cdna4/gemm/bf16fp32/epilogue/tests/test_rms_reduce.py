"""tk_rms_reduce: r[m] = 1 / sqrt(sum(partials[:, m]) / N + 1e-5), partials = [N/64, M] fp32, r = [M] fp32.
Allowed difference vs an fp64 reference: fp32 rounding of the G-term sum, the division and rsqrt,
i.e. relative error <= (G + 4) * 2^-24."""
import math

import pytest
import torch

import hk
from conftest import DEV, SHAPES, assert_rel_close, inputs, module, stream, weight

EPS = 1e-5
GROUP_CASES = [(256, 4), (4096, 64), (8192, 32), (4096, 172)]   # (M, G); G = N / 64, 172 <-> N = 11008


def rel(G):
    return (G + 4) * 2.0 ** -24


def run(partials):
    r = torch.empty(partials.shape[1], dtype=torch.float32, device=DEV)
    module("rms_reduce").dispatch(partials, r, stream())
    torch.cuda.synchronize()
    return r


def reference(partials):
    G = partials.shape[0]
    return torch.rsqrt(partials.double().sum(0) / (G * 64) + EPS)


@pytest.mark.parametrize("M,G", GROUP_CASES)
def test_shapes(M, G):
    g = torch.Generator(device=DEV).manual_seed(5)
    partials = torch.rand(G, M, generator=g, device=DEV) * 100
    assert_rel_close(run(partials).double(), reference(partials), rel(G))


def test_zeros():
    r = run(torch.zeros(64, 256, device=DEV))
    assert_rel_close(r.double(), torch.full((256,), 1 / math.sqrt(EPS), device=DEV, dtype=torch.float64), rel(64))


def test_known_answer():
    # 64 groups of 2.0: sum 128, N = 4096, mean 1/32, r = 1 / sqrt(1/32 + 1e-5).
    r = run(torch.full((64, 512), 2.0, device=DEV))
    assert_rel_close(r.double(), torch.full((512,), 1 / math.sqrt(1 / 32 + EPS), device=DEV, dtype=torch.float64), rel(64))
    with pytest.raises(RuntimeError):                    # partials and r swapped: shapes do not fit
        module("rms_reduce").dispatch(torch.empty(512, device=DEV), torch.full((64, 512), 2.0, device=DEV), stream())


@pytest.mark.parametrize("M,N,K", SHAPES)
def test_chain_with_partialrms(M, N, K):
    """hk.inv_rms (partialrms -> rms_reduce) gives 1 / rms of each row of the exact GEMM."""
    a, b = inputs(M, N, K, "random")
    module("partialrms"), module("rms_reduce")
    h = a.double() @ b.double().T
    want = torch.rsqrt(h.square().mean(1) + EPS)
    assert_rel_close(hk.inv_rms(a, weight(b)).double(), want, (64 + N // 64 + 4) * 2.0 ** -24)


def test_bad_shapes():
    with pytest.raises(RuntimeError):                    # partials' M does not match r
        module("rms_reduce").dispatch(torch.zeros(4, 257, device=DEV), torch.empty(256, device=DEV), stream())
    with pytest.raises(RuntimeError):                    # r is not a vector
        module("rms_reduce").dispatch(torch.zeros(4, 256, device=DEV), torch.empty(2, 256, device=DEV), stream())


def test_rejects_wrong_dtype():
    with pytest.raises(RuntimeError, match="must be torch"):
        module("rms_reduce").dispatch(torch.zeros(4, 256, device=DEV), torch.empty(256, dtype=torch.bfloat16, device=DEV), stream())
