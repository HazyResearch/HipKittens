"""tk_partialrms: partials[g, m] = sum of (a @ b.T)[m, cols of group g]^2 (fp32, [N/64, M]), no c.
Group g = col*4 + wc owns 32-column chunks wc and wc+4 of 256-column block col. The only allowed
difference is fp32 rounding of 64 squares and their sum: relative error <= 64 * 2^-24."""
import pytest
import torch

from conftest import BAD_SHAPES, DEV, SHAPES, assert_rel_close, inputs, module, stream

REL = 64 * 2.0 ** -24


def run(a, b):
    p = torch.zeros(b.shape[0] // 64, a.shape[0], dtype=torch.float32, device=DEV)
    module("partialrms").dispatch(a, b, p, stream())
    torch.cuda.synchronize()
    return p


def reference(a, b):
    h = a.double() @ b.double().T                                  # exact
    M, N = h.shape
    sq = h.square().view(M, N // 256, 8, 32)
    return (sq[:, :, :4].sum(-1) + sq[:, :, 4:].sum(-1)).reshape(M, N // 64).T


@pytest.mark.parametrize("kind", ["random", "large"])
@pytest.mark.parametrize("M,N,K", SHAPES)
def test_shapes(M, N, K, kind):
    a, b = inputs(M, N, K, kind)
    assert_rel_close(run(a, b).double(), reference(a, b), REL)


def test_zeros():
    a, b = inputs(256, 256, 128, "zeros")
    assert torch.count_nonzero(run(a, b)) == 0


def test_known_answer():
    a = torch.ones(512, 256, dtype=torch.bfloat16, device=DEV)
    eye = torch.eye(256, dtype=torch.bfloat16, device=DEV)
    assert torch.equal(run(a, eye), torch.full((4, 512), 64.0, device=DEV))   # 64 ones squared per group
    with pytest.raises(RuntimeError):                                          # [N/64, M] does not fit swapped a/b
        module("partialrms").dispatch(eye, a, torch.zeros(4, 512, dtype=torch.float32, device=DEV), stream())


def test_padding_does_not_leak():
    a, b = inputs(512, 1024, 256, "random")
    padded = torch.cat([b, torch.zeros(256, 256, dtype=torch.bfloat16, device=DEV)])
    p, p_pad = run(a, b), run(a, padded)
    assert torch.equal(p_pad[: p.shape[0]], p)                  # real groups unchanged
    assert torch.count_nonzero(p_pad[p.shape[0]:]) == 0         # padded groups contribute nothing


@pytest.mark.parametrize("M,N,K", BAD_SHAPES)
def test_bad_shapes(M, N, K):
    a, b = inputs(M, N, K, "random")
    with pytest.raises(RuntimeError):
        run(a, b)


def test_rejects_wrong_dtype():
    a, b = inputs(256, 256, 128, "random")
    with pytest.raises(RuntimeError, match="must be torch"):
        # bf16 [4, 256] view of an [8, 256] buffer: the fp32 writes stay in bounds until the check lands
        module("partialrms").dispatch(a, b, torch.zeros(8, 256, dtype=torch.bfloat16, device=DEV)[:4], stream())
