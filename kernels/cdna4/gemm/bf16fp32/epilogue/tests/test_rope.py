"""tk_rope: c = RoPE(a @ b.T) with interleaved pairs (2k, 2k+1), natural column order (bf16). b is natural;
cos_sin is one head's natural [M, Dh] table, shared by all heads. The output store rounds to nearest (packed
v_cvt_pk_bf16_f32). Allowed difference (assert_rope_close): one bf16 step of the result plus the fp32
rounding of the rotation's two products and sum, 4 * 2^-24 * (|x||cos| + |y||sin|), which only matters
when the two products nearly cancel."""
import pytest
import torch

import hk
from conftest import BAD_SHAPES, DEV, SHAPES, bf16_step, gemm_exact, hk_shape_error, inputs, module, stream, weight


def cos_sin_table(M, N, base=10000.0):
    """bf16 [M, N] natural interleaved table: [m, 2k] = cos(m * theta_k), [m, 2k+1] = sin(m * theta_k)."""
    k = torch.arange(N // 2, device=DEV)
    angle = torch.arange(M, device=DEV).float()[:, None] * base ** (-2.0 * k / N)[None, :]
    table = torch.empty(M, N, device=DEV)
    table[:, 0::2], table[:, 1::2] = torch.cos(angle), torch.sin(angle)
    return table.to(torch.bfloat16)


def rotate(h, cos_sin):
    """Interleaved RoPE in fp32 on natural columns, in the kernel's operation order."""
    cs = cos_sin.float()
    x, y, cos, sin = h[:, 0::2], h[:, 1::2], cs[:, 0::2], cs[:, 1::2]
    out = torch.empty_like(h)
    out[:, 0::2] = x * cos + y * sin
    out[:, 1::2] = y * cos - x * sin
    return out


def assert_rope_close(got, h, cos_sin):
    """got (bf16) vs rotate(h, cos_sin) rounded to nearest: <= 1 bf16 step + 4 * 2^-24 * magnitude, where
    magnitude is |x||cos| + |y||sin| for even outputs and |y||cos| + |x||sin| for odd outputs."""
    cs = cos_sin.float()
    x, y, cos, sin = h[:, 0::2].abs(), h[:, 1::2].abs(), cs[:, 0::2].abs(), cs[:, 1::2].abs()
    magnitude = torch.empty_like(h)
    magnitude[:, 0::2], magnitude[:, 1::2] = x * cos + y * sin, y * cos + x * sin
    want = rotate(h, cos_sin).to(torch.bfloat16).float()
    err = (got.float() - want).abs()
    bad = err > bf16_step(want) + 4 * 2.0 ** -24 * magnitude
    if bad.any():
        i = tuple(bad.nonzero()[0].tolist())
        raise AssertionError(f"{int(bad.sum())} of {bad.numel()} elements out of bound; first at {i}: got "
                             f"{got[i].item()!r}, want {want[i].item()!r}, magnitude {magnitude[i].item()!r}")


def run(a, b, cos_sin):
    module("rope")
    return hk.matmul(a, weight(b), epilogue="rope", cos_sin=cos_sin)


@pytest.mark.parametrize("dh", ["N", 128])                # one head spanning N, or 128-wide heads sharing a table
@pytest.mark.parametrize("kind", ["random", "large"])
@pytest.mark.parametrize("M,N,K", SHAPES)
def test_shapes(M, N, K, kind, dh):
    a, b = inputs(M, N, K, kind)
    cs = cos_sin_table(M, N if dh == "N" else dh)
    assert_rope_close(run(a, b, cs), gemm_exact(a, b), cs.repeat(1, N // cs.shape[1]))


def test_zeros():
    a, b = inputs(256, 256, 128, "zeros")
    assert torch.count_nonzero(run(a, b, cos_sin_table(256, 256))) == 0


def test_known_answer():
    # b = I and a 90-degree rotation (cos 0, sin 1): out[2k] = a[2k+1], out[2k+1] = -a[2k]. Exact.
    a = ((torch.arange(256, device=DEV) % 7) - 3).float().repeat(512, 1).to(torch.bfloat16)
    eye = torch.eye(256, dtype=torch.bfloat16, device=DEV)
    cs = torch.zeros(512, 256, device=DEV)
    cs[:, 1::2] = 1.0
    out = run(a, eye, cs.to(torch.bfloat16))
    want = torch.empty_like(a)
    want[:, 0::2], want[:, 1::2] = a[:, 1::2], -a[:, 0::2]
    assert torch.equal(out, want)
    with pytest.raises(ValueError, match="N % Dh"):         # a table width that does not divide N is refused
        run(a, eye, cs[:, :96].contiguous().to(torch.bfloat16))


@pytest.mark.parametrize("M,N,K", BAD_SHAPES)
def test_bad_shapes(M, N, K):
    a, b = inputs(M, N, K, "random")
    exc, msg = hk_shape_error(M, N, K)
    with pytest.raises(exc, match=msg):
        run(a, b, cos_sin_table(M, N))


def test_rejects_wrong_dtype():
    a, b = inputs(256, 256, 128, "random")
    with pytest.raises(ValueError, match="must be"):
        run(a, b, cos_sin_table(256, 256).float())


def test_rejects_misaligned_output():
    a, b = inputs(256, 256, 128, "random")
    misaligned = torch.empty(256 * 256 + 1, dtype=torch.bfloat16, device=DEV)[1:].view(256, 256)
    with pytest.raises(RuntimeError, match="4-byte alignment"):     # hk always allocates aligned output
        module("rope").dispatch(a, b, misaligned, cos_sin_table(256, 256), 256, stream())


def test_rejects_misaligned_table():
    """The epilogue reads [cos, sin] as one bf16x2 word: a table starting at an odd bf16 element is refused."""
    a, b = inputs(256, 256, 128, "random")
    cs = torch.empty(256 * 256 + 1, dtype=torch.bfloat16, device=DEV)[1:].view(256, 256)
    cs.copy_(cos_sin_table(256, 256))
    with pytest.raises(ValueError, match="4-byte aligned"):
        run(a, b, cs)
    out = torch.empty(256, 256, dtype=torch.bfloat16, device=DEV)
    with pytest.raises(RuntimeError, match="cos_sin must have 4-byte alignment"):
        module("rope").dispatch(a, b, out, cs, 256, stream())
