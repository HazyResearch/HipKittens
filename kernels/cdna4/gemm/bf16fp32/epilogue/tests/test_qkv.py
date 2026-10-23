"""hk.qkv: one "rmsnorm_rope" GEMM over three separate weights Wq, Wk, Wv (no concatenated copy), rotating only the
Q and K columns, with one head's [M, Dh] cos/sin table. Q and K must equal the separate "rmsnorm_rope" calls bitwise
(same GEMM per column, same epilogue arithmetic). V must be within one bf16 step of the "rmsnorm" call: identical
fp32 values, but this store rounds to nearest and that one truncates."""
import pytest
import torch

import hk
from conftest import DEV, bf16_step, inputs, module
from test_rmsnorm_rope import r_like
from test_rope import cos_sin_table


def weights(K, Nq, Nkv):
    """[N, K] bf16 Wq, Wk, Wv with exact products (conftest inputs)."""
    return [inputs(256, N, K, "random", seed=s)[1] for s, N in ((1, Nq), (2, Nkv), (3, Nkv))]


@pytest.mark.parametrize("M,Nq,Nkv,K", [(256, 512, 256, 128), (2048, 4096, 1024, 4096)])
def test_matches_separate_calls(M, Nq, Nkv, K):
    for name in ("rmsnorm_rope_qkv", "rmsnorm_rope", "rmsnorm_scale"):
        module(name)
    a, _ = inputs(M, 256, K, "random")
    wq, wk, wv = weights(K, Nq, Nkv)
    r = r_like(M)
    cs = cos_sin_table(M, 128)                                     # one 128-wide head's table, shared

    q, k, v = hk.qkv(a, wq, wk, wv, r=r, cos_sin=cs)

    assert torch.equal(q, hk.matmul(a, wq, "rmsnorm_rope", r=r, cos_sin=cs))
    assert torch.equal(k, hk.matmul(a, wk, "rmsnorm_rope", r=r, cos_sin=cs))
    v_ref = hk.matmul(a, wv, "rmsnorm", r=r)
    assert ((v.float() - v_ref.float()).abs() <= bf16_step(v_ref.float())).all()
    assert all(t.is_contiguous() for t in (q, k, v))


def test_rejects_mismatched_kv():
    wq, wk, _ = weights(128, 512, 256)
    a, _ = inputs(256, 256, 128, "random")
    with pytest.raises(ValueError, match="wv"):                     # V must match K's shape
        hk.qkv(a, wq, wk, weights(128, 512, 512)[1], r=r_like(256), cos_sin=cos_sin_table(256, 128))


def test_rejects_unaligned_split():
    # Nq + 2 * Nkv = 512 would be a valid N as one weight, but each part must hold whole 256-column blocks
    a, _ = inputs(256, 256, 128, "random")
    wq = torch.zeros(384, 128, dtype=torch.bfloat16, device=DEV)
    wkv = torch.zeros(64, 128, dtype=torch.bfloat16, device=DEV)
    with pytest.raises(ValueError, match="N % 256"):
        hk.qkv(a, wq, wkv, wkv, r=r_like(256), cos_sin=cos_sin_table(256, 64))


def test_rejects_head_dim_not_dividing_split():
    # Nq + 2 * Nkv = 1024 is a multiple of Dh = 512, but Nkv = 256 is not: K's heads would start mid-table
    a, _ = inputs(256, 256, 128, "random")
    wq, wk, wv = weights(128, 512, 256)
    with pytest.raises(ValueError, match="must be even and divide Nq"):
        hk.qkv(a, wq, wk, wv, r=r_like(256), cos_sin=cos_sin_table(256, 512))
