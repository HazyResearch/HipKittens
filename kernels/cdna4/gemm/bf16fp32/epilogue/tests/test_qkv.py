"""hk.qkv: one "rmsnorm_rope" GEMM over [Wq | Wk | Wv] with an identity RoPE table for V. Q and K must equal the
separate "rmsnorm_rope" calls bitwise (same GEMM per column, same epilogue arithmetic). V must be within one bf16
step of the "rmsnorm" call: identical fp32 values, but this store rounds to nearest and that one truncates."""
import pytest
import torch

import hk
from conftest import DEV, bf16_step, inputs, module, pow2_gamma
from test_rmsnorm_rope import r_like
from test_rope import cos_sin_table


def weights(K, Nq, Nkv):
    """Natural [K, N] bf16 Wq, Wk, Wv with exact products (conftest inputs)."""
    return [inputs(256, N, K, "random", seed=s)[1].t().contiguous() for s, N in ((1, Nq), (2, Nkv), (3, Nkv))]


@pytest.mark.parametrize("M,Nq,Nkv,K", [(256, 512, 256, 128), (2048, 4096, 1024, 4096)])
def test_matches_separate_calls(M, Nq, Nkv, K):
    for name in ("rmsnorm_rope", "rmsnorm_scale"):
        module(name)
    a, _ = inputs(M, 256, K, "random")
    Wq, Wk, Wv = weights(K, Nq, Nkv)
    r, g = r_like(M), pow2_gamma(K)
    cs_q, cs_k = cos_sin_table(M, Nq), cos_sin_table(M, Nkv)

    q, k, v = hk.qkv(a, hk.prepare_qkv(Wq, Wk, Wv, gamma=g), r=r, cos_sin=hk.prepare_qkv_rope_table(cs_q, cs_k))

    rope = lambda W, cs: hk.matmul(a, hk.prepare(W, layout="rope", gamma=g), "rmsnorm_rope", r=r,
                                   cos_sin=hk.prepare_rope_table(cs))
    assert torch.equal(q, rope(Wq, cs_q))
    assert torch.equal(k, rope(Wk, cs_k))
    v_ref = hk.matmul(a, hk.prepare(Wv, gamma=g), "rmsnorm", r=r)
    assert ((v.float() - v_ref.float()).abs() <= bf16_step(v_ref.float())).all()
    assert all(t.is_contiguous() for t in (q, k, v))


def test_rejects_mismatched_kv():
    Wq, Wk, _ = weights(128, 512, 256)
    with pytest.raises(ValueError, match="Wv"):                     # V must match K's width
        hk.prepare_qkv(Wq, Wk, weights(128, 512, 512)[1], gamma=pow2_gamma(128))


def test_rejects_unaligned_split():
    # Nq + 2 * Nkv = 512 passes hk.prepare's N % 256, but Q and K/V alone would not: refuse it
    K = 128
    Wq = torch.zeros(K, 384, dtype=torch.bfloat16, device=DEV)
    Wkv = torch.zeros(K, 64, dtype=torch.bfloat16, device=DEV)
    with pytest.raises(ValueError, match="N % 256"):
        hk.prepare_qkv(Wq, Wkv, Wkv, gamma=pow2_gamma(K))


def test_rejects_plain_weight():
    module("rmsnorm_rope")
    a, _ = inputs(256, 256, 128, "random")
    Wq, Wk, _ = weights(128, 512, 256)
    with pytest.raises(TypeError, match="prepare_qkv"):
        hk.qkv(a, hk.prepare(Wq, layout="rope", gamma=pow2_gamma(128)), r=r_like(256),
               cos_sin=hk.prepare_qkv_rope_table(cos_sin_table(256, 512), cos_sin_table(256, 256)))
