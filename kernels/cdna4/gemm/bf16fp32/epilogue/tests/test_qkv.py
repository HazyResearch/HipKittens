"""hk.qkv: one "rmsnorm_rope" GEMM over [Wq; Wk; Wv] rotating only the Q and K columns, with one head's
[M, Dh] cos/sin table. Q and K must equal the separate "rmsnorm_rope" calls bitwise (same GEMM per column, same
epilogue arithmetic). V must be within one bf16 step of the "rmsnorm" call: identical fp32 values, but this
store rounds to nearest and that one truncates."""
import pytest
import torch

import hk
from conftest import DEV, bf16_step, inputs, module, pow2_gamma, weight
from test_rmsnorm_rope import r_like
from test_rope import cos_sin_table


def weights(K, Nq, Nkv):
    """[N, K] bf16 Wq, Wk, Wv with exact products (conftest inputs)."""
    return [inputs(256, N, K, "random", seed=s)[1] for s, N in ((1, Nq), (2, Nkv), (3, Nkv))]


@pytest.mark.parametrize("M,Nq,Nkv,K", [(256, 512, 256, 128), (2048, 4096, 1024, 4096)])
def test_matches_separate_calls(M, Nq, Nkv, K):
    for name in ("rmsnorm_rope", "rmsnorm_scale"):
        module(name)
    a, _ = inputs(M, 256, K, "random")
    Wq, Wk, Wv = weights(K, Nq, Nkv)
    r, g = r_like(M), pow2_gamma(K)
    cs = cos_sin_table(M, 128)                                     # one 128-wide head's table, shared
    wq, wk, wv = (weight(W, gamma=g) for W in (Wq, Wk, Wv))

    q, k, v = hk.qkv(a, hk.prepare_qkv(wq, wk, wv), r=r, cos_sin=cs)

    assert torch.equal(q, hk.matmul(a, wq, "rmsnorm_rope", r=r, cos_sin=cs))
    assert torch.equal(k, hk.matmul(a, wk, "rmsnorm_rope", r=r, cos_sin=cs))
    v_ref = hk.matmul(a, wv, "rmsnorm", r=r)
    assert ((v.float() - v_ref.float()).abs() <= bf16_step(v_ref.float())).all()
    assert all(t.is_contiguous() for t in (q, k, v))


def test_rejects_mismatched_kv():
    Wq, Wk, _ = weights(128, 512, 256)
    g = pow2_gamma(128)
    with pytest.raises(ValueError, match="Wv"):                     # V must match K's width
        hk.prepare_qkv(weight(Wq, gamma=g), weight(Wk, gamma=g), weight(weights(128, 512, 512)[1], gamma=g))


def test_rejects_weight_without_gamma():
    # every part feeds "rmsnorm_rope", so an unfolded part would silently drop the norm's gain
    Wq, Wk, Wv = weights(128, 512, 256)
    g = pow2_gamma(128)
    with pytest.raises(ValueError, match="gamma folded in"):
        hk.prepare_qkv(weight(Wq, gamma=g), weight(Wk), weight(Wv, gamma=g))


def test_rejects_unaligned_split():
    # Nq + 2 * Nkv = 512 would pass N % 256 as one weight, but Q and K/V alone do not: each part is refused
    Wkv = torch.zeros(64, 128, dtype=torch.bfloat16, device=DEV)
    with pytest.raises(ValueError, match="N % 256"):
        hk.weight(Wkv, gamma_folded=True)


def test_rejects_plain_weight():
    module("rmsnorm_rope")
    a, _ = inputs(256, 256, 128, "random")
    Wq, _, _ = weights(128, 512, 256)
    with pytest.raises(TypeError, match="prepare_qkv"):
        hk.qkv(a, weight(Wq, gamma=pow2_gamma(128)), r=r_like(256), cos_sin=cos_sin_table(256, 128))


def test_rejects_head_dim_not_dividing_split():
    # Nq + 2 * Nkv = 1024 is a multiple of Dh = 512, but Nkv = 256 is not: K's heads would start mid-table
    module("rmsnorm_rope")
    a, _ = inputs(256, 256, 128, "random")
    g = pow2_gamma(128)
    wq, wk, wv = (weight(W, gamma=g) for W in weights(128, 512, 256))
    with pytest.raises(ValueError, match="must divide Nq"):
        hk.qkv(a, hk.prepare_qkv(wq, wk, wv), r=r_like(256), cos_sin=cos_sin_table(256, 512))
