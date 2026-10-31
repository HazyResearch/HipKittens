"""hk.qkv: Q and K equal separate "rmsnorm_rope" calls bitwise; V is within one bf16 step of "rmsnorm" (round to
nearest vs truncation)."""
import pytest
import torch

import hk
from conftest import (DEV, assert_norm_close, bf16_step, general_gamma, inputs, module)
from test_residual_add import residual_like
from test_residual_rms import reference as residual_rms_reference
from test_rmsnorm_rope import r_like
from test_rope import cos_sin_table, rotate

MODULES = ("rmsnorm_rope_qkv", "rmsnorm_rope", "rmsnorm_scale")


def weights(K, Nq, Nkv):
    """bf16 Wq, Wk, Wv."""
    return [inputs(256, N, K, "random", seed=s)[1] for s, N in ((1, Nq), (2, Nkv), (3, Nkv))]


def assert_matches_separate_calls(a, wq, wk, wv, r, cs, q, k, v):
    assert torch.equal(q, hk.matmul(a, wq, "rmsnorm_rope", r=r, cos_sin=cs))
    assert torch.equal(k, hk.matmul(a, wk, "rmsnorm_rope", r=r, cos_sin=cs))
    v_ref = hk.matmul(a, wv, "rmsnorm", r=r)
    assert ((v.float() - v_ref.float()).abs() <= bf16_step(v_ref.float())).all()
    assert all(t.is_contiguous() for t in (q, k, v))


@pytest.mark.parametrize("M,Nq,Nkv,K,Dh", [
    (256, 512, 256, 128, 128),
    (2048, 4096, 1024, 4096, 128),
    (256, 256, 256, 128, 128),            # Nq == Nkv (MHA)
    (512, 2048, 512, 2048, 64),           # Dh = 64
])
def test_matches_separate_calls(M, Nq, Nkv, K, Dh):
    for name in MODULES:
        module(name)
    a, _ = inputs(M, 256, K, "random")
    wq, wk, wv = weights(K, Nq, Nkv)
    r, cs = r_like(M), cos_sin_table(M, Dh)
    assert_matches_separate_calls(a, wq, wk, wv, r, cs, *hk.qkv(a, wq, wk, wv, r=r, cos_sin=cs))


def test_side_stream():
    """qkv (launch and copies) runs on the current side stream."""
    for name in MODULES:
        module(name)
    a0, _ = inputs(4096, 256, 4096, "random")
    wq, wk, wv = weights(4096, 4096, 1024)
    r, cs = r_like(4096), cos_sin_table(4096, 128)
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        a = a0 * 1
        q, k, v = hk.qkv(a, wq, wk, wv, r=r, cos_sin=cs)
    s.synchronize()
    assert_matches_separate_calls(a0, wq, wk, wv, r, cs, q, k, v)


def test_graph_capture():
    """qkv captures into a graph and replays on new inputs."""
    for name in MODULES:
        module(name)
    a1, _ = inputs(256, 256, 128, "random", seed=1)
    a2, _ = inputs(256, 256, 128, "random", seed=2)
    wq, wk, wv = weights(128, 512, 256)
    r, cs = r_like(256), cos_sin_table(256, 128)
    static_a = a1.clone()
    hk.qkv(static_a, wq, wk, wv, r=r, cos_sin=cs)                 # warm-up
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_qkv = hk.qkv(static_a, wq, wk, wv, r=r, cos_sin=cs)
    static_a.copy_(a2)
    graph.replay()
    torch.cuda.synchronize()
    assert_matches_separate_calls(a2, wq, wk, wv, r, cs, *static_qkv)


def test_chain_gamma_into_qkv():
    """residual_rms(gamma=g) -> qkv matches RoPE(rmsnorm(h, g) @ W.t()) (fp64 r), normwise 1e-2."""
    module("residual_rms_gamma"), module("rms_reduce"), module("rmsnorm_rope_qkv")
    a, b = inputs(256, 256, 128, "random")
    res = residual_like(256, 256, 1.0)
    g = general_gamma(256, seed=13)
    out = hk.residual_rms(a, b, res, gamma=g)
    wq, wk, wv = weights(256, 512, 256)
    cs = cos_sin_table(256, 128)
    q, k, v = hk.qkv(out.hg, wq, wk, wv, r=out.r, cos_sin=cs)
    h_ref, r_ref = residual_rms_reference(a, b, res)
    xn = h_ref.double() * r_ref[:, None] * g.double()[None, :]
    for got, w, rotated in ((q, wq, True), (k, wk, True), (v, wv, False)):
        want = xn @ w.double().T
        if rotated:
            want = rotate(want.float(), cs.repeat(1, w.shape[0] // 128)).double()
        assert_norm_close(got, want, 1e-2)


def test_rejects_mismatched_kv():
    wq, wk, _ = weights(128, 512, 256)
    a, _ = inputs(256, 256, 128, "random")
    with pytest.raises(ValueError, match="wv"):                     # V shape != K shape
        hk.qkv(a, wq, wk, weights(128, 512, 512)[1], r=r_like(256), cos_sin=cos_sin_table(256, 128))


def test_rejects_unaligned_split():
    # each part must be whole 256-column blocks
    a, _ = inputs(256, 256, 128, "random")
    wq = torch.zeros(384, 128, dtype=torch.bfloat16, device=DEV)
    wkv = torch.zeros(64, 128, dtype=torch.bfloat16, device=DEV)
    with pytest.raises(ValueError, match="N % 256"):
        hk.qkv(a, wq, wkv, wkv, r=r_like(256), cos_sin=cos_sin_table(256, 64))


def test_rejects_head_dim_not_dividing_split():
    # Dh must divide Nkv too
    a, _ = inputs(256, 256, 128, "random")
    wq, wk, wv = weights(128, 512, 256)
    with pytest.raises(ValueError, match="must be even and divide Nq"):
        hk.qkv(a, wq, wk, wv, r=r_like(256), cos_sin=cos_sin_table(256, 512))


def test_rejects_misaligned_table_and_wrong_r():
    a, _ = inputs(256, 256, 128, "random")
    wq, wk, wv = weights(128, 512, 256)
    cs = torch.empty(256 * 128 + 1, dtype=torch.bfloat16, device=DEV)[1:].view(256, 128)
    cs.copy_(cos_sin_table(256, 128))
    with pytest.raises(ValueError, match="4-byte aligned"):
        hk.qkv(a, wq, wk, wv, r=r_like(256), cos_sin=cs)
    with pytest.raises(ValueError, match="r must be"):
        hk.qkv(a, wq, wk, wv, r=r_like(256).bfloat16(), cos_sin=cos_sin_table(256, 128))
