"""`torch.ops.hk_vllm.linear`: the op vLLM's linear layers are routed to.

Needs a Radeon and torch, not vLLM: without it the fallback is F.linear, which
is what vLLM itself ends in on gfx1100. Checked here is what the routing rule's
unit tests cannot see -- that both branches compute x @ W.T, that the shapes a
serving engine hands over (ragged token counts, a leading batch axis, a strided
activation) reach the kernel correctly, and that the op survives the two things
an engine does to it: torch.compile and CUDA-graph capture.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hk.integration import vllm_gemm  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a Radeon"
)

RTOL, ATOL = 2e-2, 1e-2


@pytest.fixture(scope="module", autouse=True)
def op():
    vllm_gemm.register_op()
    return torch.ops.hk_vllm.linear


def _wx(m, n, k, seed=0, dtype=torch.bfloat16):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, device="cuda", dtype=dtype, generator=g)
    w = torch.randn(n, k, device="cuda", dtype=dtype, generator=g) * k ** -0.5
    return x, w


def _ref(x, w, b=None):
    y = x.float() @ w.float().t()
    return y if b is None else y + b.float()


def _took(fn):
    before = dict(vllm_gemm.STATS)
    out = fn()
    torch.cuda.synchronize()
    return out, {k: vllm_gemm.STATS[k] - before[k] for k in before}


@pytest.mark.parametrize("m", [1, 4, 16, 17, 40, 64, 65, 80, 129, 144, 192,
                               208, 300, 384, 784, 1000, 1024, 1025, 1216, 4000])
@pytest.mark.parametrize("n,k", [(6144, 4096), (4096, 12288)])
def test_every_branch_computes_x_wt(op, m, n, k):
    path = vllm_gemm.route_for(m, n, k)
    x, w = _wx(m, n, k)
    y, took = _took(lambda: op(x, w, None))
    assert y.shape == (m, n) and y.dtype == torch.bfloat16
    assert took == {p: int(p == path) for p in took}
    torch.testing.assert_close(y.float(), _ref(x, w), rtol=RTOL, atol=ATOL)


def test_the_table_reaches_all_three_paths():
    paths = {vllm_gemm.route_for(m, 6144, 4096) for m in (4, 192, 1024)}
    assert paths == {"hk_decode", "fallback", "hk"}


def test_a_decode_batch_with_a_leading_axis_is_padded_and_sliced(op):
    x, w = _wx(6, 4096, 4096)
    y, took = _took(lambda: op(x.view(2, 3, 4096), w, None))
    assert y.shape == (2, 3, 4096) and took["hk_decode"] == 1
    torch.testing.assert_close(y.reshape(6, 4096).float(), _ref(x, w),
                               rtol=RTOL, atol=ATOL)


def test_a_leading_batch_axis_is_flattened_into_m(op):
    x, w = _wx(1024, 4096, 4096)
    x3 = x.view(2, 512, 4096)
    y, took = _took(lambda: op(x3, w, None))
    assert y.shape == (2, 512, 4096) and took["hk"] == 1
    torch.testing.assert_close(y.reshape(1024, 4096).float(), _ref(x, w),
                               rtol=RTOL, atol=ATOL)


def test_a_strided_activation_is_copied_not_read_wrong(op):
    x, w = _wx(1024, 6144, 4096)
    xt = x.t().contiguous().t()                   # same values, column-major
    assert not xt.is_contiguous()
    y, took = _took(lambda: op(xt, w, None))
    assert took["hk"] == 1
    torch.testing.assert_close(y.float(), _ref(x, w), rtol=RTOL, atol=ATOL)


def test_bias_takes_the_fallback_and_is_applied(op):
    x, w = _wx(4096, 4096, 4096)
    b = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
    y, took = _took(lambda: op(x, w, b))
    assert took["fallback"] == 1
    torch.testing.assert_close(y.float(), _ref(x, w, b), rtol=RTOL, atol=ATOL)


def test_fp16_takes_the_fallback(op):
    x, w = _wx(4096, 4096, 4096, dtype=torch.float16)
    y, took = _took(lambda: op(x, w, None))
    assert took["fallback"] == 1 and y.dtype == torch.float16


def test_the_patched_layer_function_matches_f_linear():
    x, w = _wx(4096, 6144, 4096)
    y = vllm_gemm.dispatch_unquantized_gemm()(None, x, w, None)
    torch.testing.assert_close(y.float(), _ref(x, w), rtol=RTOL, atol=ATOL)


def test_it_compiles_fullgraph(op):
    # The token count is symbolic under vLLM's compile; the branch on it has
    # to live inside the op, where the fake implementation hides it.
    x, w = _wx(1024, 4096, 4096)
    f = torch.compile(lambda a, b: op(a, b, None) * 2, fullgraph=True,
                      dynamic=True)
    torch.testing.assert_close(f(x, w).float(), 2 * _ref(x, w),
                               rtol=RTOL, atol=2 * ATOL)


@pytest.mark.parametrize("m", [1024, 4])
def test_it_replays_from_a_cuda_graph(op, m):
    # 4 is the thin path, which pads inside the captured region.
    x, w = _wx(m, 4096, 4096)
    y = op(x, w, None)                            # build before capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = op(x, w, None)
    x2, _ = _wx(m, 4096, 4096, seed=1)
    x.copy_(x2)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(y.float(), _ref(x2, w), rtol=RTOL, atol=ATOL)
