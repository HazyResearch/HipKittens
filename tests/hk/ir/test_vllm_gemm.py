"""The linear-layer routing rule, without a GPU or vLLM.

The rule is a transcription of a measurement (tools/hk-bench/vllm_gemm.py), so
what is pinned here is that it keeps the shapes the measurement said lose on
vLLM's path: M=192 measured 0.71-0.95x and M=256 was mixed, and both must stay
off the generated kernel however tempting the neighbouring wins make a simpler
threshold look.
"""

import pytest

from hk.integration import vllm_gemm

QKV, DOWN = (6144, 4096), (4096, 12288)


@pytest.mark.parametrize("m", [384, 512, 768, 1024, 1025, 4000, 4096, 16384])
def test_measured_wins_route(m):
    assert vllm_gemm.route(m, *QKV)
    assert vllm_gemm.route(m, *DOWN)


@pytest.mark.parametrize("m", [0, 1, 4, 16, 32, 127, 192, 256, 385, 448,
                               1000, 1023])
def test_measured_losses_and_unmeasured_gaps_fall_back(m):
    assert not vllm_gemm.route(m, *QKV)


def test_m_128_falls_back_although_it_measured_a_win():
    # One block-row, so it is the one small M where the kernel won (1.15-1.56x)
    # -- and a prefill step of exactly 128 tokens is too rare to justify a
    # rule with a hole in it. Pinned so that changing this is a decision.
    assert not vllm_gemm.route(128, *QKV)


@pytest.mark.parametrize("n,k", [(6100, 4096), (6144, 4000), (100, 64)])
def test_shapes_the_tiling_cannot_take_fall_back(n, k):
    assert not vllm_gemm.route(4096, n, k)


def test_bias_dtype_and_layout_fall_back():
    assert not vllm_gemm.route(4096, *QKV, bias=True)
    assert not vllm_gemm.route(4096, *QKV, bf16=False)
    assert not vllm_gemm.route(4096, *QKV, contiguous_weight=False)


@pytest.mark.parametrize("m", [1, 2, 5, 8, 15, 16, 17, 24, 32, 33, 48, 64])
def test_decode_batches_take_the_decode_kernel(m):
    assert vllm_gemm.route_decode(m, *QKV)
    assert vllm_gemm.route_decode(m, *DOWN)
    assert vllm_gemm.route_decode(m, 151936, 4096)        # the LM head


@pytest.mark.parametrize("m", [0, 65, 128])
def test_the_decode_kernel_stops_at_four_token_tiles(m):
    assert not vllm_gemm.route_decode(m, *QKV)


def test_the_decode_kernel_refuses_what_it_cannot_take():
    assert not vllm_gemm.route_decode(4, 6100, 4096)      # N not whole blocks
    assert not vllm_gemm.route_decode(4, 6144, 4000)      # K not whole slices
    assert not vllm_gemm.route_decode(4, *QKV, bias=True)
    assert not vllm_gemm.route_decode(4, *QKV, bf16=False)
    assert not vllm_gemm.route_decode(4, *QKV, contiguous_weight=False)


def test_the_two_routes_never_overlap():
    for m in range(0, 2049):
        assert not (vllm_gemm.route(m, *QKV) and vllm_gemm.route_decode(m, *QKV))


def test_the_patched_dispatcher_ignores_its_arguments():
    # Newer vLLM passes a linear-backend name; it selects among CUDA backends
    # and has no meaning on ROCm, so whatever is passed the answer is ours.
    assert vllm_gemm.dispatch_unquantized_gemm() is vllm_gemm._gemm
    assert vllm_gemm.dispatch_unquantized_gemm("auto") is vllm_gemm._gemm


def test_apply_without_vllm_reports_rather_than_raises():
    from hk.integration import _common

    if _common.importable("vllm"):
        pytest.skip("vllm is installed here; the gpu tier covers apply()")
    assert "not importable" in vllm_gemm.apply()
    assert vllm_gemm.status() == "not patched"
