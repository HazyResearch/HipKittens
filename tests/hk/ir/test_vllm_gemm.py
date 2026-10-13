"""The linear-layer routing, without a GPU or vLLM.

Routing is a measured table (hk/tuned/routes_vllm_linear_gfx1100.json, from
tools/hk-bench/route_tune.py). What is pinned here is what the table has to
keep saying for the measurement to have been worth taking: the decode kernel
owns every batch to 64, the generated GEMM's wins are a comb in M rather than
a threshold, and no route sends a shape to a kernel that cannot take it.
"""

import pytest

from hk.integration import engine, vllm_gemm

QKV, O, GATE_UP, DOWN, LM_HEAD = ((6144, 4096), (4096, 4096), (24576, 4096),
                                  (4096, 12288), (151936, 4096))
SHAPES = (QKV, O, GATE_UP, DOWN, LM_HEAD)


@pytest.fixture(autouse=True)
def shipped():
    vllm_gemm.set_routes(None)
    yield
    vllm_gemm.set_routes(None)


def test_the_shipped_table_covers_qwen3_8b():
    t = vllm_gemm.routes()
    assert t is not None and t.meta["arch"] == "gfx1100"
    assert set(SHAPES) <= set(t.shapes())
    assert t.paths() <= set(vllm_gemm.PATCH.paths)


@pytest.mark.parametrize("nk", SHAPES)
@pytest.mark.parametrize("m", [1, 2, 5, 8, 15, 16, 17, 24, 32, 33, 48, 64])
def test_decode_batches_take_the_decode_kernel(m, nk):
    assert vllm_gemm.route_for(m, *nk) == "hk_decode"


@pytest.mark.parametrize("nk", (QKV, O, GATE_UP, DOWN))
@pytest.mark.parametrize("m", [1024, 2048, 4096, 8192, 16384])
def test_prefill_sized_steps_take_the_gemm(m, nk):
    assert vllm_gemm.route_for(m, *nk) == "hk"


def test_the_gemm_wins_are_a_comb_not_a_threshold():
    # The last 128-row block is computed whole: just past a multiple of 128
    # the kernel loses, and just before the next one it wins again.
    r = [vllm_gemm.route_for(m, *QKV) for m in (128, 144, 192, 208, 256, 288, 304)]
    assert r == ["hk", "fallback", "fallback", "hk", "hk", "fallback", "hk"]
    hk_runs = [ru for ru in vllm_gemm.routes()._runs[QKV] if ru[2] == "hk" and ru[0] < 1024]
    assert len(hk_runs) >= 3


def test_an_m_between_two_disagreeing_measurements_falls_back():
    # 129 lies between 128 (a win) and 144 (a loss): not guessed.
    assert vllm_gemm.route_for(129, *QKV) == "fallback"
    # 65..79: the decode kernel stops at 64 and the GEMM was measured from 80.
    assert vllm_gemm.route_for(70, *QKV) == "fallback"


@pytest.mark.parametrize("n,k", [(6100, 4096), (6144, 4000), (100, 64)])
def test_a_shape_the_table_never_measured_falls_back(n, k):
    assert vllm_gemm.route_for(4096, n, k) == "fallback"
    assert vllm_gemm.route_for(4, n, k) == "fallback"


def test_a_table_cannot_send_a_shape_to_a_kernel_that_refuses_it():
    vllm_gemm.set_routes(engine.RouteTable({
        (6100, 4096): {"runs": [[1, 64, "hk_decode"], [80, 99999, "hk"]]},
        (6144, 4000): {"runs": [[1, 64, "hk_decode"], [80, 99999, "hk"]]},
    }))
    for nk in ((6100, 4096), (6144, 4000)):
        assert vllm_gemm.route_for(4, *nk) == "fallback"
        assert vllm_gemm.route_for(4096, *nk) == "fallback"


def test_an_unknown_path_in_a_table_falls_back():
    vllm_gemm.set_routes(engine.RouteTable({QKV: {"runs": [[1, 99999, "cutlass"]]}}))
    assert vllm_gemm.route_for(4096, *QKV) == "fallback"


def test_bias_dtype_and_layout_fall_back():
    for m in (4, 4096):
        assert vllm_gemm.route_for(m, *QKV, bias=True) == "fallback"
        assert vllm_gemm.route_for(m, *QKV, bf16=False) == "fallback"
        assert vllm_gemm.route_for(m, *QKV, contiguous_weight=False) == "fallback"
    assert vllm_gemm.route_for(0, *QKV) == "fallback"


def test_route_and_route_decode_are_views_of_one_answer():
    for m in range(0, 2049, 7):
        p = vllm_gemm.route_for(m, *QKV)
        assert vllm_gemm.route(m, *QKV) == (p == "hk")
        assert vllm_gemm.route_decode(m, *QKV) == (p == "hk_decode")


def test_the_op_declares_itself_graph_safe():
    assert vllm_gemm.PATCH.graph == "safe"
    assert vllm_gemm.STATS is vllm_gemm.PATCH.stats


def test_the_attention_backend_declares_why_it_cannot_be_captured():
    from hk.integration import vllm_backend

    assert vllm_backend.PATCH.graph == "never"
    assert "max_seq_len" in vllm_backend.PATCH.why


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
