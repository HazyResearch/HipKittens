"""The engine patch layer without a GPU: how a measured table answers an M,
and what a patch's CUDA-graph contract refuses."""

from __future__ import annotations

import json

import pytest

from hk.integration import engine

T = engine.RouteTable({
    (64, 32): {"runs": [[1, 64, "hk_decode"], [80, 128, "hk"],
                        [144, 192, "fallback"], [208, 256, "hk"]],
               "above": "hk"},
})


@pytest.mark.parametrize("m,path", [
    (1, "hk_decode"), (64, "hk_decode"),       # inside a run
    (80, "hk"), (100, "hk"), (128, "hk"),
    (144, "fallback"), (200, "fallback"),      # a measured loss, and a gap
    (208, "hk"), (256, "hk"),
    (65, "fallback"), (79, "fallback"),        # between two runs that disagree
    (129, "fallback"), (143, "fallback"),
    (257, "hk"), (100000, "hk"),               # above the largest measured M
    (0, "fallback"),                           # below the smallest
])
def test_a_table_answers_inside_runs_and_falls_back_between_them(m, path):
    assert T.lookup(m, 64, 32) == path


def test_a_shape_never_measured_falls_back_and_is_remembered():
    t = engine.RouteTable({(64, 32): {"runs": [[1, 8, "hk"]]}})
    assert t.lookup(4, 128, 32) == "fallback"
    assert (128, 32) in t.misses


def test_without_above_the_last_run_extends():
    t = engine.RouteTable({(64, 32): {"runs": [[1, 8, "fallback"], [16, 32, "hk"]]}})
    assert t.lookup(1000, 64, 32) == "hk"


def test_overlapping_runs_are_refused():
    with pytest.raises(ValueError, match="overlap"):
        engine.RouteTable({(64, 32): {"runs": [[1, 64, "hk"], [64, 80, "fallback"]]}})


def test_a_table_round_trips_through_json(tmp_path):
    p = tmp_path / "routes_vllm_linear_gfx1100.json"
    p.write_text(json.dumps({"arch": "gfx1100", "shapes": {
        "64x32": {"runs": [[1, 64, "hk_decode"]], "above": "fallback"}}}))
    t = engine.RouteTable.from_json(p)
    assert t.lookup(4, 64, 32) == "hk_decode" and t.lookup(65, 64, 32) == "fallback"
    assert t.meta["arch"] == "gfx1100"


def test_a_table_from_another_arch_is_not_loaded(tmp_path, monkeypatch):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"arch": "gfx1201", "shapes": {}}))
    monkeypatch.setenv("HK_TEST_ROUTES", str(p))
    with pytest.raises(ValueError, match="gfx1201"):
        engine.shipped_table("vllm_linear", "gfx1100", "HK_TEST_ROUTES")


def test_an_override_that_does_not_exist_is_an_error(monkeypatch, tmp_path):
    monkeypatch.setenv("HK_TEST_ROUTES", str(tmp_path / "missing.json"))
    with pytest.raises(FileNotFoundError):
        engine.shipped_table("vllm_linear", "gfx1100", "HK_TEST_ROUTES")


def test_no_table_for_an_arch_is_none(monkeypatch):
    monkeypatch.delenv("HK_TEST_ROUTES", raising=False)
    assert engine.shipped_table("vllm_linear", "gfx9999", "HK_TEST_ROUTES") is None


def test_a_patch_that_cannot_be_captured_has_to_say_why():
    with pytest.raises(ValueError, match="says why"):
        engine.Patch("p_nowhy", paths=("a",), graph="never")
    with pytest.raises(ValueError, match="graph contract"):
        engine.Patch("p_bad", paths=("a",), graph="sometimes")


def test_outside_capture_a_guard_is_transparent_and_counts():
    p = engine.Patch("p_free", paths=("a", "b"))

    def f(x):
        p.count("a")
        return x + 1

    assert p.guard(f)(1) == 2
    assert p.stats == {"a": 1, "b": 0} and p.captured == {"a": 0, "b": 0}
    assert engine.PATCHES["p_free"] is p


def test_under_capture_a_never_patch_raises_its_reason(monkeypatch):
    monkeypatch.setattr(engine, "capturing", lambda: True)
    p = engine.Patch("p_never", paths=("a",), graph="never",
                     why="it plans on max_seq_len")
    with pytest.raises(engine.GraphCaptureError, match="max_seq_len"):
        p.check_capture()
    with pytest.raises(engine.GraphCaptureError, match="p_never"):
        p.guard(lambda: None)()


def test_under_capture_counts_go_to_captured(monkeypatch):
    monkeypatch.setattr(engine, "capturing", lambda: True)
    p = engine.Patch("p_cap", paths=("a",))
    p.count("a")
    assert p.stats["a"] == 0 and p.captured["a"] == 1
    assert "captured: a 1" in p.summary()
