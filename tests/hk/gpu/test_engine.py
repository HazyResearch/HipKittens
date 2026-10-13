"""The CUDA-graph contract of `hk.integration.engine`, under a real capture.

What the unit tests cannot see: that torch's sync debug mode does fire inside
a custom op on this ROCm build, that the error reaches the caller as a
`GraphCaptureError` naming the patch rather than as a failed capture, and that
the mode is put back afterwards.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hk.integration import engine  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a Radeon"
)


def _capture(fn):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    return graph, out


def _syncing(x):
    if x.sum().item() > 0:                         # a host round trip
        return x * 2
    return x


def test_a_host_sync_in_a_safe_op_raises_under_capture_naming_it():
    op = engine.Op("hk_test::syncs", "syncs(Tensor x) -> Tensor",
                   lambda x: torch.empty_like(x), paths=("hk",))
    op.define(_syncing)
    x = torch.ones(64, device="cuda")
    torch.testing.assert_close(op(x), 2 * x)       # outside capture it is fine
    prev = torch.cuda.get_sync_debug_mode()
    with pytest.raises(engine.GraphCaptureError, match="hk_test::syncs"):
        _capture(lambda: op(x))
    assert torch.cuda.get_sync_debug_mode() == prev


def test_a_never_patch_refuses_capture_with_its_reason():
    p = engine.Patch("hk_test_never", paths=("a",), graph="never",
                     why="it plans on max_seq_len")
    f = p.guard(lambda x: x + 1)
    x = torch.ones(64, device="cuda")
    f(x)
    with pytest.raises(engine.GraphCaptureError, match="max_seq_len"):
        _capture(lambda: f(x))


def test_a_safe_patch_captures_replays_and_counts_capture_apart():
    p = engine.Patch("hk_test_safe", paths=("a",))

    def body(x):
        p.count("a")
        return x * 3

    f = p.guard(body)
    x = torch.ones(64, device="cuda")
    f(x)
    graph, y = _capture(lambda: f(x))
    x.fill_(2)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(y, torch.full_like(x, 6))
    assert p.stats["a"] == 1 and p.captured["a"] == 1
