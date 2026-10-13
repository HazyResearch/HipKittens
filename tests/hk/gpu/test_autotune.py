"""Autotune end to end: compile a space, time it, record it, use the record.

The IR tier (`tests/hk/ir/test_autotune.py`) covers the search with the build
stage stubbed out, which is most of the logic. What it cannot cover is that the
thing the tuner hands to the benchmark is a kernel that launches, that the
record it writes is the one production reads back, and that a schedule rejected
by the resource gate was rejected by *hipcc* rather than by a mock. So this
file runs one small space for real.

Small on purpose: every candidate here is a compile, and a compile is tens of
seconds. The claim under test is that the path works, not that a particular
tiling wins.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import hk  # noqa: E402
from hk import bf16  # noqa: E402 -- module level: the @hk.kernel annotations
                     # below are strings (PEP 563) and are resolved against
                     # this module's globals, not against _make's locals.
from hk.autotune import Tuner, space  # noqa: E402

pytestmark = pytest.mark.gpu

N = 1 << 20


@pytest.fixture
def records(tmp_path, monkeypatch):
    """Records in a temp dir, and no shipped ones, so a real tuning record
    written by another run on this machine cannot decide these tests."""
    monkeypatch.setenv("HK_TUNE_DIR", str(tmp_path))
    monkeypatch.setattr(hk.autotune, "shipped_dir", lambda: tmp_path / "none")
    hk.autotune.forget_records()
    yield tmp_path
    hk.autotune.forget_records()


def _make(*, ROWS, COLS):
    """An elementwise add, tiled two ways. Both are correct; one of them walks
    the row faster than the other."""
    name = f"tune_add_{ROWS}x{COLS}"

    def body(a: hk.Tensor[bf16], b: hk.Tensor[bf16], o: hk.Tensor[bf16]):
        t = hk.RT(bf16, ROWS, COLS)
        idx = hk.tile_coord(hk.arch.block_idx.z, 0, hk.arch.block_idx.y,
                            hk.arch.block_idx.x)
        o[idx] = a[idx].load(t) + b[idx].load(t)

    body.__name__ = name
    return hk.kernel(body, arch="gfx1100", warps=1, name=name,
                     grid=lambda p: (hk.cdiv(p.o.cols, COLS),
                                     hk.cdiv(p.o.rows, ROWS), p.o.batch))


def _bench(key: str):
    rows, cols = (int(x) for x in key.split("x"))
    a, b = (torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16)
            for _ in range(2))
    o = torch.empty_like(a)

    def run(kernel, warmup=3, iters=20):
        for _ in range(warmup):
            kernel(a, b, o)
        torch.cuda.synchronize()
        beg, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        beg.record()
        for _ in range(iters):
            kernel(a, b, o)
        end.record()
        torch.cuda.synchronize()
        return beg.elapsed_time(end) / iters

    return run


def _tuner():
    return Tuner("test_add", _make, space(ROWS=[16, 32], COLS=[32, 64]),
                 dict(ROWS=16, COLS=64), bench=_bench, register=False)


def test_a_space_compiles_times_and_records(records):
    t = _tuner()
    res = t.tune(key="2048x2048", rounds=2)

    assert len(res.rows) == 4
    assert res.survivors, res.report()
    assert res.best is not None and res.best.ms > 0
    # Every surviving row carries what the gate saw: these are tiny kernels, so
    # no scratch and plenty of registers left.
    for r in res.survivors:
        assert r.scratch == 0 and r.vgpr and r.occupancy

    # The record is what production reads back, not a separate copy of the
    # answer: the schedule it names is the one the tuner's own kernel() hands
    # out for that key.
    assert (records / "test_add.json").exists()
    assert t.schedule_for("2048x2048") == {**t.default, **res.best.schedule}
    assert t.kernel("2048x2048") is t._kernel_for(
        {**t.default, **res.best.schedule})


def test_the_tuned_kernel_still_computes_the_right_answer(records):
    # A fast schedule that is wrong is not a win. The tuner never checks
    # numerics -- it times -- so this is the check that says a space may only
    # contain kernels that agree.
    t = _tuner()
    t.tune(key="1024x1024", rounds=1)
    a, b = (torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
            for _ in range(2))
    o = torch.empty_like(a)
    t.kernel("1024x1024")(a, b, o)
    torch.testing.assert_close(o, a + b, rtol=3e-2, atol=1e-2)


def test_aot_prebuilds_every_schedule_production_can_reach(records):
    from hk.runtime.warm import warm_cache

    t = _tuner()
    t.tune(key="2048x2048", rounds=1)
    jobs = t.aot_jobs()
    assert 1 <= len(jobs) <= 2          # the default, plus the record if it won
    res = warm_cache(jobs)
    assert res.ok, res.errors
    # Prebuilt means cached: a second pass compiles nothing.
    assert all(b.cached for b in warm_cache(jobs).built.values())
