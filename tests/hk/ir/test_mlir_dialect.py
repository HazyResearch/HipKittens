"""The hk MLIR dialect, driven from Python, against the Python IR it replaces.

Skipped where hk._mlir has not been built (scripts/build_hk_mlir.sh).
"""

from __future__ import annotations

import importlib

import pytest

pytest.importorskip("hk._mlir")

from hk.compiler import gate_a, mlir as M  # noqa: E402
from hk.compiler.from_kernel_ir import convert  # noqa: E402
from hk.target import TARGETS  # noqa: E402


def _facts(arch: str) -> dict:
    from hk._mlir.dialects import hk as hkd

    out = {}
    for line in hkd.target_facts(arch).splitlines():
        k, _, v = line.partition("=")
        out.setdefault(k, []).append(v)
    return out


@pytest.mark.parametrize("arch", sorted(TARGETS))
def test_the_cpp_target_table_is_the_python_one(arch):
    t, f = TARGETS[arch], _facts(arch)
    for field in ("wave", "max_waves_per_simd", "vgprs_per_simd", "vgpr_granule",
                  "sgprs_per_wave", "lds_per_workgroup", "lds_pool_bytes",
                  "simds_per_lds_pool"):
        assert int(f[field][0]) == getattr(t, field), field
    assert bool(int(f["verified"][0])) == t.verified
    mma = {k[4:]: v[0] for k, v in f.items() if k.startswith("mma.")}
    assert list(mma) == list(t.mma)
    for name, shape in t.mma.items():
        assert mma[name] == (f"{shape.m},{shape.n},{shape.k},"
                             f"{shape.operand_vgprs},{shape.accum_vgprs}")
    assert tuple(f["fast_smem_layout"]) == t.fast_smem_layouts


def _shipped():
    jobs = []
    for mod in ("norm", "fused", "quant", "elementwise", "attn", "sdpa", "paged"):
        m = importlib.import_module(f"hk.ops.{mod}")
        jobs += [(f"{mod}.{n}", k) for n, k in sorted(getattr(m, "KERNELS", {}).items())]
    from hk.ops import decode_gemm, gemm, paged

    jobs.append(("gemm.default", gemm.TUNER._kernel_for(gemm.DEFAULT)))
    jobs.append(("gemm.thin", gemm.TUNER._kernel_for(gemm.THIN)))
    jobs.append(("paged.decode", paged.paged_decode_kernel("rd_paged_decode")))
    jobs.append(("paged.split", paged.paged_decode_split_kernel("rd_paged_split")))
    for t, s in sorted(decode_gemm.SCHEDULES.items()):
        jobs.append((f"decode_gemm.t{t}", decode_gemm.kernel_for(tokens=t, **s)))
    return jobs


def test_every_shipped_kernel_translates_byte_identically():
    """Gate A: the C++ passes and translation agree with the Python ones."""
    differ = []
    for label, k in _shipped():
        kir = k.trace()
        try:
            gate_a.check(kir, python_warnings=kir.warnings)
        except gate_a.GateError as e:
            differ.append(f"{label}: {str(e)[:400]}")
    assert not differ, "\n".join(differ[:5])


def test_the_textual_ir_round_trips_with_its_names():
    from hk.ops import norm

    kir = norm.KERNELS["rms_bf16_w4"].trace(TPW=2, EXACT=1)
    mod = convert(kir)
    M.run_pipeline(mod)
    before = M.translate(mod, ['"kittens.cuh"'])
    with M.context():
        text = mod.operation.get_asm(enable_debug_info=True)
        again = M.ir.Module.parse(text)
    assert M.translate(again, ['"kittens.cuh"']) == before
    assert "hk.row_sum" in text and "!hk.rt<f32, 16x64, row>" in text


def test_a_pass_error_reads_as_the_python_one(monkeypatch):
    # With the gate on, a C++ message that differed would surface as a
    # GateError instead of the VerifyError asked for here.
    import hk
    from hk import bf16
    from hk.ir.verify import VerifyError

    monkeypatch.setenv("HK_GATE_A", "1")

    def body(a, o):
        s = hk.alloc_shared(hk.st(bf16, 16, 16), name="s")
        t = hk.load_shared(s, hk.rt(bf16, 16, 16), wait=False)
        hk.store(o, hk.exp(t), hk.elem_coord(0, 0, 0, 0))

    body.__annotations__ = {"a": hk.GL[bf16], "o": hk.GL[bf16]}
    k = hk.kernel(body, warps=1, name="early_read", grid=lambda p: (1, 1, 1))
    with pytest.raises(VerifyError, match="whose LDS reads are still in flight"):
        k.trace()
