"""Which arch a kernel is built for, and fp8 WMMA tiles on the one that has them.

hk.ops no longer names gfx1100: a kernel that names no arch is built for
hk.target.current_arch() -- $HK_ARCH, else the GPU in the machine, else
gfx1100. fp8 register tiles and fp8 mma exist on gfx1201 only; gfx1100 refuses
them with the reason, at trace time. The gfx1201 build here is a compile gate:
no gfx12 part was available to run it.
"""

from __future__ import annotations

import shutil

import pytest

import hk
from hk import VerifyError, fp8e4m3, fp8e5m2, fp32
from hk.target import current_arch, get_target


def _fp8_kernel(arch, b_dtype=fp8e4m3, mma=True, name="fp8_probe"):
    def body(a, b, c):
        ta = a[hk.elem_coord(0, 0, hk.arch.block_idx.x * 16, 0)].load(
            hk.RT(fp8e4m3, 16, 64))
        acc = hk.zeros(hk.RT(fp32, 16, 16, "col"))
        if mma:
            tb = b[hk.elem_coord(0, 0, 0, 0)].load(hk.RT(b_dtype, 16, 64))
            acc += ta @ tb.T
        c[hk.elem_coord(0, 0, hk.arch.block_idx.x * 16, 0)] = acc

    body.__name__ = name
    body.__annotations__ = {"a": hk.Tensor[fp8e4m3], "b": hk.Tensor[b_dtype],
                            "c": hk.Tensor[fp32]}
    return hk.kernel(body, arch=arch, warps=1, name=name,
                     grid=lambda p: (p.a.rows // 16, 1, 1))


def test_ops_kernels_are_built_for_the_process_arch():
    from hk.ops import gemm

    k = gemm.KERNELS["gemm_bf16_128x128"]
    assert k.arch == current_arch()
    assert k.trace().arch == current_arch()
    assert current_arch() in ("gfx1100", "gfx1201")


def test_an_fp8_mma_on_gfx1100_says_where_it_exists():
    with pytest.raises(TypeError, match="gfx1201"):
        _fp8_kernel("gfx1100").trace()


def test_an_fp8_tile_on_gfx1100_is_refused_even_without_an_mma():
    with pytest.raises(VerifyError, match="storage type"):
        _fp8_kernel("gfx1100", mma=False, name="fp8_tile_only").trace()


def test_gfx1201_takes_fp8_tiles_and_mixed_fp8_operands():
    src = _fp8_kernel("gfx1201", b_dtype=fp8e5m2, name="fp8_mixed").source()
    assert "kittens::rt<fp8e4m3, 16, 64" in src
    assert "kittens::rt<fp8e5m2, 16, 64" in src
    assert "kittens::mma_ABt(" in src
    assert "fp8e4m3" in get_target("gfx1201").mma
    assert "fp8e4m3" not in get_target("gfx1100").mma


@pytest.mark.skipif(shutil.which("hipcc") is None and
                    not __import__("os").path.exists("/opt/rocm/bin/hipcc"),
                    reason="needs hipcc")
def test_an_fp8_kernel_compiles_for_gfx1201_without_spilling():
    r = _fp8_kernel("gfx1201", b_dtype=fp8e5m2, name="fp8_gate").build().kernels[0]
    assert r.scratch == 0 and r.vgpr_spill == 0
