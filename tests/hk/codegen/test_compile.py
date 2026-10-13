"""Generate, compile, and read the resource remarks back.

Needs hipcc; does not need a GPU or torch, because nothing here launches. That
is the point of this tier: the thing most likely to break -- generated code that
compiles but spills, or compiles for the wrong arch -- is caught on a laptop.

Cold, each kernel costs a few seconds of hipcc. The content-hash cache makes
every rerun free, so keep the cache directory between runs.
"""

from __future__ import annotations

import pytest

import hk
from hk import bf16, fp32
from hk.ops.elementwise import KERNELS
from hk.runtime import resources
from hk.runtime.compile import cache_key


@hk.kernel(
    arch="gfx1100",
    grid=lambda p: (hk.cdiv(p.o.cols, 64), hk.cdiv(p.o.rows, 16), 1),
)
def add2(a: hk.Tensor[bf16], b: hk.Tensor[bf16], o: hk.Tensor[bf16], *,
         ROWS=16, COLS=64):
    t = hk.RT(bf16, ROWS, COLS)
    idx = hk.tile_coord(0, 0, hk.arch.block_idx.y, hk.arch.block_idx.x)
    o[idx] = a[idx].load(t) + b[idx].load(t)


@pytest.fixture(scope="module")
def build():
    return add2.build()


def test_it_compiles_to_a_shared_object(build):
    assert build.so_path.exists()
    assert build.so_path.suffix == ".so"


def test_the_resource_remarks_are_parsed(build):
    """If this comes back empty the spill gate is vacuous -- it would pass
    every kernel, including the ones that spill."""
    assert build.kernels, "no -Rpass-analysis remarks were parsed"
    (k,) = [k for k in build.kernels if "add2" in k.name]
    assert k.scratch == 0 and not k.spills
    assert k.vgpr > 0 and (k.occ_by_reg or k.occupancy) > 0


def test_the_second_build_is_a_cache_hit(build):
    again = add2.build()
    assert again.key == build.key
    assert again.so_path == build.so_path


def test_the_cache_key_covers_the_compiler_flags():
    """The include-path bug this guards against was invisible for a while: the
    fix changed the flags but not the key, so the broken .so was served from
    cache and the fix looked like it had not worked."""
    t = hk.get_target("gfx1100")
    src = add2.source()
    assert cache_key(src, t) != cache_key(src, t, extra=("-DFOO=1",))


def test_a_different_constexpr_is_a_different_build():
    wide = add2.build(COLS=128)
    assert wide.key != add2.build().key
    assert "rt<bf16, 16, 128," in add2.source(COLS=128)


# -- the gates ----------------------------------------------------------------


def test_the_occupancy_gate_can_fail_a_build():
    """The no-spill rule is unconditional and (correctly) never fires on these
    kernels, so exercise the mechanism through the optional gate instead. A
    gate that has never been seen to fail is not known to work."""
    k = add2.specialize()
    k.min_occupancy = 99
    with pytest.raises(resources.ResourceError, match="min_occupancy"):
        k.build()


def test_the_vgpr_gate_can_fail_a_build():
    k = add2.specialize()
    k.max_vgprs = 4
    with pytest.raises(resources.ResourceError, match="max_vgprs"):
        k.build()


def test_a_failed_build_keeps_its_source():
    """The generated .cpp is the artifact you take to the disassembler; losing
    it on failure is losing the only thing worth looking at."""
    from hk.runtime.compile import CompileError

    @hk.kernel(arch="gfx1100", grid=lambda p: (1, 1, 1))
    def broken(o: hk.Tensor[bf16]):
        i = hk.tile_coord()
        o[i] = hk.zeros(hk.RT(bf16, 16, 16))

    src = broken.source().replace("kittens::store", "kittens::no_such_op")
    with pytest.raises(CompileError) as e:
        hk.runtime.build(src, "gfx1100", name="broken")
    assert "failed-" in str(e.value)


def test_an_unknown_arch_is_refused_rather_than_defaulted():
    """Guessing --offload-arch either fails at load time or, on a near-enough
    arch, computes the wrong answer."""
    with pytest.raises(KeyError, match="unknown arch"):
        hk.get_target("gfx9999")


# -- every shipped kernel -----------------------------------------------------


@pytest.mark.parametrize("name", sorted(KERNELS))
def test_shipped_kernel_builds_without_spilling(name):
    b = KERNELS[name].build()
    assert b.kernels
    for k in b.kernels:
        assert not k.spills, f"{name}: {k.line()}"
