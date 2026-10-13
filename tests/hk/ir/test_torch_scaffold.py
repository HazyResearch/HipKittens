"""The torch registration, as text.

No hipcc and no torch here: what this tier can check is that the generated
source says the right things -- in particular the schema, which is the one
part of a custom op that is wrong *silently*. A schema that forgets to mark an
output mutable compiles, loads, runs, and then gives wrong answers only under
`torch.compile`, which is exactly where it would be found last.
"""

import pytest

import hk
from hk import bf16, fp32
from hk.codegen import scaffold


@hk.kernel(arch="gfx1100",
           grid=lambda p: (hk.cdiv(p.o.cols, 64), hk.cdiv(p.o.rows, 16), 1))
def _add(a: hk.Tensor[bf16], b: hk.Tensor[bf16], o: hk.Tensor[bf16], *,
         ROWS=16, COLS=64):
    t = hk.RT(bf16, ROWS, COLS)
    idx = hk.tile_coord(0, 0, hk.arch.block_idx.y, hk.arch.block_idx.x)
    o[idx] = a[idx].load(t) + b[idx].load(t)


@hk.kernel(arch="gfx1100",
           grid=lambda p: (hk.cdiv(p.o.cols, 64), hk.cdiv(p.o.rows, 16), 1))
def _two_out(a: hk.Tensor[bf16], o: hk.Tensor[bf16], o2: hk.Tensor[fp32],
             *, ROWS=16, COLS=64):
    t = hk.RT(bf16, ROWS, COLS)
    idx = hk.tile_coord(0, 0, hk.arch.block_idx.y, hk.arch.block_idx.x)
    v = a[idx].load(t)
    o[idx] = v
    o2[idx] = v.to(fp32)


@hk.kernel(arch="gfx1100", grid=lambda p: (1, 1, 1))
def _no_store(a: hk.Tensor[bf16], *, ROWS=16, COLS=64):
    idx = hk.tile_coord(0, 0, 0, 0)
    a[idx].load(hk.RT(bf16, ROWS, COLS))


def test_the_output_is_the_tensor_the_kernel_stores_into():
    ir = _add.trace()
    assert scaffold.written_tensors(ir) == ["o"]


def test_outputs_are_marked_mutable_in_the_schema():
    assert scaffold.torch_schema(_add.trace()) == (
        "_add(Tensor a, Tensor b, Tensor(a!) o) -> ()"
    )


def test_two_outputs_get_different_alias_letters():
    # One letter for both would declare them aliases of each other, and torch
    # would be entitled to assume a write to one is a write to the other.
    s = scaffold.torch_schema(_two_out.trace())
    assert s == "_two_out(Tensor a, Tensor(a!) o, Tensor(b!) o2) -> ()"


def test_a_kernel_that_writes_nothing_is_refused():
    with pytest.raises(ValueError, match="stores into no global"):
        scaffold.torch_library(_no_store.trace())


def test_the_registration_uses_a_fragment_not_a_whole_namespace():
    # Every kernel is its own .so and they share one namespace; TORCH_LIBRARY
    # would make the second .so to load throw.
    src = scaffold.torch_library(_add.trace())
    assert "TORCH_LIBRARY_FRAGMENT(hk, m)" in src
    assert "TORCH_LIBRARY(hk" not in src


def test_the_launch_is_on_torchs_stream():
    src = scaffold.torch_library(_add.trace())
    assert "launch_on(g, c10::hip::getCurrentHIPStream());" in src
    # ... and never the no-stream form, which is the default stream.
    assert "launch(g)" not in src


def test_there_is_a_meta_implementation_so_torch_compile_can_trace_it():
    src = scaffold.torch_library(_add.trace())
    assert "TORCH_LIBRARY_IMPL(hk, Meta, m)" in src
    assert "TORCH_LIBRARY_IMPL(hk, CUDA, m)" in src


def test_each_tensor_is_checked_for_its_own_dtype():
    src = scaffold.torch_library(_two_out.trace())
    assert 'hk_check(o, "o", at::kBFloat16' in src
    assert 'hk_check(o2, "o2", at::kFloat' in src


def test_outputs_are_taken_by_mutable_reference_and_inputs_are_not():
    src = scaffold.torch_library(_add.trace())
    assert "const at::Tensor &a, const at::Tensor &b, at::Tensor &o" in src


def test_the_namespace_and_op_name_are_settable():
    src = scaffold.torch_library(_add.trace(), ns="myns", op_name="myop")
    assert "TORCH_LIBRARY_FRAGMENT(myns, m)" in src
    assert 'm.def("myop(Tensor a, Tensor b, Tensor(a!) o) -> ()");' in src
    assert "void myop_impl(" in src


def test_the_kernel_body_is_the_same_one_pybind_gets():
    # The whole point of scaffold.py being separate from the translation
    # (hk-translate): two deployments of one kernel, not two kernels.
    ir = _add.trace()
    body = scaffold.bare(ir)
    kernel_src = body[body.index("struct globals"):]
    assert kernel_src in scaffold.torch_library(ir)
    assert kernel_src in scaffold.pybind_module(ir)


def test_torch_is_a_registered_scaffold():
    assert scaffold.render(_add.trace(), "torch") == scaffold.torch_library(
        _add.trace()
    )


def test_a_shipped_op_renders():
    # An elementwise op from hk.ops, end to end through the real module rather
    # than the toy above.
    k = hk.ops.elementwise.KERNELS["add_bf16"]
    src = scaffold.torch_library(k.trace())
    assert "TORCH_LIBRARY_FRAGMENT(hk, m)" in src
    assert "-> ()" in scaffold.torch_schema(k.trace())


def test_warm_cache_can_build_the_torch_scaffold(monkeypatch):
    """The AOT half of the torch path.

    A torch registration is compiled with libtorch's include and ABI flags,
    and the flags are in the cache key -- so a cache full of pybind modules
    spares a server nothing. `warm_cache` therefore has to be able to ask for
    a different scaffold and different flags, and to pass both through
    unchanged; this checks that it does rather than silently warming the
    pybind entry under a torch-shaped label.
    """
    from hk.runtime import warm as warm_mod

    seen = {}

    def fake_build(source, arch, *, name, max_vgprs, min_occupancy,
                   extra_flags=()):
        seen["source"] = source
        seen["extra_flags"] = tuple(extra_flags)
        return object()

    monkeypatch.setattr(warm_mod, "build", fake_build)
    res = warm_mod.warm_cache([("add", _add, {})], workers=1,
                              scaffold="torch", extra_flags=("-DFOO",))
    assert res.ok, res.errors
    assert "TORCH_LIBRARY_FRAGMENT" in seen["source"]
    assert seen["extra_flags"] == ("-DFOO",)


def test_warm_cache_still_defaults_to_pybind(monkeypatch):
    from hk.runtime import warm as warm_mod

    seen = {}

    def fake_build(source, arch, *, name, max_vgprs, min_occupancy,
                   extra_flags=()):
        seen["source"] = source
        seen["extra_flags"] = tuple(extra_flags)
        return object()

    monkeypatch.setattr(warm_mod, "build", fake_build)
    warm_mod.warm_cache([("add", _add, {})], workers=1)
    assert "PYBIND11_MODULE" in seen["source"]
    assert seen["extra_flags"] == ()
