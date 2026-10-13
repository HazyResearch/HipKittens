"""`x[idx].load_scalar()`: indexing by a value fetched from memory.

This is the primitive a paged kernel needs and the one the IR did not have.
`block_idx` and arithmetic on it can express "block number 7"; a page table
can only be expressed by reading an integer out of a tensor and using it as an
address. The tests here are a gather -- `out[i] = x[table[i]]` -- because that
is the smallest thing that cannot be written without it.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import hk  # noqa: E402

pytestmark = pytest.mark.gpu

ROWS, COLS = 16, 64


def _gather_kernel():
    def body(table: hk.Tensor[hk.i32], x: hk.Tensor[hk.bf16],
             o: hk.Tensor[hk.bf16]):
        # One workgroup per output tile. Which input tile it reads is a
        # number in memory, not a number the grid knows.
        src = table[hk.elem_coord(0, 0, 0, hk.arch.block_idx.z)].load_scalar()
        t = hk.RT(hk.bf16, ROWS, COLS)
        o[hk.elem_coord(hk.arch.block_idx.z, 0, 0, 0)] = (
            x[hk.elem_coord(src, 0, 0, 0)].load(t))

    return hk.kernel(body, arch="gfx1100", warps=1, name="gather_probe",
                     grid=lambda p: (1, 1, p.o.batch))


def _run(perm):
    n = len(perm)
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(n, 1, ROWS, COLS, device="cuda", dtype=torch.bfloat16,
                    generator=g)
    table = torch.tensor(perm, device="cuda", dtype=torch.int32)
    out = torch.empty_like(x)
    _gather_kernel()(table, x, out)
    return x, out


def test_a_permutation_is_followed():
    perm = [3, 0, 2, 1]
    x, out = _run(perm)
    for i, src in enumerate(perm):
        assert torch.equal(out[i], x[src]), f"row {i} took the wrong page"


def test_the_identity_is_the_identity():
    x, out = _run(list(range(5)))
    assert torch.equal(out, x)


def test_repeated_indices_are_allowed():
    # A page table may point two requests at the same block; nothing in the
    # op says the indices are distinct.
    perm = [2, 2, 2]
    x, out = _run(perm)
    assert torch.equal(out[0], x[2]) and torch.equal(out[2], x[2])


def test_the_index_is_uniform_so_the_kernel_does_not_spill():
    # The reason the op exists in this form. A page index carried per lane
    # costs a VGPR for every value derived from it, and on this architecture a
    # spill in a hand-scheduled kernel is a wrong answer, not a slow one.
    b = _gather_kernel().build()
    k = b.kernels[0]
    assert k.scratch == 0, f"scratch {k.scratch}"
