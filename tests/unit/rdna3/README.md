# HipKittens unit tests — RDNA3 (gfx1100)

Unit tests for `include/rdna3`, run on an RX 7900 / W7900 class card.

### Requirements

- A gfx1100 GPU (wave32, 64KB LDS per workgroup, 256 arch VGPRs per wave)
- ROCm 6.4 or newer (developed against 7.2.4)
- A C++20 compiler

### Building and running

```bash
make -j32
./unit_tests
```

`GPU_TARGET`, `COMP_LEVEL`, `TEST_INTENSITY` and `TEST_DEFINES` are set in the
Makefile here and default to `RDNA3`, `profile`, `2` and `-DTEST_ALL`. Raise
`TEST_INTENSITY` to sweep larger tiles; it costs compile time roughly
quadratically. To narrow the run, replace `-DTEST_ALL` with a subsection such as
`-DTEST_WARP_MEMORY` or a single test like
`-DTEST_WARP_REGISTER_TILE_MMA`.

`./unit_tests printout` dumps failing tensors into `outputs/`, which you have to
create first.

### How this tree differs from `tests/unit/cdna4`

It is a port of that tree, so most files are recognisable, but three things had
to change and are worth knowing before you add a test.

**One tile shape.** The CDNA trees parameterize tests over `ducks::rt_shape` and
`ducks::st_shape` because MFMA comes in several geometries. gfx11 has exactly
one matrix instruction, `v_wmma_*_16x16x16_*`, so `include/rdna3` declares one
member of each namespace and every sweep here passes `rt_16x16` / `st_16x16`.
The shape parameters remain in the test signatures — the harness in
`../common/testing_commons` is shared with the CDNA trees and threads them
through positionally — but they are not a degree of freedom on this
architecture.

**`swap_layout` means something else.** On CDNA the layout-swap test really
sweeps *shapes*, often with both layouts the same. Here the only thing
`swap_layout` can change is the layout, and the destination layout is pinned to
`transpose<L>`, so `test_swap_layout` sweeps row→col and col→row instead.

**Replication shows up in the vector tests.** WMMA operands are mirrored across
the two wave halves and accumulators are not, so a `float` tile and a `bf16`
tile of the same logical shape keep their values in different lanes. Anything
that converts between the two — `copy` on `rt_base`, `copy` on an
`rv_layout::align` vector — moves data rather than just casting it, and the
tests for those paths are the ones most likely to catch a layout regression.
`include/rdna3/ops/warp/register/tile/conversions.cuh` opens with the picture
the whole port is derived from; read it first.

**The mma tests count.** `mma_wrapper_2d` in `warp/register/tile/mma.cu` has its
own copy of the harness wrapper, and the cdna4 original sets `this_result` and
then never pushes it, so an mma failure prints but is missing from the tally at
the end. This tree pushes it. If you diff the two files, that is one of the
changes.

### Adding a test

Each `.cu` here wraps its scaffolding in an anonymous namespace. That is not
decoration: every file in the tree names its helpers the same way
(`test_generator`, `vec_load_store`, …), and as templates with external linkage
they collapse into one definition at link time, so whole sections silently run
some other file's tests. Keep the anonymous namespace, and keep
`<scope>::tests()` outside it.

The tree mirrors the layout of `include/rdna3`, which is how coverage gaps stay
visible.
