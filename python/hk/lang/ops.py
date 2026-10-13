"""Tile ops.

Each one records an Op and returns a Value. The names mirror
include/rdna3/ops/warp -- `row_max` here is `kittens::row_max` there -- so that
a kernel translated from C++ reads the same, and so that the emitter's job stays
a lookup rather than a translation.

Value-style (`c = a + b`) rather than the destination-passing style the C++ uses
(`add(c, a, b)`): the IR wants to know what is live, and destination-passing
hides that. The emitter turns values back into destinations when it writes C++.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

from ..ir import builder
from ..ir.nodes import (
    CoordType,
    DType,
    GlobalType,
    GranuleSetType,
    RegTileType,
    RegVecType,
    ScalarType,
    SharedTileType,
    SharedVecType,
    StageBufferType,
    Value,
    bf16,
    fp32,
    i32,
)

Scalar = Union[int, float, Value]


def _b():
    return builder.current()


# ---------------------------------------------------------------- types


def rt(dtype: DType, rows: int, cols: int, layout: str = "row") -> RegTileType:
    """A register tile *type*. Pass it to load/zeros to get a value."""
    return RegTileType(dtype, rows, cols, layout)


def st(dtype: DType, rows: int, cols: int, count: int = 1) -> SharedTileType:
    """A shared (LDS) tile type. `count` > 1 is a multi-buffer."""
    return SharedTileType(dtype, rows, cols, count)


def col_vec(tile: RegTileType) -> RegVecType:
    """The type of one value per *row* of `tile` -- what row_max/row_sum give
    back, and what add_row/mul_row take."""
    return RegVecType(tile, "col")


def row_vec(tile: RegTileType) -> RegVecType:
    """The type of one value per *column* of `tile`."""
    return RegVecType(tile, "row")


# ---------------------------------------------------------------- indices


class _BlockIdx:
    def __getattr__(self, axis: str) -> Value:
        if axis not in ("x", "y", "z"):
            raise AttributeError(f"block_idx has no axis {axis!r}")
        return _b().hoisted(
            ("block_idx", axis), "block_idx", ScalarType(i32),
            name=f"bid_{axis}", axis=axis,
        )


block_idx = _BlockIdx()


def warp_id() -> Value:
    """This warp's index within the workgroup."""
    return _b().hoisted("warp_id", "warp_id", ScalarType(i32), name="warp_id")


def lane_id() -> Value:
    return _b().hoisted("lane_id", "lane_id", ScalarType(i32), name="lane_id")


def coord(b: Scalar = 0, d: Scalar = 0, r: Scalar = 0, c: Scalar = 0, *, unit: str = "tile") -> Value:
    """A (batch, depth, row, col) index.

    unit="tile" indexes in whole tiles (C++ `coord<RT>`), unit="element" in
    elements (`coord<>`). The two are not interchangeable and picking the wrong
    one reads the wrong memory without any diagnostic, which is why the unit is
    part of the value's type rather than a convention.
    """
    if unit not in ("tile", "element"):
        raise ValueError(f"coord unit must be 'tile' or 'element', got {unit!r}")
    return _b().emit(
        "coord",
        [_as_value(x, i32) for x in (b, d, r, c)],
        result_type=CoordType(unit),
        name="idx",
    )


def tile_coord(b: Scalar = 0, d: Scalar = 0, r: Scalar = 0, c: Scalar = 0) -> Value:
    return coord(b, d, r, c, unit="tile")


def elem_coord(b: Scalar = 0, d: Scalar = 0, r: Scalar = 0, c: Scalar = 0) -> Value:
    return coord(b, d, r, c, unit="element")


def _as_value(x: Scalar, dtype: DType) -> Value:
    if isinstance(x, Value):
        return x
    return _b().constant(x, ScalarType(dtype))


# ---------------------------------------------------------------- extents
#
# The host-side grid lambda reads extents off `p`; these are the device-side
# equivalent, for the loop bounds and tail arithmetic inside the kernel.

_AXES = ("batch", "depth", "rows", "cols")


def _extent(t: Value, axis: str) -> Value:
    _expect(t, GlobalType, f"{axis}() argument")
    return _b().emit("extent", [t], result_type=ScalarType(i32), name=axis, axis=axis)


def batch(t: Value) -> Value:
    """The tensor's first extent, at runtime. `g.x.batch()`."""
    return _extent(t, "batch")


def depth(t: Value) -> Value:
    return _extent(t, "depth")


def rows(t: Value) -> Value:
    return _extent(t, "rows")


def cols(t: Value) -> Value:
    """The tensor's last extent, at runtime -- the hidden dimension, usually."""
    return _extent(t, "cols")


# ---------------------------------------------------------------- scalars


#: Device-side integer arithmetic. Separate from the tile maps because they are
#: a different machine: these are SALU on one uniform value, the maps are VALU
#: across a whole tile, and giving them the same name would hide which one a
#: line costs. The operators on a scalar Value route here (see Value._bin), so
#: `cb * COLS` in a kernel is an s_mul.
SCALARS = ("add", "sub", "mul", "div", "mod", "min", "max", "cdiv",
           "lt", "le", "gt", "ge", "eq", "ne")

#: The subset of SCALARS that yields 0/1 rather than a number. Kept separate so
#: `hk.if_` can insist on one: `if_(q_row)` is a plausible typo for
#: `if_(s_gt(q_row, 0))` and would silently invert on the row-zero workgroup.
PREDICATES = ("lt", "le", "gt", "ge", "eq", "ne")


def _scalar_binary(op: str, a: Scalar, b: Scalar) -> Value:
    va = _as_value(a, i32)
    vb = _as_value(b, i32)
    for v, side in ((va, "left"), (vb, "right")):
        if not isinstance(v.type, ScalarType):
            raise TypeError(
                f"s_{op} {side} operand is {v.type}; scalar arithmetic is for "
                f"indices and loop bounds, not tiles. Use hk.{op} for a tile."
            )
    return _b().emit(f"s_{op}", [va, vb], result_type=va.type, name=op)


def _make_scalar(name):
    def f(a: Scalar, b: Scalar) -> Value:
        return _scalar_binary(name, a, b)

    f.__name__ = f"s_{name}"
    return f


for _n in SCALARS:
    globals()[f"s_{_n}"] = _make_scalar(_n)
del _n

s_min.__doc__ = (
    "Integer min. The tail idiom: `s_min(cb * COLS, cols - COLS)` backs the "
    "last block up so every access stays in bounds. What it does *not* fix is "
    "double counting -- see left_fill."
)
s_cdiv.__doc__ = "Ceiling division, as a device-side expression."
for _n in PREDICATES:
    globals()[f"s_{_n}"].__doc__ = (
        f"Integer comparison, 0 or 1. The only thing to pass to hk.if_ -- see "
        f"hk.lang.control.if_ for why a bare index is refused."
    )
del _n


# ---------------------------------------------------------------- memory


def load(src: Value, idx: Value, tile, *, out: Optional[Value] = None,
         bounded: bool = False) -> Value:
    """Global -> register tile, or global -> register vector.

    One function for both because `kittens::load` is one overload set, and
    because the vector case is not exotic: a per-row scale or a per-row
    softmax statistic is a vector in memory and a vector in registers, and
    routing it through a 1-column tile would change its lane layout.

    The element type converts on the way in, for free -- a bf16 tensor loaded
    into an fp32 tile costs exactly what a bf16 tile would. That is the
    sanctioned way to get fp32 accumulation.

    `bounded=True` predicates the tile on `src`'s rows: rows past the last one
    read as zero and their memory is never touched, so a tile may hang off the
    end of the tensor -- the last 16 rows of a 5-row batch, say. It is opt-in
    because it is not free in registers: always on, it cost the paged decode
    kernel 66 VGPRs of spill for tiles that are never ragged.
    """
    _expect(src, GlobalType, "load source")
    _expect(idx, CoordType, "load index")
    if not isinstance(tile, _ELEMENTWISE):
        raise TypeError(f"load destination type is {tile!r}, expected a tile or a vector")
    if bounded and not isinstance(tile, RegTileType):
        raise TypeError(f"load(bounded=True) predicates tile rows; {tile!r} is not a tile")
    # `out=` matters here for the same reason it does on load_shared: a global
    # read issued in one loop iteration and consumed in the next has to land in
    # a register that exists outside the loop. Without it the prefetch is
    # unwriteable -- the value traced inside the body is a different value every
    # iteration, and the IR rejects one escaping.
    return _emit("load_global", [src, idx], tile, out=out, name="ld",
                 bounded=bool(bounded))


def store(dst: Value, val: Value, idx: Value, *, bounded: bool = False) -> None:
    """Register tile or vector -> global.

    `bounded=True` writes only the tile's rows that lie inside `dst`; see
    `load`. This is what lets a GEMM's last M block run past M instead of
    backing up over its neighbour.
    """
    _expect(dst, GlobalType, "store destination")
    _expect(idx, CoordType, "store index")
    _expect(val, _ELEMENTWISE, "store value")
    if bounded and not isinstance(val.type, RegTileType):
        raise TypeError(f"store(bounded=True) predicates tile rows; {val.type!r} is not a tile")
    _b().emit("store_global", [dst, val, idx], bounded=bool(bounded))


def store_scalar(dst: Value, val: Value, idx: Value) -> None:
    """One entry of a *uniform* register vector -> one element of a global.

    The op a folded reduction needs and nothing else does. `store` writes the
    whole vector -- 16 consecutive elements -- which is right when a tile's 16
    rows are 16 rows of the tensor. Under FOLD they are 16 chunks of one row,
    the workgroup's answer is a single number, and there is exactly one slot
    in the output for it.

    Requiring `val.type.uniform` is the whole point of the op existing
    separately. Writing one entry of a vector whose entries differ picks an
    arbitrary lane's value and is silently wrong on every row; the type says
    which vectors that cannot happen to, and the only thing that produces one
    is lang.collective.fold_rows.

    `idx` is in elements. Every warp's lane 0 writes, so a W-warp workgroup
    issues W identical stores to the same address -- the same benign duplicate
    the library already relies on for a backed-up block, and cheaper than
    branching the workgroup on warp 0.
    """
    _expect(dst, GlobalType, "store_scalar destination")
    _expect(idx, CoordType, "store_scalar index")
    _expect(val, RegVecType, "store_scalar value")
    if idx.type.unit != "element":
        raise TypeError(
            f"store_scalar takes an element coordinate, got {idx.type}. It "
            f"writes one element, so a tile-unit index would be off by the "
            f"tile size."
        )
    if not val.type.uniform:
        raise TypeError(
            f"store_scalar needs a uniform vector, got {val.type}. Every entry "
            f"of a uniform vector holds the same value, so writing one of them "
            f"is writing the answer; writing one entry of {val.type} would pick "
            f"an arbitrary row's value and be wrong on the other fifteen -- "
            f"silently, since the shapes all agree. Uniformity comes from "
            f"hk.fold_rows, which is what a FOLD'd kernel already calls to "
            f"combine its row; carry that vector here instead of re-deriving "
            f"one, or use hk.store if the sixteen entries really are sixteen "
            f"rows."
        )
    _b().emit("store_scalar", [dst, val, idx])


def load_scalar(src: Value, idx: Value) -> Value:
    """One element of a global -> a wave-uniform integer.

    The read counterpart of `store_scalar`, and the primitive a paged kernel
    needs: `block_table[req][i]`, `seq_lens[req]` and `slot_mapping[t]` are all
    "fetch an integer from memory, then index with it". Without it the IR can
    only index by `block_idx` and arithmetic on it, which cannot express a page
    table at all.

    The result is a plain scalar, so it composes with `s_add`, `elem_coord`
    and `hk.range` exactly like `block_idx` does -- a loop whose trip count is
    a loaded value, or a tile coordinate whose batch index is a page number,
    needs nothing further.

    Uniform by construction (`readfirstlane` in the emitted call): the value is
    the same for every lane, which is what keeps the address arithmetic built
    on it in SGPRs. A page index that is accidentally per-lane costs a VGPR for
    every value derived from it, and on this architecture that is how a kernel
    that fits becomes a kernel that spills.
    """
    _expect(src, GlobalType, "load_scalar source")
    _expect(idx, CoordType, "load_scalar index")
    if idx.type.unit != "element":
        raise TypeError(
            f"load_scalar takes an element coordinate, got {idx.type}. It "
            f"reads one element, so a tile-unit index would be off by the "
            f"tile size -- use hk.elem_coord."
        )
    if src.type.dtype not in (i32,):
        raise TypeError(
            f"load_scalar reads an integer index, so the global must be i32, "
            f"not {src.type.dtype.name}. A page table, a sequence length and a "
            f"slot mapping are all int32 tensors; if you want a float out of "
            f"memory, that is hk.load into a vector."
        )
    return _b().emit("load_scalar", [src, idx], ScalarType(i32),
                     name="scalar")


def lds_loads(tile: RegTileType, src: SharedTileType) -> int:
    """How many ds_read instructions `load_shared` issues for this pair.

    Zero means the pair misses the vectorised path and goes through the
    elementwise fallback, which is 16 ds_read_u16 per base tile instead of 2
    ds_read_b128 -- and, more importantly here, cannot be issued
    asynchronously at all. Transcribed from `lds_vectorizable` and `lds_loads`
    in include/rdna3/ops/warp/memory/tile/shared_to_register.cuh.
    """
    ok = (tile.layout == "row" and tile.dtype.bits == 16
          and src.dtype == tile.dtype)
    return (tile.rows // 16) * (tile.cols // 16) * 2 if ok else 0


def load_shared(src: Value, tile: RegTileType, *, wait: bool = True,
                out: Optional[Value] = None) -> Value:
    """LDS -> register tile.

    `wait=False` issues the reads and returns without retiring them, which is
    how a kernel overlaps one K-slice's reads with the previous slice's math.
    It comes with an obligation: the tile must be named in an `lds_wait_for`
    before anything reads it. That is checked -- see hk.ir.passes.lds_pipeline
    -- because it is the single most expensive mistake this architecture
    offers. `s_waitcnt` carries no register dependence, so a WMMA scheduled
    above the wait is not a compile error, not a spill, and not deterministic;
    it is a wrong answer at full speed with ScratchSize still 0.

    Hitting the vectorised path depends on `tile`'s layout; hk.ir.verify warns
    about a miss, and `wait=False` on a pair that misses it is an error rather
    than a silent downgrade, because the C++ static_asserts it.
    """
    _expect(src, SharedTileType, "shared load source")
    if not isinstance(tile, RegTileType):
        raise TypeError(f"load_shared destination type is {tile!r}, expected an rt")
    st = src.type
    if (tile.rows, tile.cols) != (st.rows, st.cols):
        raise TypeError(
            f"load_shared: {tile} from {st}. kittens::load static_asserts that "
            f"the register and shared tiles have the same height and width; a "
            f"sub-tile is a different shared tile, not a smaller load."
        )
    n = lds_loads(tile, st)
    if not wait and not n:
        raise TypeError(
            f"load_shared(wait=False) from {st} into {tile}: this pair does not "
            f"take the vectorised path, so there is nothing to keep in flight "
            f"-- the fallback goes through the compiler, which places its own "
            f"waits. Asynchronous reads need a 'row' layout 16-bit register "
            f"tile whose element type matches the shared tile's."
        )
    return _emit("load_shared", [src], tile, out=out, name="lds",
                 wait=wait, lds_loads=n)


def granules(src: Value, name: str = "gr") -> Value:
    """Factor one LDS tile's addressing into `swizzle_bytes/16` registers.

    Use this instead of `subtile` + `load_shared` whenever a loop reads many
    fragments out of the same shared tile. Those addresses are loop-invariant,
    so the compiler hoists every one of them out of the loop and then spills
    them -- 312 B/lane in this IR's first attention kernel, 372 in the
    handwritten one before the same fix. `load_frag` off a granule set folds
    the fragment coordinate into ds_read_b128's immediate offset instead, where
    it costs nothing.

    The set is per-lane and per-tile. It is register pressure for as long as it
    is live (`GranuleSetType.n` registers), which the budget model counts.
    """
    _expect(src, SharedTileType, "granule source")
    return _b().emit("granules", [src],
                     result_type=GranuleSetType(src.type), name=name)


def _frag_coord(gr: Value, row, col, what: str):
    st = gr.type.tile
    if not isinstance(col, int):
        raise TypeError(
            f"{what}: col={col!r} must be a Python int. The column is the "
            f"granule selector and half of ds_read_b128's immediate; a runtime "
            f"column has no decomposition and would need its own address."
        )
    rows_ok = isinstance(row, int)
    if not rows_ok and what == "load_frag":
        raise TypeError(
            f"load_frag: row={row!r} must be a Python int. Only the *write* "
            f"side takes a runtime row -- there the row is a warp's slice of "
            f"the head dimension, and it is passed as an addend. Reading at a "
            f"runtime row would hoist one address per site, which is the thing "
            f"this op exists to avoid."
        )
    for n, v, whole in (("row", row, st.rows), ("col", col, st.cols)):
        if isinstance(v, int) and not (0 <= v and 16 * v + 16 <= whole):
            raise ValueError(
                f"{what}: {n}={v} is out of {st}. Fragment coordinates are in "
                f"units of 16."
            )
    return rows_ok


def load_frag(gr: Value, row: int, col: int, tile: RegTileType, *,
              out: Optional[Value] = None) -> Value:
    """One 16x16 operand fragment out of a granule set, at fragment (row, col).

    Two `ds_read_b128`, neither of which waits -- there is no `wait=True` form
    because the whole point is to batch the waits, and `lds_wait_for` is how
    they are retired. Everything `load_shared(wait=False)` obliges you to do,
    this obliges you to do.
    """
    _expect(gr, GranuleSetType, "load_frag granules")
    if not isinstance(tile, RegTileType):
        raise TypeError(f"load_frag destination type is {tile!r}, expected an rt")
    st = gr.type.tile
    if (tile.rows, tile.cols) != (16, 16) or tile.layout != "row" or tile.dtype != st.dtype:
        raise TypeError(
            f"load_frag into {tile} from {st}: lds_read_frag writes eight "
            f"contiguous elements of one row into data[0..7], which is a 16x16 "
            f"'row' tile of the shared tile's own dtype and nothing else."
        )
    _frag_coord(gr, row, col, "load_frag")
    return _emit("load_frag", [gr], tile, out=out, name="frag",
                 row=row, col=col, wait=False, lds_loads=2)


def store_frag(gr: Value, val: Value, row: Scalar, col: int) -> None:
    """One 16x16 fragment into a granule set's tile, at fragment (row, col).

    `row` may be a runtime value and `col` may not, for the reason
    `_frag_coord` gives: the column picks the granule, the row is an addend.
    """
    _expect(gr, GranuleSetType, "store_frag granules")
    _expect(val, RegTileType, "store_frag value")
    st = gr.type.tile
    t = val.type
    if (t.rows, t.cols) != (16, 16) or t.layout != "row" or t.dtype != st.dtype:
        raise TypeError(
            f"store_frag of {t} into {st}: lds_write_frag reads eight "
            f"contiguous elements out of data[0..7], which is a 16x16 'row' "
            f"tile of the shared tile's own dtype and nothing else."
        )
    const_row = _frag_coord(gr, row, col, "store_frag")
    ops = [gr, val] if const_row else [gr, val, _as_value(row, i32)]
    _b().emit("store_frag", ops, row=(row if const_row else None), col=col)


def lds_wait_for(*tiles: Value) -> None:
    """Retire the outstanding LDS reads of `tiles` and re-bind their registers.

    There is deliberately no `n`: the count is derived, because getting it
    wrong by one is a wrong answer and there is no reason a human should be
    computing it. LDS returns in order, so retiring a tile means waiting until
    only the reads issued *after* it are still outstanding, and the pass that
    tracks the issue queue knows that number exactly. See
    hk.ir.passes.lds_pipeline.

    There is also deliberately no bare `lds_wait`. The wait alone does not
    order a consuming WMMA against the reads it is waiting on -- see
    `lds_bind` in include/rdna3/ops/warp/memory/util/util.cuh -- so the only
    form this DSL offers is the one that names the tiles it retires.
    """
    if not tiles:
        raise TypeError(
            "lds_wait_for() needs the tiles it retires. A wait that names "
            "nothing is the bug this op exists to prevent: it retires the "
            "reads without putting back the register dependence, and the "
            "consuming WMMA is then free to schedule above it."
        )
    for t in tiles:
        _expect(t, RegTileType, "lds_wait_for operand")
    _b().emit("lds_wait_for", list(tiles))


def store_shared(dst: Value, val: Value) -> None:
    """Register tile -> LDS."""
    _expect(dst, SharedTileType, "shared store destination")
    _expect(val, RegTileType, "shared store value")
    _b().emit("store_shared", [dst, val])


def alloc_shared(tile: SharedTileType, name: str = "smem",
                 count: Optional[int] = None) -> Value:
    """Reserve an LDS tile, or `count` of them contiguously.

    Offsets are assigned by the lds_alloc pass, which is also what checks the
    64 KB budget. A multi-buffer is one allocation and not N because the two
    buffers of a double-buffered pair have to differ in exactly one address
    bit for the XOR swap to be legal, and only a single size-aligned
    allocation guarantees that.

    Index it to get one buffer: `buf[tic]`. The index may be a runtime value.
    """
    if count is not None:
        tile = SharedTileType(tile.dtype, tile.rows, tile.cols, count)
    return _b().emit("alloc_shared", result_type=tile, name=name)


def shared_at(buf: Value, index: Scalar) -> Value:
    """One buffer of a multi-buffer LDS allocation. Also spelled `buf[i]`."""
    _expect(buf, SharedTileType, "shared buffer")
    ty = buf.type
    if ty.count == 1:
        raise TypeError(
            f"{buf} is a single LDS tile, not a stack of them -- indexing it "
            f"would be `As[0]` on something declared `st As;`. Pass count= to "
            f"alloc_shared if you meant a multi-buffer."
        )
    if isinstance(index, int) and not 0 <= index < ty.count:
        raise IndexError(f"buffer {index} of {buf}, which holds {ty.count}")
    idx = _as_value(index, i32)
    return _b().emit("shared_at", [buf, idx], result_type=ty.element(), name="sb")


# ---------------------------------------------------------------- cross-warp
#
# Everything above this line is per-warp. `row_sum` reduces within one wave and
# nothing in its type says which wave, so a workgroup that splits a row across
# its warps holds W partial answers and no way to see the other W-1. These
# three ops plus the barrier are the only way across, and they are deliberately
# low-level: the thing you want is lang.collective.cross_warp, which is built
# out of them and gets the barrier placement right.


def shared_vec(vt: RegVecType, count: int = 1) -> SharedVecType:
    """The LDS type a register vector of type `vt` round-trips through.

    Derived from the rv rather than spelled independently because
    `kittens::load(rv, sv)` static_asserts `SV::length == RV::length` and a
    mismatch is a 40-line template error naming neither.
    """
    if not isinstance(vt, RegVecType):
        raise TypeError(f"shared_vec takes a register vector type, got {vt!r}")
    return SharedVecType(vt.dtype, vt.length, count)


def alloc_shared_vec(ty: SharedVecType, name: str = "svec") -> Value:
    """Reserve `ty.count` LDS vectors, contiguous. Same allocator, same budget
    check, as alloc_shared."""
    if not isinstance(ty, SharedVecType):
        raise TypeError(f"alloc_shared_vec takes a SharedVecType, got {ty!r}")
    return _b().emit("alloc_shared_vec", result_type=ty, name=name)


def store_shared_vec(dst: Value, val: Value, index: Scalar = 0) -> None:
    """Register vector -> one entry of an LDS vector stack."""
    _expect(dst, SharedVecType, "shared vector store destination")
    _expect(val, RegVecType, "shared vector store value")
    _b().emit("store_shared_vec", [dst, val, _as_value(index, i32)])


def load_shared_vec(src: Value, vt: RegVecType, index: Scalar = 0) -> Value:
    """One entry of an LDS vector stack -> a register vector.

    Every lane of every warp may read the same entry; LDS broadcasts, so a
    W-way combine costs W reads and not W*32.
    """
    _expect(src, SharedVecType, "shared vector load source")
    if not isinstance(vt, RegVecType):
        raise TypeError(f"load_shared_vec destination type is {vt!r}, expected an rv")
    return _b().emit(
        "load_shared_vec", [src, _as_value(index, i32)], result_type=vt, name="ldv"
    )


def subtile(src: Value, rows: int, cols: int,
            row: Scalar = 0, col: Scalar = 0) -> Value:
    """A `rows` x `cols` window of an LDS tile, indexed in units of itself.

    `kittens::subtile_inplace`. This is how a warp gets its slice of a
    workgroup-sized block: eight warps share one 128x32 A-tile in LDS, and each
    reads the 32x16 piece its accumulator covers. The window is a view -- no
    copy, no allocation -- so the shared tile's swizzle still applies and
    `load_shared` still hits the vectorised path.

    `row` and `col` may be runtime values; they usually are, since one of them
    is normally derived from `warp_id`.
    """
    _expect(src, SharedTileType, "subtile source")
    st = src.type
    if st.count != 1:
        raise TypeError(f"subtile of {st}, a stack of {st.count}. Index it first.")
    for n, v, whole in (("rows", rows, st.rows), ("cols", cols, st.cols)):
        if v <= 0 or v % 16:
            raise ValueError(f"subtile {n}={v} must be a positive multiple of 16")
        if whole % v:
            raise ValueError(
                f"subtile {n}={v} does not divide the {whole} of {st}. "
                f"subtile_inplace indexes in units of the subtile, so a "
                f"window that does not tile the parent has no index to give."
            )
    return _b().emit(
        "subtile", [src, _as_value(row, i32), _as_value(col, i32)],
        result_type=SharedTileType(st.dtype, rows, cols), name="sub",
    )


def setprio(level: int) -> None:
    """`s_setprio`. Raise this wave's issue priority for the next few ops.

    The one scheduling hint in the DSL, and it is here because it measurably
    matters on this target: bracketing a WMMA sequence with setprio(1)/setprio(0)
    keeps a wave that is issuing dense matrix ops from being interleaved out by
    one that is only issuing memory, which is worth a few percent in the GEMM.
    It has no effect on correctness in either direction.
    """
    if not isinstance(level, int) or not 0 <= level <= 3:
        raise TypeError(f"setprio({level!r}): the priority field is 2 bits, 0..3")
    _b().emit("setprio", level=level)


def sched_barrier() -> None:
    """`__builtin_amdgcn_sched_barrier(0)`: nothing crosses this line.

    Not a memory fence and not a wait -- it costs no instruction at all. It
    bounds the *scheduler*, and the reason it is in the op set rather than left
    to the compiler is register allocation, not latency. The attention kernel
    has a staging half and a math half in the same straight-line iteration;
    with nothing between them LLVM hoists the staging loads up into the math,
    both live sets are held at once, and the allocator gives up -- 256 VGPRs
    and ~480 bytes/lane of scratch. On a kernel that hand-manages s_waitcnt
    that is not slower, it is wrong. The overlap given up was never the point:
    staging is hidden across waves, by the SIMD having another wave to run.
    """
    _b().emit("sched_barrier")


# ------------------------------------------------------------------ staging
#
# gfx11 has no global->LDS DMA. The asynchronous path is a pair of ops with the
# bytes parked in VGPRs in between, which is why there are three names here and
# not one: the register buffer has a lifetime, and that lifetime is the thing
# the schedule is built out of.


def stage_buffer(dst: Value, threads: int, depth: int = 1,
                 name: str = "buf") -> Value:
    """Declare the register buffer for a group copy into `dst`.

    Takes the LDS allocation rather than its type, for two reasons. The buffer
    size is `stage_calls` of the tile shape, so deriving it from the
    destination is the only way the two cannot disagree -- and a buffer one
    float4 short is an out-of-bounds write to the stack, not an error anyone
    would see. The second reason is that the C++ needs the destination as a
    template argument it can deduce `ST` from, and an `st` is an LDS object:
    the emitter cannot conjure one, so the IR has to carry it.

    Declared, not produced: this op emits an array declaration and nothing
    else, and `stage_load` writes into it. It has to work that way because the
    point of staging is that the load and the store are in *different* loop
    iterations, and a value produced inside a traced loop body cannot be read
    by the next pass through it.

    Put it at the top of the kernel, outside any `hk.range`.
    """
    _expect(dst, SharedTileType, "stage_buffer destination")
    ty = StageBufferType(dst.type.element(), threads, depth)
    return _b().emit("stage_buffer", [dst], result_type=ty, name=name)


def stage_load(buf: Value, src: Value, idx: Value, slot: int = 0) -> None:
    """Issue the global loads for one K-tile into `buf`. Does not wait.

    `slot` indexes the buffer's depth and must be a Python int: a runtime slot
    index puts the whole array in scratch, which on this target is not slow,
    it is a different kernel.
    """
    _expect(buf, StageBufferType, "stage_load buffer")
    _expect(src, GlobalType, "stage_load source")
    _expect(idx, CoordType, "stage_load index")
    ty = buf.type
    if not isinstance(slot, int) or not 0 <= slot < ty.depth:
        raise TypeError(
            f"stage_load slot={slot!r}: must be a Python int in [0, {ty.depth}). "
            f"It is a register array index, so it is resolved at compile time "
            f"or it is not resolved at all."
        )
    _b().emit("stage_load", [buf, src, idx], slot=slot)


def stage_commit(dst: Value, buf: Value, slot: int = 0, wait: bool = False) -> None:
    """Write a staged buffer to LDS.

    `wait=False` -- the default, and what the GEMM uses -- leaves the ds_writes
    outstanding. Whatever reads that buffer next must be separated from this by
    a barrier *and* by a wait that retires the writes, because `s_barrier` does
    not order memory. Pass `wait=True` to drain here instead.

    You must call `vm_wait` before this: the bytes have to have landed in the
    registers it is about to write out. That is checked by the lds_pipeline
    pass rather than left to the author, for the usual reason -- missing it is
    a wrong answer, not a crash.
    """
    _expect(dst, SharedTileType, "stage_commit destination")
    _expect(buf, StageBufferType, "stage_commit buffer")
    ty = buf.type
    if dst.type.count != 1:
        raise TypeError(
            f"stage_commit into {dst}, which is a stack of {dst.type.count}. "
            f"Index it: `stage_commit(As[toc], buf)`."
        )
    if (dst.type.rows, dst.type.cols, dst.type.dtype) != (
            ty.tile.rows, ty.tile.cols, ty.tile.dtype):
        raise TypeError(
            f"stage_commit: buffer stages {ty.tile} but the destination is "
            f"{dst.type}. The buffer size is computed from the tile shape, so "
            f"these cannot differ -- a mismatch writes past the array."
        )
    if not isinstance(slot, int) or not 0 <= slot < ty.depth:
        raise TypeError(f"stage_commit slot={slot!r}: must be an int in [0, {ty.depth})")
    _b().emit("stage_commit", [dst, buf], slot=slot, wait=wait)


def vm_wait(keep: int = 0) -> None:
    """Retire outstanding vector-memory loads, leaving `keep` in flight.

    `s_waitcnt vmcnt(keep)`. The counter is 6 bits, so `keep` is bounded at 63;
    a deeper pipeline than that cannot be expressed in the instruction and the
    check here is the only place that says so.

    Unlike `lds_wait_for` this one takes a number and not a list of tiles,
    which is not an inconsistency: the staged bytes go to a `float4` array that
    nothing reads directly, so there is no register dependence to re-establish
    -- the dependence that matters is on the buffer, and `stage_commit` has it
    as an operand.
    """
    if not isinstance(keep, int) or not 0 <= keep <= 63:
        raise TypeError(
            f"vm_wait(keep={keep!r}): vmcnt is 6 bits, so keep must be an int "
            f"in [0, 63]. A pipeline deeper than that has to be restructured, "
            f"not expressed."
        )
    _b().emit("vm_wait", keep=keep)


def wg_arrive(counter: Value, idx: Value, expected: Scalar) -> Value:
    """Count this workgroup's arrival at `counter[idx]`; return how many
    workgroups arrived there before it.

    The reduction across workgroups the library otherwise has no way to say:
    each workgroup stores its partial result, arrives, and the one that gets
    back `expected - 1` arrived last and finishes -- reading the partials in a
    fixed order, so the answer does not depend on which workgroup that was.
    Every store issued before the call is visible to that last arrival
    (release, then acquire). The last arrival resets the counter to 0, so an
    int32 tensor allocated zeroed can be handed to every launch, including the
    replays of a captured CUDA graph, with no memset in between.

    The result is a uniform scalar, for `hk.if_`. It contains barriers: every
    warp of the workgroup must reach it, which is why it is an op and not
    something to put inside an `if_` on the warp index.
    """
    _expect(counter, GlobalType, "wg_arrive counter")
    _expect(idx, CoordType, "wg_arrive index")
    if idx.type.unit != "element":
        raise TypeError(f"wg_arrive takes an element coordinate, got {idx.type}")
    if counter.type.dtype not in (i32,):
        raise TypeError(f"wg_arrive counts in an int32 tensor, not {counter.type.dtype.name}")
    return _b().emit("wg_arrive", [counter, idx, _as_value(expected, i32)],
                     ScalarType(i32), name="arrived")


def barrier(drain: bool = False) -> None:
    """`s_barrier`. Orders *execution* across the workgroup -- not memory.

    That distinction is the whole reason this takes an argument. `s_barrier`
    does not retire anything: an LDS op still in flight when a warp reaches it
    is still in flight afterwards, racing whatever the other warps do next. On
    a double-buffered kernel that is always the opposite access -- outstanding
    ds_writes against the next tile's ds_reads, outstanding ds_reads against
    the next tile's ds_writes -- so it is a race with no safe outcome, and one
    that shows up only when the scheduler happens to tighten the window.

    `hk.ir.passes.lds_pipeline` therefore refuses a barrier with a non-empty
    queue. The cheap fix is usually to retire the ops in a wait the math
    needed anyway; `drain=True` is the other one, an explicit `lgkmcnt(0)`
    here, which is what the handwritten GEMM spends after its interleaved
    store. Making it an argument rather than a separate op is deliberate: the
    drain is only ever wanted *immediately* before a barrier, and splitting
    them would let something be scheduled in between.

    Emitted as an op rather than folded into the collective helpers because a
    barrier is a control-flow fact -- every warp of the workgroup must reach
    it -- and a pass that reorders or hoists ops has to be able to see it.
    """
    _b().emit("barrier", drain=bool(drain))


# ---------------------------------------------------------------- creation


# `out=` on a constant fill is not a convenience: a tile that has to be reset at
# the top of every iteration of an hk.range body cannot be re-created there,
# because a value traced inside the body is a different value each iteration and
# the one outside is what the accumulator math names. Attention's score tile is
# exactly that -- zeroed per KV block, accumulated across eight WMMAs -- and
# without this it silently sums every block's scores into the next.


def zeros(tile: RegTileType, *, out: Optional[Value] = None) -> Value:
    return _emit("zero", [], tile, out=out, name="z")


def full(tile: RegTileType, value: float, *, out: Optional[Value] = None) -> Value:
    return _emit("full", [], tile, out=out, name="f", value=value)


def neg_infty(tile: RegTileType, *, out: Optional[Value] = None) -> Value:
    return _emit("neg_infty", [], tile, out=out, name="ninf")


# ---------------------------------------------------------------- maps
#
# `dtypes` is not decoration. base_ops.cuh specialises each operation for the
# element types it can actually compute, and asking for one it does not have is
# a *link* error -- "undefined hidden symbol kittens::base_ops::gelu::op", at
# the end of a 30 s compile, naming a template instantiation rather than the
# line of Python that asked for it. Listing the support here turns that into a
# trace-time message. Keep it in step with base_ops.cuh; the codegen tests
# compile every entry, so a stale row here fails the build rather than a user.

_ALL_FLOAT = ("bf16", "fp16", "fp32")


@dataclass(frozen=True)
class Map:
    """One elementwise op: its name in include/rdna3 and what it supports."""

    cpp: str
    dtypes: Tuple[str, ...] = _ALL_FLOAT


#: Binary elementwise ops, DSL name -> the kittens op.
BINARY = {
    "add": Map("add"),
    "sub": Map("sub"),
    "mul": Map("mul"),
    "div": Map("div"),
    "max": Map("max"),
    "min": Map("min"),
}

#: Unary elementwise ops.
UNARY = {
    "exp": Map("exp"),
    "exp2": Map("exp2"),
    "log": Map("log"),
    "log2": Map("log2"),
    "abs": Map("abs"),
    "relu": Map("relu"),
    "copy": Map("copy"),
    # gelu/dgelu are specialised for float only -- there is no packed bf16
    # implementation. Cast first if you want them on a 16-bit tile.
    "gelu": Map("gelu", ("fp32",)),
    "dgelu": Map("dgelu", ("fp32",)),
    # Float only by choice, not by omission: these exist for the normalization
    # tail, which accumulates in fp32 whatever the tensor is stored as. See the
    # comment on base_ops::sqrt.
    "sqrt": Map("sqrt", ("fp32",)),
    "rsqrt": Map("rsqrt", ("fp32",)),
    "silu": Map("silu", ("fp32",)),
}


def _check_dtype(op: str, m: Map, ty) -> None:
    if ty.dtype.name not in m.dtypes:
        raise TypeError(
            f"{op} has no {ty.dtype.name} implementation in include/rdna3 "
            f"(it is specialised for {', '.join(m.dtypes)}). Cast with "
            f"hk.cast(x, hk.fp32) first, or load into an fp32 tile -- the "
            f"global load converts for free. This is caught here because the "
            f"alternative is an undefined-symbol link error at the end of the "
            f"compile that names a template, not this line."
        )


def s_any_ne(a: Value, b: Value) -> Value:
    """Wave-level "did any entry of `a` differ from `b`", as a condition.

    The one thing in the attention kernel that cannot be written with the
    elementwise ops: it is a reduction *out of* the register file and into
    control flow. The online softmax rescales the accumulator by
    `exp2(m_old - m_new)` on every KV block, and after the first few blocks the
    running max is usually already the global max -- alpha is 1 and all 64
    `v_mul_f32` at head_dim 128 are wasted. Guarding them needs a predicate
    that is true if *any* entry grew, because the rescale is a tile operation
    and a wave cannot do it half way.

    Wave-uniform by construction, so this is a scalar branch and not a
    divergent one: `m_*` is a vector over the q range of the warp's own tile
    and every lane holds all of it, so the comparison has the same answer in
    every lane before the `__any` ever runs. The `__any` is there for the
    compiler, which has no way to know that.
    """
    for v, side in ((a, "left"), (b, "right")):
        _expect(v, RegVecType, f"s_any_ne {side} operand")
    if a.type.plain != b.type.plain:
        raise TypeError(
            f"s_any_ne: operands are {a.type} and {b.type}. The comparison is "
            f"entry by entry, so the two have to be the same shape."
        )
    return _b().emit("s_any_ne", [a, b], result_type=ScalarType(i32),
                     name="grew")


#: Tiles and vectors take the same op names -- `kittens::add` is overloaded on
#: both -- so the DSL does not split them either.
_ELEMENTWISE = (RegTileType, RegVecType)


def _emit(op: str, operands, ty, *, out: Optional[Value], name: str, **attrs) -> Value:
    """Produce a new value, or write into the caller's."""
    if out is None:
        return _b().emit(op, operands, result_type=ty, name=name, **attrs)
    # `.plain` on both sides, then the one-way check: writing a uniform result
    # into a vector not known to be uniform is sound and loses only a fact,
    # while the reverse would launder an unproven one into a type that
    # store_scalar trusts.
    if out.type.plain != ty.plain or (
        getattr(out.type, "uniform", False) and not getattr(ty, "uniform", False)
    ):
        raise TypeError(
            f"{op}: out is {out.type} but the result is {ty}. An out= that does "
            f"not match is a different tile, not a conversion."
        )
    return _b().emit_into(op, out, operands, **attrs)


def _binary(op: str, a: Value, b: Scalar, out: Optional[Value] = None) -> Value:
    _expect(a, _ELEMENTWISE, f"{op} left operand")
    _check_dtype(op, BINARY[op], a.type)
    if isinstance(b, Value):
        if isinstance(b.type, ScalarType):
            # A runtime scalar -- `acc / cols`, where the width is not known
            # until launch. kittens::mul is templated on the right operand, so
            # this is the same call with a cast; it is an operand rather than
            # an attribute because its value only exists on the device.
            return _emit(op, [a, b], a.type, out=out, name=op, scalar_rhs=True)
        if not isinstance(b.type, _ELEMENTWISE):
            raise TypeError(
                f"{op}: right operand is {b.type}, expected a tile, a vector or a number"
            )
        if b.type.plain != a.type.plain:
            raise TypeError(
                f"{op}: operand types differ, {a.type} vs {b.type}. "
                f"Elementwise ops do not broadcast; use add_row/add_col for that."
            )
        # Uniform survives only if both sides have it: a uniform vector times a
        # vector whose entries differ is a vector whose entries differ. The
        # scalar form below keeps it, because a constant is uniform.
        return _emit(op, [a, b], _meet(a.type, b.type), out=out, name=op)
    return _emit(op, [a], a.type, out=out, name=op, scalar=b)


def _meet(a, b):
    """The result type of an elementwise op on `a` and `b`: `a`'s, minus any
    refinement `b` does not also carry."""
    return a if getattr(b, "uniform", True) else a.plain


def _unary(op: str, a: Value, out: Optional[Value] = None) -> Value:
    _expect(a, _ELEMENTWISE, f"{op} operand")
    _check_dtype(op, UNARY[op], a.type)
    return _emit(op, [a], a.type, out=out, name=op)


def _make_binary(name):
    def f(a: Value, b: Scalar, *, out: Optional[Value] = None) -> Value:
        return _binary(name, a, b, out)

    f.__name__ = name
    f.__doc__ = (
        f"Elementwise {name} on a tile or a vector. Maps to "
        f"kittens::{BINARY[name].cpp}. `out=` writes an existing value."
    )
    return f


def _make_unary(name):
    def f(a: Value, *, out: Optional[Value] = None) -> Value:
        return _unary(name, a, out)

    f.__name__ = name
    f.__doc__ = (
        f"Elementwise {name} on a tile or a vector. Maps to "
        f"kittens::{UNARY[name].cpp}. `out=` writes an existing value."
    )
    return f


for _n in BINARY:
    globals()[_n] = _make_binary(_n)
for _n in UNARY:
    globals()[_n] = _make_unary(_n)
del _n


def neg(a: Value) -> Value:
    """Elementwise -a.

    A multiply by -1, not a library call: maps.cuh has `neg_infty` (a fill) but
    no negate, and the scalar overload of `mul` is the same VALU instruction a
    dedicated negate would be.
    """
    return _binary("mul", a, -1.0)


# ---------------------------------------------------------------- reductions
#
# The naming is include/rdna3's and it reads backwards the first time: `row_sum`
# sums *along* each row and so yields one value *per* row, which is a column
# vector. The broadcasts pair with it -- `mul_row(t, v)` scales row i of t by
# v[i] -- so a normalization is row_sum then mul_row, never a mix.

REDUCTIONS = {
    "row_max": ("col", "max"),
    "row_min": ("col", "min"),
    "row_sum": ("col", "sum"),
    "row_prod": ("col", "prod"),
    "col_max": ("row", "max"),
    "col_min": ("row", "min"),
    "col_sum": ("row", "sum"),
    "col_prod": ("row", "prod"),
}


def _reduce(op: str, src: Value, out: Optional[Value], accumulate: bool) -> Value:
    kind, _ = REDUCTIONS[op]
    _expect(src, RegTileType, f"{op} operand")
    if accumulate and out is None:
        raise TypeError(
            f"{op}(accumulate=True) needs out= -- there is nothing to "
            f"accumulate into otherwise. This is the loop form: make the "
            f"vector before the loop, accumulate into it inside."
        )
    # The accumulator's element type is the tile's, deliberately not a
    # parameter: an rv's inner_dim comes from WMMA_REPLICATION of its element
    # type, so an fp32 vector and a bf16 tile disagree about which lane holds
    # what. To accumulate a bf16 tensor in fp32, load it into an fp32 tile --
    # kittens::load converts on the way in and costs nothing extra.
    vt = RegVecType(src.type, kind)
    if out is None:
        return _b().emit(op, [src], result_type=vt, name=op)
    if out.type != vt:
        raise TypeError(
            f"{op}: out is {out.type}, but reducing {src.type} gives {vt}. "
            f"A vector's layout is tied to the tile it came off; a mismatched "
            f"one would reduce the wrong lanes."
        )
    return _b().emit_into(op, out, [src], accumulate=accumulate)


def _make_reduce(name):
    def f(
        src: Value,
        *,
        out: Optional[Value] = None,
        accumulate: bool = False,
    ) -> Value:
        return _reduce(name, src, out, accumulate)

    kind, _ = REDUCTIONS[name]
    f.__name__ = name
    f.__doc__ = (
        f"Reduce along each {'row' if kind == 'col' else 'column'} of a tile, "
        f"giving one value per {'row' if kind == 'col' else 'column'} "
        f"(a {kind} vector). `out=` with accumulate=True folds into an "
        f"existing vector, which is how a loop over blocks reduces."
    )
    return f


for _n in REDUCTIONS:
    globals()[_n] = _make_reduce(_n)
del _n

# ---------------------------------------------------------------- broadcasts

BROADCASTS = {}
for _axis in ("row", "col"):
    for _o in ("add", "sub", "mul", "div"):
        BROADCASTS[f"{_o}_{_axis}"] = (_axis, _o)
del _axis, _o


def _broadcast(op: str, src: Value, vec: Value, out: Optional[Value]) -> Value:
    axis, _ = BROADCASTS[op]
    _expect(src, RegTileType, f"{op} tile operand")
    _expect(vec, RegVecType, f"{op} vector operand")
    want = "col" if axis == "row" else "row"
    if vec.type.kind != want:
        raise TypeError(
            f"{op} takes a {want} vector (one value per {axis}), got "
            f"{vec.type}. row ops are indexed by row, so they take the vector "
            f"row_sum/row_max produce."
        )
    if vec.type.tile.rows != src.type.rows or vec.type.tile.cols != src.type.cols:
        raise TypeError(
            f"{op}: the vector came off a {vec.type.tile} but is being applied "
            f"to a {src.type}. The lane replication would not line up."
        )
    return _emit(op, [src, vec], src.type, out=out, name=op)


def _make_broadcast(name):
    def f(src: Value, vec: Value, *, out: Optional[Value] = None) -> Value:
        return _broadcast(name, src, vec, out)

    axis, o = BROADCASTS[name]
    f.__name__ = name
    f.__doc__ = f"{o} `vec[i]` into every element of {axis} i of the tile."
    return f


for _n in BROADCASTS:
    globals()[_n] = _make_broadcast(_n)
del _n


def _broadcast_fill(op: str, tile_type: RegTileType, vec: Value, want: str) -> Value:
    _expect(vec, RegVecType, f"{op} vector")
    if vec.type.kind != want:
        raise TypeError(f"{op} takes a {want} vector, got {vec.type}")
    if vec.type.tile.rows != tile_type.rows or vec.type.tile.cols != tile_type.cols:
        raise TypeError(
            f"{op}: the vector came off a {vec.type.tile} but is being "
            f"broadcast into a {tile_type}."
        )
    return _b().emit(op, [vec], result_type=tile_type, name="bc")


def as_vec(vec: Value, tile: RegTileType, *, uniform: bool = False) -> Value:
    """Re-attach a register vector to a tile of a different shape.

    `rt<T,R,C>::col_vec` is `rv<T, R, layout>` -- it depends on the tile's row
    count and layout and *not* on its column count (rt.cuh:79). So the vector
    that falls out of reducing a 16x16 tile is already, in C++, the exact type
    that broadcasts back into a 16x64 one; only the IR's name for it differs,
    because RegVecType carries its tile so that a mismatched layout is
    unspellable.

    This is the escape hatch for that, and it is deliberately narrow: it checks
    the three things that make the two C++ types identical (element type,
    length, layout) and refuses anything else, and the generated code carries a
    static_assert so that a future change to rt.cuh's aliases becomes a compile
    error here rather than a vector that reduces the wrong lanes.

    It exists for one caller: folding a row that was reshaped onto the tile's
    rows. See lang.collective.fold_rows.

    `uniform=True` is the other half of that: fold_rows ends with a reduction
    broadcast back across the vector's entries, so at this one point in the
    library every entry provably holds the same value, and this is where that
    fact enters the type system. It is a keyword rather than something a caller
    could stumble into, and fold_rows is the only place in the tree that passes
    it. See RegVecType.uniform.
    """
    _expect(vec, RegVecType, "as_vec source")
    if not isinstance(tile, RegTileType):
        raise TypeError(f"as_vec target is {tile!r}, expected a register tile type")
    want = RegVecType(tile, vec.type.kind, vec.type.uniform or uniform)
    if want == vec.type:
        return vec
    if (want.dtype, want.length, tile.layout) != (
        vec.type.dtype, vec.type.length, vec.type.tile.layout
    ):
        raise TypeError(
            f"as_vec cannot retype {vec.type} (off {vec.type.tile}) as {want} "
            f"(off {tile}): an rv's C++ type is (element, length, layout) and "
            f"these differ. Route it through an sv instead -- that is a real "
            f"relayout, and it costs what a relayout costs."
        )
    return _b().emit("retype_vec", [vec], result_type=want, name="rv")


def broadcast_row(tile_type: RegTileType, vec: Value) -> Value:
    """A tile whose every row is `vec[row]`. kittens::broadcast_row."""
    return _broadcast_fill("broadcast_row", tile_type, vec, "col")


def broadcast_col(tile_type: RegTileType, vec: Value) -> Value:
    """A tile whose every column is `vec[col]`. kittens::broadcast_col."""
    return _broadcast_fill("broadcast_col", tile_type, vec, "row")


# ---------------------------------------------------------------- fills
#
# The other half of the out-of-bounds idiom. include/rdna3 does not bounds-check
# a global load, so the established way to read a partial block (attn.cpp:735)
# is to back the *address* up until the whole tile is in bounds. That reads
# real data, never garbage -- but it reads some of it twice, and a reduction
# along the axis you backed up would then count those elements twice. These
# ops overwrite the overlap with the reduction's identity, which makes the
# double count vanish instead of having to be subtracted out.
#
#     lo  = hk.s_min(cb * COLS, n - COLS)   # backed-up start
#     pad = cb * COLS - lo                  # 0 except in the last block
#     t   = hk.left_fill(hk.load(x, hk.elem_coord(0, 0, r, lo), tt), pad, 0.0)

FILLS = {
    "left_fill": "col",    # columns < idx
    "right_fill": "col",   # columns >= idx
    "upper_fill": "row",   # rows < idx
    "lower_fill": "row",   # rows >= idx
    # The same shape of op, about the shifted diagonal rather than an axis.
    # `triu(t, d, -inf)` is attention's causal predicate when the scores are
    # stored transposed: a row of S^T is a kv index and a column is a query,
    # so "kv may not exceed q" is `col < row - d`, with `d` the offset between
    # the two blocks' starting indices. `make_causal_t` is the d == 0 case,
    # which is all you need when every extent is 16-aligned and not enough for
    # a ragged one.
    "triu": "diag",        # col < row - idx
    "tril": "diag",        # col > row - idx
}


def _fill(op: str, src: Value, idx: Scalar, value: float, out: Optional[Value]) -> Value:
    """A fill is a *mask applied to a tile*, and it writes in place.

    Not a style choice -- conversions.cuh:530 reads

        if (col_idx <= 0) return;

    and returns without touching `dst`. An empty mask is the common case (every
    column block but the last), so a fresh destination would be left holding
    whatever those registers happened to contain, and the kernel would produce
    inf and NaN for exactly the shapes that do divide. Writing `src` means the
    early return leaves the right values in place, which is what the library's
    `dst[in,out]` annotation is telling you.

    An explicit `out=` that is some other tile is still allowed; it costs a
    `kittens::copy` first, for the same reason.
    """
    _expect(src, RegTileType, f"{op} tile operand")
    v = _as_value(idx, i32)
    if not isinstance(v.type, ScalarType):
        raise TypeError(f"{op} index is {v.type}, expected a scalar")
    if out is None:
        out = src
    elif out is not src:
        if out.type != src.type:
            raise TypeError(
                f"{op}: out is {out.type} but the tile is {src.type}. A fill "
                f"keeps shape, layout and dtype."
            )
        _b().emit_into("cast", out, [src])
    return _b().emit_into(op, out, [src, v], value=value)


def _make_fill(name):
    def f(src: Value, idx: Scalar, value: float = 0.0, *, out: Optional[Value] = None) -> Value:
        return _fill(name, src, idx, value, out)

    axis = FILLS[name]
    f.__name__ = name
    if axis == "diag":
        kept = "col <= row - idx" if name == "triu" else "col >= row - idx"
        f.__doc__ = (
            f"Overwrite every element with `value` except where `{kept}` -- the "
            f"main diagonal shifted down by `idx`. `idx` may be a runtime value, "
            f"and the degenerate shifts behave: at `idx <= -rows` the whole tile "
            f"is filled, at `idx >= cols` none of it is."
        )
    else:
        side = {"left_fill": "before", "right_fill": "from",
                "upper_fill": "above", "lower_fill": "from"}[name]
        f.__doc__ = (
            f"Overwrite every element {side} {axis} `idx` with `value`, keeping "
            f"the rest. Pass the reduction's identity (0 for sum, -inf for max, "
            f"1 for prod) when you are masking an overlap out of a reduction."
        )
    return f


for _n in FILLS:
    globals()[_n] = _make_fill(_n)
del _n


# ---------------------------------------------------------------- vectors


def zeros_vec(vt: RegVecType, *, out: Optional[Value] = None) -> Value:
    return _emit("zero", [], vt, out=out, name="zv")


def full_vec(vt: RegVecType, value: float, *, out: Optional[Value] = None) -> Value:
    return _emit("full", [], vt, out=out, name="fv", value=value)


def neg_infty_vec(vt: RegVecType, *, out: Optional[Value] = None) -> Value:
    """The identity for max. Starting an online max at zero is a bug that only
    shows up on all-negative rows, which is why this exists."""
    return _emit("neg_infty", [], vt, out=out, name="ninfv")


# ---------------------------------------------------------------- wmma
#
# The four variants are not four conveniences. gfx1100's WMMA takes its two
# operands in fixed fragment layouts, and which of the four names applies is
# decided by the layouts you already have -- so the choice of variant is how a
# kernel avoids a transpose, and picking the wrong one is how it buys eight
# ds_read_u16 per base tile instead of two ds_read_b128. See check_layouts.
#
# Every rule below is transcribed from the static_asserts in
# include/rdna3/ops/warp/register/tile/mma.cuh. They are checked here because
# the alternative is a template instantiation error forty lines deep that names
# `rt_base<...>` and not the line that wrote it.


@dataclass(frozen=True)
class _MmaSpec:
    """One variant: what layouts it takes and where D's shape comes from."""

    a_layout: str
    b_layout: str
    d_rows: str   # attribute of A giving D.rows
    d_cols: str   # attribute of B giving D.cols
    a_red: str    # attribute of A that is the reduction extent
    b_red: str    # and of B; the two must agree


MMA = {
    # A is transposed exactly when it is read col-layout, and likewise B --
    # which is why "the transpose" costs nothing here: it is a different name
    # for the same registers, not a movement of them.
    "mma_AB":   _MmaSpec("row", "col", "rows", "cols", "cols", "rows"),
    "mma_ABt":  _MmaSpec("row", "row", "rows", "rows", "cols", "cols"),
    "mma_AtB":  _MmaSpec("col", "col", "cols", "cols", "rows", "rows"),
    "mma_AtBt": _MmaSpec("col", "row", "cols", "rows", "rows", "cols"),
}


def _mma(variant: str, a: Value, b: Value, acc: Value, out: Optional[Value]) -> Value:
    spec = MMA[variant]
    for v, what in ((a, "A"), (b, "B"), (acc, "C")):
        _expect(v, RegTileType, f"{variant} {what} operand")

    if a.type.dtype != b.type.dtype:
        raise TypeError(
            f"{variant}: A is {a.type.dtype} and B is {b.type.dtype}. WMMA takes "
            f"one operand type, not a mixed pair."
        )
    if a.type.dtype.bits != 16:
        raise TypeError(
            f"{variant}: operands are {a.type.dtype}. gfx1100 WMMA takes bf16 or "
            f"fp16 operands only -- there is no fp8 or int8 form, and the "
            f"fp16-accumulate form is not wired up in include/rdna3."
        )
    if acc.type.dtype != fp32:
        raise TypeError(
            f"{variant}: the accumulator is {acc.type.dtype}. gfx1100 accumulates "
            f"in fp32; on this architecture that is also free, because a WMMA "
            f"operand is mirrored across the two wave halves and a bf16 rt_base "
            f"therefore costs the same 8 VGPRs as an fp32 one."
        )

    for v, want, what in ((a, spec.a_layout, "A"), (b, spec.b_layout, "B")):
        if v.type.layout != want:
            other = next(n for n, sp in MMA.items()
                         if (a.type.layout, b.type.layout) == (sp.a_layout, sp.b_layout))
            raise TypeError(
                f"{variant} wants {what} in '{want}' layout and it is "
                f"'{v.type.layout}'. With the layouts you have, the variant that "
                f"applies is {other} -- which computes a different product, so "
                f"if it is not the one you want, the fix is to stage the operand "
                f"the other way round, not to relabel it. swap_layout moves data "
                f"on this architecture."
            )
    if acc.type.layout != "col":
        raise TypeError(
            f"{variant}: the accumulator is '{acc.type.layout}' layout and WMMA "
            f"writes 'col'. Every mma_* in include/rdna3 takes a col_layout D "
            f"and C."
        )

    red_a, red_b = getattr(a.type, spec.a_red), getattr(b.type, spec.b_red)
    if red_a != red_b:
        raise TypeError(
            f"{variant}: reduction extents disagree -- A.{spec.a_red}={red_a}, "
            f"B.{spec.b_red}={red_b}."
        )
    rows, cols = getattr(a.type, spec.d_rows), getattr(b.type, spec.d_cols)
    ty = RegTileType(fp32, rows, cols, "col")
    if acc.type != ty:
        raise TypeError(
            f"{variant}: A {a.type} times B {b.type} is {ty}, but the "
            f"accumulator is {acc.type}."
        )
    return _emit(variant, [a, b, acc], ty, out=out, name="mma")


def _make_mma(name):
    spec = MMA[name]

    def f(a: Value, b: Value, acc: Value, *, out: Optional[Value] = None) -> Value:
        return _mma(name, a, b, acc, out)

    f.__name__ = name
    at = "^T" if "At" in name else ""
    bt = "^T" if name.endswith("t") and "Bt" in name else ""
    f.__doc__ = (
        f"`acc + A{at} @ B{bt}`, mapping to kittens::{name}.\n\n"
        f"Takes A in '{spec.a_layout}' layout, B in '{spec.b_layout}', and a "
        f"col-layout fp32 accumulator. Result is "
        f"(A.{spec.d_rows}, B.{spec.d_cols}) with A.{spec.a_red} == "
        f"B.{spec.b_red}.\n\n"
        f"`out=acc` accumulates in place, which is the GEMM inner-loop form: it "
        f"emits `kittens::{name}(acc, a, b, acc)` and does not re-declare acc. "
        f"Without it the op declares a fresh destination, which in a loop is "
        f"almost always a bug the IR would otherwise let you write."
    )
    return f


mma_AB = _make_mma("mma_AB")
mma_ABt = _make_mma("mma_ABt")
mma_AtB = _make_mma("mma_AtB")
mma_AtBt = _make_mma("mma_AtBt")


def cast(a: Value, dtype: DType, out: Optional[Value] = None) -> Value:
    """Change a tile's element type, keeping its shape and layout. This is
    kittens::copy between differently-typed tiles.

    `out=` matters more here than for most ops. A cast is the one place a
    kernel allocates a tile it does not name, and a tile is 32 VGPRs a lane:
    writing the same cast at several points and leaving each to produce its
    own destination asks the register allocator to prove the lifetimes
    disjoint, which it does until it suddenly does not and the kernel spills.
    Passing one destination states it instead.
    """
    _expect(a, RegTileType, "cast operand")
    ty = RegTileType(dtype, a.type.rows, a.type.cols, a.type.layout)
    return _emit("cast", [a], ty, out=out, name="cvt")


# ---------------------------------------------------- transpose and masking


def transpose(a: Value, *, out: Optional[Value] = None) -> Value:
    """Transpose a register tile by *relabelling*, moving no data.

    A base tile's storage depends only on its dtype; the layout tag is purely
    how `(lane_low, pos)` is read -- `(row, col)` for row and `(col, row)` for
    col. So reading the same registers under the opposite tag already is the
    transpose, and pairing that with sending `tiles[i][j]` to `tiles[j][i]`
    transposes the whole tile for the cost of the register moves alone
    (`conversions.cuh:253`). The result has the mirrored shape *and the
    opposite layout*, which is the part worth saying out loud: this is how a
    `col` accumulator becomes a `row` operand for the next mma without a
    round trip through LDS.

    The library spells this `kittens::transpose`; the cost is register moves,
    not shuffles. If you want the shape mirrored while *keeping* the layout --
    which really does move data -- that is `transpose_sep`.
    """
    _expect(a, RegTileType, "transpose operand")
    ty = RegTileType(a.type.dtype, a.type.cols, a.type.rows,
                     "col" if a.type.layout == "row" else "row")
    return _emit("transpose", [a], ty, out=out, name="tr")


def transpose_sep(a: Value, *, out: Optional[Value] = None) -> Value:
    """Transpose a register tile, keeping its layout. Moves data.

    The "sep" is the library's: dst and src must occupy separate registers
    (`conversions.cuh:228`), so this never writes in place and `out=a` is
    refused. That is not a formality -- the per-base-tile `transpose` it calls
    permutes within a fragment as it goes, and an aliased destination is
    overwritten mid-permutation.

    Mind the register cost before reaching for it on a whole tile. The
    handwritten attention epilogue transposes the output accumulator one
    fragment at a time precisely because the whole-tile form needs a second
    `[Q_BLOCK, HEAD_DIM]` tile live alongside the first -- 128 VGPRs between
    them, which measured 392 bytes/lane of scratch and 99 spilled VGPRs. If
    the shape is big, `hk.compile`'s spill gate will catch it, but it is
    cheaper to know first.
    """
    _expect(a, RegTileType, "transpose_sep operand")
    if out is a:
        raise TypeError(
            "transpose_sep: out= is the same tile as the source. The library "
            "requires separate registers -- its per-fragment transpose permutes "
            "as it goes and would overwrite values it has not read yet. Pass a "
            "different tile, or use hk.transpose, which relabels instead of "
            "moving and has no such restriction."
        )
    ty = RegTileType(a.type.dtype, a.type.cols, a.type.rows, a.type.layout)
    return _emit("transpose_sep", [a], ty, out=out, name="trs")


#: `make_causal` keeps the lower triangle (`col <= row`); `make_causal_t`
#: keeps the upper (`col >= row`). The `_t` one is what a kernel wants when it
#: has computed S transposed -- which on RDNA3 is the orientation that makes
#: the softmax reduction a column reduction and keeps the accumulator in col
#: layout -- because transposing the tile transposes the mask with it.
CAUSAL = {"make_causal": "col <= row", "make_causal_t": "col >= row"}


def _causal(op: str, src: Value, value: float, out: Optional[Value]) -> Value:
    _expect(src, RegTileType, f"{op} operand")
    if src.type.rows != src.type.cols:
        raise TypeError(
            f"{op}: the tile is {src.type.rows}x{src.type.cols}, not square. "
            f"A diagonal only means something on a square tile; on anything "
            f"else `row == col` picks out a line whose position depends on "
            f"which of the two extents you measured it in, and the library "
            f"picks one without saying so. Mask the off-diagonal blocks with "
            f"left_fill/right_fill, which take the index explicitly, and use "
            f"this on the square block that straddles the diagonal."
        )
    # Unlike the triangular fills, this one has no early return: it is a
    # map_by_coord over every element (`conversions.cuh:451`), so a fresh
    # destination is fully written and `out=None` is safe.
    return _emit(op, [src], src.type, out=out, name=op, value=value)


def _make_causal(name):
    def f(src: Value, value: float = 0.0, *, out: Optional[Value] = None) -> Value:
        return _causal(name, src, value, out)

    f.__name__ = name
    f.__doc__ = (
        f"Keep the elements where `{CAUSAL[name]}` and overwrite the rest with "
        f"`value`. Pass the reduction's identity -- for a softmax that is "
        f"-inf, not 0, because the masked entries go through exp2 before they "
        f"are summed."
    )
    return f


for _n in CAUSAL:
    globals()[_n] = _make_causal(_n)
del _n


# ---------------------------------------------------------------- helpers


def _expect(v: Value, ty, what: str) -> None:
    if not isinstance(v, Value):
        raise TypeError(f"{what} must be a traced value, got {type(v).__name__}")
    if not isinstance(v.type, ty):
        want = ty if isinstance(ty, tuple) else (ty,)
        raise TypeError(
            f"{what} is {v.type}, expected {' or '.join(t.__name__ for t in want)}"
        )


__all__ = [
    "s_any_ne",
    "rt", "st", "col_vec", "row_vec",
    "block_idx", "warp_id", "lane_id",
    "coord", "tile_coord", "elem_coord",
    "batch", "depth", "rows", "cols",
    *(f"s_{s}" for s in SCALARS), *FILLS,
    "load", "store", "store_scalar", "load_scalar", "load_shared",
    "store_shared", "alloc_shared",
    "granules", "load_frag", "store_frag",
    "shared_at", "subtile", "setprio", "lds_loads", "lds_wait_for",
    "stage_buffer", "stage_load", "stage_commit", "vm_wait",
    "shared_vec", "alloc_shared_vec", "load_shared_vec", "store_shared_vec",
    "barrier", "sched_barrier", "wg_arrive",
    "zeros", "full", "neg_infty", "cast", "neg",
    "transpose", "transpose_sep", *CAUSAL,
    "zeros_vec", "full_vec", "neg_infty_vec",
    "broadcast_row", "broadcast_col", "as_vec",
    *BINARY, *UNARY, *REDUCTIONS, *BROADCASTS, *MMA,
]
