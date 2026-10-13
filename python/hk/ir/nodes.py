"""The HK IR: types, values, ops, and a kernel.

Deliberately small. This is a *tile* IR -- the unit of everything here is a
tile, not a scalar and not a thread. That is the HipKittens abstraction and it
is the reason the IR can be this small while still describing a kernel that runs
at 77 TFLOPs: the hard parts live in include/rdna3, and the IR's job is to pick
the right call and prove the choice is legal.
"""

from __future__ import annotations

import itertools
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------- dtypes


@dataclass(frozen=True)
class DType:
    name: str
    bits: int
    cpp: str  # the name include/rdna3 uses
    kind: str  # "float" | "int"

    def __repr__(self) -> str:  # keeps IR dumps readable
        return self.name

    @property
    def bytes(self) -> int:
        return self.bits // 8


bf16 = DType("bf16", 16, "bf16", "float")
fp16 = DType("fp16", 16, "half", "float")
fp32 = DType("fp32", 32, "float", "float")
i32 = DType("i32", 32, "int", "int")
#: A storage type only. gfx1100 has no int8 WMMA and include/rdna3 has no
#: rt<int8>, so `hk.GL[i8]` is legal and `hk.rt(i8, ...)` is not -- an int8
#: tensor is something you quantize into and dequantize out of, and the
#: arithmetic in between happens in fp32.
i8 = DType("i8", 8, "int8", "int")
#: The two OCP fp8 formats, as WMMA operands. Only gfx1201 has an fp8 WMMA
#: (v_wmma_f32_16x16x16_{fp8,bf8}_{fp8,bf8}); on gfx1100 these are storage
#: types at most, and `mma_*` on them says so. Compile-gated, not run: no
#: gfx12 part was available.
fp8e4m3 = DType("fp8e4m3", 8, "fp8e4m3", "float")
fp8e5m2 = DType("fp8e5m2", 8, "fp8e5m2", "float")
FP8 = (fp8e4m3, fp8e5m2)

DTYPES = {d.name: d for d in (bf16, fp16, fp32, i32, i8, fp8e4m3, fp8e5m2)}

# ---------------------------------------------------------------- types


class Type:
    """Base for everything a Value can have."""

    @property
    def plain(self) -> "Type":
        """This type with any *refinement* dropped.

        A refinement is a property the IR proves and the C++ cannot spell --
        today there is exactly one, `RegVecType.uniform`. Two types that differ
        only in a refinement generate identical C++, so every site that is
        asking "do these compile to the same thing" compares `.plain`, and only
        the sites enforcing the refinement look at the flag itself.
        """
        return self


@dataclass(frozen=True)
class ScalarType(Type):
    dtype: DType

    def __repr__(self) -> str:
        return f"scalar<{self.dtype}>"

    def cpp(self) -> str:
        return self.dtype.cpp


@dataclass(frozen=True)
class RegTileType(Type):
    """A register tile: rt<T, rows, cols, layout>.

    `layout` is row or col and is not cosmetic -- it decides which mma variant
    applies and whether a shared->register load hits ds_read_b128 or degrades to
    16 scalar reads. See hk.ir.verify.
    """

    dtype: DType
    rows: int
    cols: int
    layout: str = "row"

    def __post_init__(self):
        if self.layout not in ("row", "col"):
            raise ValueError(f"layout must be row or col, got {self.layout!r}")
        # fp8 is let through here and judged per kernel by hk.ir.verify, which
        # knows the target: gfx1201 has fp8 WMMA tiles, gfx1100 has none.
        if self.dtype.bits < 16 and self.dtype not in FP8:
            raise TypeError(
                f"there is no {self.dtype} register tile. include/rdna3 defines "
                f"rt_base only for 16- and 32-bit elements -- gfx1100 has no "
                f"int8 WMMA and no packed narrow ALU -- so a narrow type "
                f"is a global you quantize into, not something you compute in. "
                f"Load it into an fp32 tile; the load converts on the way."
            )
        for n, v in (("rows", self.rows), ("cols", self.cols)):
            if v <= 0 or v % 16:
                raise ValueError(
                    f"register tile {n}={v}: every extent is in units of a "
                    f"16x16 WMMA fragment, so it must be a positive multiple of 16"
                )

    def __repr__(self) -> str:
        return f"rt<{self.dtype},{self.rows}x{self.cols},{self.layout}>"

    @property
    def base_tiles(self) -> int:
        return (self.rows // 16) * (self.cols // 16)

    def cpp(self) -> str:
        return (
            f"kittens::rt<{self.dtype.cpp}, {self.rows}, {self.cols}, "
            f"kittens::ducks::rt_layout::{self.layout}>"
        )

    def transposed(self) -> "RegTileType":
        return RegTileType(
            self.dtype, self.cols, self.rows, "col" if self.layout == "row" else "row"
        )


@dataclass(frozen=True)
class SharedTileType(Type):
    """An LDS tile: st<T, rows, cols>. Shared tiles carry no layout tag -- the
    swizzle is a property of the type in the C++ library."""

    dtype: DType
    rows: int
    cols: int
    #: How many of these the allocation holds. >1 is a multi-buffer: the
    #: allocator hands out one contiguous, size-aligned block and the emitter
    #: writes the `[N]`. Contiguity is the point -- the XOR buffer swap flips
    #: one address bit, which is only legal if the pair differs in one bit.
    count: int = 1

    def __post_init__(self):
        for n, v in (("rows", self.rows), ("cols", self.cols)):
            if v <= 0 or v % 16:
                raise ValueError(f"shared tile {n}={v} must be a positive multiple of 16")
        if self.count <= 0:
            raise ValueError(f"shared tile count={self.count} must be positive")

    def __repr__(self) -> str:
        n = f"[{self.count}]" if self.count != 1 else ""
        return f"st<{self.dtype},{self.rows}x{self.cols}>{n}"

    @property
    def nbytes(self) -> int:
        return self.rows * self.cols * self.dtype.bytes * self.count

    def element(self) -> "SharedTileType":
        """One buffer of the stack -- what an index into it has as its type."""
        return SharedTileType(self.dtype, self.rows, self.cols)

    def cpp(self) -> str:
        """The *element* type, for every count, for the reason SharedVecType
        gives: `st<T,R,C>[N]` is a declarator and not a type name."""
        return f"kittens::st<{self.dtype.cpp}, {self.rows}, {self.cols}>"


@dataclass(frozen=True)
class GranuleSetType(Type):
    """The per-lane LDS addresses of one shared tile, factored.

    `kittens::lds_granules<ST>` -- see
    include/rdna3/ops/warp/memory/util/granule.cuh. A fragment read off a
    shared tile normally forms its own full address; that is fine for a tile or
    two and fatal for a loop that reads dozens of fragments out of the same
    tile, because every one of those addresses is loop-invariant and they all
    get hoisted and then spilled. The attention kernel measured 372 B/lane of
    scratch that way, and this IR measured 312 doing it through
    `subtile` + `load_shared`.

    `st::idx` is affine in the fragment coordinate, so the entire (row, col)
    dependence folds into ds_read_b128's 16-bit immediate and what is left is
    `swizzle_bytes/16` lane-dependent addresses for the whole tile, however
    many fragments come out of it. That number is `n` below, and it is real
    register pressure, which is why this is a type and not a code-generation
    detail.
    """

    tile: "SharedTileType"

    def __post_init__(self):
        if self.tile.dtype.bytes != 2:
            raise ValueError(
                f"granules of {self.tile}: the decomposition is derived and "
                f"checked for 2-byte tiles only."
            )
        if self.tile.count != 1:
            raise ValueError(
                f"granules of {self.tile}, a stack of {self.tile.count}. "
                f"Index it first -- the two buffers of a pair differ in one "
                f"address bit, so one granule set serves both via a swap, but "
                f"it still has to be built against a particular buffer."
            )
        if self.swizzle_bytes < 64:
            raise ValueError(
                f"granules of {self.tile}: swizzle_bytes is "
                f"{self.swizzle_bytes}. At 32 the row stride is small enough "
                f"that 16 rows reach bit 9 of the offset, the swizzle key "
                f"changes with the row, and the row term stops being a "
                f"compile-time addend. Needs >= 32 columns of a 2-byte tile."
            )

    @property
    def swizzle_bytes(self) -> int:
        """Transcribed from `st<T,R,C>::swizzle_bytes` in
        include/rdna3/types/shared/st.cuh. 2-byte case only, which is all this
        type admits."""
        width = self.tile.cols // 16
        return 128 if width % 4 == 0 else 64 if width % 2 == 0 else 32

    @property
    def n(self) -> int:
        """Addresses held, hence VGPRs held, for as long as the set is live."""
        return self.swizzle_bytes // 16

    def __repr__(self) -> str:
        return f"granules<{self.tile}>x{self.n}"

    def cpp(self) -> str:
        return f"kittens::lds_granules<{self.tile.cpp()}>"


@dataclass(frozen=True)
class StageBufferType(Type):
    """`float4 buf[calls]` -- the VGPR half of a global->LDS copy.

    gfx11 has no global->LDS DMA, so the asynchronous path to LDS is two ops
    with the data sitting in registers in between: issue the global loads into
    this buffer, do something else, then write the buffer to LDS. That makes
    the buffer a *live range in the register file* for as long as the copy is
    in flight, which is why it is a type here rather than a detail of the load:
    its size is register pressure, and register pressure on this target is the
    difference between occupancy 2 and a spill.

    `calls` is transcribed from `stage_calls` in
    include/rdna3/ops/warp/memory/tile/global_to_shared.cuh. It must agree with
    the C++ exactly -- it is the caller's side of the contract, and a buffer
    one float4 short is an out-of-bounds write to the stack.
    """

    tile: "SharedTileType"
    threads: int
    #: Batches kept in flight. The buffer is declared `[depth][calls]` and the
    #: slot index must be a compile-time constant, or it lands in scratch.
    depth: int = 1

    def __post_init__(self):
        if self.threads <= 0 or self.threads % 32:
            raise ValueError(f"stage buffer threads={self.threads} must be a multiple of 32")
        if self.depth <= 0:
            raise ValueError(f"stage buffer depth={self.depth} must be positive")
        if self.tile.count != 1:
            raise ValueError(
                f"stage buffer models one shared tile, not {self.tile}. A "
                f"multi-buffer is several destinations for the same staged "
                f"bytes; stage once and commit to whichever buffer is free."
            )

    @property
    def calls(self) -> int:
        per_float4 = 16 // self.tile.dtype.bytes
        n = self.tile.rows * self.tile.cols
        return -(-(n // per_float4) // self.threads)

    @property
    def floats(self) -> int:
        """float4s held, hence 4x this many VGPRs while the copy is in flight."""
        return self.calls * self.depth

    def __repr__(self) -> str:
        d = f"[{self.depth}]" if self.depth != 1 else ""
        return f"stage<{self.tile},{self.threads}t>{d}x{self.calls}"

    def cpp(self) -> str:
        return "float4"


@dataclass(frozen=True)
class SharedVecType(Type):
    """An LDS vector: sv<T, length>.

    The only thing in the IR that exists for cross-warp communication. A
    register vector is per-warp by construction -- `row_sum` reduces within one
    wave and nothing about the type says which wave -- so a workgroup that
    splits a row across its warps has to land the partials somewhere both can
    read, and this is that somewhere.

    `count` is how many warps' worth are stacked: the allocator hands out one
    contiguous block and the emitter indexes it, rather than N separate
    allocations that the 64 KB accounting would have to chase separately.
    """

    dtype: DType
    length: int
    count: int = 1

    def __post_init__(self):
        if self.length <= 0 or self.length % 16:
            raise ValueError(
                f"shared vector length={self.length} must be a positive multiple "
                f"of 16 -- sv asserts length % TILE_ROW_DIM == 0"
            )
        if self.count <= 0:
            raise ValueError(f"shared vector count={self.count} must be positive")

    def __repr__(self) -> str:
        n = f"[{self.count}]" if self.count != 1 else ""
        return f"sv<{self.dtype},{self.length}>{n}"

    @property
    def nbytes(self) -> int:
        return self.length * self.count * self.dtype.bytes

    def element(self) -> "SharedVecType":
        """One entry of the stack -- the type the emitter declares."""
        return SharedVecType(self.dtype, self.length, 1)

    def cpp(self) -> str:
        """The *element* type, for every count.

        `sv<T,L>[N]` is a declarator, not a type name: it is spellable in the
        line that declares the array and nowhere else. Since every other use
        site -- the load, the store -- names one entry anyway, cpp() gives the
        entry and the emitter writes the `[N]` where C++ allows it.
        """
        return f"kittens::sv<{self.dtype.cpp}, {self.length}>"


@dataclass(frozen=True, eq=False)
class RegVecType(Type):
    """A register vector: one entry per row (kind="col") or per column
    (kind="row") of the tile it came off.

    It carries that tile rather than just a length, and the emitter names it
    `typename <tile>::col_vec`, because an rv's layout is not free: `row_max`
    on an rt<float,16,64,row> produces a vector in a specific replication
    across lanes, and a vector declared with any other layout compiles and
    reduces the wrong lanes. Deriving the type from the tile makes the wrong
    one unspellable.

    The naming follows include/rdna3 and is worth stating once because it reads
    backwards: `row_max` reduces *along* a row, giving one value *per* row,
    which is a column vector. `add_row(dst, src, v)` takes that same column
    vector.
    """

    tile: RegTileType
    kind: str  # "col" -> one per row; "row" -> one per column
    #: Every entry holds the same value. Not a property of the C++ type -- the
    #: storage is identical -- but a property of how the value was produced,
    #: and the only producer that establishes it is lang.collective.fold_rows,
    #: whose whole mechanism is a reduction broadcast back across the entries.
    #:
    #: It is in the type rather than on the Value because it has to survive
    #: `out=`: the IR is not SSA (emit_into puts the destination at
    #: operands[0]), so a value can be its own ancestor and a backwards walk to
    #: rediscover uniformity does not terminate. Carried in the type, the
    #: elementwise ops propagate it for free -- scaling a uniform vector by a
    #: constant leaves it uniform -- and `store_scalar`, the one op that is
    #: wrong on anything else, simply requires it.
    #:
    #: Uniform is a *subtype*: anywhere a plain vector is wanted a uniform one
    #: will do, and never the reverse. See Type.plain.
    uniform: bool = False

    def __post_init__(self):
        if self.kind not in ("col", "row"):
            raise ValueError(f"vector kind must be col or row, got {self.kind!r}")

    @property
    def plain(self) -> "RegVecType":
        return self if not self.uniform else RegVecType(self.tile, self.kind)

    @property
    def dtype(self) -> DType:
        return self.tile.dtype

    @property
    def length(self) -> int:
        return self.tile.rows if self.kind == "col" else self.tile.cols

    def __repr__(self) -> str:
        u = ",uniform" if self.uniform else ""
        return f"rv<{self.dtype},{self.length},{self.kind}{u}>"

    def _key(self):
        """What makes two of these the same C++ type.

        Not the tile. `rt<T,R,C,L>::row_vec` is `rv<T, C, rt_base<T,L>::
        row_vec_layout>` (rt.cuh:78) -- the *number of rows* does not appear,
        and neither does anything else about the tile beyond its dtype and its
        layout. So the row_vec of a 16x16 col tile and the row_vec of a 32x16
        col tile are one type, and the attention kernel needs them to be: its
        running max is produced by reducing 16x16 score fragments and consumed
        by `mul_col` on a 32x16 slice of the accumulator, which in C++ is the
        same `rv_fl<16>` on both sides.

        Carrying the tile in the field is still right -- it is what `cpp()`
        spells and what makes the wrong lane layout unspellable -- but letting
        dataclass equality fall out of it would reject that kernel for a
        difference the compiler cannot see.
        """
        return (self.dtype, self.length, self.kind, self.tile.layout,
                self.uniform)

    def __eq__(self, other) -> bool:
        if not isinstance(other, RegVecType):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def cpp(self) -> str:
        return f"typename {self.tile.cpp()}::{self.kind}_vec"


@dataclass(frozen=True)
class GlobalType(Type):
    """A global tensor, always 4D (b, d, r, c) -- gl<T,-1,-1,-1,-1>. Fewer dims
    are left-padded with 1 by the pybind layer, matching pyutils.cuh."""

    dtype: DType

    def __repr__(self) -> str:
        return f"gl<{self.dtype}>"

    def cpp(self) -> str:
        return f"kittens::gl<{self.dtype.cpp}, -1, -1, -1, -1>"


@dataclass(frozen=True)
class CoordType(Type):
    """A (b, d, r, c) index. `unit` is "tile" when the coordinate is in units of
    the tile being addressed (coord<RT>), "element" when it is in elements
    (coord<>). Mixing them up reads the wrong memory and is silent, so it is in
    the type."""

    unit: str = "tile"

    def __repr__(self) -> str:
        return f"coord<{self.unit}>"


# ---------------------------------------------------------------- values / ops


_ids = itertools.count()


@dataclass(eq=False)
class Value:
    type: Type
    name: str = ""
    producer: Optional["Op"] = None
    id: int = field(default_factory=lambda: next(_ids))

    def __repr__(self) -> str:
        return f"%{self.name or self.id}"

    # Arithmetic sugar. These are defined here rather than in lang/ops so that
    # an IR built by a pass reads the same as one built by tracing.
    #
    # A scalar dispatches to the s_* family, not to the tile maps: `cb * COLS`
    # inside a kernel is index arithmetic on one uniform register, and writing
    # it as `hk.s_mul(cb, COLS)` to distinguish it from a tile multiply would
    # only make the index expressions unreadable. The dispatch is on the type,
    # so which machine a line runs on is still decided by the operand, not by
    # which spelling the author happened to pick.
    def _bin(self, other, op):
        from ..lang import ops

        if isinstance(self.type, ScalarType):
            return getattr(ops, f"s_{op}")(self, other)
        return getattr(ops, op)(self, other)

    def _rbin(self, other, op):
        from ..lang import ops

        if isinstance(self.type, ScalarType):
            return getattr(ops, f"s_{op}")(other, self)
        raise TypeError(
            f"{op}: a tile or vector has to be the left operand ({other!r} {op} "
            f"{self!r}); the maps in include/rdna3 are not commutative in their "
            f"argument order even where the operation is."
        )

    def __add__(self, o):
        return self._bin(o, "add")

    def __radd__(self, o):
        return self._rbin(o, "add")

    def __sub__(self, o):
        return self._bin(o, "sub")

    def __rsub__(self, o):
        return self._rbin(o, "sub")

    def __mul__(self, o):
        return self._bin(o, "mul")

    def __rmul__(self, o):
        return self._rbin(o, "mul")

    def __truediv__(self, o):
        return self._bin(o, "div")

    # Integer-only, so they are not offered on a tile: `//` on a float tile
    # would have to round, and kittens::div does not.
    def __floordiv__(self, o):
        return self._int_only(o, "div", "//")

    def __mod__(self, o):
        return self._int_only(o, "mod", "%")

    def _int_only(self, other, op, sym):
        from ..lang import ops

        if not isinstance(self.type, ScalarType):
            raise TypeError(
                f"{sym} is index arithmetic and {self.type} is not an index. "
                f"There is no rounding division on a tile."
            )
        return getattr(ops, f"s_{op}")(self, other)

    def __neg__(self):
        from ..lang import ops

        if isinstance(self.type, ScalarType):
            return ops.s_sub(0, self)
        return ops.neg(self)

    def __getitem__(self, i):
        """One buffer of a multi-buffer LDS allocation.

        The index may be a runtime value -- it is an LDS address computation,
        not a register index, so unlike the stage buffer's slot there is no
        reason it has to be a constant.
        """
        from ..lang import ops

        return ops.shared_at(self, i)


@dataclass(eq=False)
class Op:
    """One operation.

    Note what this IR is *not*: it is not SSA. A Value is a register, and an op
    may write into one that already exists -- `attrs["inplace"]` marks the
    first operand as the destination. That is how the target language works
    (`kittens::row_sum(acc, src, acc)`), and modelling it as SSA with
    loop-carried arguments would mean the emitter had to undo the modelling
    again on the way out. The cost is that program order is meaningful: a pass
    may not reorder across an in-place write.
    """

    opcode: str
    operands: List[Value] = field(default_factory=list)
    results: List[Value] = field(default_factory=list)
    attrs: Dict[str, Any] = field(default_factory=dict)
    #: Where in the user's Python this op came from. Carried so a verifier
    #: failure points at the line that wrote the kernel, not at the emitter.
    loc: Optional[str] = None
    #: Nested ops, for structured control flow. None for everything else --
    #: an empty list means "a loop with an empty body", which is different.
    body: Optional[List["Op"]] = None
    #: Values defined by the op for its region, e.g. a loop's induction
    #: variable. In scope only inside `body`.
    block_args: List[Value] = field(default_factory=list)

    def __repr__(self) -> str:
        res = ", ".join(repr(r) for r in self.results)
        ops_ = ", ".join(repr(o) for o in self.operands)
        at = "".join(f" {k}={v!r}" for k, v in sorted(self.attrs.items()))
        lhs = f"{res} = " if res else ""
        return f"{lhs}{self.opcode}({ops_}){at}"

    @property
    def result(self) -> Value:
        if len(self.results) != 1:
            raise ValueError(f"{self.opcode} has {len(self.results)} results, not 1")
        return self.results[0]

    @property
    def dst(self) -> Optional[Value]:
        """The value this op writes, whether it made it or was handed it."""
        if self.attrs.get("inplace"):
            return self.operands[0]
        return self.results[0] if self.results else None

    def walk(self):
        """This op and every op nested inside it, in program order."""
        yield self
        for inner in self.body or ():
            yield from inner.walk()


def capture_loc(skip: int = 3) -> Optional[str]:
    """The user's line, skipping frames inside hk itself."""
    for frame in reversed(traceback.extract_stack()[:-skip]):
        if "/hk/" not in frame.filename.replace("\\", "/"):
            return f"{frame.filename}:{frame.lineno} in {frame.name}"
    return None


# ---------------------------------------------------------------- kernel


@dataclass
class Param:
    """One kernel argument. Tensors become members of the globals struct;
    constexpr params are baked into the generated source and so are part of the
    compilation cache key."""

    name: str
    type: Type
    is_const: bool = False
    const_value: Any = None


@dataclass
class KernelIR:
    name: str
    params: List[Param] = field(default_factory=list)
    body: List[Op] = field(default_factory=list)
    consts: Dict[str, Any] = field(default_factory=dict)
    arch: str = "gfx1100"
    warps: int = 1
    #: Host-side grid expression, three entries (x, y, z).
    grid: Tuple[Any, Any, Any] = (1, 1, 1)
    #: Bytes of dynamic LDS, filled in by the lds_alloc pass.
    lds_bytes: int = 0
    #: Bytes of static LDS the library declares on an op's behalf (the slot
    #: wg_arrive broadcasts through), also from lds_alloc. The hardware
    #: limit is on the sum of the two.
    lds_static: int = 0
    #: (name, offset, size) per shared tile, from lds_alloc.
    lds_map: List[Tuple[str, int, int]] = field(default_factory=list)
    #: Non-fatal findings from hk.ir.verify -- things that make the kernel slow
    #: rather than wrong.
    warnings: List[Any] = field(default_factory=list)

    @property
    def threads(self) -> int:
        return self.warps * 32

    def tensors(self) -> List[Param]:
        return [p for p in self.params if isinstance(p.type, GlobalType)]

    def const_params(self) -> List[Param]:
        return [p for p in self.params if p.is_const]

    def walk(self):
        """Every op in the kernel, nested ones included, in program order."""
        for op in self.body:
            yield from op.walk()

    def ops(self, opcode: str) -> List[Op]:
        """Every op with this opcode, at any nesting depth -- a check that
        looked only at the top level would silently ignore a loop body."""
        return [o for o in self.walk() if o.opcode == opcode]

    def value_names(self) -> Dict[int, str]:
        """A unique, deterministic name for every value in the kernel.

        Shared by `dump()` and the C++ emitter so the textual IR and the
        generated source call the same value the same thing -- when a kernel
        spills you read both, and two naming schemes make that harder than it
        needs to be. Deterministic means derived from position, not from the
        global value counter, so a golden test does not depend on what else the
        process traced first.
        """
        names: Dict[int, str] = {}
        counts: Dict[str, int] = {}

        def bind(v: Value, base: str) -> None:
            n = counts.get(base, 0)
            counts[base] = n + 1
            names[id(v)] = base if n == 0 else f"{base}{n}"

        for p in self.params:
            names[p.name] = p.name  # keyed by name; params are not Values
        for op in self.walk():
            for a in op.block_args:
                bind(a, a.name or "arg")
            for r in op.results:
                bind(r, r.name or "v")
        return names

    def dump(self) -> str:
        """Textual IR. Stable enough to golden-test against."""
        names = self.value_names()

        def fmt(v: Value) -> str:
            return "%" + names.get(id(v), v.name or str(v.id))

        sig = ", ".join(
            f"{p.name}: {p.type}" + (f" = {p.const_value!r}" if p.is_const else "")
            for p in self.params
        )
        head = (
            f"kernel @{self.name}({sig})\n"
            f"  arch={self.arch} warps={self.warps} "
            f"threads={self.threads} lds={self.lds_bytes}B grid={self.grid}\n"
        )
        lines: List[str] = []

        def emit(ops: List[Op], depth: int) -> None:
            pad = "  " * depth
            for op in ops:
                res = ", ".join(fmt(r) for r in op.results)
                ins = ", ".join(fmt(o) for o in op.operands)
                at = "".join(f" {k}={v!r}" for k, v in sorted(op.attrs.items()))
                args = (
                    f" [{', '.join(fmt(a) for a in op.block_args)}]"
                    if op.block_args
                    else ""
                )
                open_ = " {" if op.body is not None else ""
                lines.append(
                    f"{pad}{res + ' = ' if res else ''}"
                    f"{op.opcode}({ins}){at}{args}{open_}\n"
                )
                if op.body is not None:
                    emit(op.body, depth + 1)
                    lines.append(f"{pad}}}\n")

        emit(self.body, 1)
        return head + "".join(lines)
