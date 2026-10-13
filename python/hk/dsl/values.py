"""DSL values: what a kernel body computes with.

Each wraps an MLIR value and the type it has, and each operation on one
appends an op to the kernel being traced. Scalars are SSA (`Int`, `Bool`);
tiles and vectors are buffers, so they also have in-place forms -- `t += u`,
`t.exp_()`, `into=` -- that write the registers they already occupy, which is
what a loop-carried accumulator needs.

    v  = x.at[row, col].load(hk.RT(hk.fp32, 16, 64))   # global -> registers
    sq += (v * v).sum(axis=1)                           # row_sum, accumulated
    o.at[row, col] = (v * scale).to(hk.bf16)            # mul_row, cast, store
    acc += a @ b.T                                      # mma_ABt into acc
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

from .._mlir import ir
from .._mlir.dialects import arith, hk as hkd
from .module import mlir_type
from ..types import (
    FP8, CoordType, DType, GlobalType, GranuleSetType, RegTileType, RegVecType,
    ScalarType, SharedTileType, SharedVecType, StageBufferType, fp32, i32,
)
from ..target import get_target
from . import builder as B

Scalar = Union[int, float, "Int"]

#: Elementwise ops include/rdna3 specialises only for fp32. Asking for one it
#: does not have is a link error at the end of a long compile, naming a
#: template rather than the line that asked for it.
_FP32_ONLY = ("gelu", "dgelu", "sqrt", "rsqrt", "silu")
_ALL_FLOAT = ("bf16", "fp16", "fp32")


def _t():
    return B.current()


def _i32():
    return ir.IntegerType.get_signless(32)


def _check_map_dtype(op: str, ty) -> None:
    want = ("fp32",) if op in _FP32_ONLY else _ALL_FLOAT
    if ty.dtype.name not in want:
        raise TypeError(
            f"{op} has no {ty.dtype.name} implementation in include/rdna3 "
            f"(it is specialised for {', '.join(want)}). Cast with "
            f"t.to(hk.fp32) first, or load into an fp32 tile -- the global "
            f"load converts for free. This is caught here because the "
            f"alternative is an undefined-symbol link error at the end of the "
            f"compile that names a template, not this line.")


# ---------------------------------------------------------------- base


class Value:
    """A device value of the kernel being traced."""

    __slots__ = ("v", "ty")

    def __init__(self, v: ir.Value, ty):
        self.v = v
        self.ty = ty

    __hash__ = object.__hash__

    @property
    def type(self):
        return self.ty

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.ty!r}>"


def wrap(v: ir.Value, ty) -> "Value":
    """The DSL value for an MLIR value of hk type `ty`."""
    if isinstance(ty, RegTileType):
        return Tile(v, ty)
    if isinstance(ty, RegVecType):
        return Vec(v, ty)
    if isinstance(ty, SharedTileType):
        return SharedTile(v, ty)
    if isinstance(ty, SharedVecType):
        return SharedVec(v, ty)
    if isinstance(ty, GranuleSetType):
        return Granules(v, ty)
    if isinstance(ty, StageBufferType):
        return Stage(v, ty)
    if isinstance(ty, CoordType):
        return Coord(v, ty)
    if isinstance(ty, GlobalType):
        return Global(v, ty)
    if isinstance(ty, ScalarType):
        return Int(v, ty)
    raise TypeError(f"no DSL value for {ty!r}")


# ---------------------------------------------------------------- scalars


def const(x: int) -> "Int":
    """An i32 literal, hoisted and emitted once per kernel."""
    t = _t()
    ty = ScalarType(i32)
    key = ("const", type(x).__name__, x, repr(ty))

    def build():
        return arith.ConstantOp(_i32(), int(x), loc=t.loc("c")).result

    return Int(t.hoisted(key, build), ty)


def as_int(x: Scalar) -> "Int":
    if isinstance(x, Int):
        return x
    if isinstance(x, Bool):
        return x.to_int()
    if isinstance(x, bool) or isinstance(x, int):
        return const(int(x))
    raise TypeError(f"expected an index (an int or a device Int), got {x!r}")


_ARITH = {"add": "AddIOp", "sub": "SubIOp", "mul": "MulIOp", "div": "DivSIOp",
          "mod": "RemSIOp", "min": "MinSIOp", "max": "MaxSIOp",
          "cdiv": "CeilDivSIOp"}
_PRED = {"lt": "slt", "le": "sle", "gt": "sgt", "ge": "sge", "eq": "eq",
         "ne": "ne"}


def scalar_op(op: str, a: Scalar, b: Scalar) -> "Int":
    t = _t()
    va, vb = as_int(a), as_int(b)
    cls = getattr(arith, _ARITH[op])
    return Int(cls(va.v, vb.v, loc=t.loc(op)).result, ScalarType(i32))


def compare(op: str, a: Scalar, b: Scalar) -> "Bool":
    t = _t()
    va, vb = as_int(a), as_int(b)
    pred = getattr(arith.CmpIPredicate, _PRED[op])
    return Bool(arith.CmpIOp(pred, va.v, vb.v, loc=t.loc(op)).result,
                ScalarType(i32))


class Int(Value):
    """A wave-uniform 32-bit integer: an index, an extent, a loop counter.

    Python's operators build device arithmetic; `//` and `%` are C++'s
    truncating division and remainder, which agree with Python's on the
    non-negative values an index takes.
    """

    __slots__ = ()

    def __add__(self, o): return scalar_op("add", self, o)
    def __radd__(self, o): return scalar_op("add", o, self)
    def __sub__(self, o): return scalar_op("sub", self, o)
    def __rsub__(self, o): return scalar_op("sub", o, self)
    def __mul__(self, o): return scalar_op("mul", self, o)
    def __rmul__(self, o): return scalar_op("mul", o, self)
    def __floordiv__(self, o): return scalar_op("div", self, o)
    def __rfloordiv__(self, o): return scalar_op("div", o, self)
    def __mod__(self, o): return scalar_op("mod", self, o)
    def __rmod__(self, o): return scalar_op("mod", o, self)
    def __neg__(self): return scalar_op("sub", 0, self)
    def __lt__(self, o): return compare("lt", self, o)
    def __le__(self, o): return compare("le", self, o)
    def __gt__(self, o): return compare("gt", self, o)
    def __ge__(self, o): return compare("ge", self, o)
    def __eq__(self, o): return compare("eq", self, o)  # type: ignore[override]
    def __ne__(self, o): return compare("ne", self, o)  # type: ignore[override]
    __hash__ = Value.__hash__

    def __truediv__(self, o):
        raise TypeError("an index divides with //; there is no rounding "
                        "division on a device integer")

    def __bool__(self):
        raise TypeError(
            "a device Int was used where Python needs a bool. A runtime "
            "condition is a comparison (`i < n`), and it can only steer an "
            "`if` or a loop the kernel compiles, not Python logic at trace "
            "time.")

    def __index__(self):
        raise TypeError(
            "a device Int was used where Python needs an int -- a tile "
            "extent, a Python range, a list index. Those are decided at "
            "trace time; make the value a constexpr parameter, or use it as "
            "a loop bound (`for i in range(n)` compiles a loop for a device "
            "n).")

    __int__ = __index__


class Bool(Value):
    """A comparison's answer: steers an `if`, or combines with & | ~."""

    __slots__ = ()

    def _logic(self, op: str, o) -> "Bool":
        t = _t()
        if isinstance(o, bool):
            o = Bool(arith.ConstantOp(ir.IntegerType.get_signless(1), int(o),
                                      loc=t.loc()).result, self.ty)
        if not isinstance(o, Bool):
            raise TypeError(f"{op} combines two conditions, not {o!r}")
        cls = {"and": arith.AndIOp, "or": arith.OrIOp, "xor": arith.XOrIOp}[op]
        return Bool(cls(self.v, o.v, loc=t.loc(op)).result, self.ty)

    def __and__(self, o): return self._logic("and", o)
    def __rand__(self, o): return self._logic("and", o)
    def __or__(self, o): return self._logic("or", o)
    def __ror__(self, o): return self._logic("or", o)
    def __xor__(self, o): return self._logic("xor", o)

    def __invert__(self):
        return self._logic("xor", True)

    def to_int(self) -> Int:
        t = _t()
        return Int(arith.ExtUIOp(_i32(), self.v, loc=t.loc()).result,
                   ScalarType(i32))

    def __bool__(self):
        raise TypeError(
            "a runtime condition was used where Python needs a bool -- an "
            "`and`, an `or`, a `not`, or an `if` with a `return` or `break` "
            "inside. Combine conditions with &, | and ~, and branch with a "
            "plain `if`, which the kernel compiles.")


def i1(x) -> ir.Value:
    """`x` as a branch condition.

    A comparison, or a wave-level any_ne -- not a bare index. `if c:` reads
    like "if c is in range" and means "if c is not zero", which on the c == 0
    warp is the opposite, so that is refused rather than compiled."""
    t = _t()
    if isinstance(x, Bool):
        return x.v
    if isinstance(x, Int):
        owner = x.v.owner if isinstance(x.v, ir.OpResult) else None
        name = owner.name if owner is not None else ""
        if name not in ("hk.any_ne", "arith.constant"):
            raise TypeError(
                "a branch condition is an index, not a comparison. Write the "
                "test you mean -- `if i > 0:` -- rather than relying on "
                "nonzero: for an index, the two differ exactly at 0.")
        return arith.TruncIOp(ir.IntegerType.get_signless(1), x.v,
                              loc=ir.Location.unknown()).result
    raise TypeError(f"a branch condition is a comparison, got {x!r}")


# ---------------------------------------------------------------- coords


class Coord(Value):
    """A (b, d, r, c) index, in tiles or in elements."""

    __slots__ = ()

    @property
    def unit(self) -> str:
        return self.ty.unit


def coord(b: Scalar = 0, d: Scalar = 0, r: Scalar = 0, c: Scalar = 0, *,
          unit: str = "tile") -> Coord:
    """A (batch, depth, row, col) index. `unit` is "tile" (C++ `coord<RT>`)
    or "element" (`coord<>`); picking the wrong one reads the wrong memory
    without any diagnostic, which is why it is in the type."""
    if unit not in ("tile", "element"):
        raise ValueError(f"coord unit must be 'tile' or 'element', got {unit!r}")
    t = _t()
    parts = [as_int(x).v for x in (b, d, r, c)]
    ty = CoordType(unit)
    return Coord(hkd.CoordOp(mlir_type(ty), *parts, loc=t.loc("idx")).result, ty)


def tile_coord(b: Scalar = 0, d: Scalar = 0, r: Scalar = 0, c: Scalar = 0) -> Coord:
    return coord(b, d, r, c, unit="tile")


def elem_coord(b: Scalar = 0, d: Scalar = 0, r: Scalar = 0, c: Scalar = 0) -> Coord:
    return coord(b, d, r, c, unit="element")


def _pad4(idx) -> tuple:
    idx = idx if isinstance(idx, tuple) else (idx,)
    if len(idx) > 4:
        raise IndexError(f"a global is 4D (b, d, r, c); got {len(idx)} indices")
    return (0,) * (4 - len(idx)) + idx


# ---------------------------------------------------------------- globals


class _Indexer:
    __slots__ = ("tensor", "unit")

    def __init__(self, tensor: "Global", unit: str):
        self.tensor, self.unit = tensor, unit

    def __getitem__(self, idx) -> "View":
        return View(self.tensor, coord(*_pad4(idx), unit=self.unit))

    def __setitem__(self, idx, value) -> None:
        self[idx].store(value)


class Global(Value):
    """A global tensor parameter, 4D (b, d, r, c).

    `x[idx]` is a view at a coordinate; `x.at[r, c]` builds an element
    coordinate and `x.tile[r, c]` a tile one, left-padding with zeros.
    """

    __slots__ = ("name",)

    def __init__(self, v, ty, name: str = ""):
        super().__init__(v, ty)
        self.name = name

    def _extent(self, axis: str) -> Int:
        t = _t()
        return Int(hkd.ExtentOp(_i32(), self.v, getattr(hkd.Extent, axis),
                                loc=t.loc(axis)).result, ScalarType(i32))

    @property
    def batch(self) -> Int: return self._extent("batch")
    @property
    def depth(self) -> Int: return self._extent("depth")
    @property
    def rows(self) -> Int: return self._extent("rows")
    @property
    def cols(self) -> Int: return self._extent("cols")

    @property
    def dtype(self) -> DType:
        return self.ty.dtype

    @property
    def at(self) -> _Indexer:
        return _Indexer(self, "element")

    @property
    def tile(self) -> _Indexer:
        return _Indexer(self, "tile")

    def __getitem__(self, idx) -> "View":
        if not isinstance(idx, Coord):
            raise TypeError(f"index a global with a coordinate (hk.coord, "
                            f"x.at[...], x.tile[...]), got {idx!r}")
        return View(self, idx)

    def __setitem__(self, idx, value) -> None:
        self[idx].store(value)


class View:
    """A global at a coordinate: what a load reads and a store writes."""

    __slots__ = ("tensor", "idx")

    def __init__(self, tensor: "Global", idx: Coord):
        self.tensor, self.idx = tensor, idx

    def load(self, tile, *, into: Optional["Value"] = None,
             bounded: bool = False):
        """Global -> register tile or vector. The element type converts on the
        way in, for free; `bounded` predicates rows past the global's last
        (they read zero and touch no memory)."""
        if not isinstance(tile, (RegTileType, RegVecType)):
            raise TypeError(f"load destination type is {tile!r}, expected a "
                            f"tile or a vector type")
        if bounded and not isinstance(tile, RegTileType):
            raise TypeError(f"load(bounded=True) predicates tile rows; {tile!r} "
                            f"is not a tile")
        return _make(hkd.LoadOp, tile, (self.tensor.v, self.idx.v), into=into,
                     name="ld", bounded=bool(bounded))

    def store(self, value, *, bounded: bool = False) -> None:
        """Register tile or vector -> global."""
        value = _vec_of(value)
        if not isinstance(value, (Tile, Vec)):
            raise TypeError(f"store value is {value!r}, expected a tile or a vector")
        if bounded and not isinstance(value, Tile):
            raise TypeError("store(bounded=True) predicates tile rows; the "
                            "value is not a tile")
        t = _t()
        hkd.StoreOp(self.tensor.v, value.v, self.idx.v, bounded=bool(bounded),
                    loc=t.loc())

    def load_scalar(self) -> Int:
        """One int32 element -> a wave-uniform scalar (a page index, a length)."""
        if self.idx.unit != "element":
            raise TypeError(
                f"load_scalar takes an element coordinate, got coord<"
                f"{self.idx.unit}>. It reads one element, so a tile-unit index "
                f"would be off by the tile size -- index with x.at[...] or "
                f"hk.elem_coord.")
        if self.tensor.dtype is not i32:
            raise TypeError(
                f"load_scalar reads an integer index, so the global must be "
                f"i32, not {self.tensor.dtype.name}. A page table, a sequence "
                f"length and a slot mapping are all int32 tensors; a float out "
                f"of memory is a load into a vector.")
        t = _t()
        return Int(hkd.LoadScalarOp(_i32(), self.tensor.v, self.idx.v,
                                    loc=t.loc("scalar")).result, ScalarType(i32))

    def store_scalar(self, vec: "Vec") -> None:
        """One entry of a *uniform* register vector -> one element."""
        if self.idx.unit != "element":
            raise TypeError(
                f"store_scalar takes an element coordinate, got coord<"
                f"{self.idx.unit}>. It writes one element, so a tile-unit index "
                f"would be off by the tile size -- index with x.at[...].")
        if not isinstance(vec, Vec) or not vec.ty.uniform:
            got = vec.ty if isinstance(vec, Value) else vec
            raise TypeError(
                f"store_scalar needs a uniform vector, got {got}. Every entry "
                f"of a uniform vector holds the same value, so writing one of "
                f"them is writing the answer; writing one entry of anything "
                f"else would pick an arbitrary row's value and be wrong on the "
                f"other fifteen -- silently, since the shapes all agree. "
                f"Uniformity comes from hk.collective.fold_rows, which is what "
                f"a FOLD'd kernel already calls to combine its row; carry that "
                f"vector here, or use x[idx].store(v) if the sixteen entries "
                f"really are sixteen rows.")
        t = _t()
        hkd.StoreScalarOp(self.tensor.v, vec.v, self.idx.v, loc=t.loc())


# ---------------------------------------------------------------- tiles


def _make(cls, ty, operands, *, into=None, name: str, **attrs):
    """Make a fresh value of type `ty`, or write `into` an existing one."""
    t = _t()
    if into is None:
        op = cls(mlir_type(ty), *operands, loc=t.loc(name), **attrs)
        return wrap(op.result, ty)
    if not isinstance(into, Value):
        raise TypeError(f"into= takes a tile or vector, got {into!r}")
    plain = getattr(into.ty, "plain", into.ty)
    want = getattr(ty, "plain", ty)
    if plain != want or (getattr(into.ty, "uniform", False)
                         and not getattr(ty, "uniform", False)):
        raise TypeError(f"into is {into.ty} but the result is {ty}. An into= "
                        f"that does not match is a different tile, not a "
                        f"conversion.")
    cls(None, *operands, dst=into.v, loc=t.loc(name), **attrs)
    return into


def _meet(a, b):
    """The result type of an elementwise op: a's, minus a refinement b lacks."""
    return a if getattr(b, "uniform", True) else a.plain


_BINARY = {"add": hkd.AddOp, "sub": hkd.SubOp, "mul": hkd.MulOp,
           "div": hkd.DivOp, "max": hkd.MaxOp, "min": hkd.MinOp}
_UNARY = {"exp": hkd.ExpOp, "exp2": hkd.Exp2Op, "log": hkd.LogOp,
          "log2": hkd.Log2Op, "abs": hkd.AbsOp, "relu": hkd.ReluOp,
          "copy": hkd.CopyOp, "gelu": hkd.GeluOp, "dgelu": hkd.DgeluOp,
          "sqrt": hkd.SqrtOp, "rsqrt": hkd.RsqrtOp, "silu": hkd.SiluOp}
_BROADCAST = {("add", "row"): hkd.AddRowOp, ("sub", "row"): hkd.SubRowOp,
              ("mul", "row"): hkd.MulRowOp, ("div", "row"): hkd.DivRowOp,
              ("add", "col"): hkd.AddColOp, ("sub", "col"): hkd.SubColOp,
              ("mul", "col"): hkd.MulColOp, ("div", "col"): hkd.DivColOp}
_REDUCE = {("max", "row"): hkd.RowMaxOp, ("min", "row"): hkd.RowMinOp,
           ("sum", "row"): hkd.RowSumOp, ("prod", "row"): hkd.RowProdOp,
           ("max", "col"): hkd.ColMaxOp, ("min", "col"): hkd.ColMinOp,
           ("sum", "col"): hkd.ColSumOp, ("prod", "col"): hkd.ColProdOp}


def binary(op: str, a: "Value", b, *, into=None):
    """Elementwise `op` on a tile or vector and another of the same type, a
    device scalar, or a number. A tile and a vector broadcast: a column
    vector (one value per row) applies along rows (`mul_row`), a row vector
    along columns."""
    if not isinstance(a, (Tile, Vec)):
        raise TypeError(f"{op}: left operand is {a!r}, expected a tile or a vector")
    if isinstance(a, Tile) and isinstance(b, Vec) and op in ("add", "sub", "mul", "div"):
        return broadcast(op, a, b, into=into)
    _check_map_dtype(op, a.ty)
    if isinstance(b, Int):
        return _make(_BINARY[op], a.ty, (a.v,), into=into, name=op, rhs=b.v)
    if isinstance(b, (int, float)) and not isinstance(b, bool):
        return _make(_BINARY[op], a.ty, (a.v,), into=into, name=op,
                     scalar=float(b))
    if not isinstance(b, (Tile, Vec)):
        raise TypeError(f"{op}: right operand is {b!r}, expected a tile, a "
                        f"vector or a number")
    if b.ty.plain != a.ty.plain:
        raise TypeError(f"{op}: operand types differ, {a.ty} vs {b.ty}. "
                        f"Elementwise ops do not broadcast between tiles.")
    return _make(_BINARY[op], _meet(a.ty, b.ty), (a.v,), into=into, name=op,
                 rhs=b.v)


def unary(op: str, a: "Value", *, into=None):
    if not isinstance(a, (Tile, Vec)):
        raise TypeError(f"{op}: operand is {a!r}, expected a tile or a vector")
    _check_map_dtype(op, a.ty)
    return _make(_UNARY[op], a.ty, (a.v,), into=into, name=op)


def broadcast(op: str, t: "Tile", v: "Vec", *, into=None) -> "Tile":
    """`op` vec[i] into every element of row i (column vector) or column i
    (row vector) of the tile."""
    axis = "row" if v.ty.kind == "col" else "col"
    if v.ty.tile.rows != t.ty.rows or v.ty.tile.cols != t.ty.cols:
        raise TypeError(f"{op}_{axis}: the vector came off a {v.ty.tile} but is "
                        f"being applied to a {t.ty}. The lane replication "
                        f"would not line up.")
    return _make(_BROADCAST[(op, axis)], t.ty, (t.v, v.v), into=into,
                 name=f"{op}_{axis}")


class Reduction:
    """`t.sum(axis=1)` before it lands: a vector `+=` folds it in place."""

    __slots__ = ("kind", "axis", "src")

    def __init__(self, kind: str, axis: str, src: "Tile"):
        self.kind, self.axis, self.src = kind, axis, src

    @property
    def op(self) -> str:
        return f"{self.axis}_{self.kind}"

    @property
    def vtype(self) -> RegVecType:
        return RegVecType(self.src.ty, "col" if self.axis == "row" else "row")

    def value(self, *, into=None, accumulate: bool = False) -> "Vec":
        if accumulate and into is None:
            raise TypeError(f"{self.op}(accumulate=True) needs into=")
        if into is not None and into.ty != self.vtype:
            raise TypeError(f"{self.op}: into is {into.ty}, but reducing "
                            f"{self.src.ty} gives {self.vtype}. A vector's "
                            f"layout is tied to the tile it came off.")
        return _make(_REDUCE[(self.kind, self.axis)], self.vtype, (self.src.v,),
                     into=into, name=self.op, accumulate=bool(accumulate))

    def __getattr__(self, name):
        # Anything else a vector does: materialize, then do it.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.value(), name)

    def __add__(self, o): return self.value() + o
    def __sub__(self, o): return self.value() - o
    def __mul__(self, o): return self.value() * o
    def __truediv__(self, o): return self.value() / o
    def __getitem__(self, key): return self.value()[key]


def _vec_of(x) -> "Vec":
    return x.value() if isinstance(x, Reduction) else x


class _Elementwise(Value):
    __slots__ = ()

    def __add__(self, o): return binary("add", self, _vec_of(o))
    def __sub__(self, o): return binary("sub", self, _vec_of(o))
    def __mul__(self, o): return binary("mul", self, _vec_of(o))
    def __truediv__(self, o): return binary("div", self, _vec_of(o))

    def __radd__(self, o): return binary("add", self, o)
    def __rmul__(self, o): return binary("mul", self, o)

    def __iadd__(self, o): return self._inplace("add", o)
    def __isub__(self, o): return self._inplace("sub", o)
    def __imul__(self, o): return self._inplace("mul", o)
    def __itruediv__(self, o): return self._inplace("div", o)

    # Index arithmetic only: `//` on a float tile would have to round, and
    # kittens::div does not.
    def __floordiv__(self, o): return self._index_only("//")
    def __rfloordiv__(self, o): return self._index_only("//")
    def __mod__(self, o): return self._index_only("%")
    def __rmod__(self, o): return self._index_only("%")

    def _index_only(self, sym):
        raise TypeError(f"{sym} is index arithmetic and {self.ty} is not an "
                        f"index. There is no rounding division on a tile.")

    def _inplace(self, op, o):
        binary(op, self, _vec_of(o), into=self)
        return self

    def __neg__(self):
        return binary("mul", self, -1.0)

    def maximum(self, o, *, into=None): return binary("max", self, o, into=into)
    def minimum(self, o, *, into=None): return binary("min", self, o, into=into)

    # unary maps, functional and in place
    def exp(self, *, into=None): return unary("exp", self, into=into)
    def exp2(self, *, into=None): return unary("exp2", self, into=into)
    def log(self, *, into=None): return unary("log", self, into=into)
    def log2(self, *, into=None): return unary("log2", self, into=into)
    def abs(self, *, into=None): return unary("abs", self, into=into)
    def relu(self, *, into=None): return unary("relu", self, into=into)
    def gelu(self, *, into=None): return unary("gelu", self, into=into)
    def dgelu(self, *, into=None): return unary("dgelu", self, into=into)
    def sqrt(self, *, into=None): return unary("sqrt", self, into=into)
    def rsqrt(self, *, into=None): return unary("rsqrt", self, into=into)
    def silu(self, *, into=None): return unary("silu", self, into=into)

    def copy(self, *, into=None): return unary("copy", self, into=into)

    def exp_(self): return unary("exp", self, into=self)
    def exp2_(self): return unary("exp2", self, into=self)
    def rsqrt_(self): return unary("rsqrt", self, into=self)

    # fills of the whole value
    def zero_(self): return zeros(self.ty, into=self)
    def fill_(self, v: float): return full(self.ty, v, into=self)
    def neg_infty_(self): return neg_infty(self.ty, into=self)


class Tile(_Elementwise):
    """A register tile: kittens::rt<T, rows, cols, layout>."""

    __slots__ = ()

    @property
    def dtype(self) -> DType:
        return self.ty.dtype

    # reductions -- axis=1 reduces along each row: one value per row
    def _reduce(self, kind: str, axis: int) -> Reduction:
        if axis not in (0, 1):
            raise ValueError(f"axis is 1 (along each row, one value per row) "
                             f"or 0 (along each column), got {axis}")
        return Reduction(kind, "row" if axis == 1 else "col", self)

    def sum(self, axis: int = 1) -> Reduction: return self._reduce("sum", axis)
    def max(self, axis: int = 1) -> Reduction: return self._reduce("max", axis)
    def min(self, axis: int = 1) -> Reduction: return self._reduce("min", axis)
    def prod(self, axis: int = 1) -> Reduction: return self._reduce("prod", axis)

    def to(self, dtype: DType, *, into=None) -> "Tile":
        """Change the element type, keeping shape and layout (kittens::copy)."""
        ty = RegTileType(dtype, self.ty.rows, self.ty.cols, self.ty.layout)
        return _make(hkd.CastOp, ty, (self.v,), into=into, name="cvt")

    @property
    def T(self) -> "Transposed":
        return Transposed(self)

    def __matmul__(self, other) -> "MatMul":
        return MatMul(self, other)

    # masks, in place: an empty mask is a no-op in the library, so a fresh
    # destination would keep garbage -- `into=` a new tile copies the source
    # in first
    def _fill(self, op: str, idx: Scalar, value: float, into=None):
        cls = {"left_fill": hkd.LeftFillOp, "right_fill": hkd.RightFillOp,
               "upper_fill": hkd.UpperFillOp, "lower_fill": hkd.LowerFillOp,
               "triu": hkd.TriuOp, "tril": hkd.TrilOp}[op]
        out = self if into is None else into
        if out is not self:
            if out.ty != self.ty:
                raise TypeError(f"{op}: into is {out.ty} but the tile is {self.ty}")
            _make(hkd.CopyOp, self.ty, (self.v,), into=out, name="copy")
        t = _t()
        cls(None, self.v, as_int(idx).v, float(value), dst=out.v,
            loc=t.loc(op))
        return out

    def left_fill_(self, idx, value=0.0, *, into=None): return self._fill("left_fill", idx, value, into)
    def right_fill_(self, idx, value=0.0, *, into=None): return self._fill("right_fill", idx, value, into)
    def upper_fill_(self, idx, value=0.0, *, into=None): return self._fill("upper_fill", idx, value, into)
    def lower_fill_(self, idx, value=0.0, *, into=None): return self._fill("lower_fill", idx, value, into)
    def triu_(self, idx, value=0.0, *, into=None): return self._fill("triu", idx, value, into)
    def tril_(self, idx, value=0.0, *, into=None): return self._fill("tril", idx, value, into)

    def make_causal(self, value: float = 0.0, *, into=None, transposed: bool = False):
        if self.ty.rows != self.ty.cols:
            raise TypeError(f"make_causal: the tile is {self.ty.rows}x"
                            f"{self.ty.cols}, not square; mask the off-diagonal "
                            f"blocks with left_fill_/right_fill_.")
        op = "make_causal_t" if transposed else "make_causal"
        cls = hkd.MakeCausalTOp if transposed else hkd.MakeCausalOp
        return _make(cls, self.ty, (self.v, float(value)), into=into, name=op)


class Transposed:
    """`t.T`: an mma operand read the other way, or a relabelling transpose."""

    __slots__ = ("tile",)

    def __init__(self, tile: Tile):
        self.tile = tile

    def value(self, *, into=None) -> Tile:
        return transpose(self.tile, into=into)

    def __matmul__(self, other) -> "MatMul":
        return MatMul(self, other)


class MatMul:
    """`a @ b` before it lands: `acc += a @ b.T` is one mma into acc."""

    __slots__ = ("a", "b")

    def __init__(self, a, b):
        self.a, self.b = a, b

    def variant(self):
        at = isinstance(self.a, Transposed)
        bt = isinstance(self.b, Transposed)
        a = self.a.tile if at else self.a
        b = self.b.tile if bt else self.b
        return "mma_" + ("At" if at else "A") + ("Bt" if bt else "B"), a, b


_MMA_CLS = {"mma_AB": hkd.MmaABOp, "mma_ABt": hkd.MmaABtOp,
            "mma_AtB": hkd.MmaAtBOp, "mma_AtBt": hkd.MmaAtBtOp}
_MMA_SPEC = {  # (A layout, B layout, D.rows from A, D.cols from B, A red, B red)
    "mma_AB": ("row", "col", "rows", "cols", "cols", "rows"),
    "mma_ABt": ("row", "row", "rows", "rows", "cols", "cols"),
    "mma_AtB": ("col", "col", "cols", "cols", "rows", "rows"),
    "mma_AtBt": ("col", "row", "cols", "rows", "rows", "cols"),
}


def mma(variant: str, a: Tile, b: Tile, acc: Tile, *, into=None) -> Tile:
    """`acc + A x B` on the WMMA unit. The variant is decided by the layouts
    you have; the wrong one is how a kernel buys eight ds_read_u16 per base
    tile instead of two ds_read_b128."""
    spec = _MMA_SPEC[variant]
    for v, what in ((a, "A"), (b, "B"), (acc, "C")):
        if not isinstance(v, Tile):
            raise TypeError(f"{variant} {what} operand is {v!r}, expected a tile")
    fp8 = a.ty.dtype in FP8 and b.ty.dtype in FP8
    if a.ty.dtype != b.ty.dtype and not fp8:
        raise TypeError(f"{variant}: A is {a.ty.dtype} and B is {b.ty.dtype}. "
                        f"WMMA takes one operand type, not a mixed pair "
                        f"(fp8 e4m3/e5m2 excepted).")
    arch = _t().arch
    if fp8 and a.ty.dtype.name not in get_target(arch).mma:
        raise TypeError(f"{variant}: fp8 operands need an fp8 WMMA, and {arch} "
                        f"has none -- gfx1201 (RDNA4) does. On {arch} keep fp8 "
                        f"as a storage type and dequantize to bf16 before the "
                        f"mma.")
    if not fp8 and a.ty.dtype.bits != 16:
        raise TypeError(f"{variant}: operands are {a.ty.dtype}; WMMA takes bf16 "
                        f"or fp16 (fp8 on gfx1201).")
    if acc.ty.dtype != fp32 or acc.ty.layout != "col":
        raise TypeError(f"{variant}: the accumulator is {acc.ty}; WMMA "
                        f"accumulates into a col-layout fp32 tile.")
    for v, want, what in ((a, spec[0], "A"), (b, spec[1], "B")):
        if v.ty.layout != want:
            other = next(n for n, sp in _MMA_SPEC.items()
                         if (a.ty.layout, b.ty.layout) == sp[:2])
            raise TypeError(
                f"{variant} wants {what} in '{want}' layout and it is "
                f"'{v.ty.layout}'. With the layouts you have, the variant that "
                f"applies is {other} -- which computes a different product, so "
                f"if it is not the one you want, the fix is to stage the operand "
                f"the other way round, not to relabel it. swap_layout moves "
                f"data on this architecture.")
    if getattr(a.ty, spec[4]) != getattr(b.ty, spec[5]):
        raise TypeError(f"{variant}: reduction extents disagree -- A.{spec[4]}="
                        f"{getattr(a.ty, spec[4])}, B.{spec[5]}="
                        f"{getattr(b.ty, spec[5])}.")
    ty = RegTileType(fp32, getattr(a.ty, spec[2]), getattr(b.ty, spec[3]), "col")
    if acc.ty != ty:
        raise TypeError(f"{variant}: A {a.ty} times B {b.ty} is {ty}, but the "
                        f"accumulator is {acc.ty}.")
    return _make(_MMA_CLS[variant], ty, (a.v, b.v, acc.v), into=into, name="mma")


def _tile_iadd(self: Tile, o):
    if isinstance(o, MatMul):
        variant, a, b = o.variant()
        mma(variant, a, b, self, into=self)
        return self
    return _Elementwise._inplace(self, "add", o)


Tile.__iadd__ = _tile_iadd  # type: ignore[assignment]


class Vec(_Elementwise):
    """A register vector: one value per row (col kind) or column (row kind)
    of the tile it came off."""

    __slots__ = ()

    def __iadd__(self, o):
        if isinstance(o, Reduction) and o.kind == "sum":
            o.value(into=self, accumulate=True)
            return self
        return self._inplace("add", o)

    def __imul__(self, o):
        if isinstance(o, Reduction) and o.kind == "prod":
            o.value(into=self, accumulate=True)
            return self
        return self._inplace("mul", o)

    def max_(self, r: Reduction) -> "Vec":
        """Fold `t.max(axis=...)` into this vector: the online max."""
        if not isinstance(r, Reduction) or r.kind != "max":
            return binary("max", self, _vec_of(r), into=self)
        r.value(into=self, accumulate=True)
        return self

    def min_(self, r: Reduction) -> "Vec":
        if not isinstance(r, Reduction) or r.kind != "min":
            return binary("min", self, _vec_of(r), into=self)
        r.value(into=self, accumulate=True)
        return self

    def __getitem__(self, key) -> "Vec":
        """`v[:, None]` / `v[None, :]`: the axis a broadcast runs along,
        spelled out. A vector broadcasts along the axis it came off anyway;
        this says so at the use and checks it."""
        def whole(k):
            return isinstance(k, slice) and k == slice(None)

        want = None
        if isinstance(key, tuple) and len(key) == 2:
            if whole(key[0]) and key[1] is None:
                want = "col"
            elif key[0] is None and whole(key[1]):
                want = "row"
        if want is None:
            raise TypeError("index a register vector only as v[:, None] (one "
                            "value per row) or v[None, :] (one per column)")
        if self.ty.kind != want:
            raise TypeError(
                f"{'v[:, None]' if want == 'col' else 'v[None, :]'} is a "
                f"{'column' if want == 'col' else 'row'} vector, and this one is "
                f"{self.ty} -- one value per "
                f"{'column' if self.ty.kind == 'row' else 'row'} of its tile.")
        return self

    def any_ne(self, other: "Vec") -> Int:
        """Wave-level: did any entry differ from `other`. A uniform branch."""
        if not isinstance(other, Vec) or self.ty.plain != other.ty.plain:
            raise TypeError(
                f"any_ne: operands are {self.ty} and "
                f"{getattr(other, 'ty', other)}. The comparison is entry by "
                f"entry, so the two have to be the same shape.")
        t = _t()
        return Int(hkd.AnyNeOp(_i32(), self.v, other.v, loc=t.loc("grew")).result,
                   ScalarType(i32))


# ---------------------------------------------------------------- creation


def zeros(ty, *, into=None):
    name = "z" if isinstance(ty, RegTileType) else "zv"
    return _make(hkd.ZeroOp, ty, (), into=into, name=name)


def full(ty, value: float, *, into=None):
    name = "f" if isinstance(ty, RegTileType) else "fv"
    return _make(hkd.FullOp, ty, (float(value),), into=into, name=name)


def neg_infty(ty, *, into=None):
    name = "ninf" if isinstance(ty, RegTileType) else "ninfv"
    return _make(hkd.NegInftyOp, ty, (), into=into, name=name)


def transpose(t: Tile, *, into=None) -> Tile:
    """Transpose by relabelling: mirrored shape, opposite layout, no data
    moved beyond the register shuffle."""
    ty = RegTileType(t.ty.dtype, t.ty.cols, t.ty.rows,
                     "col" if t.ty.layout == "row" else "row")
    return _make(hkd.TransposeOp, ty, (t.v,), into=into, name="tr")


def transpose_sep(t: Tile, *, into=None) -> Tile:
    """Transpose keeping the layout. Moves data; needs separate registers."""
    if into is t:
        raise TypeError("transpose_sep needs separate registers for source "
                        "and destination; use hk.transpose to relabel instead")
    ty = RegTileType(t.ty.dtype, t.ty.cols, t.ty.rows, t.ty.layout)
    return _make(hkd.TransposeSepOp, ty, (t.v,), into=into, name="trs")


def broadcast_row(ty: RegTileType, vec: Vec) -> Tile:
    """A tile whose every row is vec[row]."""
    return _broadcast_fill(hkd.BroadcastRowOp, ty, vec, "col")


def broadcast_col(ty: RegTileType, vec: Vec) -> Tile:
    return _broadcast_fill(hkd.BroadcastColOp, ty, vec, "row")


def _broadcast_fill(cls, ty, vec, want):
    if not isinstance(vec, Vec) or vec.ty.kind != want:
        raise TypeError(f"takes a {want} vector, got {getattr(vec, 'ty', vec)}")
    if vec.ty.tile.rows != ty.rows or vec.ty.tile.cols != ty.cols:
        raise TypeError(f"the vector came off a {vec.ty.tile}, not a {ty}")
    return _make(cls, ty, (vec.v,), name="bc")


def as_vec(vec: Vec, tile: RegTileType, *, uniform: bool = False) -> Vec:
    """Re-attach a register vector to a tile of another shape, where the C++
    types are the same: element type, length and layout."""
    want = RegVecType(tile, vec.ty.kind, vec.ty.uniform or uniform)
    if want == vec.ty:
        return vec
    if (want.dtype, want.length, tile.layout) != (
            vec.ty.dtype, vec.ty.length, vec.ty.tile.layout):
        raise TypeError(f"as_vec cannot retype {vec.ty} (off {vec.ty.tile}) as "
                        f"{want} (off {tile}); route it through LDS instead.")
    return _make(hkd.RetypeVecOp, want, (vec.v,), name="rv")


# ---------------------------------------------------------------- LDS


class SharedTile(Value):
    """An LDS tile, or a stack of them (`As[i]` picks one)."""

    __slots__ = ()

    def __getitem__(self, i: Scalar) -> "SharedTile":
        if self.ty.count == 1:
            raise TypeError(f"{self.ty} is a single LDS tile, not a stack; pass "
                            f"count= to hk.lds.alloc for a multi-buffer")
        if isinstance(i, int) and not 0 <= i < self.ty.count:
            raise IndexError(f"buffer {i} of {self.ty}")
        t = _t()
        ty = self.ty.element()
        return SharedTile(hkd.SharedAtOp(mlir_type(ty), self.v, as_int(i).v,
                                         loc=t.loc("sb")).result, ty)

    def subtile(self, rows: int, cols: int, row: Scalar = 0,
                col: Scalar = 0) -> "SharedTile":
        """A rows x cols window, indexed in units of itself; a view, so the
        swizzle still applies and loads still vectorise."""
        st = self.ty
        if st.count != 1:
            raise TypeError(f"subtile of {st}, a stack; index it first")
        for n, v, whole in (("rows", rows, st.rows), ("cols", cols, st.cols)):
            if v <= 0 or v % 16:
                raise ValueError(f"subtile {n}={v} must be a positive multiple of 16")
            if whole % v:
                raise ValueError(
                    f"subtile {n}={v} does not divide the {whole} of {st}. A "
                    f"subtile is indexed in units of itself, so a window that "
                    f"does not tile the parent has no index to give.")
        t = _t()
        ty = SharedTileType(st.dtype, rows, cols)
        return SharedTile(hkd.SubtileOp(mlir_type(ty), self.v, as_int(row).v,
                                        as_int(col).v, loc=t.loc("sub")).result, ty)

    def load(self, tile: RegTileType, *, wait: bool = True, into=None) -> Tile:
        """LDS -> register tile. `wait=False` leaves the reads in flight; the
        tile must then be named in hk.lds.wait_for before it is read."""
        st = self.ty
        if (tile.rows, tile.cols) != (st.rows, st.cols):
            raise TypeError(f"load: {tile} from {st}; register and shared tiles "
                            f"must have the same extents")
        n = lds_loads(tile, st)
        if not wait and not n:
            raise TypeError(f"load(wait=False) from {st} into {tile}: this pair "
                            f"does not take the vectorised path, so there is "
                            f"nothing to keep in flight. Asynchronous reads "
                            f"need a 'row' layout 16-bit tile of the shared "
                            f"tile's element type.")
        return _make(hkd.LoadSharedOp, tile, (self.v, n), into=into, name="lds",
                     wait=bool(wait))

    def store(self, value: Tile) -> None:
        """Register tile -> LDS."""
        t = _t()
        hkd.StoreSharedOp(self.v, value.v, loc=t.loc())

    def granules(self, name: str = "gr") -> "Granules":
        """Factor this tile's addressing into swizzle_bytes/16 registers, so a
        loop reading many fragments holds those and not one address each."""
        t = _t()
        ty = GranuleSetType(self.ty)
        return Granules(hkd.GranulesOp(mlir_type(ty), self.v,
                                       loc=t.loc(name)).result, ty)


def lds_loads(tile: RegTileType, src: SharedTileType) -> int:
    """ds_read instructions load_shared issues for this pair; 0 when it
    misses the vectorised path."""
    ok = (tile.layout == "row" and tile.dtype.bits == 16 and src.dtype == tile.dtype)
    return (tile.rows // 16) * (tile.cols // 16) * 2 if ok else 0


class Granules(Value):
    """One shared tile's per-lane LDS addresses."""

    __slots__ = ()

    def load_frag(self, row: int, col: int, tile: RegTileType, *, into=None) -> Tile:
        """One 16x16 fragment at (row, col), in units of 16. Never waits."""
        st = self.ty.tile
        if (tile.rows, tile.cols) != (16, 16) or tile.layout != "row" or tile.dtype != st.dtype:
            raise TypeError(f"load_frag into {tile} from {st}: a 16x16 'row' "
                            f"tile of the shared tile's own dtype")
        if not isinstance(row, int) or not isinstance(col, int):
            raise TypeError("load_frag coordinates are Python ints: they become "
                            "ds_read_b128's immediate")
        return _make(hkd.LoadFragOp, tile, (self.v, row, col, 2), into=into,
                     name="frag", wait=False)

    def store_frag(self, value: Tile, row: Scalar, col: int) -> None:
        """One 16x16 fragment into the tile; `row` may be a runtime value."""
        if not isinstance(col, int):
            raise TypeError("store_frag col is a Python int (it picks the granule)")
        t = _t()
        if isinstance(row, int):
            hkd.StoreFragOp(self.v, value.v, col, row=row, loc=t.loc())
        else:
            hkd.StoreFragOp(self.v, value.v, col, row_index=as_int(row).v,
                            loc=t.loc())


class SharedVec(Value):
    """A stack of LDS vectors: the only way across the warps of a workgroup."""

    __slots__ = ()

    def store(self, vec: Vec, index: Scalar = 0) -> None:
        t = _t()
        hkd.StoreSharedVecOp(self.v, vec.v, as_int(index).v, loc=t.loc())

    def load(self, vt: RegVecType, index: Scalar = 0) -> Vec:
        t = _t()
        return Vec(hkd.LoadSharedVecOp(mlir_type(vt), self.v, as_int(index).v,
                                       loc=t.loc("ldv")).result, vt)


class Stage(Value):
    """float4 buf[depth][calls]: the VGPR half of a global->LDS copy."""

    __slots__ = ("dst",)


__all__ = [
    "Value", "Int", "Bool", "Coord", "Global", "View", "Tile", "Vec",
    "SharedTile", "SharedVec", "Granules", "Stage", "Reduction", "MatMul",
    "Transposed", "const", "as_int", "scalar_op", "compare", "i1", "coord",
    "tile_coord", "elem_coord", "binary", "unary", "broadcast", "mma", "zeros",
    "full", "neg_infty", "transpose", "transpose_sep", "broadcast_row",
    "broadcast_col", "as_vec", "lds_loads", "wrap",
]
