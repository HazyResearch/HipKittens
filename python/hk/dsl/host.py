"""Host-side expressions, used for grid dimensions.

A grid dimension is not a tile op -- it runs on the host, once, before launch,
and it can only mention tensor extents and constexpr parameters. Keeping it in
its own tiny expression type (rather than letting arbitrary Python into the
launch path) means the generated `grid()` is a C++ expression we can read, and
that a typo names an argument that does not exist at trace time rather than
producing a silently wrong launch geometry.

    grid=lambda p: (hk.cdiv(p.o.rows, ROWS), p.o.depth, p.o.batch)

`p.<tensor>.<batch|depth|rows|cols>` maps onto gl's accessors of the same name.
"""

from __future__ import annotations

from typing import Union

Numeric = Union[int, "HostExpr"]


class HostExpr:
    def emit(self) -> str:
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.emit()

    def _wrap(self, other: Numeric) -> "HostExpr":
        if isinstance(other, HostExpr):
            return other
        if isinstance(other, int):
            return HostConst(other)
        raise TypeError(
            f"a grid dimension may only combine tensor extents, constexpr "
            f"parameters and ints; got {type(other).__name__}"
        )

    def __add__(self, o):
        return HostBin("+", self, self._wrap(o))

    def __radd__(self, o):
        return HostBin("+", self._wrap(o), self)

    def __sub__(self, o):
        return HostBin("-", self, self._wrap(o))

    def __rsub__(self, o):
        return HostBin("-", self._wrap(o), self)

    def __mul__(self, o):
        return HostBin("*", self, self._wrap(o))

    def __rmul__(self, o):
        return HostBin("*", self._wrap(o), self)

    def __floordiv__(self, o):
        return HostBin("/", self, self._wrap(o))

    def __rfloordiv__(self, o):
        return HostBin("/", self._wrap(o), self)

    def __mod__(self, o):
        return HostBin("%", self, self._wrap(o))


class HostConst(HostExpr):
    def __init__(self, value: int):
        self.value = int(value)

    def emit(self) -> str:
        return str(self.value)


class HostDim(HostExpr):
    """One extent of one tensor parameter."""

    DIMS = ("batch", "depth", "rows", "cols")

    def __init__(self, tensor: str, dim: str):
        if dim not in self.DIMS:
            raise AttributeError(
                f"tensor {tensor!r} has no extent {dim!r}; "
                f"a global layout is 4D: {', '.join(self.DIMS)}"
            )
        self.tensor = tensor
        self.dim = dim

    def emit(self) -> str:
        return f"g.{self.tensor}.{self.dim}()"


class HostBin(HostExpr):
    def __init__(self, op: str, lhs: HostExpr, rhs: HostExpr):
        self.op, self.lhs, self.rhs = op, lhs, rhs

    def emit(self) -> str:
        return f"({self.lhs.emit()} {self.op} {self.rhs.emit()})"


def cdiv(a: Numeric, b: Numeric) -> HostExpr:
    """Ceiling division. The grid almost always wants this and integer `//`
    does not do it, which is an easy way to drop the tail block."""
    a = a if isinstance(a, HostExpr) else HostConst(a)
    b = b if isinstance(b, HostExpr) else HostConst(b)
    return HostBin("/", HostBin("-", HostBin("+", a, b), HostConst(1)), b)


class TensorRef:
    """`p.o` inside a grid lambda."""

    def __init__(self, name: str):
        self._name = name

    def __getattr__(self, dim: str) -> HostDim:
        if dim.startswith("_"):
            raise AttributeError(dim)
        return HostDim(self._name, dim)


class ParamRef:
    """`p` inside a grid lambda.

    A tensor parameter yields a TensorRef (`p.o.rows`); a constexpr parameter
    yields its value directly (`p.ROWS`), so the same object covers both and the
    lambda only ever takes one argument. Names cannot collide -- a parameter is
    either a tensor or a constexpr, never both.
    """

    def __init__(self, tensor_names, consts=None):
        self._names = set(tensor_names)
        self._consts = dict(consts or {})

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._names:
            return TensorRef(name)
        if name in self._consts:
            return HostConst(self._consts[name])
        raise AttributeError(
            f"grid expression names {name!r}, which is not a parameter of this "
            f"kernel. Tensors: {sorted(self._names)}; "
            f"constexprs: {sorted(self._consts)}"
        )
