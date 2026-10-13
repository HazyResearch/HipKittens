"""hk.math: elementwise functions of a tile or a vector."""

from __future__ import annotations

from .values import binary, unary


def _unary(name):
    def f(x, *, into=None):
        return unary(name, x, into=into)

    f.__name__ = name
    f.__doc__ = f"Elementwise {name}; kittens::{name}. `into=` writes an existing value."
    return f


exp = _unary("exp")
exp2 = _unary("exp2")
log = _unary("log")
log2 = _unary("log2")
abs = _unary("abs")  # noqa: A001
relu = _unary("relu")
gelu = _unary("gelu")
dgelu = _unary("dgelu")
sqrt = _unary("sqrt")
rsqrt = _unary("rsqrt")
silu = _unary("silu")


def _binary(name, op):
    def f(a, b, *, into=None):
        return binary(op, a, b, into=into)

    f.__name__ = name
    f.__doc__ = (f"Elementwise {name}: what the operator does, with `into=` "
                 f"naming the value it writes (which may be an operand).")
    return f


add = _binary("add", "add")
sub = _binary("sub", "sub")
mul = _binary("mul", "mul")
div = _binary("div", "div")
maximum = _binary("maximum", "max")
minimum = _binary("minimum", "min")


__all__ = ["exp", "exp2", "log", "log2", "abs", "relu", "gelu", "dgelu",
           "sqrt", "rsqrt", "silu", "add", "sub", "mul", "div", "maximum",
           "minimum"]
