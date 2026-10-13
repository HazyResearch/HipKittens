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


def maximum(a, b, *, into=None):
    return binary("max", a, b, into=into)


def minimum(a, b, *, into=None):
    return binary("min", a, b, into=into)


__all__ = ["exp", "exp2", "log", "log2", "abs", "relu", "gelu", "dgelu",
           "sqrt", "rsqrt", "silu", "maximum", "minimum"]
