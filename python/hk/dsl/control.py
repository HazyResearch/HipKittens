"""Control flow: what `for` and `if` become.

A kernel body is ordinary Python, rewritten (hk.dsl.rewrite) so that a `for`
over a range and an `if` hand their bodies to the functions here. Those
decide, from the values they are given, which kind of control flow it is:

* `for i in range(4)`, `if KIND == "softmax"`: everything is known at trace
  time, so it is Python -- the loop unrolls, the branch is taken or not.
* `for i in range(n)` with a device `n`, `if i < n`: a loop or a branch the
  kernel compiles, `scf.for` / `scf.if`.

A name a runtime loop or branch reassigns is carried: an index or a flag
becomes a loop-carried value (`iter_args`) or a branch result. A tile is a
buffer and is updated in place -- `acc += ...`, `into=` -- rather than
rebound; rebinding one inside a runtime loop is refused with that advice.

`hk.range(n, unroll=4)` asks for a runtime loop explicitly, with a pragma;
`hk.range_constexpr(n)` insists on unrolling. `with hk.if_(cond,
opaque=True)` and `with hk.scope()` keep their meaning: a branch the
optimizer cannot see through, and a region whose only job is to bound live
ranges.
"""

from __future__ import annotations

import builtins
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Sequence, Tuple

from .._mlir import ir
from .._mlir.dialects import arith, scf
from ..ir.nodes import ScalarType, i32
from . import builder as B
from .values import Bool, Int, Value, as_int, i1


class UNDEF:
    """A carried name with no value before the statement."""

    def __repr__(self) -> str:
        return "<undefined>"


UNDEF = UNDEF()  # noqa: F811 -- a singleton, named for what it is


class DynRange:
    """A range the kernel loops over at run time."""

    __slots__ = ("start", "stop", "step", "unroll")

    def __init__(self, start, stop, step, unroll: int = 1):
        self.start, self.stop, self.step, self.unroll = start, stop, step, unroll

    def __iter__(self):
        raise TypeError(
            "a runtime range is iterated by the kernel, not by Python: write "
            "the loop as a plain `for` in a function hk traces (an @hk.kernel "
            "or @hk.func), or use `with`-free hk.range(...) as the iterable of "
            "such a loop.")


def _is_static(*xs) -> bool:
    return all(not isinstance(x, Value) for x in xs)


def range(start, stop=None, step=None, *, unroll: int = 1) -> DynRange:  # noqa: A001
    """A loop the kernel runs at run time, even over constant bounds.

    `unroll` becomes `#pragma unroll N`. Leave it at 1 unless measured:
    unrolling a body that already fills the register file is how a kernel
    starts to spill, and a spill here is a wrong answer, not a slow one."""
    if stop is None:
        start, stop = 0, start
    return DynRange(start, stop, 1 if step is None else step, unroll)


def range_constexpr(start, stop=None, step=None):
    """A loop unrolled at trace time; its bounds must be Python ints."""
    if stop is None:
        start, stop = 0, start
    step = 1 if step is None else step
    if not _is_static(start, stop, step):
        raise TypeError("range_constexpr needs trace-time bounds, got a device value")
    return builtins.range(start, stop, step)


def iter_(fn, *args, **kwargs):
    """What a rewritten `for` iterates: `range` over a device bound becomes a
    runtime range; anything else is called as written."""
    if fn is builtins.range:
        if _is_static(*args):
            return builtins.range(*args)
        return range(*args)
    return fn(*args, **kwargs)


def _scalar_types(vals) -> list:
    return [isinstance(v, (Int, Bool)) for v in vals]


def for_(iterable, body: Callable, carried: Tuple, names: Sequence[str],
         iv_name: str):
    """`for <iv> in <iterable>: <body>`, with `carried` the names it rebinds."""
    if not isinstance(iterable, DynRange):
        for item in iterable:
            carried = body(item, *carried)
        return carried
    return _runtime_for(iterable, body, tuple(carried), tuple(names), iv_name)


def _promote(carried):
    """A Python int a runtime loop or branch rebinds is a device index from
    then on: `n = 0` before `for ...: n = n + 1` counts at run time."""
    return tuple(as_int(v) if isinstance(v, int) and not isinstance(v, bool)
                 else v for v in carried)


def _bound(x) -> ir.Value:
    """A loop bound: a literal stays a literal in the C++ (`for (int i = 0;
    ... ++i)`), a device value is the value."""
    if isinstance(x, int) and not isinstance(x, bool):
        return arith.ConstantOp(ir.IntegerType.get_signless(32), x,
                                loc=ir.Location.unknown()).result
    return as_int(x).v


def _runtime_for(rng: DynRange, body, carried, names, iv_name):
    t = B.current()
    lb, ub, step = (_bound(x) for x in (rng.start, rng.stop, rng.step))
    unknown = ir.Location.unknown()
    original = carried
    carried = _promote(carried)
    scalars = [i for i, v in enumerate(carried) if isinstance(v, (Int, Bool))]

    def trace_body(op, iv, args):
        with t.at(ir.InsertionPoint(op.body)):
            ins = list(carried)
            for i, a in zip(scalars, args):
                ins[i] = type(carried[i])(a, carried[i].ty)
            outs = body(Int(iv, ScalarType(i32)), *ins)
        return ins, list(outs)

    # A probe pass finds which scalars the body actually rebinds; only those
    # become iter_args. Its ops are thrown away.
    changed = []
    if scalars:
        probe = scf.ForOp(lb, ub, step, [carried[i].v for i in scalars],
                          loc=unknown)
        ins, outs = trace_body(probe, probe.induction_variable,
                               list(probe.inner_iter_args))
        changed = [i for i in scalars if outs[i] is not ins[i]]
        probe.operation.erase()

    loop = scf.ForOp(lb, ub, step, [carried[i].v for i in changed],
                     loc=t.loc())
    loop.attributes["hk.iv_name"] = ir.StringAttr.get(iv_name)
    if rng.unroll != 1:
        loop.attributes["hk.unroll"] = ir.IntegerAttr.get(
            ir.IntegerType.get_signless(64), rng.unroll)
    if changed:
        loop.attributes["hk.carried_names"] = ir.ArrayAttr.get(
            [ir.StringAttr.get(names[i]) for i in changed])
    with t.at(ir.InsertionPoint(loop.body)):
        ins = list(carried)
        for i, a in zip(changed, loop.inner_iter_args):
            ins[i] = type(carried[i])(a, carried[i].ty)
        outs = list(body(Int(loop.induction_variable, ScalarType(i32)), *ins))
        for i, (before, after) in enumerate(zip(ins, outs)):
            if i in changed:
                continue
            if after is not before and not (after is carried[i]):
                kind = ("tile" if isinstance(before, Value) else "Python value")
                raise TypeError(
                    f"`{names[i]}` is rebound inside a runtime loop, and it is "
                    f"a {kind}. A loop the kernel runs cannot change a Python "
                    f"value per iteration, and a tile is a buffer: update it "
                    f"in place -- `{names[i]} += ...`, `into={names[i]}` -- so "
                    f"every iteration writes the same registers.")
        scf.YieldOp([as_value(outs[i], ins[i]) for i in changed])
    result = list(original)
    for i, r in zip(changed, loop.results):
        result[i] = type(carried[i])(r, carried[i].ty)
    return tuple(result)


def as_value(x, like) -> ir.Value:
    if isinstance(like, Bool):
        if isinstance(x, Bool):
            return x.v
        raise TypeError(f"a condition carried through a loop became {x!r}")
    return as_int(x).v


def if_stmt(cond, then: Callable, orelse: Callable, carried: Tuple,
            names: Sequence[str]):
    """`if <cond>: <then> else: <orelse>`, with `carried` the names either
    branch assigns."""
    if not isinstance(cond, Value):
        return then(*carried) if cond else orelse(*carried)
    t = B.current()
    original = tuple(carried)
    carried = _promote(original)
    # Which names become branch results: the scalars either branch rebinds.
    # With nothing carried there is nothing to find out.
    if carried:
        probe_then = _probe(t, cond, then, carried)
        probe_else = _probe(t, cond, orelse, carried)
    else:
        probe_then = probe_else = ()
    results = []
    for i, (a, b) in enumerate(zip(probe_then, probe_else)):
        if a is carried[i] and b is carried[i]:
            continue
        both = [x for x in (a, b) if x is not UNDEF]
        if all(isinstance(x, (Int, Bool, int)) for x in both) and len(both) == 2:
            results.append(i)
            continue
        raise TypeError(
            f"`{names[i]}` is assigned in a runtime branch, and as "
            f"{type(a).__name__} / {type(b).__name__} it cannot be a branch "
            f"result. A runtime branch can produce indices and flags; a tile "
            f"is written in place in both arms instead.")
    types = [carried[i] if carried[i] is not UNDEF else probe_then[i]
             for i in results]
    op = scf.IfOp(i1(cond), [x.v.type for x in types], has_else=True,
                  loc=t.loc())
    if results:
        op.attributes["hk.result_names"] = ir.ArrayAttr.get(
            [ir.StringAttr.get(names[i]) for i in results])
    outs = []
    for block, fn in ((op.then_block, then), (op.else_block, orelse)):
        with t.at(ir.InsertionPoint(block)):
            got = list(fn(*carried))
            scf.YieldOp([as_value(got[i], types[k]) for k, i in enumerate(results)])
            outs.append(got)
    final = list(original)
    for k, i in enumerate(results):
        final[i] = type(types[k])(op.results[k], types[k].ty)
    return tuple(final)


def _probe(t, cond, fn, carried):
    """Run a branch body into a scratch region and see what it rebinds."""
    probe = scf.IfOp(i1(cond), [], has_else=False, loc=ir.Location.unknown())
    with t.at(ir.InsertionPoint(probe.then_block)):
        got = list(fn(*carried))
        scf.YieldOp([])
    probe.operation.erase()
    return got


@contextmanager
def if_(cond, *, opaque: bool = False) -> Iterator[None]:
    """A region that runs only where `cond` holds.

    `opaque=True` hides the condition from the optimizer (an empty asm that
    claims to modify it), so the branch survives even when it folds to a
    constant -- the tool for keeping two live sets apart."""
    if not isinstance(cond, (Bool, Int)):
        raise TypeError(f"if_ condition is {cond!r}, expected a device comparison")
    t = B.current()
    op = scf.IfOp(i1(cond), [], has_else=False, loc=t.loc())
    if opaque:
        op.attributes["hk.opaque"] = ir.UnitAttr.get()
    with t.at(ir.InsertionPoint(op.then_block)):
        try:
            yield
        finally:
            scf.YieldOp([])


@contextmanager
def scope() -> Iterator[None]:
    """An always-taken region that is a boundary for the register allocator:
    one VGPR and a never-taken branch, and two live sets that no longer have
    to be held at once."""
    with if_(as_int(1), opaque=True):
        yield


__all__ = ["range", "range_constexpr", "if_", "scope", "iter_", "for_",
           "if_stmt", "UNDEF", "DynRange"]
