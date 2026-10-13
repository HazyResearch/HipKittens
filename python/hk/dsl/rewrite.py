"""Rewrite a kernel's `for`, `if` and `while` so device values can steer them.

    for i in range(n):          (x,) = __hk__.for_(__hk__.iter_(range, n),
        x = x + i          ->                      __hk_for_3, ..., ("x",), "i")

    if i < n:                   (x,) = __hk__.if_stmt(i < n, __hk_then_5,
        x = 1              ->                         __hk_else_5, ..., ("x",))

    while k < n:                (k,) = __hk__.while_(__hk_test_7,
        k = k + step       ->                        __hk_while_7, ..., ("k",))

The bodies, and a loop's test, become nested functions that take and return
the names the statement rebinds -- the ones defined before it or read after
it -- and hk.dsl.control decides at trace time whether the loop is Python or
the kernel's. Only three shapes are rewritten: a `for` whose iterable is a
call to something named `range` or `range_constexpr`, an `if`, and a `while`
without an `else`. A body with a `return`, `break` or `continue` at its own
level is left as Python, so it needs a trace-time condition; a device one
raises where Python asks for its truth value, with that advice.

The rewritten function keeps the original's file, line numbers and closure
cells, so a location captured inside it is the kernel author's line.
"""

from __future__ import annotations

import ast
import inspect
import itertools
import textwrap
import types
from typing import Dict, List, Optional, Set

_CACHE: Dict[types.CodeType, Optional[types.CodeType]] = {}
_counter = itertools.count()
_RANGES = ("range", "range_constexpr")


def fetch(scope: dict, names) -> tuple:
    """The current values of `names` in a frame's locals, UNDEF if unbound."""
    from .control import UNDEF  # noqa: PLC0415

    return tuple(scope.get(n, UNDEF) for n in names)


# ---------------------------------------------------------------- analysis


class _Names(ast.NodeVisitor):
    """Names loaded and stored by statements, at this scope's level."""

    def __init__(self, into_nested_loads: bool = True):
        self.loads: Set[str] = set()
        self.stores: Set[str] = set()
        self._nested = into_nested_loads

    def visit_Name(self, node):
        (self.loads if isinstance(node.ctx, ast.Load) else self.stores).add(node.id)

    def _nested_scope(self, node, name=None):
        if name:
            self.stores.add(name)
        if self._nested:
            inner = _Names()
            for child in ast.iter_child_nodes(node):
                inner.visit(child)
            self.loads |= inner.loads

    def visit_FunctionDef(self, node):
        for d in node.decorator_list:
            self.visit(d)
        self._nested_scope(node, node.name)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        self._nested_scope(node)

    def visit_ClassDef(self, node):
        self._nested_scope(node, node.name)

    def _comp(self, node):
        self._nested_scope(node)

    visit_ListComp = visit_SetComp = visit_DictComp = visit_GeneratorExp = _comp

    def visit_Import(self, node):
        for a in node.names:
            self.stores.add((a.asname or a.name).split(".")[0])

    visit_ImportFrom = visit_Import


def _loads(nodes) -> Set[str]:
    v = _Names()
    for n in nodes:
        v.visit(n)
    return v.loads


def _stores(nodes) -> Set[str]:
    v = _Names()
    for n in nodes:
        v.visit(n)
    return v.stores


def _targets(t) -> Set[str]:
    return {n.id for n in ast.walk(t) if isinstance(n, ast.Name)}


def _definite(s: ast.stmt) -> Set[str]:
    """Names `s` assigns on every path through it."""
    if isinstance(s, ast.Assign):
        return set().union(*(_targets(t) for t in s.targets
                             if isinstance(t, (ast.Name, ast.Tuple, ast.List))))
    if isinstance(s, (ast.AnnAssign, ast.AugAssign)) and isinstance(s.target, ast.Name):
        return {s.target.id}
    if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {s.name}
    if isinstance(s, (ast.Import, ast.ImportFrom)):
        return _stores([s])
    if isinstance(s, ast.If) and s.orelse:
        a = set().union(*(_definite(x) for x in s.body))
        b = set().union(*(_definite(x) for x in s.orelse))
        return a & b
    if isinstance(s, ast.With):
        out = set().union(*(_definite(x) for x in s.body))
        for item in s.items:
            if item.optional_vars is not None:
                out |= _targets(item.optional_vars)
        return out
    return set()


def _exposed(stmts, assigned=frozenset()) -> Set[str]:
    """Names `stmts` read before assigning them: what they need from before.

    A loop body or a branch may not run, so what they assign does not
    shadow a later read; a straight-line assignment does."""
    assigned = set(assigned)
    out: Set[str] = set()
    for s in stmts:
        out |= _exposed_stmt(s, assigned)
        assigned |= _definite(s)
    return out


def _exposed_stmt(s: ast.stmt, assigned: Set[str]) -> Set[str]:
    if isinstance(s, ast.For):
        inner = assigned | _targets(s.target)
        return ((_loads([s.iter]) - assigned) | _exposed(s.body, inner)
                | _exposed(s.orelse, assigned))
    if isinstance(s, ast.While):
        return ((_loads([s.test]) - assigned) | _exposed(s.body, assigned)
                | _exposed(s.orelse, assigned))
    if isinstance(s, ast.If):
        return ((_loads([s.test]) - assigned) | _exposed(s.body, assigned)
                | _exposed(s.orelse, assigned))
    if isinstance(s, ast.With):
        inner = set(assigned)
        uses = set()
        for item in s.items:
            uses |= _loads([item.context_expr]) - assigned
            if item.optional_vars is not None:
                inner |= _targets(item.optional_vars)
        return uses | _exposed(s.body, inner)
    if isinstance(s, ast.Try):
        out = _exposed(s.body, assigned) | _exposed(s.orelse, assigned)
        out |= _exposed(s.finalbody, assigned)
        for h in s.handlers:
            out |= _exposed(h.body, assigned)
        return out
    if isinstance(s, ast.Assign):
        uses = _loads([s.value])
        for t in s.targets:
            if not isinstance(t, (ast.Name, ast.Tuple, ast.List)):
                uses |= _loads([t])  # x[i] = ... reads x and i
        return uses - assigned
    if isinstance(s, ast.AugAssign):
        return (_loads([s.value]) | _targets(s.target) | _loads([s.target])) - assigned
    return _loads([s]) - assigned


class _Jumps(ast.NodeVisitor):
    """Whether a block returns, breaks or continues at its own level."""

    def __init__(self):
        self.found = False

    def visit_Return(self, node): self.found = True
    def visit_Break(self, node): self.found = True
    def visit_Continue(self, node): self.found = True
    def visit_Yield(self, node): self.found = True
    def visit_YieldFrom(self, node): self.found = True
    def visit_Global(self, node): self.found = True
    def visit_Nonlocal(self, node): self.found = True

    # A loop's own break/continue are its business; a nested def's return too.
    def visit_For(self, node):
        _Jumps._only_returns(self, node.body + node.orelse)

    visit_While = visit_AsyncFor = visit_For

    def visit_FunctionDef(self, node): pass
    visit_AsyncFunctionDef = visit_Lambda = visit_ClassDef = visit_FunctionDef

    @staticmethod
    def _only_returns(outer, body):
        for n in ast.walk(ast.Module(body=body, type_ignores=[])):
            if isinstance(n, (ast.Return, ast.Yield, ast.YieldFrom)):
                outer.found = True


def _jumps(body) -> bool:
    j = _Jumps()
    for n in body:
        j.visit(n)
    return j.found


def _range_call(it) -> bool:
    if not isinstance(it, ast.Call):
        return False
    f = it.func
    name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
    return name in _RANGES


# ---------------------------------------------------------------- transform


def _hk(attr: str) -> ast.Attribute:
    return ast.Attribute(value=ast.Name(id="__hk__", ctx=ast.Load()), attr=attr,
                         ctx=ast.Load())


def _tuple(names, ctx) -> ast.Tuple:
    return ast.Tuple(elts=[ast.Name(id=n, ctx=ctx()) for n in names], ctx=ctx())


def _strs(names) -> ast.Tuple:
    return ast.Tuple(elts=[ast.Constant(value=n) for n in names], ctx=ast.Load())


def _fetch(names) -> ast.Call:
    locs = ast.Call(func=ast.Name(id="locals", ctx=ast.Load()), args=[], keywords=[])
    return ast.Call(func=_hk("fetch"), args=[locs, _strs(names)], keywords=[])


def _def(name: str, params, body, like) -> ast.FunctionDef:
    args = ast.arguments(posonlyargs=[], args=[ast.arg(arg=p) for p in params],
                         vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None,
                         defaults=[])
    fn = _funcdef(name, args, body)
    return ast.copy_location(fn, like)


def _funcdef(name, args, body) -> ast.FunctionDef:
    fn = ast.FunctionDef(name=name, args=args, body=body, decorator_list=[],
                         returns=None)
    if "type_params" in ast.FunctionDef._fields:
        fn.type_params = []
    return fn


def _assign(names, value, like) -> ast.stmt:
    if names:
        node = ast.Assign(targets=[_tuple(names, ast.Store)], value=value)
    else:
        node = ast.Expr(value=value)
    return ast.copy_location(node, like)


class _Rewriter:
    def block(self, stmts: List[ast.stmt], defined: Set[str],
              after: Set[str]) -> List[ast.stmt]:
        out: List[ast.stmt] = []
        defined = set(defined)
        for i, s in enumerate(stmts):
            later = _exposed(stmts[i + 1:]) | after
            out.extend(self.stmt(s, defined, later))
            defined |= _stores([s])
        return out

    def stmt(self, s: ast.stmt, defined: Set[str], later: Set[str]) -> List[ast.stmt]:
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self.function(s)
            return [s]
        if (isinstance(s, ast.For) and isinstance(s.target, ast.Name)
                and not s.orelse and _range_call(s.iter) and not _jumps(s.body)):
            return self.for_(s, defined, later)
        if isinstance(s, ast.If) and not _jumps(s.body) and not _jumps(s.orelse):
            return self.if_(s, defined, later)
        if isinstance(s, ast.While) and not s.orelse and not _jumps(s.body):
            return self.while_(s, defined, later)
        # Anything else: rewrite inside its blocks, leave it as written.
        for field in ("body", "orelse", "finalbody"):
            block = getattr(s, field, None)
            if isinstance(block, list) and block and isinstance(block[0], ast.stmt):
                inner_after = later | _exposed(block)
                setattr(s, field, self.block(block, defined, inner_after))
        for h in getattr(s, "handlers", []) or []:
            h.body = self.block(h.body, defined, later | _exposed(h.body))
        return [s]

    def function(self, fn: ast.FunctionDef) -> None:
        a = fn.args
        params = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}
        params |= {x.arg for x in (a.vararg, a.kwarg) if x is not None}
        fn.body = self.block(fn.body, params, set())

    def for_(self, s: ast.For, defined, later) -> List[ast.stmt]:
        iv = s.target.id
        assigned = _stores(s.body) - {iv}
        # Carried: what the body needs from before it (read before it is
        # assigned -- from the loop's entry or the iteration before) and what
        # is read after the loop. A name the body always assigns first is
        # local to an iteration, whatever it held before.
        carried = sorted(assigned & (later | _exposed(s.body, {iv})))
        # The body runs again: what it reads before assigning is needed from
        # the iteration before.
        body_after = later | _exposed(s.body, {iv})
        body = self.block(s.body, defined | {iv}, body_after)
        body.append(ast.Return(value=_tuple(carried, ast.Load)))
        name = f"__hk_for_{s.lineno}_{next(_counter)}"
        fn = _def(name, [iv, *carried], body, s)
        it = s.iter
        it = ast.Call(func=_hk("iter_"), args=[it.func, *it.args], keywords=it.keywords)
        call = ast.Call(func=_hk("for_"),
                        args=[it, ast.Name(id=name, ctx=ast.Load()),
                              _fetch(carried), _strs(carried), ast.Constant(value=iv)],
                        keywords=[])
        return [fn, _assign(carried, call, s)]

    def if_(self, s: ast.If, defined, later) -> List[ast.stmt]:
        assigned = _stores(s.body) | _stores(s.orelse)
        # Carried: what is read after the branch, and what an arm reads
        # before assigning it (an arm is a function, so such a name has to
        # come in as an argument).
        carried = sorted(assigned & (later | _exposed(s.body) | _exposed(s.orelse)))
        k = next(_counter)
        fns = []
        for kind, block in (("then", s.body), ("else", s.orelse)):
            body = self.block(block, defined, later) if block else []
            body.append(ast.Return(value=_tuple(carried, ast.Load)))
            fns.append(_def(f"__hk_{kind}_{s.lineno}_{k}", carried, body, s))
        call = ast.Call(func=_hk("if_stmt"),
                        args=[s.test, ast.Name(id=fns[0].name, ctx=ast.Load()),
                              ast.Name(id=fns[1].name, ctx=ast.Load()),
                              _fetch(carried), _strs(carried)],
                        keywords=[ast.keyword(arg="has_else",
                                              value=ast.Constant(value=bool(s.orelse)))])
        return [*fns, _assign(carried, call, s)]

    def while_(self, s: ast.While, defined, later) -> List[ast.stmt]:
        # Carried: what the body assigns that the test reads, that the body
        # reads before assigning (from the iteration before), or that is read
        # after the loop.
        needed = later | _exposed(s.body) | _loads([s.test])
        carried = sorted(_stores(s.body) & needed)
        k = next(_counter)
        body = self.block(s.body, defined, needed)
        body.append(ast.Return(value=_tuple(carried, ast.Load)))
        test = _def(f"__hk_test_{s.lineno}_{k}", carried,
                    [ast.copy_location(ast.Return(value=s.test), s)], s)
        fn = _def(f"__hk_while_{s.lineno}_{k}", carried, body, s)
        call = ast.Call(func=_hk("while_"),
                        args=[ast.Name(id=test.name, ctx=ast.Load()),
                              ast.Name(id=fn.name, ctx=ast.Load()),
                              _fetch(carried), _strs(carried)],
                        keywords=[])
        return [test, fn, _assign(carried, call, s)]


def _compile(fn) -> Optional[types.CodeType]:
    try:
        src = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError):
        return None
    tree = ast.parse(src)
    fdef = next((n for n in tree.body if isinstance(n, (ast.FunctionDef,
                                                       ast.AsyncFunctionDef))), None)
    if fdef is None or fdef.name != fn.__code__.co_name:
        return None  # a lambda, or source that is not this function
    fdef.decorator_list = []
    _Rewriter().function(fdef)
    freevars = list(fn.__code__.co_freevars)
    wrapper = _funcdef(
        "__hk_wrap",
        ast.arguments(posonlyargs=[], args=[ast.arg(arg=n) for n in
                                            [*freevars, "__hk__"]],
                      vararg=None, kwonlyargs=[], kw_defaults=[],
                      kwarg=None, defaults=[]),
        [fdef, ast.Return(value=ast.Name(id=fdef.name, ctx=ast.Load()))])
    mod = ast.Module(body=[wrapper], type_ignores=[])
    ast.fix_missing_locations(mod)
    ast.increment_lineno(mod, fn.__code__.co_firstlineno - 1)
    code = compile(mod, fn.__code__.co_filename, "exec")
    wrap_code = next(c for c in code.co_consts if isinstance(c, types.CodeType))
    return next(c for c in wrap_code.co_consts
                if isinstance(c, types.CodeType) and c.co_name == fdef.name)


def rewrite(fn):
    """`fn` with its `for`, `if` and `while` routed through hk.dsl.control. Cached on
    the code object; a function whose source cannot be read is returned
    as it is."""
    code = fn.__code__
    if code not in _CACHE:
        _CACHE[code] = _compile(fn)
    new_code = _CACHE[code]
    if new_code is None:
        return fn
    from . import control  # noqa: PLC0415

    cells = dict(zip(code.co_freevars, fn.__closure__ or ()))
    hk_cell = types.CellType(_runtime(control))
    closure = tuple(cells[n] if n in cells else hk_cell
                    for n in new_code.co_freevars)
    new = types.FunctionType(new_code, fn.__globals__, fn.__name__,
                             fn.__defaults__, closure)
    new.__kwdefaults__ = fn.__kwdefaults__
    new.__dict__.update(fn.__dict__)
    return new


_RUNTIME: list = []


def _runtime(control):
    if not _RUNTIME:
        ns = types.SimpleNamespace(
            for_=control.for_, iter_=control.iter_, if_stmt=control.if_stmt,
            while_=control.while_, fetch=fetch, UNDEF=control.UNDEF)
        _RUNTIME.append(ns)
    return _RUNTIME[0]


__all__ = ["rewrite", "fetch"]
