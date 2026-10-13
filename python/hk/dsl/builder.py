"""The trace: one kernel being built into an hk MLIR module.

A kernel body runs once, as Python, with DSL values standing in for device
values; every operation on them appends an op at the current insertion point.
Two rules carry over from the original frontend because the generated C++ is
read and diffed:

* Operand-free values that are the same everywhere -- literals, blockIdx,
  the warp index -- are emitted once, at the top of the kernel, the newest
  first, so a loop body that names `block_idx.y` three times reads it once.
* Every value gets a base name (`ld`, `mul`, `row_sum`, ...), which the C++
  translation deduplicates in program order: ld, ld1, ld2.
"""

from __future__ import annotations

import contextlib
import os
import threading
import traceback
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from .._mlir import ir
from ..compiler import mlir as M

_tls = threading.local()

#: The frontend's own code. A kernel can live anywhere -- hk/ops ships some --
#: so the test for "the user's line" is the implementation's directories, not
#: a path that merely mentions hk. contextlib is the frame between a `with
#: hk.scope():` and the region it opens.
_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_INTERNAL = tuple(os.path.join(_PKG, d) + os.sep for d in ("dsl", "compiler", "_mlir"))
_CONTEXTLIB = os.path.abspath(contextlib.__file__)


def _internal(filename: str) -> bool:
    f = os.path.abspath(filename)
    return f.startswith(_INTERNAL) or f == _CONTEXTLIB


def capture_loc() -> Optional[str]:
    """The kernel author's line: the innermost frame outside the frontend,
    however many frontend frames the op went through on the way. The function
    named is the author's too: a loop or branch body the rewriter lifted into
    `__hk_for_*` reports the function it was written in."""
    frames = traceback.extract_stack()
    for i in range(len(frames) - 1, -1, -1):
        if _internal(frames[i].filename):
            continue
        j = i
        while j > 0 and (frames[j].name.startswith("__hk_")
                         or _internal(frames[j].filename)):
            j -= 1
        f = frames[i]
        return f"{f.filename}:{f.lineno} in {frames[j].name}"
    return None


class Trace:
    """The module, the kernel func, and where the next op goes."""

    def __init__(self, name: str, arch: str):
        self.name = name
        self.arch = arch
        self.ctx = M.context()
        self.module: Optional[ir.Module] = None
        self.func = None
        self.entry: Optional[ir.Block] = None
        self._ips: List[ir.InsertionPoint] = []
        self._hoisted: Dict[Any, Any] = {}
        #: Tensor parameters by name -> the kernel's block argument.
        self.params: Dict[str, ir.Value] = {}

    # -- insertion -------------------------------------------------------

    @property
    def ip(self) -> ir.InsertionPoint:
        return self._ips[-1]

    @contextmanager
    def at(self, ip: ir.InsertionPoint):
        self._ips.append(ip)
        try:
            with ip:
                yield
        finally:
            self._ips.pop()

    def loc(self, name: str = "") -> ir.Location:
        return M.loc(capture_loc(), name)

    def hoisted(self, key, build):
        """An operand-free op, emitted once at the top of the kernel.

        `build(loc)` makes the op at the current insertion point, which is
        moved to the kernel's start for the duration: the newest hoisted op
        goes first, as it always has.
        """
        got = self._hoisted.get(key)
        if got is None:
            with self.at(ir.InsertionPoint.at_block_begin(self.entry)):
                got = self._hoisted[key] = build()
        return got


def current() -> Trace:
    t = getattr(_tls, "trace", None)
    if t is None:
        raise RuntimeError(
            "no kernel is being traced. hk device operations are only valid "
            "inside a function decorated with @hk.kernel (or called from one).")
    return t


def current_or_none() -> Optional[Trace]:
    return getattr(_tls, "trace", None)


@contextmanager
def tracing(t: Trace):
    prev = getattr(_tls, "trace", None)
    _tls.trace = t
    try:
        with t.ctx:
            yield t
    finally:
        _tls.trace = prev


__all__ = ["Trace", "current", "current_or_none", "tracing", "capture_loc"]
