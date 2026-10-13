"""The process's MLIR context, and how hk's types and locations are spelled.

One context per process, with hk, arith, scf and func loaded and the hk passes
registered. Types are built by parsing their assembly and cached by string:
the dialect is the authority on what a type is, and the Python side never
grows a second definition of it to drift.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional, Sequence

from .._mlir import ir, passmanager
from .._mlir.dialects import hk as hkd

_lock = threading.Lock()
_CTX: List[ir.Context] = []
_TYPES: Dict[str, ir.Type] = {}

#: hk's dtype names -> MLIR element types.
ELEMENT = {
    "bf16": "bf16", "fp16": "f16", "fp32": "f32", "i32": "i32", "i8": "i8",
    "fp8e4m3": "f8E4M3FN", "fp8e5m2": "f8E5M2",
}


def context() -> ir.Context:
    """The context every hk module is built in. Made on first use."""
    if not _CTX:
        with _lock:
            if not _CTX:
                ctx = ir.Context()
                hkd.register_dialects(ctx)
                hkd.register_passes()
                _CTX.append(ctx)
    return _CTX[0]


def type_(spelling: str) -> ir.Type:
    """`!hk.rt<f32, 16x64, row>` and friends, parsed once."""
    t = _TYPES.get(spelling)
    if t is None:
        with context():
            t = _TYPES[spelling] = ir.Type.parse(spelling)
    return t


def element(dtype_name: str) -> str:
    return ELEMENT[dtype_name]


def loc(py_loc: Optional[str], name: str = "") -> ir.Location:
    """A location hk can read back: the value name outermost, then the
    Python function, then file:line. `py_loc` is "path:line in func"."""
    with context():
        inner = ir.Location.unknown()
        if py_loc:
            where, _, fn = py_loc.rpartition(" in ")
            path, _, line = where.rpartition(":")
            if path and line.isdigit():
                inner = ir.Location.fused(
                    [ir.Location.file(path, int(line), 0)],
                    ir.StringAttr.get(fn))
        if name:
            return ir.Location.name(name, inner)
        return inner


def run_pipeline(module: ir.Module) -> None:
    """hk-verify, hk-lds-alloc, hk-lds-pipeline, hk-check-budgets."""
    with context():
        pm = passmanager.PassManager.parse("builtin.module(hk-kernel-pipeline)")
        pm.run(module.operation)


def kernel_of(module: ir.Module) -> ir.Operation:
    for op in module.body.operations:
        if "hk.kernel" in op.attributes:
            return op
    raise ValueError("module has no hk.kernel func")


def translate(module: ir.Module, includes: Sequence[str]) -> str:
    """The kernel as HipKittens C++, without the module boundary."""
    with context():
        return hkd.translate_to_cpp(kernel_of(module), list(includes))


def error_text(e: Exception) -> str:
    """The message of the first error an MLIRError carries, without the
    location MLIR prepends -- hk's messages carry their own."""
    diags = getattr(e, "error_diagnostics", None)
    if diags:
        return str(diags[0].message)
    return str(e)


__all__ = ["context", "type_", "element", "loc", "run_pipeline", "kernel_of",
           "translate", "error_text", "ir"]
