"""The equivalence gate between the Python IR and the hk dialect.

Temporary, like from_kernel_ir. For one traced kernel: transliterate it, run
the C++ passes, translate, and require the same outcome the Python pipeline
had -- the same error message if it rejected the kernel, else the same
warnings and byte-identical C++ for every scaffold's includes.

`HK_GATE_A=1` runs this on every kernel any trace produces (see
hk.ir.passes.run_all), which is how the whole no-GPU test suite becomes the
gate's corpus, error paths included.
"""

from __future__ import annotations

import difflib
from typing import List, Optional

from ..codegen import cpp
from . import mlir as M
from .from_kernel_ir import convert

INCLUDE_SETS = (('"kittens.cuh"', '"pyutils/hk_bind.cuh"'), ('"kittens.cuh"',))


class GateError(AssertionError):
    pass


def _diff(a: str, b: str, n: int = 40) -> str:
    lines = difflib.unified_diff(a.splitlines(), b.splitlines(), "python",
                                 "mlir", lineterm="")
    return "\n".join(list(lines)[:n])


def check(kir, python_error: Optional[str] = None,
          python_warnings: Optional[List[str]] = None) -> None:
    """Raise GateError unless the MLIR path agrees with the Python one."""
    try:
        mod = convert(kir)
    except (KeyError, NotImplementedError):
        if python_error is not None:
            return  # malformed IR the Python verifier already refused
        raise
    try:
        M.run_pipeline(mod)
    except Exception as e:  # MLIRError
        mlir_error = M.error_text(e)
        if python_error is None:
            raise GateError(f"{kir.name}: the C++ passes reject a kernel the "
                            f"Python passes accepted:\n{mlir_error}") from None
        if mlir_error != python_error:
            raise GateError(f"{kir.name}: both reject it, with different "
                            f"messages:\n python: {python_error}\n mlir:   "
                            f"{mlir_error}") from None
        return
    if python_error is not None:
        raise GateError(f"{kir.name}: the Python passes reject it and the C++ "
                        f"passes do not:\n {python_error}")
    k = M.kernel_of(mod)
    with M.context():
        warns = ([M.ir.StringAttr(a).value for a in k.attributes["hk.warnings"]]
                 if "hk.warnings" in k.attributes else [])
    want = [str(w) for w in (python_warnings or [])]
    if warns != want:
        raise GateError(f"{kir.name}: warnings differ:\n python: {want}\n "
                        f"mlir:   {warns}")
    for incs in INCLUDE_SETS:
        old = cpp.emit(kir, includes=incs)
        new = M.translate(mod, incs)
        if old != new:
            raise GateError(f"{kir.name}: generated C++ differs\n" + _diff(old, new))


__all__ = ["check", "GateError"]
