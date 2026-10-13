"""Lowering passes, run in order between tracing and codegen.

Each pass is a function `KernelIR -> None` that mutates in place. The order is
fixed and meaningful: allocation has to happen before the budget check can say
anything, and both have to happen before codegen can name an offset.
"""

from __future__ import annotations

from typing import List

from .. import verify as _verify
from ..nodes import KernelIR
from .lds_alloc import lds_alloc
from .lds_pipeline import lds_pipeline

#: In order. lds_pipeline runs before the budget check because it can
#: reject the kernel outright, and there is no point costing a tiling
#: that is wrong.
PIPELINE = [lds_alloc, lds_pipeline]


def run_all(ir: KernelIR) -> List["_verify.Warning_"]:
    """Lower and verify. Raises VerifyError if the kernel is wrong; returns
    warnings about kernels that are merely slow."""
    _verify.check_structure(ir)
    # Before the passes, so the error points at the op the body wrote rather
    # than at whatever lowering turned it into. The rule is about the traced
    # program and no pass introduces a workgroup op inside a branch.
    _verify.check_divergence(ir)
    _verify.check_dtypes(ir)
    for p in PIPELINE:
        p(ir)
    warns = _verify.check_layouts(ir) + _verify.check_budgets(ir)
    ir.warnings = warns
    return warns


__all__ = ["run_all", "lds_alloc", "lds_pipeline", "PIPELINE"]
