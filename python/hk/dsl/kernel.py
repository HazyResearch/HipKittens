"""`@hk.kernel` and `@hk.func`: functions the DSL traces into hk modules.

Parameter rules, unchanged: positional parameters are tensors, annotated
`hk.Tensor[dtype]`; keyword-only parameters are constexprs, baked into the
generated source and ordinary Python values in the body -- which is what
lets `range(D // 16)` unroll and `if KIND == "softmax"` pick a branch at
trace time.

A trace is an MLIR module: the kernel `func.func` and its grid, verified and
run through the hk pass pipeline. `k.trace().dump()` prints it; `k.source()`
is the HipKittens C++ hk-translate writes from it.
"""

from __future__ import annotations

import functools
from typing import Any, Dict, List, Optional

from .._mlir import ir
from .._mlir.dialects import func as func_d
from ..compiler import mlir as M
from ..compiler.from_kernel_ir import begin_kernel, build_grid
from ..ir.nodes import DType, GlobalType, Param, ScalarType
from ..ir.verify import VerifyError
from ..lang.kernel import GL, Kernel as _KernelBase, _const_dtype
from . import builder as B
from .rewrite import rewrite
from .values import Global

#: The annotation for a tensor parameter: `x: hk.Tensor[hk.bf16]`.
Tensor = GL

#: Marks a keyword-only parameter as a constexpr. Optional: every keyword-only
#: parameter is one; the marker documents it at the definition.
Constexpr = "hk.constexpr"


def _loc_text(loc) -> str:
    """"file:line in func" out of a location the frontend built."""
    try:
        if loc.is_a_name():
            loc = loc.child_loc
        if loc.is_a_fused():
            meta = loc.fused_metadata if hasattr(loc, "fused_metadata") else None
            locs = loc.locations
            if locs and locs[0].is_a_file():
                f = locs[0]
                fn = ir.StringAttr(meta).value if meta is not None else "?"
                return f"{f.filename}:{f.start_line} in {fn}"
    except Exception:  # noqa: BLE001 -- best effort, never mask the error
        pass
    return ""


def verify_error(e: Exception) -> VerifyError:
    """An MLIRError as the VerifyError a kernel author reads."""
    diags = getattr(e, "error_diagnostics", None) or []
    if not diags:
        return VerifyError(str(e))
    d = diags[0]
    msg = str(d.message)
    if "\n    at " not in msg:
        at = _loc_text(d.location)
        if at:
            msg += f"\n    at {at}"
    return VerifyError(msg)


class TracedKernel:
    """One specialization of a kernel, traced and lowered: its hk module and
    what the module boundary needs to know about it."""

    def __init__(self, name: str, arch: str, warps: int, consts: Dict[str, Any],
                 tensors: List, module: ir.Module):
        self.name = name
        self.arch = arch
        self.warps = warps
        self.consts = consts
        self.module = module
        self._tensors = [Param(n, t) for n, t in tensors]
        self.params = self._tensors + [
            Param(k, ScalarType(_const_dtype(v)), is_const=True, const_value=v)
            for k, v in consts.items()]
        k = M.kernel_of(module)
        with M.context():
            self.lds_bytes = ir.IntegerAttr(k.attributes["hk.lds_bytes"]).value
            self.warnings = [ir.StringAttr(a).value
                             for a in ir.ArrayAttr(k.attributes["hk.warnings"])]

    @property
    def threads(self) -> int:
        return self.warps * 32

    def tensors(self) -> List[Param]:
        return list(self._tensors)

    def written_tensors(self) -> List[str]:
        """The tensors the kernel stores into, in declaration order."""
        k = M.kernel_of(self.module)
        names = [p.name for p in self._tensors]
        args = list(k.regions[0].blocks[0].arguments)
        out = set()

        def visit(op):
            if op.name in ("hk.store", "hk.store_scalar", "hk.group_store"):
                dst = op.operands[0]
                for i, a in enumerate(args):
                    if a == dst:
                        out.add(names[i])
            for r in op.regions:
                for b in r.blocks:
                    for o in b.operations:
                        visit(o)

        with M.context():
            for o in k.regions[0].blocks[0].operations:
                visit(o)
        return [n for n in names if n in out]

    def cpp(self, includes) -> str:
        return M.translate(self.module, includes)

    def dump(self) -> str:
        """The module as MLIR, values named as the C++ names them."""
        with M.context():
            try:
                return self.module.operation.get_asm(use_name_loc_as_prefix=True)
            except TypeError:
                return str(self.module)

    def __repr__(self) -> str:
        return f"<hk.TracedKernel {self.name} arch={self.arch} warps={self.warps}>"


class Kernel(_KernelBase):
    """A tile kernel. Call it to launch; `trace`, `source` and `build` stop
    earlier, and need no GPU."""

    def trace(self, **const_overrides: Any) -> TracedKernel:
        consts = self._resolve_consts(const_overrides)
        key = self._key(consts)
        got = self._ir_cache.get(key)
        if got is not None:
            return got
        arch = self.arch
        t = B.Trace(self.name, arch)
        body = rewrite(self.fn)
        with B.tracing(t), ir.Location.unknown():
            mod, f, entry = begin_kernel(self.name, arch, self.warps, consts,
                                         self._tensors)
            t.module, t.func, t.entry = mod, f, entry
            args = {n: Global(a, ty, n)
                    for (n, ty), a in zip(self._tensors, entry.arguments)}
            with t.at(ir.InsertionPoint(entry)):
                body(**args, **consts)
                func_d.ReturnOp([], loc=ir.Location.unknown())
            build_grid(mod, self.name, self._tensors, self._trace_grid(consts))
        try:
            with M.context():
                mod.operation.verify()
            M.run_pipeline(mod)
        except ir.MLIRError as e:
            raise verify_error(e) from None
        traced = self._ir_cache[key] = TracedKernel(
            self.name, arch, self.warps, dict(consts), list(self._tensors), mod)
        return traced

    def source(self, scaffold: str = "pybind", **const_overrides: Any) -> str:
        from ..codegen import scaffold as sc  # noqa: PLC0415

        traced = self.trace(**const_overrides)
        return sc.render(traced, scaffold,
                         emit=lambda _ir, includes: traced.cpp(includes))

    def __repr__(self) -> str:
        return f"<hk.Kernel {self.name} arch={self.arch} warps={self.warps}>"


def kernel(fn=None, *, arch: Optional[str] = None, warps: int = 1, grid=None,
           name: Optional[str] = None, max_vgprs: Optional[int] = None,
           min_occupancy: Optional[int] = None):
    """Mark a function as a tile kernel."""

    def wrap(f):
        return Kernel(f, arch=arch, warps=warps, grid=grid, name=name,
                      max_vgprs=max_vgprs, min_occupancy=min_occupancy)

    return wrap(fn) if fn is not None else wrap


def func(fn):
    """A device helper: traced inline where it is called, with its `for` and
    `if` rewritten like a kernel's."""

    @functools.wraps(fn)
    def call(*args, **kwargs):
        if B.current_or_none() is None:
            raise RuntimeError(f"{fn.__name__} is an @hk.func; call it from a "
                               f"kernel being traced")
        return rewrite(fn)(*args, **kwargs)

    call.__hk_func__ = fn
    return call


__all__ = ["kernel", "func", "Kernel", "TracedKernel", "Tensor", "Constexpr",
           "verify_error"]
