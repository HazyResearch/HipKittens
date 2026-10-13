"""The skeleton of a traced kernel: hk's Python types as dialect types, the
kernel func a body is traced into, and the grid as a host func."""

from __future__ import annotations

from .._mlir import ir
from .._mlir.dialects import arith, func, hk as hkd
from ..compiler import mlir as M
from ..types import (
    CoordType, GlobalType, GranuleSetType, RegTileType, RegVecType, ScalarType,
    SharedTileType, SharedVecType, StageBufferType, Type,
)
from . import host

_HOST_ARITH = {"+": "AddIOp", "-": "SubIOp", "*": "MulIOp", "/": "DivSIOp",
               "%": "RemSIOp"}


def mlir_type(t: Type) -> ir.Type:
    """The hk dialect type for an hk type."""
    e = lambda d: M.element(d.name)  # noqa: E731
    if isinstance(t, RegTileType):
        s = f"!hk.rt<{e(t.dtype)}, {t.rows}x{t.cols}, {t.layout}>"
    elif isinstance(t, SharedTileType):
        n = f", {t.count}" if t.count != 1 else ""
        s = f"!hk.st<{e(t.dtype)}, {t.rows}x{t.cols}{n}>"
    elif isinstance(t, RegVecType):
        u = ", uniform" if t.uniform else ""
        s = (f"!hk.rv<{e(t.dtype)}, {t.tile.rows}x{t.tile.cols}, "
             f"{t.tile.layout}, {t.kind}{u}>")
    elif isinstance(t, SharedVecType):
        n = f", {t.count}" if t.count != 1 else ""
        s = f"!hk.sv<{e(t.dtype)}, {t.length}{n}>"
    elif isinstance(t, GlobalType):
        s = f"!hk.gl<{e(t.dtype)}>"
    elif isinstance(t, CoordType):
        s = f"!hk.coord<{t.unit}>"
    elif isinstance(t, GranuleSetType):
        s = f"!hk.granules<{e(t.tile.dtype)}, {t.tile.rows}x{t.tile.cols}>"
    elif isinstance(t, StageBufferType):
        d = f", {t.depth}" if t.depth != 1 else ""
        s = (f"!hk.stage<{e(t.tile.dtype)}, {t.tile.rows}x{t.tile.cols}, "
             f"{t.threads}{d}>")
    elif isinstance(t, ScalarType):
        s = M.element(t.dtype.name)
    else:
        raise TypeError(f"no hk dialect type for {t!r}")
    return M.type_(s)


def _const_attr(v):
    if isinstance(v, bool):
        return ir.BoolAttr.get(v)
    if isinstance(v, int):
        return ir.IntegerAttr.get(ir.IntegerType.get_signless(64), v)
    if isinstance(v, float):
        return ir.FloatAttr.get_f64(v)
    return ir.StringAttr.get(str(v))


def begin_kernel(name: str, arch: str, warps: int, consts: dict, tensors):
    """A module holding an empty kernel func: one argument per (name, type)
    tensor, the hk.* attributes set. Returns (module, func, entry block)."""
    mod = ir.Module.create()
    arg_types = [mlir_type(ty) for _, ty in tensors]
    arg_attrs = ir.ArrayAttr.get([
        ir.DictAttr.get({"hk.name": ir.StringAttr.get(n)}) for n, _ in tensors])
    with ir.InsertionPoint(mod.body):
        f = func.FuncOp(name, (arg_types, []))
        f.arg_attrs = arg_attrs
        f.attributes["hk.kernel"] = ir.UnitAttr.get()
        f.attributes["hk.arch"] = ir.StringAttr.get(arch)
        f.attributes["hk.warps"] = ir.IntegerAttr.get(
            ir.IntegerType.get_signless(64), warps)
        f.attributes["hk.consts"] = ir.DictAttr.get(
            {k: _const_attr(v) for k, v in consts.items()})
        entry = f.add_entry_block(arg_locs=[ir.Location.name(n) for n, _ in tensors])
    return mod, f, entry


def build_grid(mod: ir.Module, name: str, tensors, dims) -> None:
    """`<name>__grid`: the launch grid as a host function of the tensors,
    built from the grid lambda's host expressions."""
    i32 = ir.IntegerType.get_signless(32)
    arg_types = [mlir_type(ty) for _, ty in tensors]
    arg_attrs = ir.ArrayAttr.get([
        ir.DictAttr.get({"hk.name": ir.StringAttr.get(n)}) for n, _ in tensors])
    with ir.InsertionPoint(mod.body):
        g = func.FuncOp(f"{name}__grid", (arg_types, [i32] * 3))
    g.arg_attrs = arg_attrs
    g.attributes["hk.grid_of"] = ir.FlatSymbolRefAttr.get(name)
    entry = g.add_entry_block()
    args = {n: a for (n, _), a in zip(tensors, entry.arguments)}

    def expr(e):
        if isinstance(e, int):
            e = host.HostConst(e)
        if isinstance(e, host.HostConst):
            return arith.ConstantOp(i32, e.value).result
        if isinstance(e, host.HostDim):
            return hkd.ExtentOp(i32, args[e.tensor],
                                getattr(hkd.Extent, e.dim)).result
        if isinstance(e, host.HostBin):
            cls = getattr(arith, _HOST_ARITH[e.op])
            return cls(expr(e.lhs), expr(e.rhs)).result
        raise TypeError(f"grid dimension {e!r}")

    with ir.InsertionPoint(entry):
        func.ReturnOp([expr(d) for d in dims])


__all__ = ["mlir_type", "begin_kernel", "build_grid"]
