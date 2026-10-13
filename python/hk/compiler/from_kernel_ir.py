"""The old Python KernelIR, transliterated into the hk dialect.

Temporary. It exists for the equivalence gate: every kernel the old frontend
traces is converted op for op, run through the C++ passes and the C++
translation, and the C++ that comes out has to be byte-identical to what the
old Python emitter wrote. Pass-derived attributes -- LDS offsets, lgkmcnt
counts, the warnings -- are dropped on the way in, so the gate checks the C++
passes as well as the translation.
"""

from __future__ import annotations

from typing import Dict, Optional

from .._mlir import ir
from .._mlir.dialects import arith, func, hk as hkd, scf
from ..ir.nodes import (
    CoordType, GlobalType, GranuleSetType, KernelIR, Op, RegTileType,
    RegVecType, ScalarType, SharedTileType, SharedVecType, StageBufferType,
    Type, Value,
)
from ..lang import host
from . import mlir as M

_SCALAR_ARITH = {
    "add": "AddIOp", "sub": "SubIOp", "mul": "MulIOp", "div": "DivSIOp",
    "mod": "RemSIOp", "min": "MinSIOp", "max": "MaxSIOp", "cdiv": "CeilDivSIOp",
}
_PREDICATES = {"lt": "slt", "le": "sle", "gt": "sgt", "ge": "sge",
               "eq": "eq", "ne": "ne"}
_HOST_ARITH = {"+": "AddIOp", "-": "SubIOp", "*": "MulIOp", "/": "DivSIOp",
               "%": "RemSIOp"}

_BINARY = {"add": "AddOp", "sub": "SubOp", "mul": "MulOp", "div": "DivOp",
           "max": "MaxOp", "min": "MinOp"}
_UNARY = {"exp": "ExpOp", "exp2": "Exp2Op", "log": "LogOp", "log2": "Log2Op",
          "abs": "AbsOp", "relu": "ReluOp", "copy": "CopyOp", "gelu": "GeluOp",
          "dgelu": "DgeluOp", "sqrt": "SqrtOp", "rsqrt": "RsqrtOp",
          "silu": "SiluOp", "cast": "CastOp", "transpose": "TransposeOp",
          "transpose_sep": "TransposeSepOp"}
_REDUCE = {"row_max": "RowMaxOp", "row_min": "RowMinOp", "row_sum": "RowSumOp",
           "row_prod": "RowProdOp", "col_max": "ColMaxOp", "col_min": "ColMinOp",
           "col_sum": "ColSumOp", "col_prod": "ColProdOp"}
_BROADCAST = {f"{o}_{a}": f"{o.capitalize()}{a.capitalize()}Op"
              for o in ("add", "sub", "mul", "div") for a in ("row", "col")}
_FILLS = {"left_fill": "LeftFillOp", "right_fill": "RightFillOp",
          "upper_fill": "UpperFillOp", "lower_fill": "LowerFillOp",
          "triu": "TriuOp", "tril": "TrilOp"}
_MMA = {"mma_AB": "MmaABOp", "mma_ABt": "MmaABtOp", "mma_AtB": "MmaAtBOp",
        "mma_AtBt": "MmaAtBtOp"}


def mlir_type(t: Type) -> ir.Type:
    """The hk dialect type for an old IR type."""
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


class _Converter:
    def __init__(self, kir: KernelIR):
        self.kir = kir
        self.vals: Dict[int, ir.Value] = {}

    # -- values -------------------------------------------------------

    def v(self, x: Value) -> ir.Value:
        return self.vals[id(x)]

    def i32(self, x: Value) -> ir.Value:
        """`x` where an i32 is wanted. A comparison is i1 in MLIR and an int
        in the old IR; the width change is unnamed, so the C++ inlines it."""
        val = self.v(x)
        if ir.IntegerType(val.type).width == 1:
            return arith.ExtUIOp(ir.IntegerType.get_signless(32), val,
                                 loc=M.loc(None)).result
        return val

    def i1(self, x: Value) -> ir.Value:
        val = self.v(x)
        if ir.IntegerType(val.type).width != 1:
            return arith.TruncIOp(ir.IntegerType.get_signless(1), val,
                                  loc=M.loc(None)).result
        return val

    def bind(self, x: Value, op) -> None:
        self.vals[id(x)] = op.result if hasattr(op, "result") else op

    # -- kernel -------------------------------------------------------

    def module(self) -> ir.Module:
        kir = self.kir
        tensors = [p for p in kir.params if isinstance(p.type, GlobalType)]
        with M.context(), ir.Location.unknown():
            mod, f, entry = begin_kernel(kir.name, kir.arch, kir.warps,
                                         kir.consts,
                                         [(p.name, p.type) for p in tensors])
            for p, a in zip(tensors, entry.arguments):
                self.vals[("param", p.name)] = a
            with ir.InsertionPoint(entry):
                self.region(kir.body)
                func.ReturnOp([])
            build_grid(mod, kir.name, [(p.name, p.type) for p in tensors],
                       kir.grid)
        return mod

    # -- ops ----------------------------------------------------------

    def region(self, ops) -> None:
        for op in ops:
            self.op(op)

    def operand(self, x: Value) -> ir.Value:
        if isinstance(x.type, GlobalType):
            return self.vals[("param", x.name)]
        return self.v(x)

    def op(self, op: Op) -> None:
        name = op.results[0].name or "v" if op.results else ""
        L = M.loc(op.loc, name)
        a = op.attrs
        oc = op.opcode
        inplace = bool(a.get("inplace"))
        ops = list(op.operands)
        dst = self.operand(ops[0]) if inplace else None
        srcs = ops[1:] if inplace else ops
        rtype = mlir_type(op.results[0].type) if op.results else None
        res = None
        S = lambda i: self.operand(srcs[i])  # noqa: E731

        if oc == "const":
            t = mlir_type(op.results[0].type)
            v = a["value"]
            attr = (ir.IntegerAttr.get(t, int(v)) if isinstance(t, ir.IntegerType)
                    else ir.FloatAttr.get(t, float(v)))
            res = arith.ConstantOp(t, attr, loc=L)
        elif oc == "block_idx":
            res = hkd.BlockIdxOp(rtype, getattr(hkd.Axis, a["axis"]), loc=L)
        elif oc == "warp_id":
            res = hkd.WarpIdOp(rtype, loc=L)
        elif oc == "lane_id":
            res = hkd.LaneIdOp(rtype, loc=L)
        elif oc == "coord":
            res = hkd.CoordOp(rtype, *[self.i32(x) for x in srcs], loc=L)
        elif oc == "extent":
            res = hkd.ExtentOp(rtype, S(0), getattr(hkd.Extent, a["axis"]), loc=L)
        elif oc.startswith("s_") and oc[2:] in _SCALAR_ARITH:
            cls = getattr(arith, _SCALAR_ARITH[oc[2:]])
            res = cls(self.i32(srcs[0]), self.i32(srcs[1]), loc=L)
        elif oc.startswith("s_") and oc[2:] in _PREDICATES:
            pred = getattr(arith.CmpIPredicate, _PREDICATES[oc[2:]])
            res = arith.CmpIOp(pred, self.i32(srcs[0]), self.i32(srcs[1]), loc=L)
        elif oc == "s_any_ne":
            res = hkd.AnyNeOp(rtype, S(0), S(1), loc=L)
        elif oc == "load_global":
            res = hkd.LoadOp(None if inplace else rtype, S(0), S(1), dst=dst,
                             bounded=bool(a.get("bounded")), loc=L)
        elif oc == "store_global":
            hkd.StoreOp(S(0), S(1), S(2), bounded=bool(a.get("bounded")), loc=L)
        elif oc == "store_scalar":
            hkd.StoreScalarOp(S(0), S(1), S(2), loc=L)
        elif oc == "load_scalar":
            res = hkd.LoadScalarOp(rtype, S(0), S(1), loc=L)
        elif oc == "wg_arrive":
            res = hkd.WgArriveOp(rtype, S(0), S(1), self.i32(srcs[2]), loc=L)
        elif oc == "alloc_shared":
            res = hkd.AllocSharedOp(rtype, loc=L)
        elif oc == "alloc_shared_vec":
            res = hkd.AllocSharedVecOp(rtype, loc=L)
        elif oc == "shared_at":
            res = hkd.SharedAtOp(rtype, S(0), self.i32(srcs[1]), loc=L)
        elif oc == "subtile":
            res = hkd.SubtileOp(rtype, S(0), self.i32(srcs[1]),
                                self.i32(srcs[2]), loc=L)
        elif oc == "load_shared":
            res = hkd.LoadSharedOp(None if inplace else rtype, S(0),
                                   a["lds_loads"], dst=dst,
                                   wait=bool(a.get("wait", True)), loc=L)
        elif oc == "store_shared":
            hkd.StoreSharedOp(S(0), S(1), loc=L)
        elif oc == "granules":
            res = hkd.GranulesOp(rtype, S(0), loc=L)
        elif oc == "load_frag":
            res = hkd.LoadFragOp(None if inplace else rtype, S(0), a["row"],
                                 a["col"], a["lds_loads"], dst=dst,
                                 wait=bool(a.get("wait", False)), loc=L)
        elif oc == "store_frag":
            row = a["row"]
            hkd.StoreFragOp(S(0), S(1), a["col"],
                            row_index=None if row is not None else self.i32(srcs[2]),
                            row=row, loc=L)
        elif oc == "lds_wait_for":
            hkd.LdsWaitForOp([self.operand(x) for x in srcs], loc=L)
        elif oc == "store_shared_vec":
            hkd.StoreSharedVecOp(S(0), S(1), self.i32(srcs[2]), loc=L)
        elif oc == "load_shared_vec":
            res = hkd.LoadSharedVecOp(rtype, S(0), self.i32(srcs[1]), loc=L)
        elif oc == "stage_buffer":
            res = hkd.StageBufferOp(rtype, S(0), loc=L)
        elif oc == "stage_load":
            hkd.StageLoadOp(S(0), S(1), S(2), a["slot"], loc=L)
        elif oc == "stage_commit":
            hkd.StageCommitOp(S(0), S(1), a["slot"], wait=bool(a.get("wait")),
                              loc=L)
        elif oc == "vm_wait":
            hkd.VmWaitOp(a["keep"], loc=L)
        elif oc == "group_load":
            hkd.GroupLoadOp(S(0), S(1), S(2), a["threads"], loc=L)
        elif oc == "group_store":
            hkd.GroupStoreOp(S(0), S(1), S(2), a["threads"], loc=L)
        elif oc == "barrier":
            hkd.BarrierOp(drain=bool(a.get("drain")), loc=L)
        elif oc == "sched_barrier":
            hkd.SchedBarrierOp(loc=L)
        elif oc == "setprio":
            hkd.SetPrioOp(a["level"], loc=L)
        elif oc == "zero":
            res = hkd.ZeroOp(None if inplace else rtype, dst=dst, loc=L)
        elif oc == "neg_infty":
            res = hkd.NegInftyOp(None if inplace else rtype, dst=dst, loc=L)
        elif oc == "full":
            res = hkd.FullOp(None if inplace else rtype, float(a["value"]),
                             dst=dst, loc=L)
        elif oc in _BINARY:
            cls = getattr(hkd, _BINARY[oc])
            rhs = None
            if len(srcs) > 1:
                rhs = self.i32(srcs[1]) if a.get("scalar_rhs") else S(1)
            scalar = float(a["scalar"]) if "scalar" in a else None
            res = cls(None if inplace else rtype, S(0), dst=dst, rhs=rhs,
                      scalar=scalar, loc=L)
        elif oc in _UNARY:
            res = getattr(hkd, _UNARY[oc])(None if inplace else rtype, S(0),
                                           dst=dst, loc=L)
        elif oc in _REDUCE:
            res = getattr(hkd, _REDUCE[oc])(None if inplace else rtype, S(0),
                                            dst=dst,
                                            accumulate=bool(a.get("accumulate")),
                                            loc=L)
        elif oc in _BROADCAST:
            res = getattr(hkd, _BROADCAST[oc])(None if inplace else rtype, S(0),
                                               S(1), dst=dst, loc=L)
        elif oc in ("broadcast_row", "broadcast_col"):
            cls = hkd.BroadcastRowOp if oc == "broadcast_row" else hkd.BroadcastColOp
            res = cls(rtype, S(0), loc=L)
        elif oc == "retype_vec":
            res = hkd.RetypeVecOp(rtype, S(0), loc=L)
        elif oc in _FILLS:
            res = getattr(hkd, _FILLS[oc])(None if inplace else rtype, S(0),
                                           self.i32(srcs[1]), float(a["value"]),
                                           dst=dst, loc=L)
        elif oc in ("make_causal", "make_causal_t"):
            cls = hkd.MakeCausalOp if oc == "make_causal" else hkd.MakeCausalTOp
            res = cls(None if inplace else rtype, S(0), float(a["value"]),
                      dst=dst, loc=L)
        elif oc in _MMA:
            res = getattr(hkd, _MMA[oc])(None if inplace else rtype, S(0), S(1),
                                         S(2), dst=dst, loc=L)
        elif oc == "for":
            self.for_(op, L)
            return
        elif oc == "if":
            self.if_(op, L)
            return
        else:
            raise NotImplementedError(f"no hk dialect op for {oc!r}")

        if op.results:
            self.bind(op.results[0], res)

    def for_(self, op: Op, L) -> None:
        i32 = ir.IntegerType.get_signless(32)
        zero = arith.ConstantOp(i32, 0, loc=M.loc(None)).result
        one = arith.ConstantOp(i32, 1, loc=M.loc(None)).result
        f = scf.ForOp(zero, self.i32(op.operands[0]), one, loc=L)
        (iv,) = op.block_args
        f.attributes["hk.iv_name"] = ir.StringAttr.get(iv.name or "arg")
        unroll = op.attrs.get("unroll", 1)
        if unroll != 1:
            f.attributes["hk.unroll"] = ir.IntegerAttr.get(
                ir.IntegerType.get_signless(64), unroll)
        self.vals[id(iv)] = f.induction_variable
        with ir.InsertionPoint(f.body):
            self.region(op.body)
            scf.YieldOp([])

    def if_(self, op: Op, L) -> None:
        i = scf.IfOp(self.i1(op.operands[0]), loc=L)
        if op.attrs.get("opaque"):
            i.attributes["hk.opaque"] = ir.UnitAttr.get()
        with ir.InsertionPoint(i.then_block):
            self.region(op.body)
            scf.YieldOp([])


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


def convert(kir: KernelIR) -> ir.Module:
    """`kir` as an hk module: the kernel func and its grid func."""
    return _Converter(kir).module()


__all__ = ["convert", "mlir_type", "begin_kernel", "build_grid"]
