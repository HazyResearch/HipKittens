"""Rules the IR has to satisfy, checked before anything is compiled.

Two severities, and the distinction is the point:

* **Error** -- the kernel would be *wrong*, or would not build. Raises.
* **Warning** -- the kernel would be *slow* in a way we can name. Reported, not
  raised, because a deliberate slow path is a legitimate thing to write and a
  verifier that cannot be overruled gets disabled.

The register budget is an **estimate**. The authoritative number comes from the
compiler, after codegen, via hk.runtime.compile -> hip_resources.py. The
estimate exists to fail fast and to guide the autotuner, and it says so in every
message it produces; it is never presented as measured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..target import get_target
from .nodes import (
    FP8,
    GranuleSetType,
    StageBufferType,
    CoordType,
    GlobalType,
    KernelIR,
    RegTileType,
    SharedTileType,
    Value,
)


class VerifyError(Exception):
    """The kernel is wrong or unbuildable. Never downgrade one of these."""


@dataclass
class Warning_:
    code: str
    message: str
    loc: str = ""

    def __str__(self) -> str:
        at = f"\n    at {self.loc}" if self.loc else ""
        return f"[{self.code}] {self.message}{at}"


def _fail(op, msg: str) -> "VerifyError":
    at = f"\n    at {op.loc}" if getattr(op, "loc", None) else ""
    return VerifyError(f"{msg}{at}")


# ---------------------------------------------------------------- structure


def check_structure(ir: KernelIR) -> None:
    defined = {p.name for p in ir.params if isinstance(p.type, GlobalType)}
    seen: set = set()

    for op in ir.body:
        for v in op.operands:
            if isinstance(v.type, GlobalType):
                if v.name not in defined:
                    raise _fail(op, f"{op.opcode} reads unknown tensor {v.name!r}")
            elif v.producer is None:
                raise _fail(op, f"{op.opcode} operand {v!r} has no definition")
            elif id(v) not in seen:
                raise _fail(
                    op,
                    f"{op.opcode} uses {v!r} before it is defined. This normally "
                    f"means a value escaped a loop body -- carry it with "
                    f"hk.range(..., init=[...]) instead.",
                )
        for r in op.results:
            seen.add(id(r))

    if ir.warps < 1:
        raise VerifyError(f"warps={ir.warps}: a kernel needs at least one warp")

    tgt = get_target(ir.arch)
    max_warps = tgt.max_waves_per_simd * tgt.simds_per_lds_pool
    if ir.warps > max_warps:
        raise VerifyError(
            f"warps={ir.warps} exceeds {max_warps}, the most {tgt.arch} can put "
            f"in one workgroup ({tgt.max_waves_per_simd} waves/SIMD x "
            f"{tgt.simds_per_lds_pool} SIMDs)"
        )


# ---------------------------------------------------------------- layout


def _flat(ir: KernelIR) -> List:
    """Every op in the kernel, nested ones included, in program order.

    Regions are flattened rather than skipped. For the register estimate that
    is an approximation -- a loop body's live set is scanned once, as if it ran
    once -- but the alternative in force until the GEMM arrived was to not look
    inside loops at all, which on a kernel whose entire body is one K loop
    meant estimating zero.
    """
    return [inner for op in ir.body for inner in op.walk()]


def check_layouts(ir: KernelIR) -> List[Warning_]:
    """The one that pays for itself: a col-layout 16-bit operand read out of LDS
    costs 8x the instructions of a row-layout one, and nothing about the result
    says so."""
    tgt = get_target(ir.arch)
    warns: List[Warning_] = []

    for op in _flat(ir):
        if op.opcode != "load_shared":
            continue
        d = op.dst
        if d is None:
            continue
        dst = d.type
        if not isinstance(dst, RegTileType):
            continue
        if dst.dtype.bits == 16 and dst.layout not in tgt.fast_smem_layouts:
            warns.append(
                Warning_(
                    "slow-smem-layout",
                    f"loading {dst} from LDS: a {dst.dtype} operand in "
                    f"'{dst.layout}' layout does not vectorise. Expect 16 scalar "
                    f"ds_read_u16 per base tile instead of 2 ds_read_b128 -- 8x "
                    f"the instructions. Fast layouts on {tgt.arch}: "
                    f"{', '.join(tgt.fast_smem_layouts)}. If you need the "
                    f"transpose, stage it transposed rather than loading and "
                    f"swapping: swap_layout moves data on this architecture, it "
                    f"is not a relabel.",
                    op.loc or "",
                )
            )
    return warns


# ---------------------------------------------------------------- budgets


def estimate_vgprs(ir: KernelIR) -> int:
    """Peak live register-tile VGPRs, by linear scan.

    An estimate and nothing more: it counts register tiles, staging buffers
    and granule sets -- the things whose size the IR knows -- so it misses
    loop induction variables, the addressing of anything not behind a
    granule set, and whatever else the compiler decides to keep. Treat it as a lower bound -- useful to reject a
    tiling that cannot possibly fit, useless to certify one that can.
    """
    tgt = get_target(ir.arch)
    ops = _flat(ir)
    last_use = {}
    for i, op in enumerate(ops):
        for v in op.operands:
            last_use[id(v)] = i

    live = {}
    peak = 0
    for i, op in enumerate(ops):
        for r in op.results:
            if isinstance(r.type, RegTileType):
                live[id(r)] = tgt.tile_vgprs(r.type.dtype.bits, r.type.rows, r.type.cols)
            elif isinstance(r.type, GranuleSetType):
                # Not a tile, but real and long-lived: a granule set exists to
                # be hoisted out of the loop, so it is live across all of it.
                live[id(r)] = r.type.n
            elif isinstance(r.type, StageBufferType):
                # float4s held while a global->LDS copy is in flight.
                live[id(r)] = 4 * r.type.floats
        peak = max(peak, sum(live.values()))
        for v in op.operands:
            if last_use.get(id(v)) == i:
                live.pop(id(v), None)
    return peak


def check_dtypes(ir: KernelIR) -> None:
    """fp8 register tiles exist only where the target has an fp8 WMMA."""
    tgt = get_target(ir.arch)
    for op in ir.walk():
        for v in (*op.operands, *op.results):
            ty = v.type
            if isinstance(ty, RegTileType) and ty.dtype in FP8 and ty.dtype.name not in tgt.mma:
                raise _fail(op, (
                    f"{v.name} is an {ty.dtype} register tile, and {ir.arch} has no "
                    f"fp8 WMMA or fp8 tile to put it in -- gfx1201 (RDNA4) does. "
                    f"On {ir.arch} fp8 is a storage type: load it into an fp32 or "
                    f"bf16 tile, which converts on the way in."))


def check_budgets(ir: KernelIR) -> List[Warning_]:
    tgt = get_target(ir.arch)
    warns: List[Warning_] = []

    lds_total = ir.lds_bytes + ir.lds_static
    if lds_total > tgt.lds_per_workgroup:
        static = (f" ({ir.lds_static} B of it static, declared by the library "
                  f"for wg_arrive)" if ir.lds_static else "")
        raise VerifyError(
            f"LDS request is {lds_total} B{static} but one workgroup on "
            f"{tgt.arch} gets at most {tgt.lds_per_workgroup} B -- the launch "
            f"fails with HSA_STATUS_ERROR_INVALID_ALLOCATION. Shrink a staged "
            f"tile, drop double buffering, or move a tile back into registers."
        )

    est = estimate_vgprs(ir)
    budget = tgt.vgprs_per_simd // tgt.max_waves_per_simd * tgt.max_waves_per_simd
    if est > tgt.vgprs_per_simd:
        msg = (
            f"register tiles alone need an estimated {est} VGPRs, more than the "
            f"{tgt.vgprs_per_simd} a SIMD has. This cannot be scheduled."
        )
        if tgt.verified:
            raise VerifyError(msg)
        warns.append(Warning_("vgpr-budget", msg + " (target unverified)"))
    elif est > 256:
        warns.append(
            Warning_(
                "vgpr-budget",
                f"estimated {est} VGPRs of live register tiles "
                f"(occupancy {tgt.occupancy(est)} waves/SIMD, headroom "
                f"{tgt.vgpr_headroom(est)}). This is an estimate that counts tile "
                f"storage only; the real number comes from the compiler after "
                f"codegen. If it spills, the kernel is wrong, not slow.",
            )
        )

    occ_lds = tgt.occupancy_by_lds(ir.lds_bytes, ir.warps) if ir.lds_bytes else None
    occ_reg = tgt.occupancy(est) if est else None
    if occ_lds is not None and occ_reg is not None and occ_lds < occ_reg:
        warns.append(
            Warning_(
                "lds-bound",
                f"LDS binds occupancy below registers: {occ_lds} waves/SIMD from "
                f"{ir.lds_bytes} B of LDS vs {occ_reg} from registers. The "
                f"compiler's occupancy remark reports the register number only "
                f"and will look better than what you get.",
            )
        )
    return warns


# ---------------------------------------------------------------- entry


#: Ops every warp in the workgroup has to reach. A warp that skips one and its
#: neighbours that do not are deadlocked: `s_barrier` waits for a wave count,
#: and the group staging calls have barriers inside them.
_WORKGROUP_OPS = ("barrier", "stage_load", "stage_commit", "group_load",
                  "group_store")


def check_divergence(ir: KernelIR) -> None:
    """Nothing workgroup-wide inside an `if`.

    Divergence itself is fine -- that is what exec masks are for -- and the
    attention kernel depends on it: the warps with no V^T band to stage skip
    the staging, and under causal the warps with nothing above the diagonal
    skip the math. What is not fine is skipping a rendezvous. The failure is
    a hang rather than a wrong answer, which is worse to debug and trivial to
    rule out here."""

    def walk(ops, inside: bool) -> None:
        for op in ops:
            if inside and op.opcode in _WORKGROUP_OPS:
                raise _fail(
                    op,
                    f"{op.opcode} is inside an hk.if_. It is a workgroup "
                    f"operation -- every warp has to reach it or the ones that "
                    f"do wait forever. Hoist it out and put only the work it "
                    f"guards inside the branch.",
                )
            if op.body is not None:
                walk(op.body, inside or op.opcode == "if")

    walk(ir.body, False)


def verify(ir: KernelIR) -> List[Warning_]:
    """Run every check. Raises on the first error; returns the warnings."""
    check_structure(ir)
    check_divergence(ir)
    return check_layouts(ir) + check_budgets(ir)
