"""bf16 GEMM for decode batches: C[M, N] = X[M, K] @ W[N, K]^T, M <= 64.

At decode M is the batch, the math is free, and the whole cost is reading W
once. The tiled GEMM in `ops.gemm` does that at 418-602 GB/s on Qwen3-8B's
layers even at a 16-row tiling, against ~800 GB/s that a plain streaming read
gets. Its schedule is the reason: one 18 KB batch of loads in flight per
workgroup, and a grid of N/128 workgroups -- 32 on a 4096-wide layer -- so
Little's law caps it near 32 x 18 KB per microsecond of latency. Split-K on
that schedule was measured in the C++ (`thin_config`) and tops out at 372-633.

This kernel is laid out for the read instead:

  * **W goes straight to registers.** It is used once, so staging it through
    LDS buys nothing, and W*X^T puts W's rows on the WMMA's 16-row side the way
    the paged decode kernel puts K there. The accumulator is C^T; storing its
    transpose costs register moves, not shuffles.
  * **One 16-row block of W per workgroup, its K split across the warps.**
    That is N/16 workgroups of `warps` waves -- 6144 waves on a 6144-wide
    layer instead of 48 workgroups -- and the waves are what cover latency.
  * **The split is interleaved, not blocked.** On round t warp w reads K-chunk
    t*warps + w, so at any moment a workgroup is reading `warps` adjacent
    chunks of the same 16 rows: 2 KB of each row contiguous at 16 warps,
    instead of 128 B islands 8 KB apart.

The warps' partial sums meet once, at the end, through LDS. X is read from
global every round; it is a few hundred KB and stays in L2.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .. import lang as _ops
from ..lang import GL, kernel as _kernel
from ..ir.nodes import bf16, fp32
from ..target import get_target

_TARGET = get_target("gfx1100")


def decode_gemm_kernel(name: str, *, tokens: int = 1, slices: int = 4,
                       warps: int = 16, combine_depth: int = 0,
                       resident: int = 0,
                       max_vgprs: int = 256, min_occupancy: int = 1):
    """`tokens` 16-row token tiles (M <= 16*tokens), `slices` 16-deep K slices
    per load, `warps` waves sharing each 16-row block of W.

    `combine_depth` is how many partial sums the combine reads before it
    waits; 0 is all of them. Fewer exists for wide token tiles, where all of
    them at once is 8*tokens VGPRs each and spills.

    `resident` caps the workgroups one WGP holds at once, by sizing the
    combine's LDS so that no more fit; 0 leaves it to the registers. Fewer
    concurrent streams read faster here: at 16 warps, 2 per WGP beat 4 by 11%
    on a 12288-deep layer and 7% on a 4096-wide one, for the same code.

    X and C must have exactly 16*tokens rows allocated: the loads and the
    store are whole tiles, unpredicated. K must be a multiple of
    16*slices*warps.
    """
    m_rows = 16 * tokens
    kc = 16 * slices

    ty_w = _ops.rt(bf16, 16, kc, "row")
    ty_x = _ops.rt(bf16, m_rows, kc, "row")
    ty_acc = _ops.rt(fp32, 16, m_rows, "col")      # C^T: W rows x tokens
    ty_out = _ops.rt(fp32, m_rows, 16, "row")      # its transpose
    st_part = _ops.st(fp32, m_rows, 16)
    in_flight = combine_depth or warps

    slots = warps
    if resident:
        # Largest whole number of partials that still lets only `resident`
        # workgroups share the pool.
        slots = max(warps, _TARGET.lds_pool_bytes // resident // st_part.nbytes)
        if warps == 1 or slots * st_part.nbytes > _TARGET.lds_per_workgroup:
            raise ValueError(
                f"resident={resident} needs {slots * st_part.nbytes} B of LDS "
                f"per workgroup; one may hold {_TARGET.lds_per_workgroup}, "
                f"and a one-warp workgroup has no combine to size")

    def body(x, w, c):
        n0 = _ops.block_idx.x * 16
        wid = _ops.warp_id()
        rounds = _ops.cols(x) // (kc * warps)

        acc = _ops.zeros(ty_acc)
        for t in _ops.range(rounds):
            k = (t * warps + wid) * kc
            w_t = _ops.load(w, _ops.elem_coord(0, 0, n0, k), ty_w)
            x_t = _ops.load(x, _ops.elem_coord(0, 0, 0, k), ty_x)
            _ops.mma_ABt(w_t, x_t, acc, out=acc)

        out = _ops.transpose(acc)
        if warps > 1:
            parts = _ops.alloc_shared(st_part, count=slots, name="parts")
            _ops.store_shared(_ops.shared_at(parts, wid), out)
            _ops.barrier(drain=True)
            with _ops.if_(_ops.s_eq(wid, 0)):
                for i in range(1, warps):
                    if i > 1 and (i - 1) % in_flight == 0:
                        _ops.sched_barrier()
                    other = _ops.load_shared(_ops.shared_at(parts, i), ty_out)
                    _ops.add(out, other, out=out)
                _ops.store(c, out, _ops.elem_coord(0, 0, 0, n0))
        else:
            _ops.store(c, out, _ops.elem_coord(0, 0, 0, n0))

    body.__name__ = name
    body.__annotations__ = {"x": GL[bf16], "w": GL[bf16], "c": GL[bf16]}

    return _kernel(body, arch="gfx1100", warps=warps, name=name,
                   grid=lambda p: (p.w.rows // 16, 1, 1),
                   max_vgprs=max_vgprs, min_occupancy=min_occupancy)


#: Per token-tile count, measured on Qwen3-8B's decode shapes, timed as a
#: step reads them (python/README.md has the sweep). One tile wants 16 warps,
#: two per WGP; two tiles cannot hold 15 partials of 16 VGPRs and batch them;
#: four cannot afford 16 warps at all. The wider two are held to one and two
#: workgroups per WGP by their registers already.
SCHEDULES: Dict[int, Dict[str, int]] = {
    1: dict(slices=4, warps=16, combine_depth=0, resident=2),
    2: dict(slices=2, warps=16, combine_depth=8, resident=0),
    4: dict(slices=2, warps=8, combine_depth=4, resident=0),
}

#: The widest batch, in 16-row token tiles. Every tile is another X operand
#: and another accumulator per wave, so the registers run out before the
#: bandwidth does.
MAX_TOKENS = max(SCHEDULES)

_KERNELS: Dict[Tuple[Any, ...], Any] = {}


def tokens_for(m: int) -> int:
    """The token tiles a batch of M rows is padded to."""
    return next(t for t in sorted(SCHEDULES) if m <= 16 * t)


def schedule_for(m: int, k: int) -> Optional[Dict[str, Any]]:
    """The schedule for M rows at depth K, or None if the kernel cannot take it.

    A K the measured split does not divide gets fewer warps rather than a
    refusal: every warp takes whole rounds, so the warp count has to divide
    K / (16*slices), and halving it is the nearest schedule that still does.
    """
    if not 0 < m <= 16 * MAX_TOKENS:
        return None
    tokens = tokens_for(m)
    s = dict(SCHEDULES[tokens], tokens=tokens)
    while k % (16 * s["slices"] * s["warps"]) and s["warps"] > 1:
        s["warps"] //= 2
    if k % (16 * s["slices"] * s["warps"]):
        return None
    return _settle(s)


def _settle(s: Dict[str, int]) -> Dict[str, int]:
    """What a schedule's knobs mean once its warp count is fixed: a batch at
    least as large as the combine is no batch, and one warp has no combine
    whose LDS could hold the residency down."""
    if s["combine_depth"] >= s["warps"]:
        s["combine_depth"] = 0
    if s["warps"] == 1:
        s["resident"] = 0
    return s


def kernel_for(tokens: int, slices: int, warps: int, combine_depth: int = 0,
               resident: int = 0):
    key = (tokens, slices, warps, combine_depth, resident)
    if key not in _KERNELS:
        _KERNELS[key] = decode_gemm_kernel(
            f"decode_gemm_t{tokens}_s{slices}_w{warps}_c{combine_depth}"
            f"_r{resident}",
            tokens=tokens, slices=slices, warps=warps,
            combine_depth=combine_depth, resident=resident)
    return _KERNELS[key]


def fits(m: int, n: int, k: int) -> bool:
    """Whether this kernel can take the shape at all -- not whether it wins."""
    return n % 16 == 0 and schedule_for(m, k) is not None


def matmul(x, w, **override):
    """x @ w.T for a decode batch. `w` is (N, K), as in `gemm.matmul`.

    Returns the first M rows of a buffer padded to whole token tiles; the
    padding is the kernel's, not the caller's. `override` replaces fields of
    the schedule (slices, warps, combine_depth, resident) -- for measuring,
    not for use.
    """
    import torch  # noqa: PLC0415 -- hk traces and compiles without torch

    if x.dim() != 2 or w.dim() != 2 or x.shape[1] != w.shape[1]:
        raise ValueError(f"decode matmul: want (M, K) @ (N, K).T, got "
                         f"{tuple(x.shape)} and {tuple(w.shape)}")
    if x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
        raise TypeError(f"decode matmul: bf16 only, got {x.dtype} and {w.dtype}")
    m, k = x.shape
    n = w.shape[0]
    s = schedule_for(m, k)
    if s is not None and override:
        s = _settle({**s, **override})
    if s is None or n % 16 or k % (16 * s["slices"] * s["warps"]):
        raise ValueError(
            f"decode matmul: M={m} must be in [1, {16 * MAX_TOKENS}], N={n} a "
            f"multiple of 16, and K={k} a multiple of 16 x slices x warps "
            f"(every warp takes whole rounds)")
    rows = 16 * s["tokens"]
    if m < rows:
        x = torch.nn.functional.pad(x, (0, 0, 0, rows - m))
    elif not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty((rows, n), device=x.device, dtype=torch.bfloat16)
    kernel_for(**s)(x, w.contiguous(), out)
    return out[:m]


__all__ = ["MAX_TOKENS", "SCHEDULES", "decode_gemm_kernel", "fits",
           "kernel_for", "matmul", "schedule_for", "tokens_for"]
