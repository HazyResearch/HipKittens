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

from typing import Any, Dict, Optional, Tuple

from .. import dsl as hk
from ..ir.nodes import bf16, fp32, i32
from ..target import current_arch, get_target


def decode_gemm_kernel(name: str, *, tokens: int = 1, slices: int = 4,
                       warps: int = 16, combine_depth: int = 0,
                       resident: int = 0, splits: int = 1,
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

    `splits` > 1 also cuts K across workgroups: workgroup (b, s) takes the
    s-th slice, stores its fp32 partial to `ws[s]`, and arrives at `cnt[b]`;
    the last of the `splits` to arrive sums the slots in order 0..splits-1 --
    the same order whichever workgroup that is -- and stores the result. The
    kernel then takes two more tensors, `ws` (splits, 16*tokens, N) fp32 and
    `cnt` (N/16,) int32, zeroed once; see `scratch_for`.

    X and C have the batch's M rows, any M up to 16*tokens: the X loads and
    the store are predicated on rows. K must be a multiple of
    16*slices*warps*splits.
    """
    m_rows = 16 * tokens
    kc = 16 * slices
    tgt = get_target(current_arch())

    ty_w = hk.RT(bf16, 16, kc, "row")
    ty_x = hk.RT(bf16, m_rows, kc, "row")
    ty_acc = hk.RT(fp32, 16, m_rows, "col")      # C^T: W rows x tokens
    ty_out = hk.RT(fp32, m_rows, 16, "row")      # its transpose
    st_part = hk.ST(fp32, m_rows, 16)
    in_flight = combine_depth or warps

    slots = warps
    if resident:
        # Largest whole number of partials that still lets only `resident`
        # workgroups share the pool -- less the 16 B of static LDS wg_arrive
        # declares, which counts against the same 64 KB a workgroup may take.
        budget = tgt.lds_pool_bytes // resident - (16 if splits > 1 else 0)
        slots = max(warps, budget // st_part.nbytes)
        if warps == 1 or slots * st_part.nbytes > tgt.lds_per_workgroup:
            raise ValueError(
                f"resident={resident} needs {slots * st_part.nbytes} B of LDS "
                f"per workgroup; one may hold {tgt.lds_per_workgroup}, "
                f"and a one-warp workgroup has no combine to size")

    @hk.func
    def partial(x, w, k0, rounds, finish):
        """This workgroup's 16 W rows over K[k0 : k0 + rounds*kc*warps],
        combined across its warps; `finish` gets the result in warp 0."""
        n0 = hk.arch.block_idx.x * 16
        wid = hk.arch.warp_id()
        acc = hk.zeros(ty_acc)
        for t in hk.range(rounds):
            k = k0 + (t * warps + wid) * kc
            w_t = w[hk.elem_coord(0, 0, n0, k)].load(ty_w)
            # Bounded: X has the batch's M rows, not a padded 16*tokens, and
            # the rows past M read as zero instead of past its allocation.
            x_t = x[hk.elem_coord(0, 0, 0, k)].load(ty_x, bounded=True)
            acc += w_t @ x_t.T

        out = hk.transpose(acc)
        if warps > 1:
            parts = hk.lds.alloc(st_part, count=slots, name="parts")
            parts[wid].store(out)
            hk.arch.barrier(drain=True)
            if wid == 0:
                for i in range(1, warps):
                    if i > 1 and (i - 1) % in_flight == 0:
                        hk.arch.sched_barrier()
                    out += parts[i].load(ty_out)
                finish(out, n0)
        else:
            finish(out, n0)

    def body(x: hk.Tensor[bf16], w: hk.Tensor[bf16], c: hk.Tensor[bf16]):
        def finish(out, n0):
            c[hk.elem_coord(0, 0, 0, n0)].store(out, bounded=True)

        partial(x, w, 0, x.cols // (kc * warps), finish)

    def body_split(x: hk.Tensor[bf16], w: hk.Tensor[bf16], c: hk.Tensor[bf16],
                   ws: hk.Tensor[fp32], cnt: hk.Tensor[i32]):
        s = hk.arch.block_idx.y
        per = x.cols // splits

        def finish(out, n0):
            ws[hk.elem_coord(0, s, 0, n0)] = out

        partial(x, w, s * per, per // (kc * warps), finish)

        arrived = hk.arch.wg_arrive(cnt[hk.elem_coord(0, 0, 0, hk.arch.block_idx.x)],
                                    splits)
        if arrived == splits - 1:
            if hk.arch.warp_id() == 0:
                n0 = hk.arch.block_idx.x * 16
                total = hk.zeros(ty_out)
                for j in range(splits):
                    total += ws[hk.elem_coord(0, j, 0, n0)].load(ty_out)
                c[hk.elem_coord(0, 0, 0, n0)].store(total, bounded=True)

    fn = body if splits == 1 else body_split
    fn.__name__ = name

    return hk.kernel(fn, warps=warps, name=name,
                     grid=lambda p: (p.w.rows // 16, splits, 1),
                     max_vgprs=max_vgprs, min_occupancy=min_occupancy)


#: Per token-tile count, measured on Qwen3-8B's decode shapes, timed as a
#: step reads them (python/README.md has the sweep). One tile wants 16 warps,
#: two per WGP; two tiles cannot hold 15 partials of 16 VGPRs and batch them;
#: four cannot afford 16 warps at all. The wider two are held to one and two
#: workgroups per WGP by their registers already.
SCHEDULES: Dict[int, Dict[str, int]] = {
    1: dict(slices=4, warps=16, combine_depth=0, resident=2, splits=1),
    2: dict(slices=2, warps=16, combine_depth=8, resident=0, splits=1),
    4: dict(slices=2, warps=8, combine_depth=4, resident=0, splits=1),
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
    if k % (16 * s["slices"] * s["warps"] * s["splits"]):
        s["splits"] = 1
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
               resident: int = 0, splits: int = 1):
    key = (tokens, slices, warps, combine_depth, resident, splits)
    if key not in _KERNELS:
        _KERNELS[key] = decode_gemm_kernel(
            f"decode_gemm_t{tokens}_s{slices}_w{warps}_c{combine_depth}"
            f"_r{resident}" + (f"_k{splits}" if splits > 1 else ""),
            tokens=tokens, slices=slices, warps=warps,
            combine_depth=combine_depth, resident=resident, splits=splits)
    return _KERNELS[key]


_SCRATCH: Dict[Tuple[Any, ...], Any] = {}


def scratch_for(device, splits: int, tokens: int, n: int):
    """The (ws, cnt) pair a split kernel of this shape takes, made once.

    Shared by every launch of the shape: launches on one stream run in order,
    and the last workgroup of each output block resets its counter, so `cnt`
    is zero again when the next launch starts. Made on first use -- vLLM's
    warm-up runs, before any graph is captured, are where that happens.
    """
    import torch  # noqa: PLC0415

    key = (str(device), splits, tokens, n)
    if key not in _SCRATCH:
        _SCRATCH[key] = (
            torch.empty((splits, 16 * tokens, n), device=device, dtype=torch.float32),
            torch.zeros((n // 16,), device=device, dtype=torch.int32))
    return _SCRATCH[key]


def fits(m: int, n: int, k: int) -> bool:
    """Whether this kernel can take the shape at all -- not whether it wins."""
    return n % 16 == 0 and schedule_for(m, k) is not None


def matmul(x, w, **override):
    """x @ w.T for a decode batch. `w` is (N, K), as in `gemm.matmul`.

    `override` replaces fields of the schedule (slices, warps, combine_depth,
    resident, splits) -- for measuring, not for use.
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
    if s is None or n % 16 or k % (16 * s["slices"] * s["warps"] * s["splits"]):
        raise ValueError(
            f"decode matmul: M={m} must be in [1, {16 * MAX_TOKENS}], N={n} a "
            f"multiple of 16, and K={k} a multiple of 16 x slices x warps x "
            f"splits (every warp takes whole rounds)")
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    if s["splits"] > 1:
        kernel_for(**s)(x, w.contiguous(), out,
                        *scratch_for(x.device, s["splits"], s["tokens"], n))
    else:
        kernel_for(**s)(x, w.contiguous(), out)
    return out


__all__ = ["MAX_TOKENS", "SCHEDULES", "decode_gemm_kernel", "fits",
           "kernel_for", "matmul", "schedule_for", "scratch_for", "tokens_for"]
