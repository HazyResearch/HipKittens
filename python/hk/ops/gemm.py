"""bf16 GEMM with fp32 accumulate, written in the DSL.

A transcription of kernels/rdna3/gemm/bf16fp32/gemm.cpp, which is 995 lines of
C++ and twenty `#define` knobs. The gate for this file is not "it works": it is
that the generated kernel matches the handwritten one on ScratchSize,
occupancy and TFLOPs, because an IR that can only express the slow version of a
kernel is an IR nobody will use for the fast one.

What the schedule is, in one paragraph. The workgroup holds two LDS buffers
each of A (BLOCK_M x K_STEP) and B (BLOCK_N x K_STEP), and alternates between
them. Each K-tile: read operand slices out of the resident buffer and multiply,
then write the *other* buffer from registers that were filled by global loads
issued an iteration ago, then issue the next iteration's loads. One barrier per
K-tile, not two -- the write-after-read hazard on the buffer being overwritten
is already covered by the previous iteration's barrier, so only the
write-before-read hazard needs a fresh one.

Three things the DSL does that the C++ cannot:

  * **The lgkmcnt immediates are derived.** The C++ computes them with a
    constexpr `reads_in_flight()` plus a `TAIL_OPS` term for the interleaved
    ds_writes, and a paragraph explaining why. Here `lds_wait_for` names the
    tiles it retires and the hk-lds-pipeline pass walks the issue queue. An
    immediate that is too large does not over-wait, it fails to wait at all --
    so this is the difference between a schedule you can edit and one you can
    only admire.
  * **The operand binding is not optional.** There is no bare `lds_wait` in the
    DSL to reach for.
  * **The spill gate is in the build.** `max_vgprs`/`min_occupancy` below are
    part of compiling this kernel, so a change that costs a wave fails to
    produce a `.so` rather than quietly producing a slower one.

Shapes. B is passed pre-transposed as (N, K): it keeps both shared loads
contiguous and makes `mma_ABt` the right primitive, which is the whole reason
the operand tiles are both row-layout and therefore both on the ds_read_b128
fast path.
"""

from typing import Any, Dict, Tuple

from .. import autotune as _autotune
from .. import dsl as hk
from ..target import current_arch, get_target
from ..types import bf16, fp32


def _derive(block_m: int, block_n: int, k_step: int, dot_slice: int,
            warps: int, warp_rows: int, n_split: int) -> Dict[str, int]:
    """The config arithmetic, in one place, with the C++'s static_asserts."""
    warp_cols = warps // warp_rows
    if warp_rows * warp_cols != warps:
        raise ValueError(f"warp grid {warp_rows}x{warp_cols} does not cover {warps} warps")
    reg_m = block_m // warp_rows
    reg_n = block_n // warp_cols
    split_n = reg_n // n_split
    if split_n % 16 or split_n * n_split != reg_n:
        raise ValueError(
            f"N_SPLIT={n_split} must cut the {reg_n}-wide warp tile into whole "
            f"16-column chunks"
        )
    if k_step % dot_slice:
        raise ValueError(f"DOT_SLICE={dot_slice} must divide K_STEP={k_step}")
    lds = 2 * (block_m + block_n) * k_step * 2
    cap = get_target(current_arch()).lds_per_workgroup
    if lds > cap:
        raise ValueError(f"{lds} B of LDS; a {current_arch()} workgroup has {cap}")
    return dict(
        warp_cols=warp_cols, reg_m=reg_m, reg_n=reg_n, split_n=split_n,
        slices=k_step // dot_slice, lds=lds,
    )


def gemm_kernel(name: str, *, block_m: int = 128, block_n: int = 128,
                k_step: int = 64, dot_slice: int = 16, warps: int = 8,
                warp_rows: int = 4, n_split: int = 4, wgm: int = 8,
                write_pos: int = 2,
                max_vgprs: int = 256, min_occupancy: int = 1):
    """The default tiling is `big_config` from the C++: 128x128x64, 8 warps.

    That is a 32x64 warp tile rather than the 64x64 everything else on this
    GPU runs, and the README next to the C++ has the sweep that justifies it:
    every 64x64 variant either spills or falls to 5-6 waves/SIMD, and the ones
    that avoid both do it by taking K_STEP down to 16, straight back into the
    global-latency penalty. 1.5x the LDS traffic per WMMA, bought with the
    registers to run a 64-deep K-tile at 7 waves/SIMD.

    `write_pos` is where the staging drain lands relative to the math: 2 puts
    it inside the last-but-one slice, so the ds_writes complete behind a whole
    slice of WMMAs and the waits downstream carry them. The C++ computes that
    carry by hand and calls it TAIL_OPS; here lds_pipeline derives it, which
    is the difference between a schedule you can edit and one you can only
    admire.
    """
    d = _derive(block_m, block_n, k_step, dot_slice, warps, warp_rows, n_split)
    warp_cols, reg_m, reg_n = d["warp_cols"], d["reg_m"], d["reg_n"]
    split_n, slices = d["split_n"], d["slices"]

    st_a = hk.ST(bf16, block_m, k_step)
    st_b = hk.ST(bf16, block_n, k_step)

    def body(a: hk.Tensor[bf16], b: hk.Tensor[bf16], c: hk.Tensor[bf16], *,
             WGM: hk.Constexpr = wgm):
        g = hk.group(warps)

        As = hk.lds.alloc(st_a, count=2, name="As")
        Bs = hk.lds.alloc(st_b, count=2, name="Bs")
        buf_a = g.stage_buffer(As, name="buf_a")
        buf_b = g.stage_buffer(Bs, name="buf_b")

        A_tile = hk.zeros(hk.RT(bf16, reg_m, dot_slice, "row"))
        B_tile = [hk.zeros(hk.RT(bf16, split_n, dot_slice, "row"))
                  for _ in range(n_split)]
        C_acc = [hk.zeros(hk.RT(fp32, reg_m, split_n, "col"))
                 for _ in range(n_split)]

        M = a.rows
        m_blocks = hk.cdiv(M, block_m)
        n_blocks = b.rows // block_n

        # L2 swizzle: walk WGM block-rows at a time so the B panel a group of
        # workgroups reads stays resident. One XCD here, so no chiplet term.
        wgid = hk.arch.block_idx.x
        per_group = WGM * n_blocks
        group_id = wgid // per_group
        first_m = group_id * WGM
        group_m = hk.min(m_blocks - first_m, WGM)
        row = first_m + ((wgid % per_group) % group_m)
        col = (wgid % per_group) // group_m

        # The M remainder is predicated, not backed up. Staging is a buffer
        # load whose range ends at A's last row, so rows past M read zero with
        # exactly the loads a whole tile issues -- vmcnt stays the immediate it
        # was -- and the bounded store writes only rows below M. That is also
        # what lets M be smaller than one block.
        m_base = row * block_m

        wid = hk.arch.warp_id()
        warp_row = wid // warp_cols
        warp_col = wid % warp_cols

        n_tiles = a.cols // k_step

        def a_coord(t):
            return hk.elem_coord(0, 0, m_base, t * k_step)

        def b_coord(t):
            return hk.tile_coord(0, 0, col, t)

        def stage(t):
            # Clamped rather than predicated: past the end this re-reads the
            # last tile into staging registers that are never committed, which
            # is a handful of L2-hot loads, and it keeps exactly one batch in
            # flight at all times so vmcnt stays a compile-time immediate.
            tt = hk.min(t, n_tiles - 1)
            g.stage(buf_a, a[a_coord(tt)])
            g.stage(buf_b, b[b_coord(tt)])

        def dot_tile(buf, tail=None):
            """One K-tile of math out of LDS buffer `buf`, with `tail` woven in.

            The whole slice's reads go in flight at once and the waits step
            down, so chunk s's WMMAs issue on top of chunks s+1.. still moving.
            No tile is duplicated: the overlapping reads are reads this slice
            had to do anyway, just not fenced in front of it. Double-buffering
            the operand tiles instead was tried in the C++ and is slower -- the
            second set costs 48 VGPRs and takes occupancy from 9 to 7.
            """
            A_s = As[buf]
            B_s = Bs[buf]

            def load_A(k):
                A_s.subtile(reg_m, dot_slice, warp_row, k).load(
                    A_tile.ty, wait=False, into=A_tile)

            def load_B(k, s):
                B_s.subtile(split_n, dot_slice, warp_col * n_split + s, k).load(
                    B_tile[s].ty, wait=False, into=B_tile[s])

            def run_tail():
                if tail is not None:
                    tail()

            load_A(0)
            for s in range(n_split):
                load_B(0, s)
            if slices == 1:
                # Nothing left to issue for this tile, so there is no later
                # slot to hide the drain in.
                run_tail()

            for k in range(slices):
                for s in range(n_split):
                    hk.lds.wait_for(A_tile, B_tile[s])
                    hk.arch.setprio(1)
                    C_acc[s] += A_tile @ B_tile[s].T
                    hk.arch.setprio(0)
                    # B_tile[s] is dead the instant this chunk's WMMAs have
                    # issued, so the next slice's read of it goes in flight
                    # here, a full chunk of math early, at no register cost.
                    # A_tile dies only after the slice's last chunk -- and goes
                    # out *before* that chunk's B, so the next slice's first
                    # chunk is gated on A rather than on a read behind it.
                    if k + 1 < slices:
                        if s == n_split - 1:
                            load_A(k + 1)
                        load_B(k + 1, s)
                        # The tile's last read is now in flight: hand the LDS
                        # pipe to the store and let the final slice's math
                        # cover it. Every wait after this point is downstream
                        # of two ds_writes and has to say so -- which is the
                        # one number in this kernel nobody has to write down.
                        if k + 2 == slices and s == n_split - 1:
                            run_tail()

        # Prologue: tile 0 resident, tile 1 in flight.
        g.load(As[0], a[a_coord(0)])
        g.load(Bs[0], b[b_coord(0)])
        stage(1)
        hk.arch.barrier()

        for t in range(n_tiles - 1):
            tic = t % 2
            toc = 1 - tic

            def store_tile():
                # Land the batch issued an iteration ago, drain it into the
                # buffer nobody is reading, and refill the registers at once so
                # they are never idle. GPREFETCH is 1, so vmcnt keeps nothing.
                hk.arch.vm_wait(0)
                g.commit(As[toc], buf_a)
                g.commit(Bs[toc], buf_b)
                stage(t + 2)

            if write_pos == 0:
                store_tile()
                dot_tile(tic)
            elif write_pos == 1:
                dot_tile(tic)
                store_tile()
            else:
                dot_tile(tic, store_tile)

            # The writes are still moving: dot_tile's last chunk stopped at the
            # two of them rather than at zero, on purpose, and s_barrier does
            # not order memory. Nearly free here -- a whole K-slice of WMMAs
            # has run since they issued -- and the pass refuses the barrier
            # without it, so it cannot be dropped by accident.
            hk.arch.barrier(drain=True)

        # Epilogue: the last tile is already resident and there is nothing left
        # to stage, so no tail -- and therefore no write for the waits to
        # carry. Counting writes that were never issued would be the dangerous
        # direction, and here that is structural rather than an argument the
        # caller has to remember to pass.
        dot_tile((n_tiles - 1) % 2)

        # Columns are in units of the chunk width; rows in elements.
        m_row = m_base + warp_row * reg_m
        col_tile = (col * warp_cols + warp_col) * n_split
        for s in range(n_split):
            c[hk.elem_coord(0, 0, m_row, (col_tile + s) * split_n)].store(
                C_acc[s], bounded=True)

    body.__name__ = name

    def _grid(p):
        return (hk.cdiv(p.a.rows, block_m) * (p.b.rows // block_n), 1, 1)

    if write_pos not in (0, 1, 2):
        raise ValueError(f"write_pos={write_pos} must be 0, 1 or 2")

    return hk.kernel(body, warps=warps, grid=_grid, name=name,
                     max_vgprs=max_vgprs, min_occupancy=min_occupancy)


#: The tiling the C++ calls `big_config`, and the one every shape gets until a
#: tuning record says otherwise.
DEFAULT = dict(block_m=128, block_n=128, k_step=64, dot_slice=16, warps=8,
               warp_rows=4, n_split=4, wgm=8, write_pos=2)

#: What the tuner searches. Five axes and 48 points, chosen so that each one
#: trades something real against something else rather than wandering:
#:
#:   block_m   128 is 1.5x the LDS traffic per WMMA for the registers to run a
#:             64-deep K-tile at 7 waves; 64 is the other side of that trade.
#:   k_step    how much global latency one barrier covers, against LDS.
#:   n_split   how the warp tile is cut into accumulators -- register pressure
#:             against ds_read width.
#:   wgm       the L2 swizzle: how long a B panel stays resident.
#:   write_pos where the staging drain lands relative to the math.
#:
#: dot_slice, warps and warp_rows are held at the default: moving them changes
#: the fragment shapes rather than the schedule, and every combination that
#: spills does so for a reason the resource gate reports more usefully than a
#: benchmark would. Points that cannot work -- LDS over 64 KB, a split that
#: does not divide -- stay in the space on purpose and come back as `invalid`
#: rows, because "it does not fit" is an answer.
SPACE = _autotune.space(
    block_m=[64, 128], k_step=[32, 64], n_split=[2, 4],
    wgm=[4, 8, 16], write_pos=[1, 2],
)

#: Shapes the CLI tunes by default: a square prefill GEMM, a bigger one, a
#: decode-shaped thin-M one, and the 16384-row prefill from the Qwen3 TP2 runs.
TUNE_KEYS = ["4096x4096x4096", "8192x8192x8192",
             "1024x8192x4096", "16384x4096x4096"]


def _sched_name(s: Dict[str, Any]) -> str:
    return ("gemm_bf16_{block_m}x{block_n}x{k_step}_w{warps}r{warp_rows}"
            "_s{n_split}_d{dot_slice}_g{wgm}_p{write_pos}").format(**s)


def _make(**sched):
    """One kernel per schedule. The name carries the whole schedule because the
    symbol ends up in a resource report, and a report that says
    `gemm_bf16_128x128` for six different tilings is a report you cannot read."""
    return gemm_kernel(_sched_name({**DEFAULT, **sched}), **sched)


def _bench_for(key: str):
    """`key` is "MxNxK". Returns a closure that times one kernel on it."""
    import torch  # noqa: PLC0415

    m, n, k = (int(x) for x in key.split("x"))
    g = torch.Generator(device="cuda").manual_seed(0)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=g)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=g)
    c = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)

    def run(kernel, warmup=3, iters=10):
        for _ in range(warmup):
            kernel(a, b, c)
        torch.cuda.synchronize()
        beg, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        beg.record()
        for _ in range(iters):
            kernel(a, b, c)
        end.record()
        torch.cuda.synchronize()
        return beg.elapsed_time(end) / iters

    return run


#: The tuner. Building it registers it, which is all `python3 -m hk.autotune`
#: needs to find it.
TUNER = _autotune.Tuner("gemm_bf16", _make, SPACE, DEFAULT,
                        bench=_bench_for, keys=TUNE_KEYS)

#: The default kernel, by name, for anything that wants it without going
#: through the tuner -- cache warming, resource checks, the tests.
KERNELS = {"gemm_bf16_128x128": TUNER._kernel_for(DEFAULT)}

#: What the default tiling requires of a shape. K and N have to tile exactly;
#: M does not, because the last block row is predicated (see gemm_kernel).
BLOCK_M, BLOCK_N, K_STEP = DEFAULT["block_m"], DEFAULT["block_n"], DEFAULT["k_step"]

#: For a batch of at most 16 rows: one 16-row block, so a decode-sized M does
#: not compute 128 rows to keep one. 1.19-1.62x over hipBLASLt on Qwen3-8B's
#: decode shapes at M=16 (python/README.md), though hk.ops.decode_gemm, laid
#: out for the weight stream, is faster still.
THIN = dict(block_m=16, block_n=128, k_step=64, dot_slice=16, warps=4,
            warp_rows=1, n_split=2, wgm=8, write_pos=2)


def _bucket(m: int) -> int:
    """M rounded up to a power of two, for the tuning key.

    A record per exact M would be a record per batch size, which is a cache
    that never hits. The schedule's sensitivity to M is through how many
    block-rows there are and how full the last one is, and that moves on a
    log scale.
    """
    b = 1
    while b < m:
        b *= 2
    return b


def schedule_for(m: int, n: int, k: int) -> Dict[str, Any]:
    """The tuned schedule for this shape, or the default.

    The fallback is not belt and braces. A record is keyed on a bucketed M and
    an exact N and K, so a recorded schedule has tiled *a* shape in this
    bucket, but `k_step=64` chosen at K=4096 must not turn K=4064 -- which the
    default tiles fine -- into an exception two months later. A schedule that
    cannot tile the shape in front of it loses to the one that can.
    """
    s = TUNER.schedule_for(f"{_bucket(m)}x{n}x{k}")
    if s == DEFAULT and m <= THIN["block_m"] and not (k % THIN["k_step"] or
                                                     n % THIN["block_n"]):
        return dict(THIN)
    if k % s["k_step"] or n % s["block_n"]:
        return dict(DEFAULT)
    return s


def matmul(a, b, out=None):
    """A @ B.T for bf16, fp32 accumulate.

    `b` is (N, K), not (K, N). That is not an inconvenience to be hidden: it
    keeps both shared loads contiguous and makes `mma_ABt` the primitive, which
    is what puts both operand tiles in row layout and therefore on the
    `ds_read_b128` fast path. Transposing here instead would copy, and the copy
    is a bigger cost than the GEMM saves.
    """
    import torch  # noqa: PLC0415 -- hk traces and compiles without torch

    if a.dim() != 2 or b.dim() != 2:
        raise ValueError(f"matmul: want 2D, got {a.dim()}D and {b.dim()}D")
    m, k = a.shape
    n, kb = b.shape
    sched = schedule_for(m, n, k)
    block_m, block_n, k_step = sched["block_m"], sched["block_n"], sched["k_step"]
    if kb != k:
        raise ValueError(f"matmul: A is (M={m}, K={k}) and B is (N={n}, K={kb}); "
                         f"B is passed pre-transposed, so its second axis is K")
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise TypeError(f"matmul: bf16 only for now, got {a.dtype} and {b.dtype}")
    if k % k_step or n % block_n:
        raise ValueError(
            f"matmul: K={k} must be a multiple of {k_step} and N={n} a multiple "
            f"of {block_n}. Only M may be ragged: rows of the last block past M "
            f"are loaded as zeros and not stored."
        )
    if m < 1:
        raise ValueError(f"matmul: M={m}; there is nothing to multiply")
    if out is None:
        out = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
    elif out.shape != (m, n) or out.dtype != torch.bfloat16:
        raise ValueError(f"matmul: out is {tuple(out.shape)}/{out.dtype}, "
                         f"expected ({m}, {n})/bfloat16")
    TUNER._kernel_for(sched)(a, b, out)
    return out


__all__ = ["KERNELS", "TUNER", "SPACE", "DEFAULT", "gemm_kernel", "matmul",
           "schedule_for", "BLOCK_M", "BLOCK_N", "K_STEP"]
