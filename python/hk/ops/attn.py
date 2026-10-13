"""Flash attention forward, written in the DSL.

A transcription of kernels/rdna3/attn/fwd/attn.cpp -- 1237 lines of C++ and
twenty `#define` knobs -- into something a Python caller can retune without a
`make clean`. The gate is parity: same ScratchSize, same occupancy, same
TFLOPs. An IR that can only express the slow attention is an IR nobody will
use for the fast one.

What the schedule is, in one paragraph. One workgroup owns Q_TILE query rows,
Q_BLOCK of them per warp, and never writes them to LDS: Q lives in registers
for the whole kernel. It walks the KV axis in blocks of KV_BLOCK, staging each
block's K straight into LDS and its V *transposed* into LDS, then every warp
multiplies the whole block against its own Q rows. One barrier pair per KV
block.

Three things are stored transposed, and it is the same reason each time.
S is kept as S^T, [kv, q]:

  * its reductions (the running max, the running sum) are then over the
    *element* axis of a col-layout fp32 accumulator -- register-local steps
    plus one cross-half exchange, no butterfly;
  * both statistics are `row_vec`s of S^T, and so is what `mul_col`/`div_col`
    want on the O accumulator, so no vector ever changes layout;
  * the causal predicate becomes one `v_cmp` per element: q is the lane axis
    and kv the element axis, both free.

O follows S^T, so the PV matmul is O^T += V^T . P^T, which is what forces V^T
into LDS -- and that is the third transpose, paid once per block by the stager
rather than once per warp per block by sixteen `ds_read_u16`.

The one thing this file does *not* transpose is the epilogue. The C++ moves the
accumulator's data back to [q, d] with a per-fragment `transpose`; here
`hk.transpose` relabels instead, which is the same transpose for free, because
a base tile's storage does not depend on its layout tag.
"""

import math
from typing import Any, Dict, Optional

from .. import autotune as _autotune
from .. import dsl as hk
from ..target import current_arch, get_target
from ..ir.nodes import bf16, fp32

NEG_INF = float("-inf")


def _derive(head_dim: int, q_block: int, kv_block: int, warps: int,
            vt_d_chunk: int, qk_tiles: int, pv_tiles: int) -> Dict[str, int]:
    """The config arithmetic, with the C++'s static_asserts."""
    for n, v in (("head_dim", head_dim), ("q_block", q_block),
                 ("kv_block", kv_block), ("vt_d_chunk", vt_d_chunk)):
        if v <= 0 or v % 16:
            raise ValueError(f"{n}={v}: every extent is in units of a 16x16 WMMA fragment")
    if kv_block % (16 * qk_tiles):
        raise ValueError(f"qk_tiles={qk_tiles} must tile the {kv_block}-row KV block")
    if head_dim % (16 * pv_tiles):
        raise ValueError(f"pv_tiles={pv_tiles} must tile the {head_dim}-wide head")
    if head_dim % vt_d_chunk:
        raise ValueError(f"vt_d_chunk={vt_d_chunk} must tile head_dim={head_dim}")
    if vt_d_chunk != 16:
        # The staging write would need the t-th 16x16 fragment of the
        # transposed slab, and a register tile has no subtile op -- a fragment
        # is a value here, not a view. Expressible by transposing per fragment
        # instead; not measured, so not offered.
        raise ValueError(
            f"vt_d_chunk={vt_d_chunk}: only 16 is implemented. A wider staging "
            f"slab needs the transposed tile split into fragments, and this IR "
            f"has no register subtile -- write the loop over 16-wide slabs."
        )
    d_chunks = head_dim // vt_d_chunk
    if d_chunks > warps:
        # V_ITERS > 1 in the C++. Expressible here -- it is a Python loop over
        # `warp_id + i * warps` -- but it has never been the measured config and
        # an untested branch in a kernel this tight is a liability, so it is a
        # refusal rather than a path.
        raise ValueError(
            f"head_dim/vt_d_chunk = {d_chunks} V^T chunks but only {warps} warps. "
            f"Raise vt_d_chunk or warps; one chunk per warp is the only staging "
            f"split this file implements."
        )
    lds = (kv_block * head_dim + head_dim * kv_block) * 2
    cap = get_target(current_arch()).lds_per_workgroup
    if lds > cap:
        raise ValueError(f"{lds} B of LDS; a {current_arch()} workgroup has {cap}")
    return dict(
        q_tile=q_block * warps,
        d_frags=head_dim // 16,
        kv_frags=kv_block // 16,
        d_chunks=d_chunks,
        v_bands=kv_block // 16,
        vt_frags=vt_d_chunk // 16,
        lds=lds,
    )


def attn_kernel(name: str, *, head_dim: int = 128, q_block: int = 16,
                kv_block: int = 32, warps: int = 12, vt_d_chunk: int = 16,
                qk_tiles: int = 2, pv_tiles: int = 2, causal: bool = False,
                scale: Optional[float] = None,
                max_vgprs: int = 256, min_occupancy: int = 1):
    """The default tiling is the C++'s, which its sweep.sh picked.

    `qk_tiles`/`pv_tiles` are how many 16x16 LDS reads go in flight before the
    wait that retires them. They are not a tile shape here the way they are in
    the C++ -- every tile in this file is one fragment -- they are a batch size
    for `lds_wait_for`, which is what they always were.

    `scale` is baked in. A runtime scale would be one more kernel parameter and
    one more SALU; it is a constant because the shapes a server runs are fixed
    at startup and a recompile per distinct scale is a one-time 30 s, paid off
    the moment the second request arrives.
    """
    d = _derive(head_dim, q_block, kv_block, warps, vt_d_chunk, qk_tiles, pv_tiles)
    q_tile, d_frags, kv_frags = d["q_tile"], d["d_frags"], d["kv_frags"]
    d_chunks, v_bands, vt_frags = d["d_chunks"], d["v_bands"], d["vt_frags"]

    # exp2 rather than exp: one hardware instruction. log2(e) is folded into the
    # scale, and the scale is applied to S rather than to Q, because Q is bf16
    # and multiplying it by a non-power-of-two would round the inputs twice.
    scale_l2e = (scale if scale is not None else head_dim ** -0.5) * math.log2(math.e)

    st_k = hk.ST(bf16, kv_block, head_dim)
    st_vt = hk.ST(bf16, head_dim, kv_block)

    ty_q = hk.RT(bf16, 16, 16, "row")          # one d-slice of this warp's Q
    ty_k = hk.RT(bf16, 16, 16, "row")          # one [kv, d] fragment of K
    ty_s = hk.RT(fp32, 16, q_block, "col")     # one [kv, q] fragment of S^T
    ty_p = hk.RT(bf16, 16, q_block, "col")     # the same, converted
    ty_o = hk.RT(fp32, 16, q_block, "col")     # one [d, q] fragment of O^T
    ty_v = hk.RT(bf16, 16, vt_d_chunk, "row")  # a [kv, d] slab of V, in flight
    ty_vt = hk.RT(bf16, vt_d_chunk, 16, "row")  # the same slab, transposed
    ty_vr = hk.RT(bf16, 16, 16, "row")         # one [d, kv] fragment of V^T
    ty_vec = hk.row_vec(ty_s)                  # one entry per query

    def body(q: hk.Tensor[bf16], k: hk.Tensor[bf16], v: hk.Tensor[bf16],
             o: hk.Tensor[bf16]):
        g = hk.group(warps)

        k_s = hk.lds.alloc(st_k, name="k_s")
        vt_s = hk.lds.alloc(st_vt, name="vt_s")
        buf_k = g.stage_buffer(k_s, name="buf_k")

        # Every LDS read below goes through a granule set rather than
        # `subtile` + `load`. The QK and PV loops issue d_frags*kv_frags
        # + d_frags*kv_frags fragment reads per KV block, and every one of
        # those addresses is loop-invariant -- the compiler hoists them all out
        # of the loop and then spills them (312 B/lane, measured, and a spill
        # here is a wrong answer because the LDS schedule hand-manages
        # s_waitcnt). The granule form holds 8 + 4 = 12 registers for the whole
        # kernel instead. See SharedTile.granules.
        gr_k = k_s.granules(name="gr_k")
        gr_vt = vt_s.granules(name="gr_vt")

        b = hk.arch.block_idx.z
        head = hk.arch.block_idx.y
        qb = hk.arch.block_idx.x
        n_q = q.rows
        n_kv = k.rows

        if causal:
            # Longest-processing-time-first. Under causal the work per
            # workgroup is a ramp -- workgroup i does i+1 KV blocks -- and
            # workgroups reach WGPs roughly in index order, so the machine
            # drains with a tail of the longest ones. Reversing the q index
            # starts the long ones while there is still short work left to
            # fill the slots they free. It is a permutation within one head,
            # so the grid's K/V locality is untouched: x is still the
            # fastest-varying axis. The block count is recomputed from n_q
            # rather than read off gridDim because it is the same expression
            # the grid lambda uses, and one of the two would otherwise be
            # free to drift.
            qb = hk.cdiv(n_q, q_tile) - 1 - qb

        # GQA is an addressing change and nothing else: the grid is over *query*
        # heads and several of them read the same K/V head. One scalar divide,
        # hoisted; when the head counts match it is a divide by 1.
        head_kv = head // (q.depth // k.depth)

        # Q remainder by backing the last block up, not by predicating it. Each
        # query row's output depends on nothing but that row, so the overlap is
        # computed twice from identical inputs and written twice with identical
        # values -- idempotent, not a race. Same trick as the GEMM.
        q_tile_start = hk.max(hk.min(qb * q_tile, n_q - q_tile), 0)
        wid = hk.arch.warp_id()
        q_row = q_tile_start + wid * q_block

        q_r = [q[hk.elem_coord(b, head, q_row, 16 * kk)].load(ty_q)
               for kk in range(d_frags)]

        o_c = [hk.zeros(ty_o) for _ in range(d_frags)]
        m_run = hk.neg_infty(ty_vec)
        l_run = hk.zeros(ty_vec)

        # Declared out here, and *only* these, because only these cross an
        # iteration boundary: the prefetch issues block kb+1's global reads
        # before block kb's math and lands them at kb+1's commit. Everything
        # else the loop uses -- the score tiles, the LDS windows, the bf16 copy
        # of P -- is created inside the body, where its live range ends. Hoisting
        # those out so that `into=` has somewhere to write costs 56 VGPRs that
        # are pinned across the staging code and read by nothing, which is the
        # difference between 132 B/lane of scratch and none.
        v_rows = [hk.zeros(ty_v) for _ in range(v_bands)]

        # This warp's V^T chunk: a vt_d_chunk-wide slice of the head dimension.
        # V_ITERS is 1 by construction (see _derive), so it is just the warp id,
        # and the warps past d_chunks stage nothing.
        c_chunk = wid
        stages_v = c_chunk < d_chunks

        kv_blocks = hk.cdiv(n_kv, kv_block)
        if causal:
            # Stop where the workgroup's *last* query does, not this warp's:
            # staging is a group operation with barriers in it, so the bound
            # has to be uniform across the workgroup. The slack this leaves a
            # low-numbered warp is taken by the per-fragment mask below, which
            # fills a fully-masked fragment with -inf rather than skipping it.
            # That is safe because block 0 is never fully masked for any query
            # (kv = 0 <= q always), so the running max is finite from the first
            # block and `exp2(-inf - m)` is 0 rather than NaN.
            kv_blocks = hk.min(kv_blocks, hk.cdiv(q_tile_start + q_tile, kv_block))

        def block_start(kb):
            # Clamped, not predicated: past the end this re-reads the last block
            # into staging registers that are overwritten before use, which is a
            # handful of L2-hot loads, and it keeps the global reads in bounds
            # without threading a predicate through the pipeline. The warp-level
            # V load goes through raw pointers, so out-of-range here would be an
            # out-of-bounds read rather than a harmless zero.
            kb = hk.min(kb, kv_blocks - 1)
            return hk.min(kb * kv_block, n_kv - kv_block)

        def load_kv(kv_start):
            """Issue a block's global reads. Waits for nothing."""
            g.stage(buf_k, k[hk.elem_coord(b, head_kv, kv_start, 0)])
            if stages_v:
                for w in range(v_bands):
                    v[hk.elem_coord(b, head_kv, kv_start + 16 * w,
                                    c_chunk * vt_d_chunk)].load(ty_v, into=v_rows[w])

        def commit_kv():
            """Land them: K straight into LDS, V transposed."""
            hk.arch.vm_wait(0)
            g.commit(k_s, buf_k)
            # V^T[d, kv] is what the PV matmul reads. V arrives [kv, d], so the
            # warp read a [16, vt_d_chunk] slice (perfectly coalesced),
            # transposes it in registers, and writes a [vt_d_chunk, 16] block --
            # still row layout, so still a vectorised ds_write.
            if stages_v:
                for w in range(v_bands):
                    v_t = hk.transpose_sep(v_rows[w])
                    for t in range(vt_frags):
                        gr_vt.store_frag(v_t, c_chunk * vt_frags + t, w)

        load_kv(block_start(0))

        for kb in range(kv_blocks):
            kv_lo = kb * kv_block
            kv_start = hk.min(kv_lo, n_kv - kv_block)
            skip = kv_lo - kv_start         # rows this block does not own

            # The previous iteration's reads must retire before this one
            # overwrites the buffers. Single-buffered, so this is the whole
            # write-after-read hazard.
            hk.arch.barrier()
            commit_kv()
            # s_barrier orders execution, not memory: the ds_writes above have
            # to be retired explicitly or the next warp's ds_reads race them.
            hk.arch.barrier(drain=True)
            # Issue the next block's globals *after* the barrier that published
            # this one. Nothing below touches them until the next commit, so
            # they have the whole math body to land in.
            load_kv(block_start(kb + 1))

            # Everything above is workgroup-wide; everything below is this
            # warp's. Without the boundary the allocator holds both live sets at
            # once -- see hk.scope.
            with hk.scope():
                # The score tile is per block, not running: everything else in
                # this region accumulates across blocks and this does not.
                s_f = [hk.zeros(ty_s) for _ in range(kv_frags)]

                # ---- S^T = K . Q^T ---------------------------------------
                for kk in range(d_frags):
                    for n0 in range(0, kv_frags, qk_tiles):
                        k_win = [gr_k.load_frag(n0 + i, kk, ty_k)
                                 for i in range(qk_tiles)]
                        hk.lds.wait_for(*k_win)
                        hk.arch.setprio(1)
                        for i in range(qk_tiles):
                            s_f[n0 + i] += k_win[i] @ q_r[kk].T
                        hk.arch.setprio(0)

                for n in range(kv_frags):
                    s_f[n] *= scale_l2e

                # Neither mask below is behind a branch, and that took two
                # tries to get right. Both of them used to hold one register
                # per element of the tile -- the row index for the tail mask,
                # `col - row` for the causal one -- and both of those are
                # loop-invariant, so the compiler hoisted them into the
                # preheader and spilled them. A uniform `if` hid that, at the
                # price of control flow in the middle of the hot region, and
                # for the causal mask that price was the whole Q register
                # file: 156 B/lane of spill with the branch, none without it.
                # The fix belonged in the library instead -- detail::diag_fill
                # and detail::axis_fill fold the index into the lane term so
                # the per-element part is an immediate. One register each now,
                # and nothing left worth branching around.
                if causal:
                    # mask where kv > q. A row of S^T is a kv index and a
                    # column is a query, so with d = q_row - kv_start the
                    # predicate on fragment n is `col >= row - (d - 16n)`,
                    # which is `triu`. One v_cmp per element, no cross-lane
                    # traffic and no materialized mask tile -- the dividend of
                    # keeping the scores transposed.
                    #
                    d = q_row - kv_start
                    for n in range(kv_frags):
                        s_f[n].triu_(d - 16 * n, NEG_INF)

                # This one *is* behind a branch, and the asymmetry is real.
                # `skip` is nonzero on exactly one KV block in the whole loop
                # -- the backed-up last one, and only when kv_block does not
                # divide N -- so the branch is not taken at all on a shape
                # that divides, and the mask costs nothing instead of two VALU
                # per element on every block. Measured both ways at d=128:
                # 224 VGPRs and no spill with the branch, 116 B/lane of spill
                # without it. The causal mask above runs on nearly every block
                # and gets the opposite answer for the same reason.
                if skip > 0:
                    for n in range(kv_frags):
                        # A row of S^T is a kv index, so the overlap with the
                        # previous block is an `upper_fill` and not a
                        # predicate: rows below `skip` belong to that block
                        # and would be counted twice by the softmax sum.
                        s_f[n].upper_fill_(skip - 16 * n, NEG_INF)

                # ---- online softmax --------------------------------------
                # Every reduction is over kv, which is the element axis of a
                # col-layout accumulator: register-local steps plus one
                # cross-half exchange.
                m_old = m_run.copy()
                for n in range(kv_frags):
                    m_run.max_(s_f[n].max(axis=0))
                for n in range(kv_frags):
                    s_f[n] -= m_run
                    s_f[n].exp2_()     # s_f is P^T from here
                # The rescale is behind a branch and this one earns it. After
                # the first few KV blocks the running max is already the global
                # max, alpha is 1, and every one of the column multiplies below
                # is a v_mul_f32 against nothing -- 64 of them per block at
                # head_dim 128, against the block's 32 WMMAs. The predicate is
                # wave-uniform (every lane holds the whole q range of this
                # warp's tile in m_*), so it is an s_cbranch and not a divergent
                # region, and the branch is taken rarely enough that the
                # register cost the causal mask could not afford is paid back
                # here many times over.
                if m_run.any_ne(m_old):
                    alpha = (m_old - m_run).exp2()
                    l_run *= alpha
                    for c in o_c:
                        c *= alpha
                for n in range(kv_frags):
                    l_run += s_f[n].sum(axis=0)

                # ---- O^T += V^T . P^T ------------------------------------
                # The kv fragment is the outer loop so that only one 16x16 block
                # of P^T is ever in bf16 at a time. Converting the whole score
                # tile up front costs kv_frags * 8 more VGPRs, which at
                # head_dim=128 is the difference between fitting and spilling --
                # and a spill here is wrong, not slow, because the LDS schedule
                # hand-manages s_waitcnt.
                for n in range(kv_frags):
                    p_cvt = s_f[n].to(bf16)
                    for g0 in range(0, d_frags, pv_tiles):
                        v_win = [gr_vt.load_frag(g0 + i, n, ty_vr)
                                 for i in range(pv_tiles)]
                        hk.lds.wait_for(*v_win)
                        hk.arch.setprio(1)
                        for i in range(pv_tiles):
                            o_c[g0 + i] += v_win[i] @ p_cvt
                        hk.arch.setprio(0)

        # Normalise and go back to [q, d] for the store. The C++ moves the data
        # here, one fragment at a time; `hk.transpose` relabels, which is the
        # same permutation with no instructions at all -- a base tile's storage
        # depends on its dtype and not on its layout tag, so reading a [d, q]
        # col fragment as a [q, d] row fragment *is* the transpose.
        for i, c in enumerate(o_c):
            c /= l_run
            o[hk.elem_coord(b, head, q_row, 16 * i)] = hk.transpose(c)

    body.__name__ = name

    def _grid(p):
        return (hk.cdiv(p.q.rows, q_tile), p.q.depth, p.q.batch)

    return hk.kernel(body, warps=warps, grid=_grid, name=name,
                     max_vgprs=max_vgprs, min_occupancy=min_occupancy)


#: The tiling the C++'s sweep.sh picked, and the one every shape gets until a
#: tuning record says otherwise.
DEFAULT = dict(vt_d_chunk=16, qk_tiles=2, pv_tiles=2)

#: What the tuner searches, and deliberately not more than this.
#:
#: `qk_tiles`/`pv_tiles` are how many 16x16 LDS reads go in flight before the
#: wait that retires them, and `vt_d_chunk` is how wide a band of V^T is staged
#: at a time. All three are *pure schedule*: they move instruction scheduling
#: and register pressure and nothing else. The knobs that are not here --
#: q_block, kv_block, warps -- change `Q_TILE`, which is part of what
#: `attention` will accept, so tuning them would make the set of supported
#: shapes depend on a file in a cache directory. That is a trade this kernel
#: does not need to make: it is already at 1.000-1.008x of the handwritten C++,
#: so the upside here is small and the downside is a shape that worked last
#: week.
SPACE = _autotune.space(vt_d_chunk=[16, 32], qk_tiles=[1, 2, 3, 4],
                        pv_tiles=[1, 2, 3, 4])

#: Sequence lengths the CLI tunes by default: one per order of magnitude that
#: H3 actually runs, bucketed the same way `_tune_key` buckets them.
TUNE_KEYS = ["n4096", "n16384", "n65536"]


def _bench_for(head_dim: int, causal: bool):
    """A benchmark closure factory for one (head_dim, causal) tuner.

    One shape, timed with CUDA events, warmed up first. The tuner owns the
    *order* the candidates run in -- that is the part that has to be
    interleaved for the numbers to mean anything on this chip -- and this owns
    nothing but the launch.
    """
    def for_key(key: str):
        import torch  # noqa: PLC0415

        n = int(key.lstrip("n"))
        g = torch.Generator(device="cuda").manual_seed(0)
        q, k, v = (torch.randn(1, 16, n, head_dim, device="cuda",
                               dtype=torch.bfloat16, generator=g)
                   for _ in range(3))
        o = torch.empty_like(q)

        # Fewer iterations as N grows: the kernel is O(N^2) and a 65536-long
        # sequence is 25 s of GPU for ten of them, times 32 candidates times
        # three rounds. The comparison is between candidates measured the same
        # way, so what matters is that every candidate gets the same count --
        # not that the count is the same at every length.
        iters = 10 if n <= 16384 else 3

        def run(kernel, warmup=2, iters=iters):
            for _ in range(warmup):
                kernel(q, k, v, o)
            torch.cuda.synchronize()
            beg, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            beg.record()
            for _ in range(iters):
                kernel(q, k, v, o)
            end.record()
            torch.cuda.synchronize()
            return beg.elapsed_time(end) / iters

        return run

    return for_key


def _tuner(head_dim: int, causal: bool) -> _autotune.Tuner:
    base = f"attn_fwd_d{head_dim}{'_causal' if causal else ''}"

    def make(**sched):
        # The default schedule keeps the bare name. That is not cosmetic: it is
        # the name in the compile cache, in every resource report in the README
        # and in the four-row table this kernel is judged by, and a tuner that
        # renamed it would invalidate all of them on the day it was added.
        name = base if sched == DEFAULT else f"{base}_{_autotune.label_of(sched)}"
        return attn_kernel(name, head_dim=head_dim, causal=causal, **sched)

    return _autotune.Tuner(base, make, SPACE, DEFAULT,
                           bench=_bench_for(head_dim, causal), keys=TUNE_KEYS)

#: One tuner per (head_dim, causal): the two are not schedule knobs, they are
#: different kernels, and a space that contained them would be a space most of
#: whose points answer a question nobody asked.
TUNERS = {(d, c): _tuner(d, c) for d in (128, 64) for c in (False, True)}

#: The shapes H3 runs, at the default schedule. Nothing here is traced or
#: compiled at import: `attn_kernel` only builds a Kernel object.
KERNELS = {t.name: t._kernel_for(DEFAULT) for t in TUNERS.values()}

#: Head dims with a shipped tiling.
HEAD_DIMS = (64, 128)
#: Query rows one workgroup owns: `warps * q_block` of the default tiling. A
#: shorter Q than this has no block to back up into, so it is rejected rather
#: than read out of bounds.
Q_TILE = 192
#: The KV step, and so the shortest K/V the backed-up last block can land on.
KV_BLOCK = 32

_SCALED: Dict[str, object] = {}


def _tune_key(n: int) -> str:
    """The tuning key for a sequence length: the next power of two.

    A record per exact N would be a record per request and a cache that never
    hits. What the schedule is sensitive to is how many KV blocks one workgroup
    walks before it retires, and that moves on a log scale.
    """
    b = 4096
    while b < n:
        b *= 2
    return f"n{b}"


def _kernel_for(head_dim: int, causal: bool, scale: Optional[float],
                n: int = 4096):
    """The compiled kernel for one (head_dim, causal, scale).

    `scale` is a trace-time constant (see `attn_kernel`), so a non-default one
    is a different kernel. Keying the name on the float's bit pattern rather
    than a counter keeps it stable across processes, which is what lets the
    on-disk compile cache hit on the second run instead of the second call.
    """
    suffix = "_causal" if causal else ""
    if scale is None or scale == head_dim ** -0.5:
        return TUNERS[(head_dim, causal)].kernel(_tune_key(n))
    import struct  # noqa: PLC0415 -- only on the non-default path
    bits = struct.unpack("<Q", struct.pack("<d", float(scale)))[0]
    name = f"attn_fwd_d{head_dim}{suffix}_s{bits:016x}"
    k = _SCALED.get(name)
    if k is None:
        k = _SCALED[name] = attn_kernel(name, head_dim=head_dim, causal=causal,
                                        scale=float(scale))
    return k


def _why(q, k, v, causal, *, require_contiguous: bool = True):
    """Why this call cannot run on the kernel, or '' if it can.

    Separate from `attention` because a drop-in SDPA has to *decide* without
    raising -- it falls back to torch -- while a direct call should get the
    reason as an exception. One list of rules, two policies.

    `require_contiguous=False` asks the question the drop-in needs: *would*
    this run if the tensors were made contiguous? Every other rule here reads
    shapes and dtypes only, so asking it costs nothing, and asking it first is
    what stops a call that is going to fall back from paying for three copies
    on its way to not using them. Measured on a SigLIP tower (head_dim 72,
    which this kernel does not have): copying first made the patched path 16%
    slower than no patch at all.
    """
    import torch  # noqa: PLC0415

    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        return f"bf16 only, got {q.dtype}/{k.dtype}/{v.dtype}"
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        return f"expects (B, H, N, D), got {q.dim()}/{k.dim()}/{v.dim()} dims"
    if require_contiguous and not (
            q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        return "q, k and v must be contiguous (B, H, N, D)"
    if k.shape != v.shape:
        return f"k and v must match: {tuple(k.shape)} vs {tuple(v.shape)}"
    b, h, n_q, d = q.shape
    b_k, h_kv, n_kv, d_k = k.shape
    if b_k != b or d_k != d:
        return f"batch and head_dim must match: {tuple(q.shape)} vs {tuple(k.shape)}"
    if d not in HEAD_DIMS:
        return f"head_dim {d} has no tiling; have {HEAD_DIMS}"
    if h_kv == 0 or h % h_kv:
        return f"GQA needs h_kv to divide h, got {h} query heads and {h_kv} kv heads"
    if n_q < Q_TILE:
        # The last Q block backs up to end at n_q instead of predicating, and
        # there is nothing to back into below one tile. Decode wants a split-KV
        # kernel anyway, not this one with 191 masked rows.
        return f"n_q={n_q} is below one Q tile ({Q_TILE})"
    if n_kv < KV_BLOCK:
        return f"n_kv={n_kv} is below one KV block ({KV_BLOCK})"
    if causal and n_kv != n_q:
        # Two conventions for where the diagonal sits when the lengths differ;
        # this kernel's is top-left and torch's is bottom-right.
        return f"causal cross attention: n_kv={n_kv} != n_q={n_q}"
    return ""


def attention(q, k, v, *, causal: bool = False, scale: Optional[float] = None,
              out=None):
    """Flash attention forward, (B, H, N, D) bf16, MHA or GQA.

    The operand layout is `F.scaled_dot_product_attention`'s, so a framework
    that already calls torch can be redirected without reshaping. What this
    takes and what it does not is `_why`; it raises rather than falls back,
    because a silent fallback inside an explicit `hk.attention` call is how a
    20 TFLOPs path gets mistaken for a 64 TFLOPs one.

    Ragged N is fine in both directions: the last Q block backs up to end at
    n_q and recomputes the overlap (idempotent -- a query row's output depends
    on that row alone), and the last KV block backs up and masks the rows it
    does not own.
    """
    import torch  # noqa: PLC0415 -- hk traces and compiles without torch

    why = _why(q, k, v, causal)
    if why:
        raise ValueError(f"hk.attention: {why}")
    if out is None:
        out = torch.empty_like(q)
    elif out.shape != q.shape or out.dtype != q.dtype or not out.is_contiguous():
        raise ValueError(f"hk.attention: out is {tuple(out.shape)}/{out.dtype}, "
                         f"expected a contiguous {tuple(q.shape)}/{q.dtype}")
    _kernel_for(q.shape[-1], causal, scale, q.shape[-2])(q, k, v, out)
    return out


def supported(q, k, v, causal: bool = False) -> bool:
    """Whether `attention` would run these operands on the kernel."""
    return not _why(q, k, v, causal)


__all__ = ["KERNELS", "TUNERS", "SPACE", "DEFAULT", "attn_kernel",
           "attention", "supported", "HEAD_DIMS", "Q_TILE", "KV_BLOCK"]
