"""Attention over a paged KV cache -- the decode half.

vLLM V1 does not keep K and V as dense `(B, H, N, D)` tensors. It keeps them
in fixed-size pages and hands the kernel a *page table*: `block_table[req][i]`
is where request `req`'s i-th page lives. There is no contiguous tensor on
that path, which is why an SDPA drop-in cannot reach a decoder however it is
configured, and why this file exists.

**The layout is ours, and that is the point.** `gl`'s strides are derived from
its dims, so a `(block, token, head, dim)` cache -- vLLM's default -- cannot
produce a contiguous `(block_size, head_dim)` tile for one head. Since a
backend declares its own `get_kv_cache_shape`, this one declares

    (num_blocks, num_kv_heads, block_size, head_dim)

which is exactly `gl`'s `(b, d, r, c)`, so one page of one head is a plain
tile load. The cost is that the cache update becomes a strided scatter instead
of a contiguous one -- and that is the right way round, because a decode step
*reads* `seq_len` pages per layer and *writes* one. The read is thousands of
times the traffic.

**Page size is the KV block.** `KV_BLOCK` here is the cache's page size rather
than a tiling choice, so one loop iteration is one page and the within-page
offset is always zero. vLLM's ROCm backends offer 16 and 32; 32 is what the
dense kernel already uses.

What this does not do, deliberately, in its first form: no split across the KV
axis (one workgroup per (request, kv head), walking the whole sequence), and
no prefill. Both are measurable next steps rather than guesses -- see
`python/README.md`.
"""

import math
from typing import Dict, Optional

from .. import dsl as hk
from ..ir.nodes import bf16, fp32, i32

#: The cache's page size. Also the kernel's KV block, by construction.
KV_BLOCK = 32

#: Query rows per workgroup. A decode step has one query token per request, so
#: the rows are the GQA group -- the query heads that share this kv head --
#: padded to a tile. Packing the group rather than one row is what keeps the
#: WMMA from being 1/16 occupied.
Q_TILE = 16

NEG_INF = -1e30


def paged_decode_kernel(name: str, *, head_dim: int = 128,
                        kv_block: int = KV_BLOCK,
                        scale: Optional[float] = None,
                        max_vgprs: int = 256, min_occupancy: int = 1):
    """One workgroup per (request, kv head); it walks that request's pages.

    Single warp on purpose in this form. Decode attention is bandwidth bound --
    it reads the whole KV cache and does one query row of maths against it --
    so the schedule that matters is the read, and splitting the KV axis across
    warps buys parallelism at the cost of a cross-warp merge of (o, m, l). The
    merge is the next step, not this one; what this has to be first is right.
    """
    if head_dim % 16:
        raise ValueError(f"head_dim {head_dim} must be a multiple of 16")
    if kv_block % 16:
        raise ValueError(f"kv_block {kv_block} must be a multiple of 16")
    d_frags = head_dim // 16
    kv_frags = kv_block // 16

    # exp2, with log2(e) folded into the scale, and the scale applied to S
    # rather than to Q -- Q is bf16 and scaling it would round the input twice.
    scale_l2e = (scale if scale is not None else head_dim ** -0.5) * math.log2(math.e)

    ty_q = hk.RT(bf16, 16, 16, "row")        # one d-slice of the Q group
    ty_k = hk.RT(bf16, 16, 16, "row")        # one [kv, d] fragment of a page
    ty_s = hk.RT(fp32, 16, Q_TILE, "col")    # one [kv, q] fragment of S^T
    ty_o = hk.RT(fp32, 16, Q_TILE, "col")    # one [d, q] fragment of O^T
    ty_v = hk.RT(bf16, 16, 16, "row")        # a [kv, d] fragment of V
    ty_vt = hk.RT(bf16, 16, 16, "row")       # the same, transposed
    ty_vec = hk.row_vec(ty_s)                # one entry per query row

    def body(q: hk.Tensor[bf16], k_cache: hk.Tensor[bf16], v_cache: hk.Tensor[bf16],
             block_table: hk.Tensor[i32], seq_lens: hk.Tensor[i32],
             o: hk.Tensor[bf16]):
        req = hk.arch.block_idx.z
        kv_head = hk.arch.block_idx.y

        # The only two numbers that are not in the shapes: how long this
        # request is, and where its pages are. Both come out of memory.
        seq_len = seq_lens[hk.elem_coord(0, 0, 0, req)].load_scalar()
        n_pages = hk.cdiv(seq_len, kv_block)

        q_r = [q[hk.elem_coord(req, kv_head, 0, 16 * kk)].load(ty_q)
               for kk in range(d_frags)]

        o_c = [hk.zeros(ty_o) for _ in range(d_frags)]
        m_run = hk.neg_infty(ty_vec)
        l_run = hk.zeros(ty_vec)

        for pi in range(n_pages):
            page = block_table[hk.elem_coord(0, 0, req, pi)].load_scalar()

            with hk.scope():
                # S^T = K . Q^T, one [kv, d] fragment of the page at a time.
                s_f = [hk.zeros(ty_s) for _ in range(kv_frags)]
                for kk in range(d_frags):
                    for n in range(kv_frags):
                        with hk.scope():
                            k_f = k_cache[hk.elem_coord(page, kv_head, 16 * n,
                                                        16 * kk)].load(ty_k)
                            s_f[n] += k_f @ q_r[kk].T

                for n in range(kv_frags):
                    s_f[n] *= scale_l2e

                # The tail. A row of S^T is a kv index, so positions at or past
                # the sequence length are rows to fill -- not a predicate, and
                # not a backed-up block either: pages are allocated whole and
                # the slots past the end hold whatever the allocator left.
                # Counting them would corrupt the softmax sum.
                valid = seq_len - pi * kv_block
                if valid < kv_block:
                    for n in range(kv_frags):
                        s_f[n].lower_fill_(valid - 16 * n, NEG_INF)

                # ---- online softmax, the dense kernel's, unchanged ----
                m_old = m_run.copy()
                for n in range(kv_frags):
                    m_run.max_(s_f[n].max(axis=0))
                for n in range(kv_frags):
                    s_f[n] -= m_run
                    s_f[n].exp2_()
                if m_run.any_ne(m_old):
                    alpha = (m_old - m_run).exp2()
                    l_run *= alpha
                    for c in o_c:
                        c *= alpha
                for n in range(kv_frags):
                    l_run += s_f[n].sum(axis=0)

                # ---- O^T += V^T . P^T ----
                # V arrives [kv, d] and the matmul reads [d, kv], so each
                # fragment is transposed in registers on the way in. With one
                # warp there is nobody to share a staged copy with, so it does
                # not go through LDS at all.
                for n in range(kv_frags):
                    p_cvt = s_f[n].to(bf16)
                    for g0 in range(d_frags):
                        # One V fragment live at a time. Without the scope the
                        # allocator keeps all d_frags loads in flight -- eight
                        # at head_dim 128 -- on top of the O and Q tiles that
                        # are pinned for the whole kernel, and that is exactly
                        # the register file.
                        with hk.scope():
                            v_f = v_cache[hk.elem_coord(page, kv_head, 16 * n,
                                                        16 * g0)].load(ty_v)
                            o_c[g0] += hk.transpose_sep(v_f) @ p_cvt

        for i, c in enumerate(o_c):
            c /= l_run
            o[hk.elem_coord(req, kv_head, 0, 16 * i)] = hk.transpose(c)

    body.__name__ = name
    return hk.kernel(body, warps=1, name=name,
                     grid=lambda p: (1, p.q.depth, p.q.batch),
                     max_vgprs=max_vgprs, min_occupancy=min_occupancy)


def paged_decode_split_kernel(name: str, *, head_dim: int = 128,
                              splits: int = 4, kv_block: int = KV_BLOCK,
                              scale: Optional[float] = None,
                              max_vgprs: int = 256, min_occupancy: int = 1):
    """The same walk, cut `splits` ways along the KV axis.

    Why this exists, measured rather than assumed. Without it one workgroup
    per (request, kv head) walks the whole context, so a batch-1 step on
    Qwen3-8B launches 8 workgroups of one warp each on a 48-WGP GPU. Going
    from 1024 to 4096 tokens of context cost that kernel +28.2 ms/step against
    vLLM's Triton backend's +0.97 -- 16 GB/s against 469, or 1.9% of this
    card's bandwidth against 54%. Decode attention is a bandwidth problem and
    the only way to use the bandwidth is to put the machine on it.

    Each workgroup produces a *partial* softmax over its page range: the
    unnormalised O, the running max m, and the running sum l. It does not
    divide, because a split's l is not the sequence's l. `merge_splits` below
    combines them.

    `splits` is a constexpr rather than a grid dimension read from a tensor,
    so the page range is compile-time arithmetic and a split past the end is
    a branch the workgroup takes once. A handful of instantiations covers
    every batch size; they are cached like any other schedule.
    """
    if head_dim % 16 or kv_block % 16:
        raise ValueError("head_dim and kv_block must be multiples of 16")
    d_frags, kv_frags = head_dim // 16, kv_block // 16
    scale_l2e = (scale if scale is not None else head_dim ** -0.5) * math.log2(math.e)

    ty_q = hk.RT(bf16, 16, 16, "row")
    ty_k = hk.RT(bf16, 16, 16, "row")
    ty_s = hk.RT(fp32, 16, Q_TILE, "col")
    ty_o = hk.RT(fp32, 16, Q_TILE, "col")
    ty_v = hk.RT(bf16, 16, 16, "row")
    ty_vec = hk.row_vec(ty_s)

    def body(q: hk.Tensor[bf16], k_cache: hk.Tensor[bf16], v_cache: hk.Tensor[bf16],
             block_table: hk.Tensor[i32], seq_lens: hk.Tensor[i32],
             o_part: hk.Tensor[fp32], ml_part: hk.Tensor[fp32]):
        split = hk.arch.block_idx.x
        kv_head = hk.arch.block_idx.y
        req = hk.arch.block_idx.z

        seq_len = seq_lens[hk.elem_coord(0, 0, 0, req)].load_scalar()
        n_pages = hk.cdiv(seq_len, kv_block)
        per = hk.cdiv(n_pages, splits)
        lo = split * per
        hi = hk.min(n_pages, lo + per)

        q_r = [q[hk.elem_coord(req, kv_head, 0, 16 * kk)].load(ty_q)
               for kk in range(d_frags)]
        o_c = [hk.zeros(ty_o) for _ in range(d_frags)]
        m_run = hk.neg_infty(ty_vec)
        l_run = hk.zeros(ty_vec)

        # A split past the end of a short sequence contributes nothing. It
        # still has to write, because the merge reads every split's slot --
        # m = -inf and l = 0 make it weightless there.
        for pi in range(hk.max(hi - lo, 0)):
            page = block_table[hk.elem_coord(0, 0, req, lo + pi)].load_scalar()
            with hk.scope():
                s_f = [hk.zeros(ty_s) for _ in range(kv_frags)]
                for kk in range(d_frags):
                    for n in range(kv_frags):
                        with hk.scope():
                            k_f = k_cache[hk.elem_coord(page, kv_head, 16 * n,
                                                        16 * kk)].load(ty_k)
                            s_f[n] += k_f @ q_r[kk].T
                for n in range(kv_frags):
                    s_f[n] *= scale_l2e

                valid = seq_len - (lo + pi) * kv_block
                if valid < kv_block:
                    for n in range(kv_frags):
                        s_f[n].lower_fill_(valid - 16 * n, NEG_INF)

                m_old = m_run.copy()
                for n in range(kv_frags):
                    m_run.max_(s_f[n].max(axis=0))
                for n in range(kv_frags):
                    s_f[n] -= m_run
                    s_f[n].exp2_()
                if m_run.any_ne(m_old):
                    alpha = (m_old - m_run).exp2()
                    l_run *= alpha
                    for c in o_c:
                        c *= alpha
                for n in range(kv_frags):
                    l_run += s_f[n].sum(axis=0)

                for n in range(kv_frags):
                    p_cvt = s_f[n].to(bf16)
                    for g0 in range(d_frags):
                        with hk.scope():
                            v_f = v_cache[hk.elem_coord(page, kv_head, 16 * n,
                                                        16 * g0)].load(ty_v)
                            o_c[g0] += hk.transpose_sep(v_f) @ p_cvt

        # Unnormalised. The merge divides once, by the sum of the rescaled
        # per-split sums; dividing here would throw away the only thing that
        # makes the splits combinable.
        row = req * splits + split
        for i, c in enumerate(o_c):
            o_part[hk.elem_coord(row, kv_head, 0, 16 * i)] = hk.transpose(c)
        ml_part[hk.elem_coord(row, kv_head, 0, 0)] = m_run
        ml_part[hk.elem_coord(row, kv_head, 1, 0)] = l_run

    body.__name__ = name
    return hk.kernel(body, warps=1, name=name,
                     grid=lambda p: (splits, p.q.depth, p.q.batch),
                     max_vgprs=max_vgprs, min_occupancy=min_occupancy)


def merge_splits(o_part, ml_part, splits: int, out):
    """Combine the per-split partial softmaxes into the answer.

    In torch, for now, and deliberately: it is a few elementwise passes over a
    tensor whose size is `splits` times the output, which is small next to the
    KV traffic the splits exist to parallelise. If it shows up in a profile it
    becomes a kernel; writing it as one first would be optimising a thing
    nobody has measured.
    """
    import torch  # noqa: PLC0415

    r_s, hkv, tile, d = o_part.shape
    reqs = r_s // splits
    o = o_part.view(reqs, splits, hkv, tile, d)
    m = ml_part.view(reqs, splits, hkv, 2, tile)[:, :, :, 0]      # (r,s,h,t)
    l = ml_part.view(reqs, splits, hkv, 2, tile)[:, :, :, 1]

    m_max = m.amax(dim=1, keepdim=True)
    # A split that saw nothing has m = -inf and l = 0; exp2(-inf - m) is 0, so
    # it drops out -- unless *every* split saw nothing, which cannot happen
    # because split 0 always covers page 0.
    #
    # The numerator and the denominator take *different* weights, and getting
    # that wrong is the whole difficulty of a split softmax. O_s is already a
    # sum of (probability x V) under split s's own maximum, so rescaling it to
    # the global maximum takes exp2(m_s - m) and nothing else; l_s is a sum of
    # those same probabilities and takes the same factor. Weighting O by
    # exp2(m_s - m) * l_s -- which is what this did first -- multiplies every
    # numerator by its split's sum a second time. The weights still sum to
    # something, so the answer looks like an answer: 77x off, not NaN.
    rescale = torch.exp2(m - m_max)                                # (r,s,h,t)
    num = (o * rescale.unsqueeze(-1)).sum(dim=1)                   # (r,h,t,d)
    den = (l * rescale).sum(dim=1).clamp_min(1e-30)                # (r,h,t)
    out.copy_((num / den.unsqueeze(-1)).to(out.dtype))
    return out


#: head_dim -> kernel. Nothing is traced or compiled at import.
KERNELS: Dict[int, object] = {
    64: paged_decode_kernel("paged_decode_d64", head_dim=64),
    128: paged_decode_kernel("paged_decode_d128", head_dim=128),
}

#: (head_dim, splits) -> kernel, built on demand.
_SPLIT: Dict[tuple, object] = {}


def split_kernel(head_dim: int, splits: int):
    key = (head_dim, splits)
    if key not in _SPLIT:
        _SPLIT[key] = paged_decode_split_kernel(
            f"paged_decode_d{head_dim}_s{splits}",
            head_dim=head_dim, splits=splits)
    return _SPLIT[key]


#: Workgroups to aim for, measured rather than reasoned
#: (tools/hk-bench/paged_splits.py, Qwen3-8B's shape on a W7900D: 8 kv heads,
#: head_dim 128). Each column is a split count, each cell the kernel plus its
#: merge in ms, and the last two are the achieved fraction of the card's
#: 864 GB/s at the best split:
#:
#:     ctx  batch     s1     s4    s16    s32   best   %HBM
#:    1024      1  0.312  0.173  0.135  0.132     32     4%
#:    1024     16  0.310  0.205  0.242  0.323      8    41%
#:    4096      1  0.991  0.327  0.157  0.132     32    15%
#:    4096     16  1.310  0.643  0.607  0.682      8    56%
#:   16384      1  3.835  1.047  0.366  0.246     32    32%
#:   16384     16  5.081  2.183  1.796  1.869      8    71%
#:   16384     32  5.675  3.287  3.403  3.649      4    76%
#:
#: The best split at every one of the fifteen points measured puts
#: `n_reqs * n_kv_heads * splits` at about a thousand, so that is the target.
#: The first version of this aimed for 96 -- two per WGP -- which is the right
#: instinct for a workgroup of twelve warps and wrong by an order of magnitude
#: for one of a single warp. At batch 16 it asked for one split, which is 128
#: workgroups of one warp each walking 128 pages in series.
TARGET_WORKGROUPS = 1024

#: Past this the merge's traffic grows faster than the split saves, and every
#: value is a separate compile.
MAX_SPLITS = 32


def plan_splits(n_reqs: int, n_kv_heads: int, n_pages: int) -> int:
    """How many ways to cut the KV axis, given how empty the machine is.

    Without a split the grid is `n_reqs * n_kv_heads` one-warp workgroups.
    This asks for enough splits to reach `TARGET_WORKGROUPS`, capped by the
    page count -- a split with no pages is pure overhead -- and by
    `MAX_SPLITS`.
    """
    import os  # noqa: PLC0415

    forced = os.environ.get("HK_SPLITS", "auto")
    if forced in ("off", "0", "1"):
        # Off by default, against the measurements, because of an open fault.
        #
        # Under vLLM at batch 16 / context 4096 the second `generate` aborts
        # with HSA_STATUS_ERROR_EXCEPTION 0x1016 inside an unrelated torch
        # kernel. What is known: forcing one split makes it go away and
        # forcing prefill onto torch does not, so it is this path; and the
        # kernel itself is clean in isolation -- 2000 calls with ragged
        # lengths, padded block tables and guard pages either side of both
        # outputs leave every margin byte untouched and every output finite.
        # So the kernel does not write out of bounds and something about how
        # it is driven does. Until that is found, the split is opt-in
        # (HK_SPLITS=8) and the shipped path is the one that has run a model
        # end to end without faulting.
        return 1
    if forced in ("auto", "plan"):
        have = max(1, n_reqs * n_kv_heads)
        want = -(-TARGET_WORKGROUPS // have)      # ceil
        return max(1, min(want, n_pages, MAX_SPLITS))
    return max(1, min(int(forced), n_pages, MAX_SPLITS))
