"""Row-wise normalizations: RMSNorm, LayerNorm, softmax.

The three share a skeleton -- reduce along a row, then rescale that row -- and
the skeleton is where all the difficulty is, so it is written once here and the
three kernels differ only in which reduction and which rescale.

Four things about that skeleton are worth reading before the code.

**The tensor does not have to tile.** include/rdna3 bounds-checks nothing, so a
partial block cannot simply be read: past the end of a torch allocation is a
fault, not zeros. The library's own answer (attn.cpp:735) is to back the
*address* up until the tile is wholly in bounds. That is exact for an
elementwise kernel -- the overlapping elements are simply computed twice with
the same answer -- but a reduction *along the axis you backed up* would count
those elements twice and give a wrong mean. So the overlap is masked with the
reduction's identity before it is folded in:

    lo  = s_min(cb * COLS, cols - COLS)   # backed-up start of block cb
    pad = cb * COLS - lo                  # 0 except in the last block
    t   = left_fill(load(x, (row, lo)), pad, identity)

Only for a reduction that is not idempotent, though. The overlap is real
elements of the same row, so a max or a min cannot be moved by seeing one of
them a second time; softmax's max pass and quantize's absmax pass skip the mask
entirely, and only the sums pay for it. The mask is also the reason `pad` is a
thunk: two of the three passes never ask for it.

Along rows there is nothing to mask: each row's normalization is independent of
every other, so recomputing a row in the backed-up last block writes the same
bytes. `lo_row` is a plain back-up.

Because the tile has to fit, this needs `rows >= ROWS` and `cols >= COLS`. A
hidden dimension smaller than 64 is not a case worth a second code path; the
wrapper says so rather than reading past the tensor.

**The accumulator is fp32 even when the tensor is bf16.** Not for the usual
accuracy reason alone: an rv's lane layout is derived from its element type, so
a bf16 tile and an fp32 vector do not agree about which lane holds which row,
and there is no such thing as a mixed-precision reduction here. The tensor is
loaded straight into an fp32 tile instead -- `kittens::load` runs the elements
through `base_types::convertor` on the way in -- and everything downstream is
fp32 until the store converts back.

That load is not free, and for a long time it was much less free than this
paragraph used to claim. An fp32 tile's lanes own every *other* column of a
16-element run, so a widening load issued one narrow access per element: 32
`global_load_u16` per 16x64 tile where two `global_load_b128` would do. The
answer is the same one the narrowing store already used in the other
direction -- read the whole run and pick, rather than ask the memory system
for a strided gather -- and it lives in `load`'s `widening` branch. It is also
why MAX_TPW's register figures moved: the narrow loads were live values.

**One warp per row block is not enough warps.** A 4096-row tensor at 16 rows
per block is 256 workgroups; this GPU has 192 SIMDs, so a one-warp workgroup
leaves 1.3 waves per SIMD and nothing to hide a load behind. The row is split
across WARPS warps of one workgroup instead and the partial reductions are
combined through LDS (lang.collective.cross_warp). That is parallelism bought
without an extra pass, and it is most of the difference between this and
torch.compile's numbers.

**The other half is the pass count.** Written the obvious way this is three
passes over memory for softmax and two for the others, because the values are
re-read rather than kept. torch inductor's generated kernel is *persistent*: it
holds the row in registers and reads it once. Per achieved byte the two are the
same speed, so the extra pass is the entire gap.

So this holds the row too, when it fits. `TPW` is how many 16x64 fp32 tiles
each warp keeps live; at WARPS=16 and TPW=4 that is a 4096-wide row held across
a 512-thread workgroup at 128 VGPRs a lane, and the kernel reads global memory
exactly once. When it does not fit, `TPW=0` selects the streaming variant and
the same code re-reads instead -- `visit()` below is deliberately the only
place that knows which.

A 16384-wide row does not fit either way: 16 rows of it is a megabyte. But *one*
row of it does, and `FOLD` is how a 16-row tile addresses a one-row problem --
the wrapper reshapes (R, C) to (R*16, C/16) so a tile's 16 rows are 16 chunks
of a single row, and `fold_rows` combines them at the end. See
lang.collective.fold_rows.

`plan` picks between the three, and it needs to know *which* of the three ops
is asking. rms and layer want the row held unfolded wherever that is legal;
softmax wants it folded at the same width, by half again. The reason is in
`_FOLD_FIRST_KINDS`, with the measurement.
"""


from typing import Any

from .. import dsl as hk
from ..dsl.collective import cross_warp, fold_rows
from ..ir.nodes import bf16, fp16, fp32

#: 16 rows x 64 columns of fp32 is 32 VGPRs. 64 bf16 columns is also what makes
#: the global load issue 16-byte accesses instead of going elementwise.
ROWS, COLS = 16, 64

#: Live fp32 tiles one warp may hold in the persistent variant. Four is 128
#: VGPRs of data; the reductions' temporaries bring a quantize kernel to 167,
#: which is under the 240 cliff and leaves 9 waves per SIMD at gfx1100's
#: allocation granule of 24.
#:
#: Five was tried and rejected, and the two reasons are worth keeping apart.
#:
#: It is *nearly* legal now. Five used to spill outright -- a spill here is a
#: wrong answer rather than a slowdown, see hk.build's hard gate -- and the
#: widening global load freed enough registers that the exact variant builds:
#: rms 216/occ 7, layer 216/occ 7, softmax 240/occ 6 with four registers of
#: headroom. The *inexact* variant of softmax, which carries the clamp and the
#: mask, still does not, and plan emits TPW=MAX_TPW at widths that are not a
#: whole number of blocks. So five is not available at every width it would be
#: chosen for, which is already disqualifying.
#:
#: And it would not pay where it is available. Raising the cap changes the plan
#: for exactly one band -- 4097 to 5120 columns, 65 to 80 blocks -- which
#: otherwise falls through to FOLD. Measured at the top of that band
#: (norm_plan.py, rms 16384x5120, the shape Qwen2.5-14B would hit):
#:
#:     w4  tpw2 fold16   0.467      <- what MAX_TPW=4 picks
#:     w16 tpw5 fold1    0.512      <- what MAX_TPW=5 would pick
#:     w16 tpw0 fold1    0.606
#:
#: Ten percent the wrong way. Five tiles plus the reduction temporaries is 216
#: VGPRs and occupancy 7 where four is 167 and occupancy 9, and at this width
#: the extra occupancy is worth more than the saved pass. The cliff is between
#: four and five, not between five and six.
MAX_TPW = 4

#: 16 warps is a 512-thread workgroup. Past that the workgroup stops fitting
#: alongside anything else and the barrier starts to cost more than the
#: parallelism is worth.
MAX_WARPS = 16

_NEG_INF = float("-inf")


def _grid(p):
    # One workgroup per row block. The column loop is inside the kernel: a
    # normalization is a reduction along the row, so the whole row has to be
    # seen by one workgroup, and splitting it across workgroups would need a
    # second pass through memory to combine the partials.
    return (1, hk.cdiv(p.x.rows, p.ROWS), 1)


def _block_start(idx, block: int, extent, exact: bool = False):
    """Where block `idx` starts, backed up so the whole tile is in bounds.

    Returns `(start, backed_up_by)`. The second is how many leading elements of
    the loaded tile repeat the previous block -- zero for every block but the
    last, and zero for that one too when the extent divides. It is a thunk
    because two of the three passes do not need it, and an unused `const int`
    in the generated C++ is a line the next reader has to rule out.

    `exact` says the caller has *proved*, at trace time, that no block is ever
    backed up: the extent is a whole number of blocks and no warp is handed an
    index past the last of them. Then there is no clamp and no overlap, and the
    thunk is None -- which is not the same as a thunk returning zero. A zero
    that only the host knows about still costs the kernel a `const int` it
    compares against every column, and it costs LayerNorm a whole tile copy,
    because the only reason that copy exists is to have something maskable that
    is not the held tile. See `_mask` and `_exact`.
    """
    want = idx * block
    if exact:
        return want, None
    lo = hk.min(want, extent - block)
    return lo, lambda: want - lo


def _mask(tile, pad, ident):
    """Fill the backed-up overlap of `tile` with a reduction's identity.

    A no-op when `pad` is None -- see `_block_start` -- so that a kernel can
    write the masking unconditionally and still emit nothing for a width that
    divides.
    """
    if pad is not None:
        tile.left_fill_(pad(), ident)
    return tile


def _exact(cols: int, warps: int, tpw: int) -> bool:
    """Is every column block this plan visits wholly in bounds, unclamped?

    Two conditions, and both are decided by `plan` before anything is traced:
    the row is a whole number of blocks, and the warp-slots the kernel hands
    out land exactly on those blocks. The persistent variant hands out
    WARPS*TPW of them once; the streaming one hands out WARPS per step for
    ceil(nblk/WARPS) steps, so it needs WARPS to divide nblk or the last step
    runs off the end.
    """
    if cols % COLS:
        return False
    nblk = _ceil(cols, COLS)
    return warps * tpw == nblk if tpw else nblk % warps == 0


def _ceil(a: int, b: int) -> int:
    # Not lang.host.cdiv: that builds a *traced* host expression for the grid
    # lambda, and everything in plan() is a plain Python integer decided before
    # anything is traced.
    return -(-a // b)


def _split(nblk: int):
    """(warps, tiles per warp) for `nblk` column blocks in one workgroup.

    One warp per block, capped at MAX_WARPS. Only past sixteen blocks does a
    warp hold more than one tile, and then as evenly as ceiling division
    allows.

    **This used to round the warp count down to a power of two**, which costs
    nothing when `nblk` is itself a power of two and a great deal when it is
    not: five blocks went to four warps holding two tiles each, eight
    warp-slots for five blocks, so three eighths of the warps did redundant
    work on the clamped last block while every warp paid for a second live
    tile. Measured at the width where that happens (tools/hk-bench/
    odd_warps.py, 5120 columns, which folds to five blocks -- and is
    Qwen2.5-14B's hidden size):

                      quantize 16384x5120   rms 16384x5120   softmax 8192x5120
        w6 tpw1             0.366               0.465              0.254
        w5 tpw1             0.368               0.478              0.253
        w3 tpw2             0.385               0.487              0.258
        w4 tpw2 (was)       0.407               0.474              0.265
        torch.compile       0.371               0.780              0.335

    Note which axis explains it. w3 and w6 hand out the same six slots and w6
    is the faster of the two in all three ops, so this is not slot-counting:
    it is TPW. A second live tile is 32 more VGPRs a lane, and at this width
    the occupancy is worth more than the halved workgroup count. The old rule
    could not express "five warps" and so could not reach either.

    That quantize row is the whole reason this changed -- it was the one shape
    in the Phase 2 bench where hk lost to torch.compile.

    Five warps and not six, even though six is faster in the table: six covers
    five blocks with a sixth warp that has nothing of its own and is clamped
    onto the last block, and "round the warp count *up* past the work" is a
    rule with one supporting measurement (rms, 2.8%, with quantize and softmax
    calling it noise) and no mechanism. Five covers the blocks exactly. The
    price is that rms at this one width gives up 1% against the old w4 tpw2,
    which is what the other two ops gain eleven and four and a half percent to
    pay for.

    The cost is kernels: WARPS is a constexpr, so admitting every count from 1
    to 16 takes the shipped table from 45 norm kernels to 144. They are
    compiled lazily and cached on content hash, so a process still pays only
    for the widths it sees.
    """
    warps = min(MAX_WARPS, nblk)
    return warps, _ceil(nblk, warps)


#: Below this many warps the folded view is not worth taking. Folding divides
#: the blocks a workgroup covers by 16, so a row narrow enough would fold down
#: to a one- or two-warp workgroup that still pays fold_rows' LDS round trip.
#: Four is where the measurement starts -- 4096 columns fold to four blocks,
#: which is the narrowest folded case in the sweep. It gates only the
#: fold-first branch; a row too wide to hold unfolded folds at any width,
#: because there the alternative is streaming.
_FOLD_MIN_WARPS = 4

#: Which ops take the folded view *before* trying to hold the row unfolded.
#: Both read memory exactly once; what differs is how the work is spread. At
#: 16384x4096 the unfolded plan is 1024 workgroups of 16 warps holding four
#: tiles each, the folded one 16384 workgroups of four warps holding one --
#: same tiles, four times the warps, a quarter of the live registers in any of
#: them, and the register count is what buys occupancy.
#:
#: That used to be the whole argument, and it used to cover all three ops.
#: It does not any more. Since `load` grew its widening fast path -- a 16-bit
#: global read into an fp32 tile now issues two `global_load_b128` per 16x64
#: tile where it used to issue 32 `global_load_u16` -- the unfolded hold is
#: cheaper in registers (the narrow loads were live values: TPW=4 went 207
#: VGPR to 167, occupancy 7 to 9) and far cheaper in issue slots. It now wins
#: for rms and layer wherever it is legal. Measured (norm_plan.py, burst
#: timing, ms):
#:
#:                       folded w4 tpw1   unfolded w16 tpw4
#:     rms   16384x4096      0.402              0.361
#:     layer 16384x4096      0.378              0.354
#:     rms    8192x4096      0.201              0.172
#:     rms    6144x4096      0.117              0.087
#:     softmax 8192x4096     0.191              0.290   <- the exception
#:
#: softmax is the exception, and not by a little: folded is 1.5x there, and
#: the ordering holds at 2048x4096 too (0.051 vs 0.059). Its rescale pass is
#: the one of the three that is arithmetic-bound rather than issue-bound --
#: `v_exp_f32` is quarter rate on gfx1100 -- so occupancy is worth more to it
#: than the saved issue slots are, and folding is what buys occupancy. Which
#: is the paragraph above, surviving in the one place it is still true.
_FOLD_FIRST_KINDS = ("softmax",)


def plan(cols: int, kind: str = "rms"):
    """(WARPS, TPW, FOLD) for a normalization over rows of `cols` elements.

    Three outcomes, all about the row:

      persistent          TPW in 1..MAX_TPW, FOLD=1. A whole 16-row slab held
                          in registers across the workgroup, so the row is
                          read exactly once.
      persistent + fold   FOLD=ROWS. The caller reshapes so one workgroup
                          covers one row and a tile's 16 rows are 16 chunks of
                          it. Also read once, at a quarter of the live tiles
                          per warp. Tried first only for the kinds in
                          _FOLD_FIRST_KINDS; otherwise it is the fallback for
                          a row too wide to hold unfolded.
      streaming           TPW=0, FOLD=1. Neither fits; re-read the row.

    `kind` is the one of "rms" / "layer" / "softmax" being planned, and it is
    here because the three do not want the same answer -- see
    _FOLD_FIRST_KINDS for the measurement that says so.

    **This function used to ask a fourth question and no longer does.** It
    took the bytes that had to stay live for the re-read to hit, and when they
    fit in the 96 MB Infinity Cache it capped TPW and refused to fold, on the
    argument that paying registers to avoid a second read is a straight loss
    when the second read is a cache hit at better than a terabyte a second.
    The argument is still sound and the conclusion was still wrong, for the
    same reason quant.plan's copy of it was: it priced the hold at what the
    hold used to cost. With the widening load the held variants win inside the
    cache as well, by more than they win outside it (norm_plan.py, every case
    resident; `plan` here is what the cache branch chose):

        rms   4096x4096   plan w16 tpw0 f1 0.088   w16 tpw4 f1 0.063
        layer 4096x4096   plan w16 tpw0 f1 0.086   w16 tpw4 f1 0.066
        rms   6144x4096   plan w16 tpw0 f1 0.117   w16 tpw4 f1 0.087
        softmax 2048x4096 plan w16 tpw0 f1 0.056   w4  tpw1 f16 0.051

    Twenty-four to twenty-nine percent. So the regime question is gone, and
    with it the `working_set` argument and _CACHE_MAX_TPW. What remains true
    of the old branch is only its negative half: the *flat* rule it replaced
    (TPW=0, FOLD=1 everywhere, resident or not) was wrong by forty percent at
    1024 columns, and nothing below would pick it.

    Powers of two only for WARPS, so that the number of distinct kernels a
    process compiles stays small -- every (kind, dtype, WARPS, TPW, FOLD) is a
    separate hipcc invocation the first time it is seen.

    **What this still gets wrong**, from the sweep that fixed the rest of it
    (norm_plan.py, 16 cases; the rule names the winner or ties it in 12):

        rms 32768x2048   plan w16 tpw2 f1  0.358   w2 tpw0 f16  0.335   -6%
        rms 65536x1024   plan w16 tpw1 f1  0.367   w1 tpw0 f16  0.356   -3%
        softmax 8192x5120 plan w4 tpw2 f16 0.289   w2 tpw3 f16  0.275   -5%

    The first two are the same shape of miss: at a narrow width a *folded
    streaming* workgroup wins, and this function never emits one -- FOLD is
    only ever reached with a TPW that holds. The obvious patch is a fourth
    branch, and it is not taken, because 1024 columns disagrees with itself:
    at 16384 rows the plan's w16 tpw1 f1 is the outright winner (0.058 against
    0.064 for w1 tpw0 f16), and at 65536 rows it loses. Same width, opposite
    answers, so the width cannot decide it -- and what distinguishes the two
    is total size, which is the regime question this function just finished
    deleting on four cases' worth of evidence. Three percent is not enough to
    reinstate it on two.

    The third is _split's doing: 5 blocks over powers-of-two warps is 4x2,
    and the winner is 2x3. Admitting non-powers-of-two for WARPS would fix it
    and would roughly double the kernel count.
    """
    foldable = cols % ROWS == 0 and cols // ROWS >= COLS
    if kind in _FOLD_FIRST_KINDS and foldable:
        fw, ftpw = _split(_ceil(cols // ROWS, COLS))
        if ftpw <= MAX_TPW and fw >= _FOLD_MIN_WARPS:
            return fw, ftpw, ROWS
    warps, tpw = _split(_ceil(cols, COLS))
    if tpw <= MAX_TPW:
        return warps, tpw, 1
    # Too wide to hold unfolded. Fold anyway if it is legal at all -- even a
    # two-warp folded workgroup beats re-reading the row, which is why
    # _FOLD_MIN_WARPS does not apply here.
    if foldable:
        fw, ftpw = _split(_ceil(cols // ROWS, COLS))
        if ftpw <= MAX_TPW:
            return fw, ftpw, ROWS
    return warps, 0, 1


def _row_reduce_kernel(name: str, dtype, kind: str, warps: int):
    """A kernel that reduces each row and rescales it.

    `kind` picks the arithmetic:
      rms    -- y = x * rsqrt(mean(x^2) + eps)
      layer  -- y = (x - mean(x)) * rsqrt(var(x) + eps)
      softmax-- y = exp(x - max(x)) / sum(exp(x - max(x)))
    """
    def body(x: hk.Tensor[dtype], o: hk.Tensor[dtype], *, ROWS=ROWS, COLS=COLS,
             WARPS=warps, TPW=0, FOLD=1, EXACT=0, EPS=1e-6):
        t = hk.RT(fp32, ROWS, COLS)
        vt = hk.col_vec(t)
        ncols = x.cols
        nblk = hk.cdiv(ncols, COLS)
        row_lo, _ = _block_start(hk.arch.block_idx.y, ROWS, x.rows)
        # How many elements of the *original* row this workgroup covers. With
        # FOLD the tensor has been reshaped, so the tensor's column count is a
        # sixteenth of the row length and the means below would be 16x too big.
        rowlen = ncols * FOLD if FOLD != 1 else ncols

        def cb_of(step):
            """The column block this warp handles at `step`.

            Strided rather than blocked, so that at any moment the warps of a
            workgroup are reading 16 adjacent blocks -- one contiguous 4 KB
            run per row -- instead of 16 places a row apart.
            """
            return step * WARPS + hk.arch.warp_id() if WARPS > 1 else step

        def tile(cb):
            col_lo, pad = _block_start(cb, COLS, ncols, EXACT)
            idx = hk.elem_coord(0, 0, row_lo, col_lo)
            return x[idx].load(t), pad, idx

        # Over-provisioned blocks -- WARPS*TPW can exceed nblk, and a warp
        # whose step runs past the end gets cb >= nblk -- are not a special
        # case. _block_start clamps them to the last in-bounds block, so they
        # re-read data another warp already has; the mask then makes them the
        # identity for the reduction, and the write pass writes the same bytes
        # to the same place. Both are exactly what backing up already does for
        # a non-dividing tail.
        held = [tile(cb_of(s)) for s in range(TPW)] if TPW else None
        steps = None if TPW else hk.cdiv(nblk, WARPS)

        def visit(fn):
            """Call `fn(v, pad, idx)` once per column block this warp owns.

            Persistent: the tiles are already in registers and this costs
            nothing. Streaming: a device loop that re-reads them. Keeping the
            two behind one call is what stops the variants from drifting --
            everything below this point is written once.

            The one rule `fn` has to obey: **do not write to `v` except in the
            last pass.** In the streaming variant `v` is a fresh load and
            scribbling on it is harmless; in the persistent variant it is the
            tile the write pass will store. `left_fill_` writes in place -- it
            has to, see hk.dsl.values.Tile._fill -- so masking `v` directly
            would zero the overlap columns of the *output*, and another warp's
            copy of those same columns would then race with them. A shape that
            divides never shows it, because there the mask is empty. Mask a
            derived tile instead; every reduction below already makes one.
            """
            if TPW:
                for rec in held:
                    fn(*rec)
            else:
                for s in range(steps):
                    fn(*tile(cb_of(s)))

        def combine(vec, op):
            """Finish a per-warp partial: across the workgroup's warps, then
            across the tile's rows when they are chunks of one row."""
            vec = cross_warp(vec, op, WARPS)
            return fold_rows(vec, op, WARPS) if FOLD != 1 else vec

        if kind == "softmax":
            # Pass 1: the row max, for the shift that keeps exp in range.
            #
            # Unmasked, and that is not an oversight. Backing the address up
            # makes a block re-read columns of the *same row* that its
            # neighbour already covered -- real elements, never garbage -- and
            # a max cannot be moved by an element it has already seen. Same for
            # a warp whose step ran past the last block: it re-reads that block.
            # Only the sum below needs the overlap masked, because a sum is not
            # idempotent. Not masking also means nothing here writes to `v`,
            # which is the rule visit() states.
            m = hk.full(vt, _NEG_INF)
            visit(lambda v, pad, idx: m.max_(v.max(axis=1)))
            m = combine(m, "max")

            # Pass 2: sum of exp. The mask carries through: m is finite, so a
            # masked element is exp(-inf) = 0, the identity for sum.
            s = hk.zeros(vt)

            def accumulate_exp(v, pad, idx):
                nonlocal s
                # Shift first, mask second. The broadcast already allocates
                # the tile, so the mask lands on a temporary for free and `v`
                # stays untouched. The order is not arbitrary either way:
                # masking after the shift writes -inf, and exp(-inf) is 0,
                # the identity for the sum.
                p = _mask(v - m, pad, _NEG_INF)
                p.exp_()
                s += p.sum(axis=1)

            visit(accumulate_exp)
            s = combine(s, "add")

            # Pass 3: write. Unmasked -- a repeated column is written twice
            # with the same value, which is the point of backing up.
            def write(v, pad, idx):
                p = (v - m).exp()
                p /= s
                o[idx] = p.to(dtype)

            visit(write)
            return

        sq = hk.zeros(vt)
        mean = hk.zeros(vt) if kind == "layer" else None

        def accumulate(v, pad, idx):
            nonlocal sq, mean
            # A sum is not idempotent, so unlike the max above this one does
            # have to mask the overlap. What it must not do is mask `v` -- see
            # visit() -- so the mask lands on a tile derived from it. The two
            # kinds differ in which tile that is, and it is worth the branch:
            # each ends up holding exactly one temporary.
            if kind == "layer" and pad is not None:
                # LayerNorm needs the plain sum as well, so it needs a
                # maskable copy of the held tile: mask it, sum it, then square
                # it in place and sum that.
                m_t = _mask(v.copy(), pad, 0.0)
                mean += m_t.sum(axis=1)
                m_t *= m_t
                sq += m_t.sum(axis=1)
            else:
                # No copy. RMSNorm never needed one -- the multiply allocates
                # the tile it writes to, so the mask lands on a temporary for
                # free, and masking after squaring is the same sum because 0
                # squared is 0. LayerNorm gets here too when EXACT says there
                # is nothing to mask, and then its plain sum can read the held
                # tile directly; that copy is the whole cost of the `else`
                # branch not being taken, 32 VGPRs and 32 moves per tile.
                if kind == "layer":
                    mean += v.sum(axis=1)
                sq += _mask(v * v, pad, 0.0).sum(axis=1)

        visit(accumulate)
        sq = combine(sq, "add")
        if kind == "layer":
            mean = combine(mean, "add")

        # E[x^2], and for LayerNorm var = E[x^2] - E[x]^2.
        #
        # That identity cancels: on a row with mean 100 and variance 1 the two
        # terms agree to four digits and fp32 keeps about seven, so the
        # variance carries ~1e-3 of relative error. torch uses Welford and does
        # not. It is accepted here because the alternative is a third pass over
        # the row (mean, then deviations) for an input distribution -- a
        # normalized activation with a large offset -- that a LayerNorm is
        # placed to prevent. If a model ever needs it, the fix is the third
        # pass, not a wider accumulator.
        sq /= rowlen
        if kind == "layer":
            mean /= rowlen
            sq -= mean * mean
        sq += EPS
        scale = sq.rsqrt()

        def write(v, pad, idx):
            if kind == "layer":
                v -= mean
            v *= scale
            o[idx] = v.to(dtype)

        visit(write)

    body.__name__ = name
    return hk.kernel(body, warps=warps, grid=_grid, name=name)


_DTYPES = {"bf16": bf16, "fp16": fp16, "fp32": fp32}

#: How many warps a workgroup may have. `plan` only ever returns one of these,
#: and it can return any of them: `_split` hands out one warp per column block
#: up to MAX_WARPS, so a row of five blocks asks for five warps. See _split for
#: what restricting this to powers of two cost.
WARP_COUNTS = tuple(range(1, MAX_WARPS + 1))

#: (kind, dtype suffix, warps) -> Kernel. Nothing is compiled until called, and
#: TPW/FOLD ride in as constexpr arguments so that one entry here still covers
#: every column count.
KERNELS = {}
for _s, _dt in _DTYPES.items():
    for _k in ("rms", "layer", "softmax"):
        for _w in WARP_COUNTS:
            _n = f"{_k}_{_s}_w{_w}"
            KERNELS[_n] = _row_reduce_kernel(_n, _dt, _k, _w)
del _s, _dt, _k, _w, _n


# ---------------------------------------------------------------- wrappers

_TORCH_DTYPES = {
    "torch.bfloat16": "bf16",
    "torch.float16": "fp16",
    "torch.float32": "fp32",
}


def _suffix_of(t) -> str:
    name = _TORCH_DTYPES.get(str(t.dtype))
    if name is None:
        raise TypeError(f"hk norms handle bf16/fp16/fp32, not {t.dtype}")
    return name


def _shape_of(t):
    """(rows, cols) of the normalization `t` describes."""
    if not t.is_contiguous():
        raise ValueError("hk kernels take contiguous tensors; call .contiguous()")
    cols = t.shape[-1]
    rows = t.numel() // cols if cols else 0
    if cols < COLS or rows < ROWS:
        raise ValueError(
            f"shape {tuple(t.shape)} gives a {rows}x{cols} normalization and hk's "
            f"norm kernels need at least {ROWS}x{COLS}. The tail handling backs "
            f"the block up until it is in bounds, which needs a full tile to "
            f"exist; it does not need the shape to divide."
        )
    return rows, cols


def _view(t, rows: int, cols: int, fold: int):
    """The 2D view the kernel reads.

    With fold != 1 this is the reshape that puts one row per workgroup: row r
    of the original becomes rows r*fold .. r*fold+fold-1 of the view, which is
    exactly the `fold` consecutive tile rows that fold_rows combines. It is a
    view of the same contiguous memory, so it costs nothing.

    Nothing in bytes. `Tensor.view` is about a microsecond of Python and these
    kernels run for sixty, and the common case -- a 2D tensor, FOLD 1 -- asks
    for the shape it already has: `rows` was computed as `numel // cols` and
    `cols` as `shape[-1]`, so for `t.dim() == 2` the requested shape is `t`'s
    own by construction. Hand back `t`.
    """
    if fold == 1 and t.dim() == 2:
        return t
    return t.view(rows * fold, cols // fold)


#: (kind, torch dtype, cols) -> (kernel, TPW, FOLD, EXACT).
#:
#: Everything in that tuple is a function of the three key parts and nothing
#: else, and computing it costs a `plan` call, an `_exact` call, an f-string,
#: a `str(dtype)` and two dict probes. That is small, and it is not small
#: compared to what it is in front of: a 4096x4096 rmsnorm is sixty
#: microseconds of GPU and the bare pybind launcher under all of this is
#: 5.3 us. So each piece of the Python side that can be answered once per
#: (op, dtype, width) rather than once per call is worth removing. See
#: Kernel.__call__ for the other half.
#:
#: The key holds the `torch.dtype` object rather than hk's suffix for it,
#: because deriving the suffix is `str(dtype)` plus a lookup and a dtype is
#: already hashable. The suffix lookup then runs only on a miss -- including
#: its TypeError, which is raised before anything is cached, so an unsupported
#: dtype raises on every call and not just the first. `_suffix_of` stays for
#: the benches, which hold a tensor and want the name.
_PLAN_CACHE: dict = {}


def _resolve(kind: str, dtype, cols: int):
    key = (kind, dtype, cols)
    got = _PLAN_CACHE.get(key)
    if got is None:
        suffix = _TORCH_DTYPES.get(str(dtype))
        if suffix is None:
            raise TypeError(f"hk norms handle bf16/fp16/fp32, not {dtype}")
        warps, tpw, fold = plan(cols, kind)
        got = _PLAN_CACHE[key] = (
            KERNELS[f"{kind}_{suffix}_w{warps}"], tpw, fold,
            int(_exact(cols // fold, warps, tpw)),
        )
    return got


def _run(kind: str, x, out, eps: float):
    rows, cols = _shape_of(x)
    if out is None:
        # Imported here and not at the top because `hk` traces, builds and
        # compiles without torch -- tests/hk/ir and tests/hk/codegen run on a
        # machine with no torch and no GPU. Inside the branch and not at the
        # top of the function because `import` is a sys.modules probe and a
        # store every call, and the caller that passes `out` does not need it.
        import torch  # noqa: PLC0415
        out = torch.empty_like(x)
    elif out.shape != x.shape or out.dtype != x.dtype:
        raise ValueError(
            f"{kind}: out is {tuple(out.shape)}/{out.dtype}, "
            f"expected {tuple(x.shape)}/{x.dtype}"
        )
    k, tpw, fold, exact = _resolve(kind, x.dtype, cols)
    # The constexprs go through the call rather than `specialize(...)`: the
    # call path is keyed on them and hits the trace/build/module caches, where
    # a fresh specialization would re-trace and re-resolve the module on every
    # launch. See Kernel.specialize.
    #
    # Written as two calls rather than one with a `**extra` dict because the
    # dict is the expensive part -- Kernel.__call__ keys its launcher cache on
    # the kwargs tuple, and spelling the kwargs out lets CPython build that
    # tuple directly instead of building a dict for this frame to unpack and
    # the callee to rebuild.
    xv, ov = _view(x, rows, cols, fold), _view(out, rows, cols, fold)
    if kind == "softmax":
        k(xv, ov, TPW=tpw, FOLD=fold, EXACT=exact)
    else:
        k(xv, ov, TPW=tpw, FOLD=fold, EXACT=exact, EPS=eps)
    return out


def rmsnorm(x, eps: float = 1e-6, out=None):
    """x * rsqrt(mean(x^2) + eps), over the last axis.

    No learned weight: a weight multiply is elementwise and belongs in whatever
    fuses next, and adding it here would mean a second global and a broadcast
    that the caller may not want.
    """
    return _run("rms", x, out, eps)


def layernorm(x, eps: float = 1e-5, out=None):
    """(x - mean) * rsqrt(var + eps), over the last axis, unweighted.

    `var` is the biased estimate (divide by N), which is what torch's
    F.layer_norm uses.
    """
    return _run("layer", x, out, eps)


def softmax(x, out=None):
    """Softmax over the last axis, max-shifted."""
    return _run("softmax", x, out, 0.0)


__all__ = [
    "KERNELS", "ROWS", "COLS", "MAX_TPW", "MAX_WARPS", "WARP_COUNTS",
    "plan", "rmsnorm", "layernorm", "softmax",
]
