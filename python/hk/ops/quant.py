"""Dynamic per-row int8 quantization, and its inverse.

This is the W8A8 activation path: the weights are quantized offline, the
activations cannot be, so every forward pass has to find each row's scale and
narrow the row with it. It is a bandwidth-bound pass over the activation
tensor, which is exactly what this library is good at and exactly what a
framework would otherwise spend a separate kernel launch plus a round trip
through memory on.

What had to be added to include/rdna3 to make it possible, and what did not:

* **Added** (`common/base_types.cuh`): `packing<int8>` so `gl<int8_t>`
  instantiates, and `convertor<int8, float>` / `convertor<float, int8>`. The
  narrowing convertor rounds to nearest-even and saturates, neither of which a
  C cast does -- see the comment there.
* **Not added**: an int8 *register* tile. gfx1100 has no int8 WMMA and no
  packed int8 ALU, so there is nothing an rt<int8> would be good for, and
  `hk.rt(i8, ...)` refuses with that reasoning rather than letting someone
  discover it from a template error. The tile is fp32 the whole way; the
  convertor runs in the global store.

Symmetric, per row, and the scale is `absmax / 127` -- what vLLM's
`scaled_int8_quant` does for activations. Not per-tensor: a per-tensor scale
on an activation is decided by whichever row has the worst outlier, and on a
4096-wide hidden state that is a real loss.
"""


from .. import dsl as hk
from ..dsl.collective import cross_warp, fold_rows
from ..ir.nodes import bf16, fp16, fp32, i8
from .norm import (COLS, MAX_TPW, ROWS, WARP_COUNTS, _block_start, _ceil,
                   _exact, _split, _view)

#: absmax of an all-zero row is zero, and 127/0 is an infinity that the
#: saturating convertor would turn into 127 -- a row of zeros coming back as a
#: row of 127s. Clamping the absmax instead makes the scale tiny and the
#: quantized row zero, which is the answer.
_TINY = 1e-12


def _grid_q(p):
    return (1, hk.cdiv(p.x.rows, p.ROWS), 1)


def _grid_dq(p):
    return (1, hk.cdiv(p.q.rows, p.ROWS), 1)


def plan(cols: int, working_set: int | None = None):
    """(WARPS, TPW, FOLD) for a quantization over rows of `cols` elements.

    The same three choices norm.plan makes: split the row across the warps of
    one workgroup, hold it in registers when it fits so that the two passes
    read memory once, and -- when it does not fit either way -- reshape (R, C)
    to (R*16, C/16) so that a 16-row tile covers one row of the tensor.

    FOLD took an extra op to become available here, and the reason is worth
    recording because it is the only structural difference between this and a
    normalization. Quantize has a per-row *output*: the scale. Folded, one
    workgroup owns one row, so the workgroup has exactly one scale to write --
    but `store` writes a whole register vector, sixteen consecutive elements,
    which under FOLD would be sixteen copies of this row's scale laid over the
    next fifteen rows' slots. The answer is `hk.store_scalar`, which writes one
    entry and refuses any vector not proven uniform; `fold_rows` is what proves
    it. See lang.ops.store_scalar.

    **FOLD is a last resort here, where in norm.plan it is tried first**, and
    that inversion is measured, not reasoned. norm folds a 4096-wide row even
    though it fits unfolded, because the folded plan holds a quarter of the
    tiles per warp and the registers it gives back buy occupancy. Quantize does
    not want that trade at any width that has a choice. Measured
    (tools/hk-bench/quant_plan.py, burst timing, the best plan of each kind at
    each shape):

                        unfolded          folded            compile
        16384x4096   w16 tpw4  0.299   w4  tpw0  0.317       0.299
         4096x4096   w16 tpw4  0.062   w4  tpw1  0.063       0.065
         8192x4096   w16 tpw4  0.113   w4  tpw0  0.135       0.111
         4096x16384       -- does not fit --  w16 tpw1 0.294  0.305
        65536x1024   w16 tpw1  0.292   w1  tpw0  0.303       0.303

    Unfolded wins wherever it exists; folded wins only where it does not. So
    the rule is the simple one: hold the row unfolded if it fits, fold if it
    does not, stream if neither.

    Why, at least in part. Folding 4096 columns leaves 256 per folded row,
    which is four 64-wide blocks -- so a folded workgroup at this width cannot
    use more than four warps, and `candidates` offers none. The unfolded plan
    spreads the same row over sixteen. That is a structural ceiling on the
    folded column of the table, not a bandwidth effect, and it is why the gap
    is widest exactly where the row is narrow enough to fold hard.

    An earlier version of this docstring explained the same ordering by saying
    folded plans degrade with *more* warps while unfolded ones improve. That
    was true of the numbers it was written against and is not true of these:
    at 16384x4096 the folded plans now read w1 0.406, w2 0.409, w4 0.317, which
    improves with warps like everything else. What changed in between was the
    widening global load (see `load`'s `widening` branch), which moved every
    variant and moved the narrow-workgroup ones most. The ordering survived the
    change; the explanation did not.

    **There is no cache branch here, and that is a measurement, not an
    oversight.** norm.plan used to have one: a normalization whose working set
    fits in the 96 MB Infinity Cache was held to be uninterested in the
    register-held variant, because what FOLD and TPW buy is not touching HBM
    twice and there is no HBM touch to avoid. The argument is sound; the
    conclusion was not. Read the same table down the cache/HBM axis instead of
    the fold axis: `w16 tpw4` wins at 48 MB (0.062 against 0.094 for tpw0), at
    96 MB (0.113 against 0.161) and at 192 MB (0.299 against 0.370) -- same
    winner, same margin, every regime. Avoiding the second read is not what
    the held variant is buying; it is buying issue slots and a shorter
    dependence chain, and those are worth the same whoever answers the read.
    norm.plan has since dropped its branch for the same reason.

    `working_set` stays in the signature, accepted and ignored. No caller
    passes it any more -- `quantize` used to compute one, three torch calls per
    launch for a value this function immediately discarded, and on a 60 us
    kernel that is not free. It stays because the answer to "does the regime
    matter" is a fact about this op worth recording where someone would look
    for it, and because tests/hk/ir/test_phase2.py pins that passing one
    changes nothing.
    """
    del working_set  # deliberately: see above
    warps, tpw = _split(_ceil(cols, COLS))
    if tpw <= MAX_TPW:
        return warps, tpw, 1
    # Too wide to hold a 16-row slab of. One row of it still fits, if the
    # reshape is legal: the row has to divide into ROWS chunks and each chunk
    # has to be at least one tile wide.
    if cols % ROWS == 0 and cols // ROWS >= COLS:
        fw, ftpw = _split(_ceil(cols // ROWS, COLS))
        if ftpw <= MAX_TPW:
            return fw, ftpw, ROWS
    # Neither. Re-read the row instead; `visit` is the only thing that knows.
    return warps, 0, 1


def _quantize_kernel(name: str, dtype, warps: int):
    def body(x: hk.Tensor[dtype], q: hk.Tensor[i8], s: hk.Tensor[fp32], *,
             ROWS=ROWS, COLS=COLS, WARPS=warps, TPW=0, FOLD=1, EXACT=0):
        # The tile is fp32 even though the tensor is bf16, and holding it in
        # bf16 instead is not an optimization on this part -- it is a
        # pessimization, measured. See the note above `visit`.
        t = hk.RT(fp32, ROWS, COLS)
        vt = hk.col_vec(t)
        ncols = x.cols
        nblk = hk.cdiv(ncols, COLS)
        row_lo, _ = _block_start(hk.arch.block_idx.y, ROWS, x.rows)

        def cb_of(step):
            return step * WARPS + hk.arch.warp_id() if WARPS > 1 else step

        def tile(cb):
            col_lo, pad = _block_start(cb, COLS, ncols, EXACT)
            idx = hk.elem_coord(0, 0, row_lo, col_lo)
            return x[idx].load(t), pad, idx

        held = [tile(cb_of(i)) for i in range(TPW)] if TPW else None
        steps = None if TPW else hk.cdiv(nblk, WARPS)

        def visit(fn):
            """Once per column block this warp owns; see norm.visit."""
            if TPW:
                for rec in held:
                    fn(*rec)
            else:
                for step in range(steps):
                    fn(*tile(cb_of(step)))

        amax = hk.zeros(vt)
        # Unmasked. Backing the address up re-reads columns of the same row
        # that a neighbouring block already covered, and a max of absolute
        # values cannot be raised by an element it has already seen -- only a
        # sum would need the overlap removed, and there is no sum here. `abs`
        # writes to a tile of its own, so `v` is left as it was loaded, which
        # matters because in the persistent variant it is the tile the write
        # pass stores. See norm's visit().
        #
        # Holding the tile in bf16 instead of fp32 looks like the obvious way
        # to halve what the persistent variant keeps alive, and on gfx1100 it
        # is not: a 16-bit register tile is stored in the WMMA operand
        # fragment layout, where lanes l and l+16 hold *identical* data, so a
        # bf16 tile costs exactly as many VGPRs as an fp32 one. The toolchain
        # already says so -- Target.tile_vgprs(16, ...) == tile_vgprs(32, ...)
        # -- and the compiler agrees: at TPW=2, 145 VGPRs held in bf16 against
        # 109 in fp32, because the bf16 version pays for the widening
        # temporary on top. Pinned in tests/hk/ir/test_target.py.
        visit(lambda v, pad, idx: amax.max_(v.abs().max(axis=1)))
        amax = cross_warp(amax, "max", WARPS)
        if FOLD != 1:
            # The tile's 16 rows are 16 chunks of one row, so the row's absmax
            # is the max of the 16 partial answers this vector holds. After
            # this every entry holds it -- which is both what the rescale below
            # wants (all 16 chunk-rows share the scale) and what lets the
            # single-element store be written at all.
            amax = fold_rows(amax, "max", WARPS)

        amax.maximum(_TINY, into=amax)
        scale = amax * (1.0 / 127.0)
        inv = hk.full(vt, 127.0) / amax
        if FOLD != 1:
            # One workgroup, one row of the real tensor, one scale. block_idx.y
            # *is* the row index: the folded view has rows*FOLD rows, the grid
            # is one workgroup per FOLD of them, and FOLD == ROWS so
            # _block_start never has to clamp.
            s[hk.elem_coord(0, 0, 0, hk.arch.block_idx.y)].store_scalar(scale)
        else:
            # Every warp writes the same 16 scales to the same 16 slots. That
            # is the benign duplicate write the library already relies on for a
            # backed-up block, and cheaper than branching the whole workgroup
            # on warp 0.
            s[hk.elem_coord(0, 0, 0, row_lo)] = scale

        def write(v, pad, idx):
            v *= inv
            # The rounding and the clamp are in convertor<int8, float>, which
            # the store runs per element. Nothing here has to know about them.
            q[idx] = v

        visit(write)

    body.__name__ = name
    return hk.kernel(body, warps=warps, grid=_grid_q, name=name)


def _dequantize_kernel(name: str, dtype, warps: int):
    def body(q: hk.Tensor[i8], s: hk.Tensor[fp32], o: hk.Tensor[dtype], *,
             ROWS=ROWS, COLS=COLS, WARPS=warps, EXACT=0):
        t = hk.RT(fp32, ROWS, COLS)
        vt = hk.col_vec(t)
        ncols = q.cols
        nblk = hk.cdiv(ncols, COLS)
        row_lo, _ = _block_start(hk.arch.block_idx.y, ROWS, q.rows)

        # No reduction, so no masking and no cross-warp combine: a backed-up
        # block recomputes a few elements and writes them the same. The warps
        # are here only for the wave count, and each loads its own copy of the
        # 16 scales -- 64 bytes, against the tiles it is about to stream.
        scale = s[hk.elem_coord(0, 0, 0, row_lo)].load(vt)
        for step in range(hk.cdiv(nblk, WARPS)):
            cb = step * WARPS + hk.arch.warp_id() if WARPS > 1 else step
            col_lo, _ = _block_start(cb, COLS, ncols, EXACT)
            idx = hk.elem_coord(0, 0, row_lo, col_lo)
            v = q[idx].load(t)
            v *= scale
            o[idx] = v.to(dtype)

    body.__name__ = name
    return hk.kernel(body, warps=warps, grid=_grid_dq, name=name)


_DTYPES = {"bf16": bf16, "fp16": fp16, "fp32": fp32}

#: (op, dtype suffix, warps) -> Kernel. TPW rides in as a constexpr argument,
#: so one entry covers every column count.
KERNELS = {}
for _s, _dt in _DTYPES.items():
    for _w in WARP_COUNTS:
        KERNELS[f"quantize_{_s}_w{_w}"] = _quantize_kernel(f"quantize_{_s}_w{_w}", _dt, _w)
        KERNELS[f"dequantize_{_s}_w{_w}"] = _dequantize_kernel(
            f"dequantize_{_s}_w{_w}", _dt, _w)
del _s, _dt, _w


# ---------------------------------------------------------------- wrappers

_TORCH_DTYPES = {
    "torch.bfloat16": "bf16",
    "torch.float16": "fp16",
    "torch.float32": "fp32",
}


#: (torch dtype, cols) -> (kernel, TPW, FOLD, EXACT), and its dequantize twin.
#: Same reasoning and same keying as norm._PLAN_CACHE, and more pressing here:
#: quantize is the op where the Python in front of the launch was large enough
#: to lose a benchmark the kernel wins.
_QPLAN_CACHE: dict = {}
_DQPLAN_CACHE: dict = {}


def _suffix(dtype, what: str) -> str:
    suffix = _TORCH_DTYPES.get(str(dtype))
    if suffix is None:
        raise TypeError(f"hk {what} takes bf16/fp16/fp32, not {dtype}")
    return suffix


def _resolve_q(dtype, cols: int):
    key = (dtype, cols)
    got = _QPLAN_CACHE.get(key)
    if got is None:
        warps, tpw, fold = plan(cols)
        got = _QPLAN_CACHE[key] = (
            KERNELS[f"quantize_{_suffix(dtype, 'quantize')}_w{warps}"],
            tpw, fold, int(_exact(cols // fold, warps, tpw)),
        )
    return got


def _resolve_dq(dtype, cols: int):
    key = (dtype, cols)
    got = _DQPLAN_CACHE.get(key)
    if got is None:
        warps, _ = _split(_ceil(cols, COLS))
        got = _DQPLAN_CACHE[key] = (
            KERNELS[f"dequantize_{_suffix(dtype, 'dequantize produces')}_w{warps}"],
            int(_exact(cols, warps, 0)),
        )
    return got


def _check_2d(rows: int, cols: int, what: str = "hk quantize"):
    if cols < COLS or rows < ROWS:
        raise ValueError(
            f"{what} works on at least {ROWS}x{COLS}, got {rows}x{cols}. "
            f"The shape does not have to divide -- the tail block is backed up "
            f"and the overlap masked -- but a full tile has to fit."
        )


def quantize(x, out=None, scales=None):
    """Symmetric per-row int8 quantization over the last axis.

    Returns `(q, scales)`: `q` has x's shape and dtype int8, `scales` is x's
    shape without the last axis, in fp32. `x ~ q * scales[..., None]`.
    """
    if not x.is_contiguous():
        raise ValueError("hk quantize: x must be contiguous")
    cols = x.shape[-1]
    rows = x.numel() // cols if cols else 0
    _check_2d(rows, cols)

    if out is None or scales is None:
        import torch  # noqa: PLC0415  -- see norm._run
        if out is None:
            out = torch.empty(x.shape, dtype=torch.int8, device=x.device)
        if scales is None:
            scales = torch.empty(x.shape[:-1], dtype=torch.float32,
                                 device=x.device)
    k, tpw, fold, exact = _resolve_q(x.dtype, cols)
    # The scales tensor is *not* folded: it has one entry per real row either
    # way, and under FOLD the kernel indexes it by workgroup rather than by the
    # folded row block.
    k(_view(x, rows, cols, fold), _view(out, rows, cols, fold),
      scales.view(1, rows), TPW=tpw, FOLD=fold, EXACT=exact)
    return out, scales


def dequantize(q, scales, dtype=None, out=None):
    """q * scales[..., None], back to a float dtype."""
    import torch  # noqa: PLC0415  -- unconditional: the dtype checks need it

    if q.dtype != torch.int8:
        raise TypeError(f"hk dequantize takes an int8 tensor, not {q.dtype}")
    if not q.is_contiguous():
        raise ValueError("hk dequantize: q must be contiguous")
    if scales.dtype != torch.float32:
        raise TypeError(f"hk dequantize: scales must be float32, not {scales.dtype}")
    cols = q.shape[-1]
    rows = q.numel() // cols if cols else 0
    _check_2d(rows, cols, "hk dequantize")
    if scales.numel() != rows:
        raise ValueError(
            f"hk dequantize: {scales.numel()} scales for {rows} rows"
        )

    if out is None:
        out = torch.empty(q.shape, dtype=dtype or torch.bfloat16, device=q.device)
    # Not plan(): dequantize has no reduction to fold and no persistent
    # variant -- it reads each element once already -- so the only choice is
    # how many warps share the row, which is _split's first answer.
    k, exact = _resolve_dq(out.dtype, cols)
    k(q.view(rows, cols), scales.view(1, rows), out.view(rows, cols),
      EXACT=exact)
    return out


__all__ = ["KERNELS", "plan", "quantize", "dequantize"]
