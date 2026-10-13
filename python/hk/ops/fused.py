"""Fused pointwise kernels: SwiGLU's activation half, and RoPE.

Neither has a reduction, which makes the out-of-bounds story the easy one: an
elementwise result does not depend on what else is in the tile, so backing the
last block up recomputes a few elements to the same answer and no masking is
needed anywhere. It also means nothing here needs `left_fill`, which is why
`silu_mul` takes any width at all and the norm kernels' `pad` thunk has no
counterpart in this file.

**The knob that mattered was not the one this file was built around.** The
obvious worry for a pointwise kernel is that one 16x64 tile per workgroup is not
enough work per wave -- at 8192x11008 that is 88064 workgroups of 32 threads
each doing two loads and a store, and the loads of one wave cannot overlap the
loads of the next because there is no next. So `silu_mul` took the same two
knobs the norm kernels do: WARPS warps per workgroup, each covering TILES column
blocks. Swept, all fifteen settings landed within 5% and the smallest won; with
tens of thousands of workgroups resident there was never a shortage of waves.
`plan` returns (1, 1) and says so.

What the same numbers did show is that the identical kernel ran at 716 GB/s
over an 8192x4096 tensor and 675 GB/s over an 8192x11008 one, doing the same
number of tiles in the same number of workgroups. The difference is the stride
a tile's 16 rows step by. And an elementwise kernel does not owe the caller its
row structure -- see `canonical`, which is where the speed actually came from.

RoPE is here rather than in a file of its own because the interesting part is
the same trick: the rotation pairs column j with column j + D/2, and there is
no cross-lane shuffle in the warp register API that would let one tile see
both. So the kernel loads *two* tiles from two column offsets -- which costs
nothing, since they are separate global reads either way -- and writes two. The
consequence is a real constraint, stated in the wrapper: D/2 must be a multiple
of the tile width.
"""

from functools import lru_cache as _lru_cache

from .. import dsl as hk
from ..ir.nodes import bf16, fp16, fp32

ROWS, COLS = 16, 64


# ---------------------------------------------------------------- silu_mul


#: Live fp32 tiles a warp may hold. Two per column block -- one from each
#: operand -- so TILES=4 is 8 tiles, 256 VGPRs of data, and spills. The build
#: gate would catch that, but naming the reason here is cheaper than a
#: compile.
MAX_TILES = 3


def _start(idx, block: int, extent):
    """Where block `idx` starts, backed up so the whole tile is in bounds.

    norm's version of this returns a mask thunk as well; this one does not need
    one. An elementwise result does not depend on what else is in the tile, so
    the backed-up overlap is recomputed to the same answer and written twice
    with the same bytes.
    """
    return hk.min(idx * block, extent - block)


def _silu_mul_grid(p):
    return (hk.cdiv(p.o.cols, p.COLS * p.WARPS * p.TILES),
            hk.cdiv(p.o.rows, p.ROWS), 1)


def _silu_mul_kernel(name: str, dtype, warps: int):
    def body(a: hk.Tensor[dtype], b: hk.Tensor[dtype], o: hk.Tensor[dtype], *,
             ROWS=ROWS, COLS=COLS, WARPS=warps, TILES=1):
        # fp32 throughout: base_ops::silu is specialised for float only, and
        # the load converts on the way in for free, so there is nothing to pay
        # for the extra precision but register width.
        t = hk.RT(fp32, ROWS, COLS)
        ncols = o.cols
        row_lo = _start(hk.arch.block_idx.y, ROWS, o.rows)
        base = hk.arch.block_idx.x * (WARPS * TILES)

        def block_of(i):
            """The column block warp `warp_id` covers at step `i`.

            Strided like norm's: at any step the warps of a workgroup are on
            WARPS adjacent blocks, one contiguous run, rather than WARPS places
            a group apart.
            """
            cb = base + i * WARPS if i else base
            return cb + hk.arch.warp_id() if WARPS > 1 else cb

        idx = [hk.elem_coord(0, 0, row_lo, _start(block_of(i), COLS, ncols))
               for i in range(TILES)]
        # Both loads of every tile before any of the arithmetic. Written this
        # way rather than as a load/compute/store per tile so that the loads
        # are adjacent in the generated C++ and the scheduler has TILES*2
        # outstanding, not one.
        xs = [a[i].load(t) for i in idx]
        bs = [b[i].load(t) for i in idx]
        for x, bv, i in zip(xs, bs, idx):
            x.silu(into=x)
            x *= bv
            o[i] = x.to(dtype)

    body.__name__ = name
    return hk.kernel(body, warps=warps, grid=_silu_mul_grid, name=name)


# ---------------------------------------------------------------- rope


def _rope_grid(p):
    # x is (bh, 1, seq, D); one workgroup per (half-head column block, row
    # block, bh). The column axis only covers D/2 because each workgroup
    # handles a pair of columns j and j + D/2.
    return (p.HALF // p.COLS, hk.cdiv(p.x.rows, p.ROWS), p.x.batch)


def _rope_kernel(name: str, dtype, half: int):
    """Rotary embedding, NeoX/half-split convention.

        out[j]        = x[j] * cos[j] - x[j + D/2] * sin[j]
        out[j + D/2]  = x[j + D/2] * cos[j] + x[j] * sin[j]

    `cos` and `sin` are (seq, D/2) and are shared across heads, so they index
    with batch 0 while x indexes with the head. That is the whole reason the
    kernel takes them as separate globals instead of folding them in: they are
    broadcast along an axis x is not.
    """

    def body(x: hk.Tensor[dtype], cos: hk.Tensor[fp32], sin: hk.Tensor[fp32],
             o: hk.Tensor[dtype], *, ROWS=ROWS, COLS=COLS, HALF=half):
        t = hk.RT(fp32, ROWS, COLS)
        bh = hk.arch.block_idx.z
        r = hk.arch.block_idx.y
        c = hk.arch.block_idx.x
        # The second half is HALF/COLS tiles further along the row.
        c2 = c + HALF // COLS

        lo = hk.tile_coord(bh, 0, r, c)
        hi = hk.tile_coord(bh, 0, r, c2)
        ang = hk.tile_coord(0, 0, r, c)

        xl = x[lo].load(t)
        xh = x[hi].load(t)
        cs = cos[ang].load(t)
        sn = sin[ang].load(t)

        o[lo] = (xl * cs - xh * sn).to(dtype)
        o[hi] = (xh * cs + xl * sn).to(dtype)

    body.__name__ = name
    return hk.kernel(body, warps=1, grid=_rope_grid, name=name)


_DTYPES = {"bf16": bf16, "fp16": fp16, "fp32": fp32}

#: Warp counts a silu_mul workgroup can have. Same set as the norm kernels;
#: WARPS is a launch geometry and so has to be baked into the kernel, while
#: TILES rides in as a constexpr argument.
WARP_COUNTS = tuple(1 << i for i in range(5))  # 1, 2, 4, 8, 16

KERNELS = {}
for _s, _dt in _DTYPES.items():
    for _w in WARP_COUNTS:
        KERNELS[f"silu_mul_{_s}_w{_w}"] = _silu_mul_kernel(
            f"silu_mul_{_s}_w{_w}", _dt, _w)
    KERNELS[f"rope_{_s}"] = _rope_kernel(f"rope_{_s}", _dt, COLS)
del _s, _dt, _w


# ---------------------------------------------------------------- wrappers

_TORCH_DTYPES = {
    "torch.bfloat16": "bf16",
    "torch.float16": "fp16",
    "torch.float32": "fp32",
}


def _suffix_of(t) -> str:
    name = _TORCH_DTYPES.get(str(t.dtype))
    if name is None:
        raise TypeError(f"hk fused kernels handle bf16/fp16/fp32, not {t.dtype}")
    return name


def _as_2d(t, what: str):
    if not t.is_contiguous():
        raise ValueError(f"{what} must be contiguous; call .contiguous()")
    cols = t.shape[-1]
    rows = t.numel() // cols if cols else 0
    if cols < COLS or rows < ROWS:
        raise ValueError(
            f"{what} is {rows}x{cols} and silu_mul needs at least {ROWS}x{COLS}. "
            f"The shape does not have to divide -- the last block backs its "
            f"address up and recomputes the overlap, which for a pointwise "
            f"kernel is exact -- but a whole tile has to fit."
        )
    r, c = canonical(rows, cols)
    # Already that shape -- `canonical` is frequently the identity, and
    # `Tensor.view` is about a microsecond of Python per operand.
    return t if r == rows and c == cols and t.dim() == 2 else t.view(r, c)


#: The row width silu_mul would rather walk, and the narrower ones it settles
#: for. See `canonical`. Measured by tools/hk-bench/fused_plan.py.
WIDTHS = (4096, 2048, 1024, 512, 256, 128, 64)


@_lru_cache(maxsize=None)
def canonical(rows: int, cols: int):
    """A (rows, cols) with the same elements that the memory system likes more.

    Memoized because `silu_mul` asks it three times per call, once per operand,
    and a launch at these sizes has only a couple of hundred microseconds of
    GPU behind it to hide Python in. The arguments are two ints and the answer
    is a pure function of them.

    silu_mul is elementwise, so the row structure is not part of the problem:
    a contiguous tensor may be viewed at any width dividing its element count
    and the kernel writes the same bytes to the same places. That makes the
    width a free parameter, and it is not a neutral one. The same kernel doing
    the same number of 16x64 tiles in the same number of workgroups ran at
    716 GB/s over an 8192x4096 tensor and 675 GB/s over an 8192x11008 one; the
    only difference between them is the stride a tile's 16 rows step by, 8192
    bytes against 22016.

    So a tensor whose own width is awkward is re-viewed at the widest power of
    two that divides its element count: 8192x11008 becomes 22016x4096. One that
    divides nothing in WIDTHS -- an odd hidden size -- keeps its own shape,
    which still works, just at the slower stride.

    Not applicable to the reductions in ops/norm.py: there the row *is* the
    problem and its width is not ours to choose.
    """
    n = rows * cols
    for w in WIDTHS:
        if n % w == 0 and n // w >= ROWS:
            return n // w, w
    return rows, cols


def plan(cols: int):
    """(WARPS, TILES) for a pointwise pass over rows of `cols` elements.

    Measured, not derived: see tools/hk-bench/fused_plan.py -- and what it
    measured refuted the hypothesis this knob was added for. Across
    2048x11008, 8192x11008 and 8192x4096 all fifteen (warps, tiles) settings
    landed within 5% of each other and the *smallest* of them, one warp holding
    one tile, won all three. A pointwise pass at these sizes already has 20k-90k
    workgroups in flight; there was never a shortage of waves to hide a load
    behind, and the only thing a wider workgroup changed was how many of its
    blocks were the clamped last one, which it then re-wrote.

    So: (1, 1). The knobs stay because the kernels are parameterized on them
    already and a narrower row or another arch could still want them, not
    because anything measured here does. `_canonical` below is what actually
    made this kernel faster.
    """
    del cols
    return 1, 1


#: (torch dtype, cols) -> (kernel, TILES). As in norm._PLAN_CACHE, and keyed
#: on the dtype object for the same reason: silu_mul at 2048x11008 is 216 us
#: of GPU, which a few microseconds of Python is a measurable fraction of --
#: it is one of the two rows where hk and torch.compile were within 1.5% of
#: each other.
_PLAN_CACHE: dict = {}


def _resolve(dtype, cols: int):
    got = _PLAN_CACHE.get((dtype, cols))
    if got is None:
        suffix = _TORCH_DTYPES.get(str(dtype))
        if suffix is None:
            raise TypeError(f"hk silu_mul handles bf16/fp16/fp32, not {dtype}")
        warps, tiles = plan(cols)
        got = _PLAN_CACHE[(dtype, cols)] = (
            KERNELS[f"silu_mul_{suffix}_w{warps}"], tiles)
    return got


def silu_mul(a, b, out=None):
    """silu(a) * b -- the second half of SwiGLU, with the gate already split.

    torch's equivalent is `F.silu(a) * b`; this does it in one pass, so the
    intermediate never reaches memory.
    """
    if a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError(
            f"silu_mul: operands differ -- {tuple(a.shape)}/{a.dtype} vs "
            f"{tuple(b.shape)}/{b.dtype}"
        )
    if out is None:
        import torch  # noqa: PLC0415  -- see norm._run
        out = torch.empty_like(a)
    av, bv, ov = _as_2d(a, "a"), _as_2d(b, "b"), _as_2d(out, "out")
    k, tiles = _resolve(a.dtype, av.shape[-1])
    k(av, bv, ov, TILES=tiles)
    return out


#: torch dtype -> the rope kernel for it. One f-string and one `str(dtype)`
#: per launch is not much, and a 32x2048x128 rope is 50 us.
_rope_cache: dict = {}


def rope(x, cos, sin, out=None):
    """Rotary embedding over the last axis, half-split (NeoX) convention.

    `x` is (..., seq, D) with at least one leading axis; `cos` and `sin` are
    (seq, D/2) in fp32 and are shared across every leading axis. D/2 must be a
    multiple of 64 -- head dims of 128 and 256 qualify, 64 does not, and the
    error says so rather than silently producing a rotation by the wrong angle.
    """
    import torch  # noqa: PLC0415

    if x.dim() < 3:
        raise ValueError(
            f"rope expects (batch*heads, seq, D), got {tuple(x.shape)}. Reshape "
            f"the head axis in; the kernel indexes it as the global's batch."
        )
    if not x.is_contiguous():
        raise ValueError("rope: x must be contiguous")
    seq, d = x.shape[-2], x.shape[-1]
    half = d // 2
    if d % 2 or half % COLS:
        raise ValueError(
            f"rope: head dim {d} gives a half of {half}, which is not a multiple "
            f"of {COLS}. The kernel pairs column j with column j + D/2 by loading "
            f"two tiles, so the halves have to land on tile boundaries."
        )
    if seq % ROWS:
        raise ValueError(f"rope: sequence length {seq} must be a multiple of {ROWS}")
    for nm, tbl in (("cos", cos), ("sin", sin)):
        if tuple(tbl.shape) != (seq, half):
            raise ValueError(
                f"rope: {nm} is {tuple(tbl.shape)}, expected ({seq}, {half})"
            )
        if tbl.dtype != torch.float32:
            raise TypeError(f"rope: {nm} must be float32, got {tbl.dtype}")

    if out is None:
        out = torch.empty_like(x)
    bh = x.numel() // (seq * d)
    k = _rope_cache.get(x.dtype)
    if k is None:
        k = _rope_cache[x.dtype] = KERNELS[f"rope_{_suffix_of(x)}"]
    k(x.view(bh, 1, seq, d), cos.view(1, 1, seq, half), sin.view(1, 1, seq, half),
      out.view(bh, 1, seq, d), HALF=half)
    return out


__all__ = ["KERNELS", "ROWS", "COLS", "MAX_TILES", "WARP_COUNTS", "WIDTHS",
           "canonical", "plan", "silu_mul", "rope"]
