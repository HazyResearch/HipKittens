#!/usr/bin/env python3
"""Why is hk.quantize 5x slower than hk.rmsnorm on the same tensor?

The two kernels have the same shape: one warp per 16 rows, two passes over the
row, a col_vec accumulator between them. rmsnorm reaches 650 GB/s and quantize
reaches 94. Something in the int8 half is paying for it, and there are only
three candidates -- the int8 *store*, the narrowing convertor's rint/clamp, and
the reduction itself (row_max of abs vs row_sum of x*x).

So build the three one at a time, in this process, interleaved:

  pass1     the amax reduction, storing only the per-row scale
  store_f   the full kernel with the int8 store replaced by a bf16 store
  full      hk.quantize itself
  dequant   the inverse -- an int8 *load* and a bf16 store

pass1 isolates the reduction, store_f isolates the store's element width from
everything else that surrounds it, and dequant says whether an int8 global is
slow in both directions or only on the way out.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import torch  # noqa: E402

import hk  # noqa: E402
from hk import bf16, fp32, i8  # noqa: E402
from hk.ops import quant  # noqa: E402
from hk.ops.norm import COLS, ROWS, _block_start  # noqa: E402

from phase2 import HBM_GB_S, _interleave  # noqa: E402

DEV = "cuda"
_TINY = 1e-12


def _grid(p):
    return (1, hk.cdiv(p.x.rows, p.ROWS), 1)


def _amax_body(x, s, *, ROWS=ROWS, COLS=COLS):
    """Pass 1 of quantize, and nothing else."""
    t = hk.RT(fp32, ROWS, COLS)
    vt = hk.col_vec(t)
    ncols = x.cols
    nblk = hk.cdiv(ncols, COLS)
    row_lo, _ = _block_start(hk.arch.block_idx.y, ROWS, x.rows)

    amax = hk.zeros(vt)
    for cb in hk.range(nblk):
        col_lo, pad = _block_start(cb, COLS, ncols)
        v = x[hk.elem_coord(0, 0, row_lo, col_lo)].load(t)
        amax.max_(v.left_fill_(pad(), 0.0).abs().max(axis=1))

    amax.maximum(_TINY, into=amax)
    s[hk.elem_coord(0, 0, 0, row_lo)] = amax * (1.0 / 127.0)


def _store_f_body(x, q, s, *, ROWS=ROWS, COLS=COLS):
    """hk.quantize with the int8 store swapped for a bf16 one. Same arithmetic,
    same two passes, same convertor absent."""
    t = hk.RT(fp32, ROWS, COLS)
    vt = hk.col_vec(t)
    ncols = x.cols
    nblk = hk.cdiv(ncols, COLS)
    row_lo, _ = _block_start(hk.arch.block_idx.y, ROWS, x.rows)

    def tile(cb, mask):
        col_lo, pad = _block_start(cb, COLS, ncols)
        idx = hk.elem_coord(0, 0, row_lo, col_lo)
        v = x[idx].load(t)
        return (v.left_fill_(pad(), 0.0) if mask else v), idx

    amax = hk.zeros(vt)
    for cb in hk.range(nblk):
        v, _ = tile(cb, True)
        amax.max_(v.abs().max(axis=1))

    amax.maximum(_TINY, into=amax)
    inv = hk.full(vt, 127.0) / amax
    s[hk.elem_coord(0, 0, 0, row_lo)] = amax * (1.0 / 127.0)

    for cb in hk.range(nblk):
        v, idx = tile(cb, False)
        v *= inv
        q[idx] = v.to(bf16)


def _mk(body, name, anns):
    body.__name__ = name
    body.__annotations__ = anns
    return hk.kernel(body, arch="gfx1100", warps=1, grid=_grid, name=name)


AMAX = _mk(_amax_body, "probe_amax",
           {"x": hk.Tensor[bf16], "s": hk.Tensor[fp32]})
STORE_F = _mk(_store_f_body, "probe_store_f",
              {"x": hk.Tensor[bf16], "q": hk.Tensor[bf16], "s": hk.Tensor[fp32]})


def main(rows=16384, cols=4096):
    torch.manual_seed(0)
    x = torch.randn(rows, cols, device=DEV, dtype=torch.bfloat16)
    qi = torch.empty(rows, cols, device=DEV, dtype=torch.int8)
    qf = torch.empty(rows, cols, device=DEV, dtype=torch.bfloat16)
    s = torch.empty(1, rows, device=DEV, dtype=torch.float32)
    sv = s.view(rows)
    out = torch.empty(rows, cols, device=DEV, dtype=torch.bfloat16)

    read = x.numel() * 2
    variants = {
        # bytes moved, thunk
        "pass1 (amax only)": (read + rows * 4, lambda: AMAX(x, s)),
        "store bf16": (2 * read + rows * 4, lambda: STORE_F(x, qf, s)),
        "full (store int8)": (read + x.numel() + rows * 4,
                              lambda: quant.KERNELS["quantize_bf16_w1"](x, qi, s)),
        "dequant (load int8)": (x.numel() + read + rows * 4,
                                lambda: quant.KERNELS["dequantize_bf16_w1"](qi, s, out)),
    }
    # Prime: dequant needs a populated qi, and every kernel needs its JIT done
    # before the timing loop or the first iteration measures hipcc.
    quant.KERNELS["quantize_bf16_w1"](x, qi, s)

    print(f"{rows}x{cols} bf16, W7900D, {HBM_GB_S:.0f} GB/s ceiling")
    print(f"{'variant':<22}{'ms':>9}{'GB/s':>9}{'%HBM':>8}")
    print("-" * 48)
    times = _interleave({k: v[1] for k, v in variants.items()})
    for name, (nbytes, _) in variants.items():
        ms = times[name]
        gbs = nbytes / (ms * 1e-3) / 1e9
        print(f"{name:<22}{ms:>9.3f}{gbs:>9.1f}{gbs / HBM_GB_S * 100:>7.1f}%")

    # And a correctness spot check, so a fast probe cannot be a broken one.
    ref = x.float().abs().amax(-1).clamp_min(_TINY) / 127.0
    AMAX(x, s)
    torch.cuda.synchronize()
    print("\npass1 scale matches torch:", torch.allclose(sv, ref, rtol=1e-5))


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
