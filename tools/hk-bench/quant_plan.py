#!/usr/bin/env python3
"""Which (WARPS, TPW) an int8 quantization deserves, measured instead of argued.

`quant.plan` inherited its rule from `norm.plan` without ever being swept, and
the two passes are not the same shape of problem. Quantization reads bf16 and
writes int8, so its write is half its read and the ratio the persistent variant
saves is different. Whatever norm measured does not transfer, so measure this.

FOLD is in the sweep now. It was not at first -- a folded workgroup owns one
row and so has one scale to write, which `store` cannot express -- and the
answer was `x[idx].store_scalar(v)`. See quant.plan.

The cases below sit on both sides of the 96 MB Infinity Cache, because that
used to be a question `plan` branched on -- a pass whose working set fits in
cache was held not to want the register-held variant, since the second read it
avoids would have been a hit. The sweep refuted that, here and in norm.plan;
the cases stay so the refutation keeps being checkable.

    python3 tools/hk-bench/quant_plan.py
    python3 tools/hk-bench/quant_plan.py 16384 4096
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import torch  # noqa: E402

import _timing  # noqa: E402

from hk.ops import norm, quant  # noqa: E402

DEV = "cuda"
DT = torch.bfloat16
HBM_GB_S = 864.0

#: (rows, cols). bf16 in, int8 out, fp32 scales: 3 bytes per element.
CASES = [
    (16384, 4096),   # 201 MB -- HBM
    (4096, 4096),    #  50 MB -- cache
    (8192, 4096),    # 101 MB -- astride
    (4096, 16384),   # 201 MB -- HBM, wide
    (65536, 1024),   # 201 MB -- HBM, narrow
    (16384, 5120),   # 251 MB -- HBM, and the width MAX_TPW=5 opened: 80
                     #           blocks is five tiles across sixteen warps.
]


def _ceil(a: int, b: int) -> int:
    return -(-a // b)


def candidates(cols: int):
    """Every (warps, tpw, fold) legal for this width."""
    out = []
    folds = [1]
    if cols % norm.ROWS == 0 and cols // norm.ROWS >= norm.COLS:
        folds.append(norm.ROWS)
    for fold in folds:
        nblk = _ceil(cols // fold, norm.COLS)
        for warps in norm.WARP_COUNTS:
            if warps > min(norm.MAX_WARPS, nblk):
                continue
            out.append((warps, 0, fold))
            tpw = _ceil(nblk, warps)
            if tpw <= norm.MAX_TPW:
                out.append((warps, tpw, fold))
    return sorted(set(out))


_interleave = _timing.interleave
def run(rows: int, cols: int):
    x = torch.randn(rows, cols, device=DEV, dtype=DT)
    q = torch.empty(rows, cols, device=DEV, dtype=torch.int8)
    s = torch.empty(1, rows, device=DEV, dtype=torch.float32)
    nbytes = x.numel() * 3

    variants, skipped = {}, []
    for warps, tpw, fold in candidates(cols):
        label = f"w{warps:<2d} tpw{tpw} f{fold}"
        k = quant.KERNELS[f"quantize_bf16_w{warps}"]
        c = {"TPW": tpw, "FOLD": fold,
             "EXACT": int(norm._exact(cols // fold, warps, tpw))}
        xv, qv = (norm._view(t, rows, cols, fold) for t in (x, q))
        try:
            k.build(**c)  # a spilling variant is eliminated without a GPU run
            variants[label] = lambda k=k, c=c, xv=xv, qv=qv: k(xv, qv, s, **c)
        except Exception as exc:  # noqa: BLE001
            skipped.append((label, str(exc).splitlines()[0][:70]))

    mb = nbytes / 2**20
    chosen = "w%-2d tpw%d f%d" % quant.plan(cols, nbytes)
    print(f"\nquantize {rows}x{cols} bf16->int8  {mb:.0f} MB "
          f"({'cache' if mb <= 96 else 'HBM'})   plan() picks: {chosen}")
    print(f"  {'variant':16s} {'ms':>8s} {'GB/s':>8s} {'%HBM':>6s}")
    best = _interleave(variants)
    for label, ms in sorted(best.items(), key=lambda kv: kv[1]):
        gbs = nbytes / (ms * 1e-3) / 1e9
        mark = "  <- plan()" if label == chosen else ""
        print(f"  {label:16s} {ms:8.3f} {gbs:8.1f} "
              f"{100 * gbs / HBM_GB_S:5.1f}%{mark}")
    for label, why in skipped:
        print(f"  {label:16s} {'skipped':>8s}  {why}")

    # The bar, in the same process for the same reason.
    def ref():
        amax = x.float().abs().amax(-1, keepdim=True).clamp_min(1e-12)
        return (x.float() / (amax / 127.0)).round().clamp(-128, 127).to(torch.int8)

    try:
        both = _interleave({"compile": torch.compile(ref, dynamic=False),
                            "eager": ref})
        for name, ms in both.items():
            print(f"  {name:16s} {ms:8.3f} {nbytes / (ms * 1e-3) / 1e9:8.1f}")
    except Exception as exc:  # noqa: BLE001
        print(f"  torch.compile unavailable: {exc}")
    return min(best, key=best.get)


def main(argv):
    cases = CASES
    if len(argv) == 3:
        cases = [(int(argv[1]), int(argv[2]))]
    elif len(argv) != 1:
        sys.exit(__doc__)
    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}")
    winners = {}
    for rows, cols in cases:
        winners[(rows, cols)] = run(rows, cols)
    print("\nwinner per case:")
    for (rows, cols), label in winners.items():
        print(f"  {rows:6d}x{cols:<6d} {label}")


if __name__ == "__main__":
    main(sys.argv)
