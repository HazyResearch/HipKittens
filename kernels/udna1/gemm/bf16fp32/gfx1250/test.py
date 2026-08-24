#!/usr/bin/env python3
"""Check the gfx1250 GEMM ladder against a torch reference.

This is the ladder's correctness gate: every rung against `torch.matmul` at four shapes, through the
pybind11 module each one builds. There is no CPU reference in the tree.

    make all-kernels && python3 test.py               # every rung, every shape
    python3 test.py --rung 10_gemm_epilogue              # one rung
    python3 test.py --rung 10_gemm_epilogue -s "8192 8192 8192" --terse

`--terse` prints one `bad=<count>` line per cell and exits non-zero if any rung fails.

`gemm_ladder.py` runs the same check over its own arms before it times them, in the process that
does the timing. This is the full sweep, and the tolerance both use lives in `utils.compare`.
"""

import argparse
import importlib
import sys

import torch

from utils import RUNGS, SHAPES, compare, gemm_reference, init_c, init_operand, print_title


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rung", action="append",
                   help="rung to check; repeatable. Default is every rung in utils.RUNGS.")
    p.add_argument("-s", "--shape", action="append",
                   help='"M N K"; repeatable. Default is the four shapes in utils.SHAPES.')
    p.add_argument("--terse", action="store_true",
                   help="one bad= line per rung, for a machine to read")
    a = p.parse_args()

    rungs = a.rung or RUNGS
    shapes = [tuple(int(x) for x in s.split()) for s in a.shape] if a.shape else SHAPES
    for s in shapes:
        if len(s) != 3:
            sys.exit(f"--shape wants three numbers, got {s!r}")

    torch.manual_seed(0)
    if not torch.cuda.is_available():
        sys.exit("no GPU visible to torch")

    mods = {}
    for r in rungs:
        try:
            mods[r] = importlib.import_module(r)
        except ImportError as e:
            sys.exit(f"cannot import {r}: {e}\nBuild it first: make KERNEL={r}")

    if not a.terse:
        print(f"device: {torch.cuda.get_device_properties(0).gcnArchName}  "
              f"torch {torch.__version__}")
        print_title("gfx1250 bf16 GEMM ladder vs torch", 78)
        print(f"{'rung':<24} {'shape':<20} {'bad':>10} {'max_err':>9} {'mean_err':>9} "
              f"{'gate':>7}  ok")

    failures = 0
    for (m, n, k) in shapes:
        operand_a = init_operand((m, k))
        operand_b = init_operand((n, k))
        ref = gemm_reference(operand_a, operand_b)
        for r in rungs:
            c = init_c(m, n)
            mods[r].dispatch(operand_a, operand_b, c)
            st = compare(c, ref, k)
            ok = st["bad"] == 0 and st["nonfinite"] == 0
            failures += 0 if ok else 1
            if a.terse:
                print(f"{r} {m}x{n}x{k} bad={st['bad']}/{st['n']} "
                      f"max_abs_err={st['max_abs_err']:.4f} "
                      f"{'OK' if ok else 'FAIL'}", flush=True)
            else:
                print(f"{r:<24} {f'{m}x{n}x{k}':<20} {st['bad']:>4}/{st['n']:<10} "
                      f"{st['max_abs_err']:>8.4f} {st['mean_abs_err']:>9.4f} "
                      f"{st['atol']:>7.3f}  {'PASS' if ok else 'FAIL'}", flush=True)
                if not ok:
                    print(f"{'':<24} nonfinite={st['nonfinite']} "
                          f"max|ref|={st['max_ref_abs']:.1f}")
        del operand_a, operand_b, ref
        torch.cuda.empty_cache()

    if not a.terse:
        total = len(rungs) * len(shapes)
        print(f"\n{total - failures}/{total} cells passed "
              f"(gate: |got-ref| <= 0.5*sqrt(K/8192) + 0.01*|ref|, zero elements over)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
