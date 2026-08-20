"""Shared constants and tensor helpers for the gfx1250 GEMM ladder.

The naming follows `kernels/cdna4/gemm/bf16fp32/utils.py` (`init_*`, `print_title`) so the two
directories read the same way.
"""

import math

import torch

DTYPE = torch.bfloat16
DEVICE = "cuda:0"

# Every kernel on disk, worst to best, which is also the chain `gemm_ladder.py` walks.
RUNGS = ["00_gemm_naive", "01_gemm_double_buf", "02_gemm_async", "03_gemm_128x128",
         "04_gemm_256x256", "05_gemm_deepk", "06_gemm_segment", "07_gemm_tdm",
         "08_gemm_split_bar", "09_gemm_wgc_multicast", "10_gemm_epilogue",
         "11_gemm_one_wave"]

# Legal for all twelve rungs at once, so a failure is about the rung and not the shape. The
# binding constraint is the three cluster rungs: they launch a 4x4 cluster over a grid of
# (M/256, N/256) workgroups and refuse a grid that is not a multiple of 4 in both axes, which makes
# M and N multiples of 1024. K must be a multiple of the deepest BLOCK_K, 128. At least one shape
# has to be non-square, because a layout error in C is invisible at M == N -- a transposed output
# and a consistently transposed reference agree element for element.
SHAPES = [
    (8192, 8192, 8192),   # the shape the published table is taken at
    (4096, 8192, 2048),   # non-square, M < N
    (8192, 4096, 2048),   # non-square, M > N
    (2048, 1024, 1024),   # small, and M != N
]


def init_uniform(shape, dtype=DTYPE, device=DEVICE, lo=-1.0, hi=1.0):
    """U(lo, hi) operands, which is the range `compare`'s tolerance was derived for.

    |C| grows as sqrt(K) * sigma_ab, about 181 at K=8192 for U(-1,1) and comfortably inside bf16.
    Unit-normal operands are three times wider and would need the tolerance rescaled with them.
    """
    return torch.empty(shape, dtype=dtype, device=device).uniform_(lo, hi)


def init_c(m, n, dtype=DTYPE, device=DEVICE):
    """The kernels' output tensor: logical shape (M, N), column-major memory.

    Every rung writes C at `c*M + r`, so the strides are (1, M); `pyext.h` documents why and checks
    them. Filled with NaN rather than left uninitialised so that a rung which declines to launch --
    the cluster rungs refuse a grid that is not a multiple of their cluster dimension -- fails the
    comparison instead of passing on whatever the allocator happened to hand over.
    """
    return torch.empty_strided((m, n), (1, m), dtype=dtype, device=device).fill_(float("nan"))


def print_title(title, width=30):
    print("-" * width)
    print(title)
    print("-" * width)


def gemm_reference(a, b):
    """C = a . b^T in fp32, from the same bf16 operands the kernel reads.

    fp32 rather than bf16 so the reference is better than the thing it checks: a bf16 `torch.matmul`
    rounds its own output and would contribute error the size of the error being measured.
    """
    return torch.matmul(a.float(), b.float().t())


def compare(got, ref, k):
    """Error statistics for one rung against the reference, and the count that gates it.

    The tolerance is derived rather than fitted: input quantization accumulates as a random walk
    over the reduction, so the absolute term scales as sqrt(K), anchored at 0.5 for K=8192 with
    U(-1,1) operands; the relative term is bf16's round-to-nearest-even half-ULP, 2^-9, rounded to
    1e-2. Published results were gated on `bad`, the count of elements outside that, being zero.
    NaN never exceeds a tolerance, so `nonfinite` is the half of the gate that catches a rung which
    never wrote its output at all.
    """
    atol, rtol = 0.5 * math.sqrt(k / 8192.0), 1.0e-2
    got_f = got.float()
    err = (got_f - ref).abs()
    return {
        "bad": int((err > atol + rtol * ref.abs()).sum()),
        "n": ref.numel(),
        "max_abs_err": float(err.max()),
        "mean_abs_err": float(err.mean()),
        "max_ref_abs": float(ref.abs().max()),
        "nonfinite": int((~torch.isfinite(got_f)).sum()),
        "atol": atol,
    }
