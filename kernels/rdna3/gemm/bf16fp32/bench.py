"""bf16 GEMM benchmark, HipKittens against both AMD BLAS libraries.

Three things this harness is careful about, each because getting it wrong
produced a published number that was wrong:

1.  **Both libraries.** torch on ROCm defaults to hipBLASLt, and on gfx1100
    hipBLASLt is the *slower* of the two -- rocBLAS beats it on every shape
    here, by up to 33%. Benchmarking against torch's default alone flatters
    the kernel. `torch.backends.cuda.preferred_blas_library` switches between
    them at runtime ("cublaslt" is hipBLASLt, "cublas" is rocBLAS).

2.  **Every layout, and the matching one.** Tensile tunes per transpose
    combination and the four are not equally covered on this part -- rocBLAS
    spans 56..88 TFLOPs across NN/NT/TN/TT at 8192^2x4096. HipKittens
    implements exactly one layout: A is (m,k), B is (n,k), C = A*B^T, i.e. both
    operands K-contiguous, which is torch's "NT". So there are two honest
    numbers and this table prints both:

      HK/best  against the vendor's best layout. This is the number that
               matters if you get to choose how your data is laid out, and it
               is the harder comparison -- the vendor is solving an easier
               memory problem with the same math.
      HK/same  against the vendor in HipKittens' own layout. This is the number
               that matters if your data is already K-contiguous, which is what
               an attention or MoE pipeline hands you.

    Reporting only HK/best understates the kernel; reporting only HK/same
    flatters it. The gap between the two columns is a statement about Tensile's
    NT tuning on gfx1100, not about either kernel's math.

3.  **The denominator.** There is no `PEAK_TFLOPS` constant here any more. The
    122.6 TFLOPs figure is 96 CU x 512 FLOP/clk x 2.495 GHz boost, and this
    card does not run at 2.495 GHz under a GEMM: it sits at 2.0-2.1 GHz against
    its 241 W limit. Percentages are taken against WMMA_CEILING below, which is
    measured rather than derived -- see tools/rdna-probes/wmma_peak.hip.
"""
import random

import torch

import tk_kernel
from utils import time_gemm, print_title

# Back-to-back v_wmma_f32_16x16x16_bf16 with no memory access at all, measured
# by tools/rdna-probes/wmma_peak.hip on this card. Flat from 2 to 16 independent
# accumulator chains, so it is an issue-rate ceiling and not a latency artifact.
WMMA_CEILING = 100.5

bench_shapes = [
    (4096, 4096, 4096), # (m, n, k)
    (8192, 8192, 4096),
    (4096, 8192, 2048),
    (8192, 4096, 2048),
    (2048, 4096, 4096),
    (2048, 2048, 4096),
    (2048, 2048, 2048),
]

torch.manual_seed(0)
random.seed(0)
dtype = torch.bfloat16
device = "cuda:0"


def vendor_layouts(shape, backend):
    """TFLOPs per transpose combination, for one BLAS backend.

    "NT" is the entry to compare against directly: it is the layout HipKittens
    implements, both operands K-contiguous.
    """
    m, n, k = shape
    torch.backends.cuda.preferred_blas_library(backend)
    A  = torch.randn((m, k), dtype=dtype, device=device)
    At = torch.randn((k, m), dtype=dtype, device=device)
    B  = torch.randn((k, n), dtype=dtype, device=device)
    Bt = torch.randn((n, k), dtype=dtype, device=device)
    C  = torch.empty((m, n), dtype=dtype, device=device)
    layouts = {
        "NN": lambda: torch.matmul(A,      B,      out=C),
        "NT": lambda: torch.matmul(A,      Bt.t(), out=C),
        "TN": lambda: torch.matmul(At.t(), B,      out=C),
        "TT": lambda: torch.matmul(At.t(), Bt.t(), out=C),
    }
    out = {}
    for lay, fn in layouts.items():
        for _ in range(15):
            fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(50):
            fn()
        e.record()
        torch.cuda.synchronize()
        out[lay] = 2*m*n*k / ((s.elapsed_time(e) / 50) * 1e9)
    del A, At, B, Bt, C
    torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    print_title(f"bf16 GEMM on gfx1100 -- WMMA ceiling {WMMA_CEILING} TFLOPs", 76)
    print("vendor columns: best-of-four-layouts, and NT = the layout HK implements")
    print(f"{'shape (MxNxK)':>18} | {'hipBLASLt':>13} | {'rocBLAS':>13} | "
          f"{'vendor NT':>9} | {'HipKittens':>10} | {'HK/best':>7} | "
          f"{'HK/same':>7} | {'HK/ceil':>7}")
    print("-" * 100)

    for shape in bench_shapes:
        m, n, k = shape
        hipblaslt = vendor_layouts(shape, "cublaslt")
        rocblas   = vendor_layouts(shape, "cublas")

        params = {"device": device, "dtype": dtype, "shape": shape}
        hk = 2*m*n*k / (time_gemm(params, tk_kernel.dispatch_micro, True) * 1e9)

        def top(d):
            lay = max(d, key=d.get)
            return d[lay], lay
        lt_tf, lt_lay = top(hipblaslt)
        rb_tf, rb_lay = top(rocblas)
        best = max(lt_tf, rb_tf)
        same = max(hipblaslt["NT"], rocblas["NT"])

        print(f"{m}x{n}x{k:<6} | {lt_tf:8.1f} ({lt_lay}) | {rb_tf:8.1f} ({rb_lay}) | "
              f"{same:9.1f} | {hk:10.1f} | {100*hk/best:6.0f}% | "
              f"{100*hk/same:6.0f}% | {100*hk/WMMA_CEILING:6.0f}%")
