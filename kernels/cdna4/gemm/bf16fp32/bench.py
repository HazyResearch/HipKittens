import torch
import tk_kernel
from aiter.tuned_gemm import tgemm
import random
from utils import bench_gemm, print_title

BENCHMARK_AITER = True
BENCHMARK_TORCH = True

bench_shapes = [
    (8192, 8192, 8192), # (m, n, k)
    (4096, 8192, 2048),
    (8192, 4096, 2048),
    (8192, 2048, 4096),
    (4096, 4096, 4096),
    (2048, 4096, 4096),
    (2048, 2048, 4096),
    (2048, 2048, 2048),
]

torch.manual_seed(0)
random.seed(0)
dtype = torch.bfloat16
device = "cuda:0"

if __name__ == "__main__":
    gemm_params = {}
    gemm_params["device"] = device
    gemm_params["dtype"] = dtype

    print_title("HipKittens bf16 GEMM")
    for shape in bench_shapes:
        gemm_params["shape"] = shape
        bench_gemm(gemm_params, tk_kernel.dispatch_micro, True)

    if BENCHMARK_TORCH:
        print_title("PyTorch bf16 GEMM")
        for shape in bench_shapes:
            gemm_params["shape"] = shape
            torch_gemm = lambda A, B, C: torch.matmul(A, B, out=C)
            bench_gemm(gemm_params, torch_gemm, False)

    if BENCHMARK_AITER:
        print_title("AITER bf16 GEMM")
        for shape in bench_shapes:
            gemm_params["shape"] = shape
            aiter_gemm = lambda A, B, C: tgemm.mm(A, B)
            bench_gemm(gemm_params, aiter_gemm, True)