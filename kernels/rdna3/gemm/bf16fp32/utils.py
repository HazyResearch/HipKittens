import torch

def init_randint(shape, low, high, dtype, device):
    return torch.randint(low, high, shape, dtype=dtype, device=device)

def init_randn(shape, dtype, device, scale=1):
    return scale * torch.randn(shape, dtype=dtype, device=device)

def init_empty(shape, dtype, device):
    return torch.empty(shape, dtype=dtype, device=device)

def init_zero(shape, dtype, device):
    return torch.zeros(shape, dtype=dtype, device=device)

def print_title(title, len=30):
    print("-"*len)
    print(title)
    print("-"*len)

def time_gemm(gemm_params, gemm_func, transpose_B=False, num_warmup=20, num_iter=50):
    """Average ms per call.

    Allocation stays outside the timing loop and many calls sit between one pair
    of events. The earlier version of this did the opposite -- three randn's per
    iteration and one kernel per event pair -- which charges allocation and
    launch latency to the kernel and understates everything by a few percent.
    """
    m, n, k = gemm_params["shape"]
    dtype, device = gemm_params["dtype"], gemm_params["device"]

    A = init_randn((m, k), dtype, device)
    B = init_randn((n, k) if transpose_B else (k, n), dtype, device)
    C = init_empty((m, n), dtype, device)

    for _ in range(num_warmup):
        gemm_func(A, B, C)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(num_iter):
        gemm_func(A, B, C)
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / num_iter


def bench_gemm(gemm_params, gemm_func, transpose_B=False, num_warmup=20, num_iter=50):
    m, n, k = gemm_params["shape"]
    ms = time_gemm(gemm_params, gemm_func, transpose_B, num_warmup, num_iter)
    tflops = 2*m*n*k / (ms * 1e9)
    print(f"m={m},n={n},k={k}: {tflops:.1f} TFLOPS")
    return tflops