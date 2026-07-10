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

def bench_gemm(gemm_params, gemm_func, transpose_B=False, num_warmup=500, num_iter=500):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    m, n, k = gemm_params["shape"]
    flop = 2*m*n*k
    dtype = gemm_params["dtype"]
    device = gemm_params["device"]

    A_shape = (m, k)
    B_shape = (n, k) if transpose_B else (k, n)
    C_shape = (m, n)

    A = init_randn(A_shape, dtype, device)
    B = init_randn(B_shape, dtype, device)
    C = init_empty(C_shape, dtype, device)

    for _ in range(num_warmup):
        gemm_func(A, B, C)

    elapsed_time = 0
    
    for _ in range(num_iter):
        A = init_randn(A_shape, dtype, device)
        B = init_randn(B_shape, dtype, device)
        C = init_empty(C_shape, dtype, device)
        torch.cuda.synchronize()
        start_event.record()
        gemm_func(A, B, C)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time += start_event.elapsed_time(end_event)

    avg_elapsed_time = elapsed_time / num_iter
    tflops = int(flop / (avg_elapsed_time * 1e9))
    print(f"m={m},n={n},k={k}: {tflops} TFLOPS")