import torch
import tk_kernel
import random
import time
import argparse
# from aiter.tuned_gemm import tgemm

torch.manual_seed(0)
random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--profile', type=bool, default=False)
args = parser.parse_args()

# Inputs
M = 192*40    
K = 8192
N = 8192  
# A = torch.randn(N, N, dtype=torch.bfloat16, device='cuda') / 10.0  
# B = torch.randn(N, N, dtype=torch.bfloat16, device='cuda') / 10.0  
# Bt = B.t().contiguous()  # Transpose B for the kernel

num_blocks_x = (M / 192)
num_blocks_y = (N / 256)
print(f"{num_blocks_x=} mod 32 {num_blocks_x % 32}, {num_blocks_y=} mod 32 {num_blocks_y % 32}\n")

if args.profile:
    num_warmup = 500
    num_iters = 100
else:
    num_warmup = 0
    num_iters = 1

start_event = torch.cuda.Event(enable_timing=True) # in milliseconds
end_event = torch.cuda.Event(enable_timing=True)
flops_ref = (2 * M * N * K)  # FLOPs for reference

# Reference matmul using PyTorch
for _ in range(num_warmup):
    A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda') / 10.0  
    B = torch.randn(N, K, dtype=torch.bfloat16, device='cuda') / 10.0  
    Bt = B.t().contiguous()  # Transpose B for the kernel
    C_ref = torch.matmul(A, Bt)
timings_ref = []
torch.manual_seed(0)
random.seed(0)
for _ in range(num_iters):
    C = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
    A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda') / 10.0  
    B = torch.randn(N, K, dtype=torch.bfloat16, device='cuda') / 10.0  
    Bt = B.t().contiguous()  # Transpose B for the kernel
    torch.cuda.synchronize()
    start_event.record()
    C_ref = torch.matmul(A, Bt)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    timings_ref.append(elapsed_time)
print(f"{C_ref.dtype=}")
avg_time_ref = sum(timings_ref) / len(timings_ref)
tflops_ref = flops_ref / (avg_time_ref * 1e9) 
print(f"PyTorch reference average execution time: {avg_time_ref:.4f} ms")
print(f"PyTorch reference performance: {tflops_ref:.2f} TFLOPS for {M}x{K} by {K}x{N} matrix multiplication.\n")


# for _ in range(num_warmup):
#     A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda') / 10.0  
#     B = torch.randn(N, K, dtype=torch.bfloat16, device='cuda') / 10.0  
#     C_aiter = tgemm.mm(A, B, None, None, None)
# timings_aiter = []
# for _ in range(num_iters):
#     A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda') / 10.0  
#     B = torch.randn(N, K, dtype=torch.bfloat16, device='cuda') / 10.0  
#     torch.cuda.synchronize()
#     start_event.record()
#     C_aiter = tgemm.mm(A, B, None, None, None)
#     end_event.record()
#     torch.cuda.synchronize()
#     elapsed_time = start_event.elapsed_time(end_event)
#     timings_aiter.append(elapsed_time)
# print(f"{C_aiter.dtype=}")
# avg_time_aiter = sum(timings_aiter) / len(timings_aiter)
# tflops_aiter = flops_ref / (avg_time_aiter * 1e9) 
# print(f"AITER (AMD) reference average execution time: {avg_time_aiter:.4f} ms")
# print(f"AITER (AMD) reference performance: {tflops_aiter:.2f} TFLOPS for {M}x{K} by {K}x{N} matrix multiplication.\n")


# Kernel matmul
print("Running kernel matmul...")
C = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
for _ in range(num_warmup):
    A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda') / 10.0  
    B = torch.randn(N, K, dtype=torch.bfloat16, device='cuda') / 10.0  
    Bt = B.t().contiguous()  # Transpose B for the kernel
    tk_kernel.dispatch_micro(A, B, C)
timings = []
torch.manual_seed(0)
random.seed(0)
for _ in range(num_iters):
    C = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')
    A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda') / 10.0  
    B = torch.randn(N, K, dtype=torch.bfloat16, device='cuda') / 10.0  
    Bt = B.t().contiguous()  # Transpose B for the kernel
    torch.cuda.synchronize()
    start_event.record()
    tk_kernel.dispatch_micro(A, B, C)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    timings.append(elapsed_time)
print(f"{C.dtype=}")
avg_time = sum(timings) / len(timings)
tflops = flops_ref / (avg_time * 1e9) 
print(f"Average execution time: {avg_time:.4f} ms")
print(f"Performance: {tflops:.2f} TFLOPS for {M}x{K} by {K}x{N} matrix multiplication.\n")

# Compare against reference
C_float = C.float()
C_ref_float = C_ref.float()
diff = (C_float - C_ref_float).abs()
max_error = diff.max().item()
mean_error = diff.mean().item()
error_count = (diff > 0.1).sum().item()

# breakpoint()

print(f"Reference: {C_ref_float[:3, :4]}")
print(f"Kernel: {C_float[:3, :4]}")

print(f"Max error between kernel and reference: {max_error}")
print(f"Max error: {max_error}")
print(f"Mean error: {mean_error}")
print(f"Number of large errors (>0.1): {error_count} out of {M*N} ({error_count / (M*N) * 100:.2f}%)\n")

print(f"diff[0,:128] = {diff[0,:128].max().item()}")

# breakpoint()
