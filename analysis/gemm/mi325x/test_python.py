import torch
import random
import time
import sys
import subprocess
import os
import tk_kernel
from aiter.tuned_gemm import tgemm

profiling = True
torch.manual_seed(0)
random.seed(0)

# Inputs
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
A = torch.randn(N, N, dtype=torch.bfloat16, device='cuda') / 10.0  
B = torch.randn(N, N, dtype=torch.bfloat16, device='cuda') / 10.0  
Bt = B.t().contiguous()  # Transpose B for the kernel

if profiling:
    num_warmup = 100
    num_iters = 500
else:
    num_warmup = 1
    num_iters = 0

start_event = torch.cuda.Event(enable_timing=True) # in milliseconds
end_event = torch.cuda.Event(enable_timing=True)
flops_ref = (2 * N**3)  # FLOPs for reference

if profiling:
    # Reference matmul using PyTorch
    for _ in range(num_warmup):
        C_pytorch = torch.matmul(A, Bt)
    timings_pytorch = []
    for _ in range(num_iters):
        torch.cuda.synchronize()
        start_event.record()
        C_pytorch = torch.matmul(A, Bt)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        timings_pytorch.append(elapsed_time)
    if profiling:
        print(f"{C_pytorch.dtype=}")
        avg_time_pytorch = sum(timings_pytorch) / len(timings_pytorch)
        tflops_pytorch = flops_ref / (avg_time_pytorch * 1e9) 
        print(f"PyTorch reference average execution time: {avg_time_pytorch:.4f} ms")
        print(f"PyTorch reference performance: {tflops_pytorch:.2f} TFLOPS for {N}x{N} matrix multiplication.\n")

    # Reference matmul using AITER (AMD)
    for _ in range(num_warmup):
        C_aiter = tgemm.mm(A, B, None, None, None)
    timings_aiter = []
    for _ in range(num_iters):
        torch.cuda.synchronize()
        start_event.record()
        C_aiter = tgemm.mm(A, B, None, None, None)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        timings_aiter.append(elapsed_time)
    if profiling:
        print(f"{C_aiter.dtype=}")
        avg_time_aiter = sum(timings_aiter) / len(timings_aiter)
        tflops_aiter = flops_ref / (avg_time_aiter * 1e9) 
        print(f"AITER (AMD) reference average execution time: {avg_time_aiter:.4f} ms")
        print(f"AITER (AMD) reference performance: {tflops_aiter:.2f} TFLOPS for {N}x{N} matrix multiplication.\n")

# Kernel matmul
C = torch.zeros(N, N, dtype=torch.bfloat16, device='cuda')
for _ in range(num_warmup):
    tk_kernel.dispatch_micro(A, B, C)
timings = []
for _ in range(num_iters):
    torch.cuda.synchronize()
    start_event.record()
    tk_kernel.dispatch_micro(A, B, C)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    timings.append(elapsed_time)
if profiling:
    print(f"{C.dtype=}")
    avg_time = sum(timings) / len(timings)
    tflops = flops_ref / (avg_time * 1e9) 
    print(f"Average execution time: {avg_time:.4f} ms")
    print(f"Performance: {tflops:.2f} TFLOPS for {N}x{N} matrix multiplication.\n")


# Compare against reference
if profiling:
    C_float = C.float()
    C_ref_float = C_pytorch.float()
    diff = (C_float - C_ref_float).abs()
    max_error = diff.max().item()
    mean_error = diff.mean().item()
    error_count = (diff > 0.1).sum().item()

    print(f"Max error between kernel and reference: {max_error}")
    print(f"Max error: {max_error}")
    print(f"Mean error: {mean_error}")
    print(f"Number of large errors (>0.1): {error_count}\n")

    # pos_max_diff = diff.max()
    # pos_max_diff_index = torch.where(diff == pos_max_diff)

    print("diff[:32, :32].max()", diff[:32, :32].max())
    print("diff[:32, 32:64].max()", diff[:32, 32:64].max())
    print("diff[32:64, :32].max()", diff[32:64, :32].max())
    print("diff[32:64, 32:64].max()", diff[32:64, 32:64].max())
    print()

    # print("diff[:32, 64:96].max()", diff[:32, 64:96].max())
    # print("diff[:32, 96:128].mean()", diff[:32, 96:128].mean())
    # print("diff[32:64, 64:96].max()", diff[32:64, 64:96].max())
    # print("diff[32:64, 96:128].max()", diff[32:64, 96:128].max())
    # print()

    # print("diff[64:96, :32].max()", diff[64:96, :32].max())
    # print("diff[64:96, 32:64].max()", diff[64:96, 32:64].max())
    # print("diff[64:96, 64:96].max()", diff[64:96, 64:96].max())
    # print()


    # print("diff[64:128, 64:128].max()", diff[64:128, 64:128].max())
    # print("diff[128:192, 128:192].max()", diff[128:192, 128:192].max())
    # print("diff[192:256, 192:256].max()", diff[192:256, 192:256].max())

    # # end tiles
    # print("diff[7168:7232, 7168:7232].max()", diff[7168:7232, 7168:7232].max())
    # print("diff[7232:7296, 7232:7296].max()", diff[7232:7296, 7232:7296].max())
    # print("diff[7296:7360, 7296:7360].max()", diff[7296:7360, 7296:7360].max())
    # print("diff[7360:7424, 7360:7424].max()", diff[7360:7424, 7360:7424].max())




    ############### LOGGING OUTPUTS ####################

    data_to_log = {
        "avg_time_pytorch": avg_time_pytorch,
        "tflops_pytorch": tflops_pytorch,
        "avg_time_aiter": avg_time_aiter,
        "tflops_aiter": tflops_aiter,
        "avg_time": avg_time,
        "tflops": tflops,
        # "max_error": max_error,
        # "mean_error": mean_error,
        # "error_count": error_count,
    }

    import json

    if not os.path.exists("data_to_log.json"):
        with open("data_to_log.json", "w") as f:
            json.dump({}, f, indent=4)

    with open("data_to_log.json", "r") as f:
        data = json.load(f)
        data[str(N)] = data_to_log

    with open("data_to_log.json", "w") as f:
        json.dump(data, f, indent=4)

    print(f"Results saved to data_to_log.json")

    ################ END LOGGING OUTPUTS ###############

    