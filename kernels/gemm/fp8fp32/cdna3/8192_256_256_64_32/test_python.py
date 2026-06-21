import torch
import tk_kernel
import random

profiling = True

torch.manual_seed(0)
random.seed(0)

# Inputs
N = 8192
A = (torch.randn(N, N, dtype=torch.float32, device='cuda') / 10.0).to(torch.float8_e4m3fnuz)
B = (torch.randn(N, N, dtype=torch.float32, device='cuda') / 10.0).to(torch.float8_e4m3fnuz)
Bt = B.t().contiguous()  # Transpose B for the kernel


def efficiency(flop_count, time_ms):
    flop_t = flop_count / 1e12
    time_s = time_ms / 1e3
    return flop_t / time_s


if profiling:
    num_warmup = 20
    num_iters = 20
else:
    num_warmup = 1
    num_iters = 0

start_event = torch.cuda.Event(enable_timing=True) # in milliseconds
end_event = torch.cuda.Event(enable_timing=True)
flops_ref = (2 * N**3)  # FLOPs for reference

if profiling:
    # Correctness-only reference matmul using PyTorch
    C_ref = torch.matmul(A.float(), Bt.float())

# Kernel matmul
C = torch.zeros(N, N, dtype=torch.float32, device='cuda')
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
    tflops = efficiency(flops_ref, avg_time)
    print(f"Average execution time: {avg_time:.4f} ms")
    print(f"Performance: {tflops:.2f} TFLOPS for {N}x{N} matrix multiplication.\n")


if profiling:
    diff = (C - C_ref).abs()
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

    # end tiles
    print("diff[7168:7232, 7168:7232].max()", diff[7168:7232, 7168:7232].max())
    print("diff[7232:7296, 7232:7296].max()", diff[7232:7296, 7232:7296].max())
    print("diff[7296:7360, 7296:7360].max()", diff[7296:7360, 7296:7360].max())
    print("diff[7360:7424, 7360:7424].max()", diff[7360:7424, 7360:7424].max())
