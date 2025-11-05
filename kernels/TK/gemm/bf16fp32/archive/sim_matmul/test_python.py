import torch
import tk_kernel

from triton_kernel import matmul_triton


# Inputs
N = 8192
A = torch.randn(N, N, dtype=torch.bfloat16, device='cuda') / 10.0  # Scale down to avoid overflow
B = torch.randn(N, N, dtype=torch.bfloat16, device='cuda') / 10.0  # Scale down to avoid overflow
alpha = 1.0
beta = 0.0

num_warmup = 2
num_iters = 10
start_event = torch.cuda.Event(enable_timing=True) # in milliseconds
end_event = torch.cuda.Event(enable_timing=True)
flops_ref = (2 * N**3)  # FLOPs for reference

# Reference matmul using PyTorch
C_ref = torch.matmul(A, B)
for _ in range(num_warmup):
    C_ref = torch.matmul(A, B)

timings_ref = []
for _ in range(num_iters):
    torch.cuda.synchronize()
    start_event.record()
    C_ref = torch.matmul(A, B)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    timings_ref.append(elapsed_time)
avg_time_ref = sum(timings_ref) / len(timings_ref)
tflops_ref = flops_ref / (avg_time_ref * 1e9)  # Convert to TFLOPS
print(f"PyTorch reference average execution time: {avg_time_ref:.4f} ms")
print(f"PyTorch reference performance: {tflops_ref:.2f} TFLOPS for {N}x{N} matrix multiplication.\n")


# Triton kernel matmul
for _ in range(num_warmup):
    C_triton = matmul_triton(A, B)

timings_triton = []
for _ in range(num_iters):
    torch.cuda.synchronize()
    start_event.record()
    C_triton = matmul_triton(A, B)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    timings_triton.append(elapsed_time)
avg_time_triton = sum(timings_triton) / len(timings_triton)
tflops_triton = flops_ref / (avg_time_triton * 1e9)  # Convert to TFLOPS
print(f"Triton kernel average execution time: {avg_time_triton:.4f} ms")
print(f"Triton kernel performance: {tflops_triton:.2f} TFLOPS for {N}x{N} matrix multiplication.\n")


# Kernel matmul
C = torch.zeros(N, N, dtype=torch.float32, device='cuda')
for _ in range(num_warmup):
    tk_kernel.dispatch_micro(A, B, C, alpha, beta)

timings = []
for _ in range(num_iters):
    torch.cuda.synchronize()
    start_event.record()
    tk_kernel.dispatch_micro(A, B, C, alpha, beta)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    timings.append(elapsed_time)
avg_time = sum(timings) / len(timings)
tflops = flops_ref / (avg_time * 1e9)  # Convert to TFLOPS
print(f"Average execution time: {avg_time:.4f} ms")
print(f"Performance: {tflops:.2f} TFLOPS for {N}x{N} matrix multiplication.\n")


# Compare against reference
C_float = C.float()
C_ref_float = C_ref.float()
C_triton_float = C_triton.float()
diff = (C_float - C_ref_float).abs()
max_error = diff.max().item()
mean_error = diff.mean().item()
error_count = (diff > 0.1).sum().item()

print(f"Max error between kernel and reference: {max_error}")
print(f"Max error: {max_error}")
print(f"Mean error: {mean_error}")
print(f"Number of large errors (>0.1): {error_count}\n")

diff = (C_triton_float - C_ref_float).abs()
max_error_triton = diff.max().item()
mean_error_triton = diff.mean().item()
error_count_triton = (diff > 0.1).sum().item()
print(f"Max error between Triton kernel and reference: {max_error_triton}")
print(f"Mean error: {mean_error_triton}")
print(f"Number of large errors (>0.1) in Triton kernel: {error_count_triton}")


# # print(C_float[0:64, 0:64])  # Print a small part of the result for verification
# # print(C_ref_float[0:64, 0:64])  # Print a small part of the reference for verification

# # find max region of difference
# max_diff = diff.max()
# max_diff_indices = (diff == max_diff).nonzero(as_tuple=True)
# print(f"Max difference at indices: {max_diff_indices}")

# # print region of C_float and C_ref_float around the max difference
# row, col = max_diff_indices
# print(f"Region around max difference at ({row.item()}, {col.item()}):")
# print("C_float region:")
# print(C_float[row.item()-2:row.item()+3, col.item()-2:col.item()+3])
# print("C_ref_float region:")
# print(C_ref_float[row.item()-2:row.item()+3, col.item()-2:col.item()+3])


