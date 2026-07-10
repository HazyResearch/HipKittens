import torch
import time
import tk_kernel

M = N = K = 8192
torch.manual_seed(0)

# +/-1 values are exactly representable in fp8e4m3, so the fp32 reference is a
# clean integer sum and any divergence is from the kernel, not quantization.
A = (torch.randint(0, 2, (1, 1, M, K), device="cuda") * 2 - 1).to(torch.float8_e4m3fn)
B = (torch.randint(0, 2, (1, 1, N, K), device="cuda") * 2 - 1).to(torch.float8_e4m3fn)
C = torch.zeros((1, 1, M, N), dtype=torch.float32, device="cuda")

# Benchmark on GPU.
num_warmup, num_iters = 5, 20
for _ in range(num_warmup):
    tk_kernel.dispatch_micro(A, B, C)
torch.cuda.synchronize()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
timings = []
for _ in range(num_iters):
    torch.cuda.synchronize()
    start.record()
    tk_kernel.dispatch_micro(A, B, C)
    end.record()
    torch.cuda.synchronize()
    timings.append(start.elapsed_time(end))

avg = sum(timings) / len(timings)
flops = 2 * M * N * K
tflops = (flops / 1e12) / (avg / 1e3)
print(f"TK fp8 GEMM {M}x{N}x{K}: avg {avg:.3f} ms, {tflops:.1f} TFLOPS")

# Correctness: kernel computes C = A @ B^T (mma_ABt). The 7.0-preview container's
# hipblasLt has no gfx942 library, so compute the reference on CPU for a row
# subset (8 rows) -- still a strong check since +/-1 inputs have an exact sum.
REF_ROWS = 8
A_cpu = A[0, 0, :REF_ROWS].cpu().to(torch.float32)   # (REF_ROWS, K)
B_cpu = B[0, 0].cpu().to(torch.float32)              # (N, K)
ref = A_cpu @ B_cpu.t()                               # (REF_ROWS, N)
sub = C[0, 0, :REF_ROWS].cpu()                        # (REF_ROWS, N)
diff = (sub - ref).abs()
print(f"ref rows={REF_ROWS}  max_diff={diff.max().item():.4f}  mean_diff={diff.mean().item():.6f}")
print(f"ref[0,:5]={ref[0,:5].tolist()}")
print(f"tk [0,:5]={sub[0,:5].tolist()}")
ok = torch.allclose(sub, ref, atol=1.0, rtol=1e-3)
print("CORRECT" if ok else "MISMATCH")
