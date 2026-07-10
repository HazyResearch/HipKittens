import torch
import tk_kernel

B = 64
N = 8192
D = 128

torch.random.manual_seed(42)
x = torch.randn((B, N, D), dtype=torch.bfloat16, device='cuda')


def flops(batch, seqlen, dim):
    # max: N comparisons, sub: N, exp: N, sum: N, div: N => ~5N per row
    return batch * seqlen * dim * 5


def efficiency(flop, time):
    flop = flop / 1e12
    time = time / 1e3
    return flop / time


# PyTorch reference
o_ref = torch.softmax(x.float(), dim=-1).bfloat16()

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
flops_val = flops(B, N, D)
num_warmup = 50
num_iters = 50


# Benchmark PyTorch
print("\nPyTorch:")
timings = []
for _ in range(num_warmup):
    _ = torch.softmax(x, dim=-1)
for _ in range(num_iters):
    torch.cuda.synchronize()
    start_event.record()
    _ = torch.softmax(x, dim=-1)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    timings.append(elapsed_time)
avg_time_ref = sum(timings) / len(timings)
eff_ref = efficiency(flops_val, avg_time_ref)
print(f"PyTorch average execution time: {avg_time_ref:.4f} ms")
print(f"PyTorch performance: {eff_ref:.2f} TFLOPS for {B=} {N=} {D=}.")


# Benchmark HK
print("\nHK:")
o_tk = torch.zeros_like(x)
timings = []
for _ in range(num_warmup):
    tk_kernel.dispatch_softmax(x, o_tk)
for _ in range(num_iters):
    torch.cuda.synchronize()
    start_event.record()
    tk_kernel.dispatch_softmax(x, o_tk)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    timings.append(elapsed_time)
avg_time_tk = sum(timings) / len(timings)
eff_tk = efficiency(flops_val, avg_time_tk)
print(f"HK average execution time: {avg_time_tk:.4f} ms")
print(f"HK performance: {eff_tk:.2f} TFLOPS for {B=} {N=} {D=}.")
speedup = avg_time_ref / avg_time_tk
print(f"Speedup from HK: {speedup:.2f}x")


# Correctness
print("\nCorrectness:")
o_diff = o_ref - o_tk
max_diff = o_diff.abs().max()
print(f"max_diff: {max_diff}")
if max_diff > 0.01:
    print(f"o_ref:  {o_ref[0, 0, :8]}")
    print(f"o_tk:   {o_tk[0, 0, :8]}")

# Sanity: rows should sum to ~1.0
row_sums = o_tk.float().sum(dim=-1)
print(f"Row sum min: {row_sums.min():.4f}, max: {row_sums.max():.4f} (should be ~1.0)")
