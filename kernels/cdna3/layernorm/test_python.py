import torch
import torch.nn as nn
from tqdm import trange
import numpy as np
import tk_kernel

B = 16
H = 16
N = 4096
HEAD_D = 128
D = HEAD_D * H
DROPOUT_P = 0.01

norm = nn.LayerNorm(D).cuda()
torch.random.manual_seed(42)
x = torch.randn((B, N, D), dtype=torch.bfloat16, device='cuda').requires_grad_()
residual = torch.randn((B, N, D), dtype=torch.bfloat16, device='cuda').requires_grad_()


def flops(batch, seqlen, hidden_dim):
    """Calculate FLOPs for LayerNorm operation."""
    B, N, D = batch, seqlen, hidden_dim
    mean_flops = B * N * D    
    var_flops = B * N * D * 3  # subtract, square, sum    
    norm_flops = B * N * D * 2  # subtract, divide    
    scale_shift_flops = B * N * D * 2  # multiply, add
    total_flops = mean_flops + var_flops + norm_flops + scale_shift_flops
    return total_flops


def efficiency(flop, time):
    """Calculate efficiency in TFLOPS."""
    flop = flop / 1e12  # convert to TFLOPS
    time = time / 1e3   # convert to seconds
    return flop / time


def get_output(x, residual, norm, dropout_p=DROPOUT_P):
    # 1. dropout on x
    mask = torch.bernoulli(torch.full_like(x, 1 - dropout_p))
    dropped = x * mask / (1 - dropout_p) 

    # 3. residual = dropped + residual
    residual = ( dropped + residual ) if residual is not None else dropped 

    # 4. norm
    sum_residual = residual.sum(dim=-1, keepdim=True)
    mean = residual.mean(dim=-1, keepdim=True)
    var  = residual.var(dim=-1, keepdim=True, unbiased=False)
    sqrt_var = torch.sqrt(var + norm.eps)
    residual_norm = (residual - mean) / sqrt_var

    norm_weight = norm.weight
    norm_bias = norm.bias

    o = norm_weight * residual_norm + norm_bias 
    
    return o, residual, norm_weight, norm_bias, mean, sqrt_var


def pytorch_ref(x, residual, norm):
    dropped = torch.nn.functional.dropout(x, p=DROPOUT_P)
    residual = (residual + dropped) if residual is not None else dropped
    x = norm(residual.to(dtype=norm.weight.dtype))
    return x, residual

start_event = torch.cuda.Event(enable_timing=True) # in milliseconds
end_event = torch.cuda.Event(enable_timing=True)
flops_ref = flops(B, N, D)
num_warmup = 50
num_iters = 50


# Benchmark and test correctness
# PyTorch
timings = []
print("\nPyTorch:")
for _ in range(num_warmup):
    o_ref, new_residual_ref = pytorch_ref(x, residual, norm)
for _ in range(num_iters):
    torch.cuda.synchronize()
    start_event.record()
    o_ref, new_residual_ref = pytorch_ref(x, residual, norm)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    timings.append(elapsed_time)
avg_time_ref = sum(timings) / len(timings)
eff = efficiency(flops_ref, avg_time_ref)
print(f"PyTorch average execution time: {avg_time_ref:.4f} ms")
print(f"PyTorch performance: {eff:.2f} TFLOPS for {B=} {N=} {D=}.")


#  PyTorch (Compiled)
compiled_pytorch_ref = torch.compile(pytorch_ref)
print("\nPyTorch (Compiled):")
timings_compiled = []
for _ in range(num_warmup):
    o_compiled, new_residual_compiled = compiled_pytorch_ref(x, residual, norm)
for _ in range(num_iters):
    torch.cuda.synchronize()
    start_event.record()
    o_compiled, new_residual_compiled = compiled_pytorch_ref(x, residual, norm)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    timings_compiled.append(elapsed_time)
avg_time_compiled = sum(timings_compiled) / len(timings_compiled)
eff_compiled = efficiency(flops_ref, avg_time_compiled)
print(f"PyTorch compiled average execution time: {avg_time_compiled:.4f} ms")
print(f"PyTorch compiled performance: {eff_compiled:.2f} TFLOPS for {B=} {N=} {D=}.")
speedup = avg_time_ref / avg_time_compiled
print(f"Speedup from torch.compile: {speedup:.2f}x")


# TK
print("\nTK (PyTorch):")
o_tk = torch.zeros_like(o_ref).bfloat16()
o_resid_tk = torch.zeros_like(new_residual_ref).bfloat16()
norm_weight_tk = norm.weight.detach().clone().to(dtype=torch.bfloat16, device='cuda')
norm_bias_tk = norm.bias.detach().clone().to(dtype=torch.bfloat16, device='cuda')
timings = []
for _ in range(num_warmup):
    tk_kernel.dispatch_micro(x, residual, o_tk, o_resid_tk, norm_weight_tk, norm_bias_tk)
for _ in range(num_iters):
    torch.cuda.synchronize()
    start_event.record()
    tk_kernel.dispatch_micro(x, residual, o_tk, o_resid_tk, norm_weight_tk, norm_bias_tk)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    timings.append(elapsed_time)

avg_time = sum(timings) / len(timings)
eff = efficiency(flops_ref, avg_time)
print(f"TK average execution time: {avg_time:.4f} ms")
print(f"TK performance: {eff:.2f} TFLOPS for {B=} {N=} {D=}.")
speedup = avg_time_ref / avg_time
print(f"Speedup from TK: {speedup:.2f}x")

# Correctness
print("\nCorrectness:")
o_diff = o_ref - o_tk
max_diff = o_diff.abs().max()
print(f"max_diff: {max_diff}")
if max_diff > 0.1:
    print(f"o: ", o_ref[0, 0, :8])
    print(f"o_tk: ", o_tk[0, 0, :8])

o_resid_diff = new_residual_ref - o_resid_tk
max_diff = o_resid_diff.abs().max()
print(f"max_diff: {max_diff}")
if max_diff > 0.1:
    print(f"new_residual: ", new_residual_ref[0, 0, :8])
    print(f"o_resid_tk: ", o_resid_tk[0, 0, :8])


print(f"If there's a difference, note dropout p is {DROPOUT_P}")




