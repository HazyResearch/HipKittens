import torch
import torch.nn as nn
from tqdm import trange
import numpy as np
import sys
import math
from einops import rearrange, repeat
import tk_kernel

B = 16
H = 16
N = 2048
D = 128

D_2 = D // 2

torch.random.manual_seed(42)
x = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda').requires_grad_()

def flops(batch, seqlen, hidden_dim):
    """Calculate FLOPs for LayerNorm operation."""
    B, N, D = batch, seqlen, hidden_dim
    f = B * H * N * D_2 # * cos
    f += B * H * N * D_2 # * sin
    f += B * N * D_2 # sum
    return f


def efficiency(flop, time):
    """Calculate efficiency in TFLOPS."""
    flop = flop / 1e12  # convert to TFLOPS
    time = time / 1e3   # convert to seconds
    return flop / time


def get_cos_sin(x, rotary_emb_base=10000, rotary_emb_dim=D, dtype=torch.bfloat16):
    t = torch.arange(N, device=x.device, dtype=dtype) # We want fp32 here
    inv_freq = 1.0 / (rotary_emb_base ** (torch.arange(0, rotary_emb_dim, 2, device=x.device, dtype=dtype) / rotary_emb_dim))
    freqs = torch.outer(t, inv_freq).to(dtype=dtype)
    cos_in = torch.cos(freqs).to(dtype=dtype)
    sin_in = torch.sin(freqs).to(dtype=dtype)
    ro_dim = cos_in.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    return cos_in, sin_in, ro_dim


def get_output(x, cos_in, sin_in, ro_dim):
    x_ro_dim     = x[..., :ro_dim]
    x_ro_dim_end = x[..., ro_dim:]
    x1, x2 = x_ro_dim.chunk(2, dim=-1)              # D/2, D/2
    rotated_x = torch.cat((-x2, x1), dim=-1)        # D
    cos = repeat(cos_in, "n d -> 1 n (2 d)" )
    sin = repeat(sin_in, "n d -> 1 n (2 d)" )
    o = torch.cat([x_ro_dim * cos + rotated_x * sin, x_ro_dim_end], dim=-1)
    return o


start_event = torch.cuda.Event(enable_timing=True) # in milliseconds
end_event = torch.cuda.Event(enable_timing=True)
flops_ref = flops(B, N, D)
num_warmup = 50
num_iters = 50

cos_in, sin_in, ro_dim = get_cos_sin(x)

# Benchmark and test correctness
# PyTorch
timings = []
print("\nPyTorch:")
for _ in range(num_warmup):
    o = get_output(x, cos_in, sin_in, ro_dim)
for _ in range(num_iters):
    torch.cuda.synchronize()
    start_event.record()
    o = get_output(x, cos_in, sin_in, ro_dim)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    timings.append(elapsed_time)
avg_time_ref = sum(timings) / len(timings)
eff = efficiency(flops_ref, avg_time_ref)
print(f"PyTorch average execution time: {avg_time_ref:.4f} ms")
print(f"PyTorch performance: {eff:.2f} TFLOPS for {B=} {N=} {D=}.")

#  PyTorch (Compiled)
compiled_pytorch_ref = torch.compile(get_output)
print("\nPyTorch (Compiled):")
timings_compiled = []
for _ in range(num_warmup):
    o = compiled_pytorch_ref(x, cos_in, sin_in, ro_dim)
for _ in range(num_iters):
    torch.cuda.synchronize()
    start_event.record()
    o = compiled_pytorch_ref(x, cos_in, sin_in, ro_dim)
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

# AITER
print("\nAITer (RoPE cached):")
from aiter.ops.rope import rope_cached_fwd as _aiter_cached_fwd

# Prepare inputs in AITer layout (SBHD) and cached cos/sin as [S,1,1,D//2]
x_sbhd = rearrange(x, 'b h n d -> n b h d').contiguous()
cos_aiter = cos_in.view(N, 1, 1, D // 2).to(dtype=x.dtype, device=x.device).contiguous()
sin_aiter = sin_in.view(N, 1, 1, D // 2).to(dtype=x.dtype, device=x.device).contiguous()

# AITer flags: 0 = NEOX-style rotation; reuse front part True for d//2 cos/sin; nope_first can be True (irrelevant if rotate over full D)
rotate_style = 0
reuse_freqs_front_part = True
nope_first = True

def _aiter_call(x_sbhd, cos, sin):
    return _aiter_cached_fwd(x_sbhd, cos, sin,
                                 rotate_style, reuse_freqs_front_part, nope_first)
for _ in range(num_warmup):
    o_sbhd = _aiter_call(x_sbhd, cos_aiter, sin_aiter)
timings_aiter = []
for _ in range(num_iters):
    torch.cuda.synchronize()
    start_event.record()
    o_sbhd = _aiter_call(x_sbhd, cos_aiter, sin_aiter)
    end_event.record()
    torch.cuda.synchronize()
    timings_aiter.append(start_event.elapsed_time(end_event))

avg_time_aiter = sum(timings_aiter) / len(timings_aiter)
eff_aiter = efficiency(flops_ref, avg_time_aiter)
print(f"AITer average execution time: {avg_time_aiter:.4f} ms")
print(f"AITer performance: {eff_aiter:.2f} TFLOPS for {B=} {N=} {D=}.")
print(f"Speedup vs PyTorch eager: {avg_time_ref / avg_time_aiter:.2f}x")

# aiter vs. pytorch diff
o_aiter = rearrange(o_sbhd, 's b h d -> b h s d').contiguous()
print("AITer o[0,0,0,:8]:", o_aiter[0, 0, 0, :8])
o_diff_aiter = (o - o_aiter).abs()
print("AITer max_diff:", o_diff_aiter.max().item())


# TK
# print("\nTK:")
o_tk = torch.zeros_like(x).bfloat16()
sin_tk = sin_in.to(torch.bfloat16).cuda()
cos_tk = cos_in.to(torch.bfloat16).cuda()
timings = []
for _ in range(num_warmup):
    tk_kernel.dispatch_rotary(x, o_tk, sin_tk, cos_tk)
for _ in range(num_iters):
    torch.cuda.synchronize()
    start_event.record()
    tk_kernel.dispatch_rotary(x, o_tk, sin_tk, cos_tk)
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
o_diff = o- o_tk
print("o: ", o[0, 0, 0, :8])
print("o_tk: ", o_tk[0, 0, 0, :8])
print("o_diff: ", o_diff[0, 0, :2, :8])
# print("o_diff: ", o_diff[:, :-1, :, :].max())

max_diff = o_diff.abs().max().item()
print("max_diff: ", max_diff)


# breakpoint()
