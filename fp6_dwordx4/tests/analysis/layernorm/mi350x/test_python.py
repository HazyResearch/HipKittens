import torch
import random
import sys
import os
import tk_kernel
import torch.nn as nn

profiling = True
torch.manual_seed(0)
random.seed(0)

# Inputs
B = 16
H = 16
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
HEAD_D = 128
D = HEAD_D * H
DROPOUT_P = 0.01
norm = nn.LayerNorm(D).cuda()
torch.random.manual_seed(42)
x = torch.randn((B, N, D), dtype=torch.bfloat16, device='cuda').requires_grad_()
residual = torch.randn((B, N, D), dtype=torch.bfloat16, device='cuda').requires_grad_()

if profiling:
    num_warmup = 100
    num_iters = 500
else:
    num_warmup = 1
    num_iters = 0

start_event = torch.cuda.Event(enable_timing=True) # in milliseconds
end_event = torch.cuda.Event(enable_timing=True)

def efficiency(flop, time):
    """Calculate efficiency in TFLOPS."""
    flop = flop / 1e12  # convert to TFLOPS
    time = time / 1e3   # convert to seconds
    return flop / time


def flops(batch, seqlen, hidden_dim):
    """Calculate FLOPs for LayerNorm operation."""
    B, N, D = batch, seqlen, hidden_dim
    mean_flops = B * N * D    
    var_flops = B * N * D * 3  # subtract, square, sum    
    norm_flops = B * N * D * 2  # subtract, divide    
    scale_shift_flops = B * N * D * 2  # multiply, add
    total_flops = mean_flops + var_flops + norm_flops + scale_shift_flops
    return total_flops

def pytorch_ref(x, residual, norm):
    dropped = torch.nn.functional.dropout(x, p=DROPOUT_P)
    residual = (residual + dropped) if residual is not None else dropped
    x = norm(residual.to(dtype=norm.weight.dtype))
    return x, residual


flops_ref = flops(B, N, D)

timings_pytorch = []
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
    timings_pytorch.append(elapsed_time)
avg_time_ref = sum(timings_pytorch) / len(timings_pytorch)
eff = efficiency(flops_ref, avg_time_ref)
print(f"PyTorch average execution time: {avg_time_ref:.4f} ms")
print(f"PyTorch performance: {eff:.2f} TFLOPS for {B=} {N=} {D=}.")

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

# Kernel matmul
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

avg_time_tk = sum(timings) / len(timings)
eff_tk = efficiency(flops_ref, avg_time_tk)
print(f"TK average execution time: {avg_time_tk:.4f} ms")
print(f"TK performance: {eff_tk:.2f} TFLOPS for {B=} {N=} {D=}.")
speedup = avg_time_ref / avg_time_tk
print(f"Speedup from TK: {speedup:.2f}x")

# Compare against reference
if profiling:
    try:
        C_float = o_tk.float()
        C_ref_float = o_ref.float()
        diff = (C_float - C_ref_float).abs()
        max_error = diff.max().item()
        mean_error = diff.mean().item()
        error_count = (diff > 0.1).sum().item()
        print(f"Max error between kernel and reference: {max_error}")
        print(f"Max error: {max_error}")
        print(f"Mean error: {mean_error}")
        print(f"Number of large errors (>0.1): {error_count}\n")
    except Exception as e:
        print(f"Error comparing kernel and reference: {e}")
        max_error = None
        mean_error = None
        error_count = None

    ############### LOGGING OUTPUTS ####################

    data_to_log = {
        "avg_time_pytorch": avg_time_ref,
        "tflops_pytorch": eff,
        "avg_time_compiled": avg_time_compiled,
        "tflops_compiled": eff_compiled,
        "avg_time_tk": avg_time_tk,
        "tflops_tk": eff_tk,
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

    