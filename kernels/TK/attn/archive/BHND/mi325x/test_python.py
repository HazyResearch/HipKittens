import torch
import tk_kernel
import random
import time
import math
from torch.nn.functional import scaled_dot_product_attention
from aiter.ops.triton.mha import flash_attn_func

profiling = True

torch.manual_seed(0)
random.seed(0)

# Inputs
B = 16
H = 16
N = 1024
D = 64
causal = False
dtype = torch.bfloat16
q = torch.randn(B, H, N, D, dtype=dtype, device='cuda', requires_grad=True)
k = torch.randn(B, H, N, D, dtype=dtype, device='cuda', requires_grad=True)
v = torch.randn(B, H, N, D, dtype=dtype, device='cuda', requires_grad=True)


def flops(batch, seqlen, nheads, headdim, causal, mode="fwd"):
    """Calculate FLOPs for attention operation."""
    flop = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return flop

def efficiency(flop, time):
    """Calculate efficiency in TFLOPS."""
    flop = flop / 1e12  # convert to TFLOPS
    time = time / 1e3   # convert to seconds
    return flop / time


if profiling:
    ############### LOGGING STUFF ###############

    import os
    import time
    import shutil
    import re

    def parse_makefile_targets(makefile_path):
        src = None
        with open(makefile_path, "r") as f:
            for line in f:
                if match := re.match(r"^SRC\s*=\s*(\S+)", line):
                    src = match.group(1)
        return src

    base_dir = os.path.dirname(os.path.realpath(__file__))

    # Set destination directory
    dirpath = "/workdir/data_logs/"
    timestamp = time.strftime("%m%d_%H%M%S")
    new_dir = os.path.join(dirpath, f"{timestamp}_outputs")
    os.makedirs(new_dir, exist_ok=True)

    # Files to copy (relative to base_dir)
    src_name = parse_makefile_targets(os.path.join(base_dir, "Makefile"))
    print(f"src: {src_name}")
    files_to_copy = [
        "Makefile",
        src_name, 
        "tk_kernel.cpython-313-x86_64-linux-gnu.so",
        "tk_kernel.cpython-312-x86_64-linux-gnu.so"
    ]

    for filename in files_to_copy:
        src = os.path.join(base_dir, filename)
        dst = os.path.join(new_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            print(f"Warning: {filename} not found at {src}, skipping.")

    ################ END LOGGING STUFF ###############

if profiling:
    num_warmup = 20
    num_iters = 20
else:
    num_warmup = 1
    num_iters = 0

start_event = torch.cuda.Event(enable_timing=True) # in milliseconds
end_event = torch.cuda.Event(enable_timing=True)
flops_ref = flops(B, N, H, D, causal)


if profiling:
    # # Reference matmul using PyTorch
    # for _ in range(num_warmup):
    #     out_ref = scaled_dot_product_attention(q, k, v, is_causal=causal)
    # timings_ref = []
    # for _ in range(num_iters):
    #     torch.cuda.synchronize()
    #     start_event.record()
    #     out_ref = scaled_dot_product_attention(q, k, v, is_causal=causal)
    #     end_event.record()
    #     torch.cuda.synchronize()
    #     elapsed_time = start_event.elapsed_time(end_event)
    #     timings_ref.append(elapsed_time)
    # if profiling:
    #     print(f"{out_ref.dtype=}")
    #     avg_time_ref = sum(timings_ref) / len(timings_ref)
    #     eff_ref = efficiency(flops_ref, avg_time_ref)
    #     print(f"PyTorch reference average execution time: {avg_time_ref:.4f} ms")
    #     print(f"PyTorch reference performance: {eff_ref:.2f} TFLOPS for {B=} {H=} {N=} {D=} {causal=}")
    # Reference matmul using PyTorch
    q_bnhd = q.permute(0, 2, 1, 3).contiguous()  # -> (B, N, H, D)
    k_bnhd = k.permute(0, 2, 1, 3).contiguous()
    v_bnhd = v.permute(0, 2, 1, 3).contiguous()
    for _ in range(num_warmup):
        out_ref = flash_attn_func(
            q_bnhd, k_bnhd, v_bnhd
        )
    timings_ref = []
    for _ in range(num_iters):
        torch.cuda.synchronize()
        start_event.record()
        out_ref = flash_attn_func(
            q_bnhd, k_bnhd, v_bnhd
        )
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        timings_ref.append(elapsed_time)
    if profiling:
        print(f"{out_ref.dtype=}")
        avg_time_ref = sum(timings_ref) / len(timings_ref)
        eff_ref = efficiency(flops_ref, avg_time_ref)
        print(f"AITER (AMD) reference average execution time: {avg_time_ref:.4f} ms")
        print(f"AITER (AMD) reference performance: {eff_ref:.2f} TFLOPS for {B=} {H=} {N=} {D=} {causal=}")

# Kernel matmul
out = torch.zeros(B, H, N, D, dtype=dtype, device='cuda', requires_grad=True)
for _ in range(num_warmup):
    tk_kernel.dispatch_micro(q, k, v, out)
timings = []
for _ in range(num_iters):
    torch.cuda.synchronize()
    start_event.record()
    tk_kernel.dispatch_micro(q, k, v, out)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    timings.append(elapsed_time)
if profiling:
    print(f"{out.dtype=}")
    avg_time = sum(timings) / len(timings)
    eff = efficiency(flops_ref, avg_time)
    print(f"Average execution time: {avg_time:.4f} ms")
    print(f"Performance: {eff:.2f} TFLOPS for {N}x{N} matrix multiplication.\n")


# Compare against reference
if profiling:
    out_float = out.float()
    out_ref_float = out_ref.permute(0, 2, 1, 3).contiguous().float()
    diff = (out_float - out_ref_float).abs()
    max_error = diff.max().item()
    mean_error = diff.mean().item()
    error_count = (diff > 0.1).sum().item()

    print(f"Max error between kernel and reference: {max_error}")
    print(f"Max error: {max_error}")
    print(f"Mean error: {mean_error}")
    print(f"Number of large errors (>0.1): {error_count}\n")


    ############## LOGGING OUTPUTS ####################

    data_to_log = {
        "N": N,
        "avg_time_ref": avg_time_ref,
        "tflops_ref": eff_ref,
        "avg_time": avg_time,
        "tflops": eff,
        "max_error": max_error,
        "mean_error": mean_error,
        "error_count": error_count,
    }

    import json
    with open(os.path.join(new_dir, "data_to_log.json"), "w") as f:
        json.dump(data_to_log, f, indent=4)

    ############### END LOGGING OUTPUTS ###############

    