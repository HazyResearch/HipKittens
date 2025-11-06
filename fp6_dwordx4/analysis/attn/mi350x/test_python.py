import torch
import tk_kernel
import random
import time
import math
import sys
from torch.nn.functional import scaled_dot_product_attention
import aiter

profiling = True
using_aiter = True

torch.manual_seed(0)
random.seed(0)


torch.set_printoptions(
    precision=3,        # decimals
    sci_mode=False,     # True → always scientific
    linewidth=220,      # characters per line before folding
    threshold=float("inf")  # print every element, no summarising "..."
)

# Inputs
B = 16
H = 64
H_KV = 8
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
D = 128
causal = False
dtype = torch.bfloat16
q = torch.randn(B, N, H, D, dtype=dtype, device='cuda', requires_grad=True)
k = torch.randn(B, N, H_KV, D, dtype=dtype, device='cuda', requires_grad=True)
v = torch.randn(B, N, H_KV, D, dtype=dtype, device='cuda', requires_grad=True)


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
                if match := re.match(r"^SRC\s*[?]?=\s*(\S+)", line):
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
    
    # Only include src_name in files_to_copy if it was successfully parsed
    files_to_copy = ["Makefile"]
    if src_name is not None:
        files_to_copy.append(src_name)
    files_to_copy.extend([
        "tk_kernel.cpython-313-x86_64-linux-gnu.so",
        "tk_kernel.cpython-312-x86_64-linux-gnu.so"
    ])

    for filename in files_to_copy:
        src = os.path.join(base_dir, filename)
        dst = os.path.join(new_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            print(f"Warning: {filename} not found at {src}, skipping.")

    ################ END LOGGING STUFF ###############

if profiling:
    num_warmup = 100
    num_iters = 500
else:
    num_warmup = 1
    num_iters = 0

start_event = torch.cuda.Event(enable_timing=True) # in milliseconds
end_event = torch.cuda.Event(enable_timing=True)
flops_ref = flops(B, N, H, D, causal)


if profiling:

    # # Reference matmul using PyTorch
    # for _ in range(num_warmup):
    #     out_ref_pytorch = scaled_dot_product_attention(q, k, v, is_causal=causal)
    # timings_ref = []
    # for _ in range(num_iters):
    #     torch.cuda.synchronize()
    #     start_event.record()
    #     out_ref_pytorch = scaled_dot_product_attention(q, k, v, is_causal=causal)
    #     end_event.record()
    #     torch.cuda.synchronize()
    #     elapsed_time = start_event.elapsed_time(end_event)
    #     timings_ref.append(elapsed_time)
    # if profiling:
    #     print(f"{out_ref_pytorch.dtype=}")
    #     avg_time_ref = sum(timings_ref) / len(timings_ref)
    #     eff_ref = efficiency(flops_ref, avg_time_ref)
    #     print(f"PyTorch reference average execution time: {avg_time_ref:.4f} ms")
    #     print(f"PyTorch reference performance: {eff_ref:.2f} TFLOPS for {B=} {H=} {N=} {D=} {causal=}.\n")

    # Reference matmul using AITER
    if using_aiter:
        for _ in range(num_warmup):
            out_ref = aiter.flash_attn_func(q, k, v, causal=causal, return_lse=True, deterministic=True)
        timings_ref = []
        for _ in range(num_iters):
            torch.cuda.synchronize()
            start_event.record()
            out_ref = aiter.flash_attn_func(q, k, v, causal=causal, return_lse=True, deterministic=True)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)
            timings_ref.append(elapsed_time)
        if profiling:
            print(f"{out_ref[0].dtype=}")
            avg_time_ref = sum(timings_ref) / len(timings_ref)
            eff_ref = efficiency(flops_ref, avg_time_ref)
            print(f"AITER (AMD) reference average execution time: {avg_time_ref:.4f} ms")
            print(f"AITER (AMD) reference performance: {eff_ref:.2f} TFLOPS for {B=} {H=} {N=} {D=} {causal=}.\n")

# Kernel matmul
out = torch.zeros(B, N, H, D, dtype=dtype, device='cuda', requires_grad=True)
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
    # out_ref_pytorch_float = out_ref_pytorch.float()
    # diff_pytorch = (out_float - out_ref_pytorch_float).abs()
    # max_error_pytorch = diff_pytorch.max().item()
    # mean_error_pytorch = diff_pytorch.mean().item()
    # error_count_pytorch = (diff_pytorch > 0.1).sum().item()

    # print(f"Max error between kernel and PyTorch reference: {max_error_pytorch}")
    # print(f"Mean error between kernel and PyTorch reference: {mean_error_pytorch}")
    # print(f"Number of large errors (>0.1) between kernel and PyTorch reference: {error_count_pytorch}\n")

    if using_aiter:

        out_ref_float = out_ref[0].float()
        diff = (out_float - out_ref_float)
        max_error = diff.max().item()
        mean_error = diff.mean().item()
        error_count = (diff > 0.1).sum().item()

        print(f"Max error between kernel and reference: {max_error}")
        print(f"Max error: {max_error}")
        print(f"Mean error: {mean_error}")
        print(f"Number of large errors (>0.1): {error_count}\n")


        print(out_float[0:2, 0, 0, :16])
        print(out_ref_float[0:2, 0, 0, :16])

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

    if not os.path.exists("data_to_log.json"):
        with open("data_to_log.json", "w") as f:
            json.dump({}, f, indent=4)

    with open("data_to_log.json", "r") as f:
        data = json.load(f)
        data[str(N)] = data_to_log

    with open("data_to_log.json", "w") as f:
        json.dump(data, f, indent=4)

    print(f"Results saved to data_to_log.json")

    ############### END LOGGING OUTPUTS ###############

    