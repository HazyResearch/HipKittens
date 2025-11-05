import torch
import tk_kernel
import random
import time

profiling = True

torch.manual_seed(0)
random.seed(0)

# Inputs
N = 8192
A = torch.randn(N, N, dtype=torch.bfloat16, device='cuda') / 10.0  
B = torch.randn(N, N, dtype=torch.bfloat16, device='cuda') / 10.0  
Bt = B.t().contiguous()  # Transpose B for the kernel


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
flops_ref = (2 * N**3)  # FLOPs for reference

if profiling:
    # Reference matmul using PyTorch
    for _ in range(num_warmup):
        C_ref = torch.matmul(A, Bt)
    timings_ref = []
    for _ in range(num_iters):
        torch.cuda.synchronize()
        start_event.record()
        C_ref = torch.matmul(A, Bt)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        timings_ref.append(elapsed_time)
    if profiling:
        print(f"{C_ref.dtype=}")
        avg_time_ref = sum(timings_ref) / len(timings_ref)
        tflops_ref = flops_ref / (avg_time_ref * 1e9) 
        print(f"PyTorch reference average execution time: {avg_time_ref:.4f} ms")
        print(f"PyTorch reference performance: {tflops_ref:.2f} TFLOPS for {N}x{N} matrix multiplication.\n")


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
    C_ref_float = C_ref.float()
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
    print("diff[7168:7232, 7168:7232].max()", diff[7168:7232, 7168:7232].max())
    print("diff[7232:7296, 7232:7296].max()", diff[7232:7296, 7232:7296].max())
    print("diff[7296:7360, 7296:7360].max()", diff[7296:7360, 7296:7360].max())
    print("diff[7360:7424, 7360:7424].max()", diff[7360:7424, 7360:7424].max())




    ############### LOGGING OUTPUTS ####################

    data_to_log = {
        "N": N,
        "avg_time_ref": avg_time_ref,
        "tflops_ref": tflops_ref,
        "avg_time": avg_time,
        "tflops": tflops,
        "max_error": max_error,
        "mean_error": mean_error,
        "error_count": error_count,
    }

    import json
    with open(os.path.join(new_dir, "data_to_log.json"), "w") as f:
        json.dump(data_to_log, f, indent=4)

    ################ END LOGGING OUTPUTS ###############

    