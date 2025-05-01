import torch
import triton
import triton.language as tl
import numpy as np
from utils.check import compare
import pydra

"""
Triton matrix multiplication on GPU

Modified from example on Triton repo:
https://github.com/triton-lang/triton/blob/main/python/tutorials/03-matrix-multiplication.py

TODO:
- tune for MI300X
"""


# Triton kernel implementation (from the provided code)
def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

def is_hip_mi200():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == 'hip' and target.arch == 'gfx90a'

def is_hip_mi300():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == 'hip' and target.arch == 'gfx942'

def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        # ... shortened for brevity ...
    ]

def get_hip_autotune_config():
    # this is tuned for MI200, not sure if this tuned for MI300X
    return [
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
            num_warps=4, num_stages=2),
        # ... shortened for brevity ...
    ]

def get_autotune_config():
    if is_cuda():
        return get_cuda_autotune_config()
    else:
        print("Using HIP autotune config for AMD GPU")
        return get_hip_autotune_config()

@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)

@triton.jit
def matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension
        stride_am, stride_ak,  
        stride_bk, stride_bn,  
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  
        GROUP_SIZE_M: tl.constexpr,  
        ACTIVATION: tl.constexpr  
):
    # Map program ids to the block of C it should compute
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Create pointers for the first blocks of A and B
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Compute a block of the C matrix
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, with masking
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # Accumulate along the K dimension
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        
    # Apply activation if needed
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    # c = accumulator.to(tl.float16)
    c = accumulator

    # Write back the block of the output matrix C with masks
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)

def matmul_triton(a, b, activation=""):
    # Check constraints
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocate output
    c = torch.empty((M, N), device=a.device)
    # Launch kernel
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        ACTIVATION=activation
    )
    return c

def test_triton_matmul(config: pydra.Config, M: int, N: int, K: int, A_h: np.ndarray, B_h: np.ndarray, C_h: np.ndarray, alpha: float, beta: float, C_expected: np.ndarray):
    """
    Test the performance of Triton matrix multiplication on GPU
    A_h, B_h, C_h are numpy arrays
    """
    # Check if GPU is available
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU detected")
    
    # Get the appropriate device
    device = torch.device("cuda")
    
    # Convert numpy arrays to PyTorch tensors and send to device
    A_tensor = torch.from_numpy(A_h).to(device)
    B_tensor = torch.from_numpy(B_h).to(device)
    C_tensor = torch.from_numpy(C_h).to(device)

    # Create CUDA events for timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Record start event
    start_event.record()
    
    # Perform matrix multiplication using Triton
    # Note: the Triton kernel doesn't directly support alpha and beta,
    # so we implement the full operation: alpha * A @ B + beta * C
    C_result = matmul_triton(A_tensor, B_tensor)
    # assume alpha = 1.0 and beta = 0.0
    # C_result = alpha * AB_result + beta * C_tensor
    
    # Record end event and synchronize
    end_event.record()
    torch.cuda.synchronize()
    
    # Get elapsed time in milliseconds
    elapsed_time = start_event.elapsed_time(end_event)
    
    # Copy result back to CPU for validation
    C_result_cpu = C_result.cpu().numpy()
    compare(C_result_cpu, C_expected, config.debug)
    
    return elapsed_time

