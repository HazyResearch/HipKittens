# Standard library imports
import os
import math
import random
import array
import ctypes
from enum import Enum, auto

# Third-party imports
import numpy as np
import pydra
from pydra import REQUIRED, Config


# Write results to JSON file
import json
import datetime

# HIP-related imports
from hip import hip, hiprtc, hipblas

# Local/project imports
from utils.check import hip_check, compare
from src import run_hip_blas, run_pytorch, run_triton, run_hip

"""
Evaluate the performance of HiP implementations of various kernels
"""

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERNEL_DIR = os.path.join(REPO_TOP_DIR, "kernels")

class KernelType(Enum):
    """Enum for different kernel implementations"""
    PYTORCH = auto()
    HIP = auto()
    HIP_BLAS = auto()
    TRITON = auto()
    THUNDERKITTEN = auto()
    
    def __str__(self):
        return self.name.lower()
    
    @classmethod
    def from_string(cls, name: str):
        """Convert string to enum value, case-insensitive"""
        try:
            return cls[name.upper()]
        except KeyError:
            valid_values = [k.name.lower() for k in cls]
            raise ValueError(f"Invalid kernel type: {name}. Valid values are: {', '.join(valid_values)}")


class EvalConfig(Config):
    def __init__(self):
        self.kernel = "" # name of the matmul kernel to evaluate
        self.block_size = 16

        # matrix size dimensions
        # A matrix is M x K 
        # B matrix is K x N
        # C matrix is M x N
        self.M = 1024
        self.K = 1024
        self.N = 1024
        self.alpha = 1.0
        self.beta = 0.0

        # which kernel backend to run
        self.kernel_type = KernelType.HIP

        # timing
        self.num_warmup = 3
        self.num_iterations = 10

        self.debug = False
        self.results_dir = os.path.join(REPO_TOP_DIR, "results")

    def correctness(self):
        self.num_warmup = 0
        self.num_iterations = 1
        self.M = 128
        self.K = 128
        self.N = 128
        self.debug = True

    def matmul_shape(self):
        # Standard GEMM shape for benchmarking

        self.M = 8192
        self.K = 8192
        self.N = 8192   

    # Below are shapes of operations used in Llama 70B model
    def qkv_proj_shape(self):
        # Fused QKV Projection GEMM shape

        self.M = 16384
        self.K = 8192
        self.N = 1280

    def attn_output_shape(self):
        # Attention Output Projection shape
        self.M = 16384
        self.K = 1024
        self.N = 8192

    def ffn_gemm_shape(self):
        # FFN GEMM shape
        self.M = 16384
        self.K = 3584
        self.N = 8192


    def __repr__(self):
        return f"EvalConfig({self.to_dict()})"


def test_kernel_harness(config: EvalConfig):
    """
    Driver code to test kernels
    """

    # Convert string to KernelType if needed
    if not isinstance(config.kernel_type, KernelType):
        kernel_type = KernelType.from_string(str(config.kernel_type))
    else:
        kernel_type = config.kernel_type
   

    # Define matrix dimensions
    M = config.M
    K = config.K
    N = config.N

    alpha = config.alpha
    beta = config.beta

 
    # HIP BLAS expects matrices in column-major order
    order = "F" if kernel_type == KernelType.HIP_BLAS else "C"
    A_h = np.random.rand(M, K).astype(np.float32, order=order)
    B_h = np.random.rand(K, N).astype(np.float32, order=order)
    C_h = np.random.rand(M, N).astype(np.float32, order=order)
    # A_h = np.ones((M, K), dtype=np.float32, order=order)
    # B_h = np.ones((K, N), dtype=np.float32, order=order)
    # C_h = np.ones((M, N), dtype=np.float32, order=order)

    # Compute expected result using NumPy as Golden Reference
    C_expected = alpha * np.dot(A_h, B_h) + beta * C_h

    if kernel_type in [KernelType.HIP_BLAS, KernelType.HIP]:

        # Scalars
        alpha = ctypes.c_float(config.alpha)
        beta = ctypes.c_float(config.beta)

        # Allocate device memory
        num_bytes_A = A_h.nbytes
        num_bytes_B = B_h.nbytes
        num_bytes_C = C_h.nbytes

        A_d = hip_check(hip.hipMalloc(num_bytes_A))
        B_d = hip_check(hip.hipMalloc(num_bytes_B))
        C_d = hip_check(hip.hipMalloc(num_bytes_C))

        # Copy input data to device
        hip_check(hip.hipMemcpy(A_d, A_h, num_bytes_A, hip.hipMemcpyKind.hipMemcpyHostToDevice))
        hip_check(hip.hipMemcpy(B_d, B_h, num_bytes_B, hip.hipMemcpyKind.hipMemcpyHostToDevice))
        hip_check(hip.hipMemcpy(C_d, C_h, num_bytes_C, hip.hipMemcpyKind.hipMemcpyHostToDevice))

    # setup kernel
    match kernel_type:
        case KernelType.HIP_BLAS: # HIP_BLAS
            
            return run_hip_blas.test_hip_blas_matmul(config, M, N, K, A_d, B_d, C_d, alpha, beta, C_expected)
        
        case KernelType.HIP: # hand-written kernel

            assert len(config.kernel) > 0, "Kernel name must be provided"
            return run_hip.test_hip_kernel(config, M, N, K, A_d, B_d, C_d, alpha, beta, C_expected)
        
        case KernelType.TRITON: # Triton
            return run_triton.test_triton_matmul(config, M, N, K, A_h, B_h, C_h, alpha, beta, C_expected)

        case KernelType.THUNDERKITTEN: # Thunderkitten
            raise NotImplementedError("Thunderkitten is not implemented yet")

        case KernelType.PYTORCH: # PyTorch
            # pass the numpy arrays to pytorch
            return run_pytorch.test_pytorch_matmul(config, M, N, K, A_h, B_h, C_h, alpha, beta, C_expected)
        
        case _:
            raise ValueError(f"Not implemented for kernel type: {config.kernel_type}")
    
    # Clean up device memory
    hip_check(hip.hipFree(A_d))
    hip_check(hip.hipFree(B_d))
    hip_check(hip.hipFree(C_d))


@pydra.main(base=EvalConfig)
def main(config: EvalConfig):

    print(f"Running {config.kernel_type} kernel")

    M = config.M
    N = config.N
    K = config.K
    alpha = config.alpha
    beta = config.beta

    print(f"Testing Kernel with size: M {M} x K {K} x N {N} Alpha: {alpha}, Beta: {beta}")
    print(f"Using Kernel Backend: {config.kernel_type} Kernel Name: {config.kernel}")

    # Warmup
    for _ in range(config.num_warmup):
        test_kernel_harness(config)

    num_iterations = config.num_iterations

    kernel_times = []
    for _ in range(config.num_iterations):
        kernel_times.append(test_kernel_harness(config))
    

    # Compute execution test stats
    times_array = np.array(kernel_times)

    stats = {
        "mean": float(f"{np.mean(times_array):.2f}"),
        "std": float(f"{np.std(times_array):.2f}"), 
        "min": float(f"{np.min(times_array):.2f}"),
        "max": float(f"{np.max(times_array):.2f}"),
        "median": float(f"{np.median(times_array):.2f}"),
        "total_time": float(f"{np.sum(times_array):.2f}")
    }
    
    
    # total_time = sum(kernel_times)
    # avg_time = total_time / num_iterations
    print(f"\n✅ Average Kernel Execution Time: {stats['mean']:.4f} ms over {num_iterations} runs")
    print(f"Stats: {stats}")
    flops = 2. * 1e-9 * num_iterations * M * N * K / stats['total_time']
    print(f"\nPerformance FLOPS: ({flops:.2f}) TFLOPS. size: ({M} x {K}) * ({K} x {N}).\n")
    

    results = {
        "kernel_type": str(config.kernel_type),
        "kernel_name": config.kernel,
        "matrix_dims": {
            "M": M,
            "N": N, 
            "K": K,
            "alpha": alpha,
            "beta": beta
        },
        "parameters": {
            "num_warmup": config.num_warmup,
            "num_iterations": num_iterations
        },
        "timing_stats": stats,
        "performance": {
            "tflops": float(f"{flops:.2f}")
        },
        "timestamp": datetime.datetime.now().isoformat(),
        "precision": "fp32"
    }

    # Generate filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_{config.kernel_type}_config_{config.kernel}_{M}x{K}x{N}_{timestamp}.json"
    filepath = os.path.join(config.results_dir, filename)

    # Write results to JSON file
    with open(filepath, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults written to: {filepath}")

if __name__ == "__main__":
    main()