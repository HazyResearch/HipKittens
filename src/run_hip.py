import pydra
from pydra import REQUIRED, Config

import os
from utils.io import read_file_as_bytes
from utils.check import hip_check, compare

# HIP-related imports
from hip import hip, hiprtc, hipblas

import math
import ctypes
import numpy as np

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERNEL_DIR = os.path.join(REPO_TOP_DIR, "kernels/HIP")

def test_hip_kernel(config: pydra.Config, M: int, N: int, K: int, A_d, B_d, C_d, alpha: float, beta: float, C_expected):
    """
    Test the performance of the HiP kernel implementation
    Note this is not HIPBlas, but pure HiP kernel
    """
    assert len(config.kernel) > 0, "Kernel name must be provided"

    # Define the HIP kernel (inline compilation)
    kernel_path = os.path.join(KERNEL_DIR, f"{config.kernel}.cpp")
    assert os.path.exists(kernel_path), f"Kernel file does not exist: {kernel_path}"
    source = read_file_as_bytes(kernel_path)

    # Compile kernel using HIPRTC
    prog = hip_check(hiprtc.hiprtcCreateProgram(source, b"matmul_kernel", 0, [], []))

    props = hip.hipDeviceProp_t()
    hip_check(hip.hipGetDeviceProperties(props, 0))
    arch = props.gcnArchName

    print(f"Compiling kernel for {arch}")

    cflags = [b"--offload-arch=" + arch]
    err, = hiprtc.hiprtcCompileProgram(prog, len(cflags), cflags)
    if err != hiprtc.hiprtcResult.HIPRTC_SUCCESS:
        log_size = hip_check(hiprtc.hiprtcGetProgramLogSize(prog))
        log = bytearray(log_size)
        hip_check(hiprtc.hiprtcGetProgramLog(prog, log))
        raise RuntimeError(log.decode())

    # Extract compiled binary
    code_size = hip_check(hiprtc.hiprtcGetCodeSize(prog))
    code = bytearray(code_size)
    hip_check(hiprtc.hiprtcGetCode(prog, code))
    module = hip_check(hip.hipModuleLoadData(code))
    kernel = hip_check(hip.hipModuleGetFunction(module, b"matmul_kernel"))

    # Compute block and grid
    block_size = config.block_size
    
    if (config.kernel == "warptiling_mfma"):
        BN = 128
        BM = 128
        TM = 4
        TN = 1
        WNITER = 4
        WMITER = 4

        block = hip.dim3(x=BM * BN / (WMITER * TM * WNITER * TN))
        grid = hip.dim3(x=math.ceil(N / BN), y=math.ceil(M / BM))
    elif (config.kernel == "warptiling"):
        BN = 128
        BM = 128
        TM = 1
        TN = 4
        WNITER = 4
        WMITER = 4

        block = hip.dim3(x=BM * BN / (WMITER * TM * WNITER * TN))
        grid = hip.dim3(x=math.ceil(N / BN), y=math.ceil(M / BM))

    elif (config.kernel == "resolve_bank_extra_cols"):
        BN = 128
        BM = 128
        BK = 8
        TM = 8
        TN = 8

        block = hip.dim3(x=BM * BN / (TM * TN))
        grid = hip.dim3(x=math.ceil(N / BN), y=math.ceil(M / BM))
    elif (config.kernel == "resolve_bank_conflicts"):
        BN = 128
        BM = 128
        BK = 8
        TM = 8
        TN = 8

        block = hip.dim3(x=BM * BN / (TM * TN))
        grid = hip.dim3(x=math.ceil(N / BN), y=math.ceil(M / BM))
    elif (config.kernel == "vectorize"):
        BN = 128
        BM = 128
        BK = 8
        TM = 8
        TN = 8

        block = hip.dim3(x=BM * BN / (TM * TN))
        grid = hip.dim3(x=math.ceil(N / BN), y=math.ceil(M / BM))
    elif (config.kernel == "2d_blocked_matmul"):
        BN = 256
        BM = 256
        BK = 8
        TM = 8
        TN = 8

        block = hip.dim3(x=BM * BN / (TM * TN))
        grid = hip.dim3(x=math.ceil(N / BN), y=math.ceil(M / BM))
    elif (config.kernel == "1d_blocked_matmul"):
        BN = 64
        BM = 64
        BK = 8

        block = hip.dim3(x=BM * BN / BK)
        grid = hip.dim3(x=math.ceil(N / BN), y=math.ceil(M / BM))
    elif (config.kernel == "blocked_matmul"):
        BN = 32
        BM = 32
        BK = 32

        block = hip.dim3(x=BM, y=BK)
        grid = hip.dim3(x=math.ceil(N / BN), y=math.ceil(M / BM))
    else:
        block = hip.dim3(x=block_size, y=block_size)
        grid = hip.dim3(x=math.ceil(N / block_size), y=math.ceil(M / block_size))

    # Create HIP events for timing
    start_event = hip_check(hip.hipEventCreate())
    stop_event = hip_check(hip.hipEventCreate())

    # Record start event
    hip_check(hip.hipEventRecord(start_event, 0))

    hip_check(
        hip.hipModuleLaunchKernel(
            kernel,
            *grid,
            *block,
            sharedMemBytes=0,
            stream=None,
            kernelParams=None,
            extra=(
                ctypes.c_int(M),
                ctypes.c_int(N),
                ctypes.c_int(K),
                A_d,
                B_d,
                C_d,
                alpha,
                beta,
            ),
        )
    )

    # Record stop event
    hip_check(hip.hipEventRecord(stop_event, 0))
    hip_check(hip.hipEventSynchronize(stop_event))

    # Measure elapsed time
    elapsed_time = hip_check(hip.hipEventElapsedTime(start_event, stop_event))

    # Copy result (stored in C_d) back to host
    C_out = np.zeros((M, N), dtype=np.float32)
    num_bytes_C = C_out.nbytes
    hip_check(hip.hipMemcpy(C_out, C_d, num_bytes_C, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
    compare(C_out, C_expected, config.debug)

    # Destroy events and module
    hip_check(hip.hipEventDestroy(start_event))
    hip_check(hip.hipEventDestroy(stop_event))
    hip_check(hip.hipModuleUnload(module))
    hip_check(hiprtc.hiprtcDestroyProgram(prog.createRef()))

    return elapsed_time