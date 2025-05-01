from utils.check import hip_check, compare  
import pydra

from hip import hip, hipblas
import numpy as np

def test_hip_blas_matmul(config: pydra.Config, M: int, N: int, K: int, A_d, B_d, C_d, alpha: float, beta: float, C_expected):
    """
    Test the performance of the hipBLAS gemm implementation
    """

    # Create hipBLAS handle
    handle = hip_check(hipblas.hipblasCreate())

    # Create HIP events for timing
    start_event = hip_check(hip.hipEventCreate())
    stop_event = hip_check(hip.hipEventCreate())

    # Record start event
    hip_check(hip.hipEventRecord(start_event, 0))

    hip_check(hipblas.hipblasSgemm(handle, 
        hipblas.hipblasOperation_t.HIPBLAS_OP_N,  # No transpose A
        hipblas.hipblasOperation_t.HIPBLAS_OP_N,  # No transpose B
        M, N, K,
        alpha, 
        A_d, M,
        B_d, K,
        beta, 
        C_d, M
    ))

    # Record stop event
    hip_check(hip.hipEventRecord(stop_event, 0))
    hip_check(hip.hipEventSynchronize(stop_event))

    # Measure elapsed time
    elapsed_time = hip_check(hip.hipEventElapsedTime(start_event, stop_event))

    # Copy result (stored in C_d) back to host
    C_out = np.zeros((M, N), dtype=np.float32, order="F")
    num_bytes_C = C_out.nbytes
    hip_check(hip.hipMemcpy(C_out, C_d, num_bytes_C, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
    compare(C_out, C_expected, config.debug)

    # Destroy events and module
    hip_check(hip.hipEventDestroy(start_event))
    hip_check(hip.hipEventDestroy(stop_event))

    return elapsed_time

