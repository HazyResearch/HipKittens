#include <hip_fp16.h>
extern "C" __global__ void matmul_kernel(int M, int N, int K, __half *A, __half *B, float *C, float alpha, float beta) {

    #define BlockSize 32

    // Shared memory is used to cache the tile from both input matrices
    // The tile is a square of size BlockSize x BlockSize
    __shared__ __half As[BlockSize][BlockSize];
    __shared__ __half Bs[BlockSize][BlockSize];

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int bx = blockIdx.x;
    const unsigned int by = blockIdx.y;

    const unsigned int a_cols = K;

    // b_cols must match the number of output matrix columns
    const unsigned int b_cols = blockDim.x * gridDim.x;

    // The number of tiles is deterimed by A's columns and B's rows
    const unsigned int steps = (K + BlockSize - 1) / BlockSize;

    // thread_result is the accumulation variable
    float thread_result = 0.0f;

    // Output tile indices
    const unsigned int c_row = by * BlockSize + ty;
    const unsigned int c_col = bx * BlockSize + tx;

    for (unsigned int step = 0; step < steps; step++) {

        const unsigned int a_row = by * BlockSize + ty;
        const unsigned int a_col = step * BlockSize + tx;

        // Load each element in the tile to shared memory
        if (a_row < M && a_col < K)
            As[ty][tx] = A[a_row * K + a_col];
        else
            As[ty][tx] = 0.0f;

        const unsigned int b_row = step * BlockSize + ty;
        const unsigned int b_col = bx * BlockSize + tx;

        if (b_row < K && b_col < N)
            Bs[ty][tx] = B[b_row * N + b_col];
        else
            Bs[ty][tx] = 0.0f;

        // Wait for all threads to finish loading
        __syncthreads();

        // Multiply the tile and accumulate the result
        for (unsigned int i = 0; i < BlockSize; i++) {
            thread_result += __half2float(As[ty][i]) * __half2float(Bs[i][tx]);
        }

        // Wait for all threads to finish multiplying
        __syncthreads();
    }

    if (c_row < M && c_col < N)
        C[c_row * N + c_col] = alpha * thread_result + beta * C[c_row * N + c_col];

}