#include <hip_fp16.h>
extern "C" __global__ void matmul_kernel(int M, int N, int K, __half *A, __half *B, float *C, float alpha, float beta) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}