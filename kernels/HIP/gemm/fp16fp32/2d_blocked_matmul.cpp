#include <hip_fp16.h>
extern "C" __global__ void matmul_kernel(int M, int N, int K, __half *A, __half *B, float *C, float alpha, float beta) {

    const int BM = 256;
    const int BN = 256;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    __shared__ __half As[BM * BK];
    __shared__ __half Bs[BK * BN];

    const int C_row = blockIdx.y * BM;
    const int C_col = blockIdx.x * BN;

    const int totalResultsInBlockTile = BM * BN;
    const int totalThreadsInBlock = totalResultsInBlockTile / (TM * TN);

    // BN/TN are the number of threads to span a column
    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    const int steps = (K + BK - 1) / BK;

    float thread_result[TM * TN] = {0.0f};
    __half regM[TM] = {0.0f};
    __half regN[TN] = {0.0f};

    A += C_row * K;
    B += C_col;
    C += C_row * N + C_col;

    const int sA_row = threadIdx.x / BK;
    const int sA_col = threadIdx.x % BK;
    const int A_stride = totalThreadsInBlock / BK;

    const int sB_row = threadIdx.x / BN;
    const int sB_col = threadIdx.x % BN;
    const int B_stride = totalThreadsInBlock / BN;

    for (int i = 0; i < steps; i++) {

        // Load A and B tiles to shared memory
        for (int loadOffset = 0; loadOffset < BM; loadOffset += A_stride) {
            As[(sA_row + loadOffset) * BK + sA_col] = A[(sA_row + loadOffset) * K + sA_col];
        }

        for (int loadOffset = 0; loadOffset < BK; loadOffset += B_stride) {
            Bs[(sB_row + loadOffset) * BN + sB_col] = B[(sB_row + loadOffset) * K + sB_col];
        }

        __syncthreads();

        A += BK;
        B += BK * N;

        // Multiply the tiles and accumulate the result
        for (int j = 0; j < BK; j++) {

            for (int k = 0; k < TM; k++) {
                regM[k] = As[(threadRow * TM + k) * BK + j];
            }

            for (int k = 0; k < TN; k++) {
                regN[k] = Bs[(j * BN) + (threadCol * TN) + k];
            }

            for (int row = 0; row < TM; row++) {
                for (int col = 0; col < TN; col++) {
                    thread_result[row * TN + col] += __half2float(regM[row]) * __half2float(regN[col]);
                }
            }

        }

        __syncthreads();
    }

    // Store the result in the output matrix
    for (int row = 0; row < TM; row++) {
        for (int col = 0; col < TN; col++) {
            C[(threadRow * TM + row) * N + (threadCol * TN + col)] = 
                alpha * thread_result[row * TN + col] + beta * C[(threadRow * TM + row) * N + (threadCol * TN + col)];

        }
    }
}