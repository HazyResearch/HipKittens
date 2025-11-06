extern "C" __global__ void matmul_kernel(int M, int N, int K, float *A, float *B, float *C, float alpha, float beta) {

    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    __shared__ float As[BM * BK];
    const int extraCols = 4;
    __shared__ float Bs[BK * (BN + extraCols)];

    const int C_row = blockIdx.y * BM;
    const int C_col = blockIdx.x * BN;

    const int totalResultsInBlockTile = BM * BN;
    const int totalThreadsInBlock = totalResultsInBlockTile / (TM * TN);

    // BN/TN are the number of threads to span a column
    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    const int steps = (K + BK - 1) / BK;

    float thread_result[TM * TN] = {0.0f};
    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};

    A += C_row * K;
    B += C_col;
    C += C_row * N + C_col;

    const int sA_row = threadIdx.x / (BK / 4);
    const int sA_col = threadIdx.x % (BK / 4);

    const int sB_row = threadIdx.x / (BN / 4);
    const int sB_col = threadIdx.x % (BN / 4);

    for (int i = 0; i < steps; i++) {

        // Load A and B tiles to shared memory
        // Transpose As
        float4 tmp = reinterpret_cast<float4 *>(&A[sA_row * K + sA_col * 4])[0];
        As[(sA_col * 4 + 0) * BM + sA_row] = tmp.x;
        As[(sA_col * 4 + 1) * BM + sA_row] = tmp.y;
        As[(sA_col * 4 + 2) * BM + sA_row] = tmp.z;
        As[(sA_col * 4 + 3) * BM + sA_row] = tmp.w;

        tmp = reinterpret_cast<float4 *>(&B[sB_row * N + sB_col * 4])[0];
        Bs[sB_row * (BN + extraCols) + sB_col * 4 + 0] = tmp.x;
        Bs[sB_row * (BN + extraCols) + sB_col * 4 + 1] = tmp.y;
        Bs[sB_row * (BN + extraCols) + sB_col * 4 + 2] = tmp.z;
        Bs[sB_row * (BN + extraCols) + sB_col * 4 + 3] = tmp.w;
        __syncthreads();

        A += BK;
        B += BK * N;

        // Multiply the tiles and accumulate the result
        for (int j = 0; j < BK; j++) {

            for (int k = 0; k < TM; k++) {
                regM[k] = As[(j * BM) + (threadRow * TM) + k];
            }

            for (int k = 0; k < TN; k++) {
                regN[k] = Bs[j * (BN + extraCols) + threadCol * TN + k];
            }

            for (int row = 0; row < TM; row++) {
                for (int col = 0; col < TN; col++) {
                    thread_result[row * TN + col] += regM[row] * regN[col];
                }
            }

        }

        __syncthreads();
    }

    // Store the result in the output matrix
    for (int row = 0; row < TM; row++) {
        for (int col = 0; col < TN; col += 4) {

            float4 tmp = reinterpret_cast<float4 *>(
                &C[(threadRow * TM + row) * N + (threadCol * TN + col)]
            )[0];

            tmp.x = alpha * thread_result[row * TN + col] + beta * tmp.x;
            tmp.y = alpha * thread_result[row * TN + col + 1] + beta * tmp.y;
            tmp.z = alpha * thread_result[row * TN + col + 2] + beta * tmp.z;
            tmp.w = alpha * thread_result[row * TN + col + 3] + beta * tmp.w;

            reinterpret_cast<float4 *>(
                &C[(threadRow * TM + row) * N + threadCol * TN + col])[0] = tmp;
        }
    }
}