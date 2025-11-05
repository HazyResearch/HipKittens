#include <hip_fp16.h>

namespace wt {
using half_vec4 = __attribute__((__vector_size__(4 * sizeof(__fp16)))) __fp16;
using float_vec4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

template <const int BM, const int BN, const int BK, const int rowStrideA,
          const int rowStrideB>
__device__ void loadFromGmem(int N, int K, const __fp16 *A, const __fp16 *B,
                             __fp16 *As, __fp16 *Bs, int innerRowA, int innerColA,
                             int innerRowB, int innerColB) {
  for (uint offset = 0; offset < BM; offset += rowStrideA) {
    const half_vec4 tmp = reinterpret_cast<const half_vec4 *>(
        &A[(innerRowA + offset) * K + innerColA * 4])[0];

    As[(innerColA * 4) * BM + innerRowA + offset] = tmp[0];
    As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp[1];
    As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp[2];
    As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp[3];
  }

  for (uint offset = 0; offset < BK; offset += rowStrideB) {
    reinterpret_cast<half_vec4 *>(
        &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] = 
        reinterpret_cast<const half_vec4 *>(
            &B[(innerRowB + offset) * N + innerColB * 4])[0];
  }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
          const int TM, const int TN>
__device__ void
processFromSmem(float *threadResults, const __fp16 *As,
                const __fp16 *Bs, const uint warpRow, const uint warpCol,
                const uint threadRowInWarp, const uint threadColInWarp) {
  
  half_vec4 A_in = {0.0f};
  half_vec4 B_in = {0.0f};

  for (uint dotIdx = 0; dotIdx < BK; dotIdx += 16) {
    // execute warptile matmul
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {

        A_in[0] = As[(dotIdx * BM) + (warpRow * WM) + (wSubRowIdx * WSUBM) + ((4 * threadRowInWarp) * BN) + threadColInWarp];
        A_in[1] = As[(dotIdx * BM) + (warpRow * WM) + (wSubRowIdx * WSUBM) + ((4 * threadRowInWarp + 1) * BN) + threadColInWarp];
        A_in[2] = As[(dotIdx * BM) + (warpRow * WM) + (wSubRowIdx * WSUBM) + ((4 * threadRowInWarp + 2) * BN) + threadColInWarp];
        A_in[3] = As[(dotIdx * BM) + (warpRow * WM) + (wSubRowIdx * WSUBM) + ((4 * threadRowInWarp + 3) * BN) + threadColInWarp];

        B_in[0] = Bs[(dotIdx * BN) + (warpCol * WN) + (wSubColIdx * WSUBN) + ((4 * threadRowInWarp) * BN) + threadColInWarp];
        B_in[1] = Bs[(dotIdx * BN) + (warpCol * WN) + (wSubColIdx * WSUBN) + ((4 * threadRowInWarp + 1) * BN) + threadColInWarp];
        B_in[2] = Bs[(dotIdx * BN) + (warpCol * WN) + (wSubColIdx * WSUBN) + ((4 * threadRowInWarp + 2) * BN) + threadColInWarp];
        B_in[3] = Bs[(dotIdx * BN) + (warpCol * WN) + (wSubColIdx * WSUBN) + ((4 * threadRowInWarp + 3) * BN) + threadColInWarp];

        float_vec4 acc = {0.0f};
        acc = __builtin_amdgcn_mfma_f32_16x16x16f16(A_in, B_in, acc, 0, 0, 0);

        threadResults[(wSubRowIdx) * (WNITER * TM) + (wSubColIdx * TM)] += acc[0];
        threadResults[(wSubRowIdx) * (WNITER * TM) + (wSubColIdx * TM) + 1] += acc[1];
        threadResults[(wSubRowIdx) * (WNITER * TM) + (wSubColIdx * TM) + 2] += acc[2];
        threadResults[(wSubRowIdx) * (WNITER * TM) + (wSubColIdx * TM) + 3] += acc[3];
      }
    }
  }
}

__device__ void print_shared_matrix(const __fp16* shared_matrix, int rows, int cols, const char* label = "Shared Matrix") {
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        printf("\n=== %s ===\n", label);
        for (int r = 0; r < rows; ++r) {
            printf("Row %2d: ", r);
            for (int c = 0; c < cols; ++c) {
                // Convert __fp16 to float for readable output
                printf("%7.3f ", __half2float(shared_matrix[r * cols + c]));
            }
            printf("\n");
        }
        printf("====================\n");
    }
}

} // namespace wt

/*
 * @tparam BM The threadblock size for M dimension SMEM caching.
 * @tparam BN The threadblock size for N dimension SMEM caching.
 * @tparam BK The threadblock size for K dimension SMEM caching.
 * @tparam WM M dim of continuous tile computed by each warp
 * @tparam WN N dim of continuous tile computed by each warp
 * @tparam WMITER The number of subwarp tiling steps in M dimension.
 * @tparam WNITER The number of subwarp tiling steps in N dimension.
 * @tparam TM The per-thread tile size for M dimension.
 * @tparam TN The per-thread tile size for N dimension.
 */
 const int WARPSIZE = 64; // warpSize is not constexpr
extern "C" __global__ void matmul_kernel(int M, int N, int K, __fp16 *A, __fp16 *B, float *C, float alpha, float beta) {

    const int BM = 128;
    const int BN = 128;
    const int BK = 16;
    const int WM = 64;
    const int WN = 64;
    const int WNITER = 4;
    const int TM = 4;
    const int TN = 1;

    const int NUM_THREADS = 256;

    __shared__ __fp16 As[BM * BK];
    __shared__ __fp16 Bs[BK * BN];

    const int C_row = blockIdx.y * BM;
    const int C_col = blockIdx.x * BN;

    // Placement of the warp in the threadblock tile
    const uint warpIdx = threadIdx.x / WARPSIZE; // the warp this thread is in
    const uint warpCol = warpIdx % (BN / WN); 
    const uint warpRow = warpIdx / (BN / WN);

    // size of the warp subtile
    constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr uint WSUBM = WM / WMITER; // 32/2=16
    constexpr uint WSUBN = WN / WNITER; // 32/2=16

    // Placement of the thread in the warp subtile
    const uint threadIdxInWarp = threadIdx.x % WARPSIZE;         // [0, 63]
    const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN); // i%(64/8)
    const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN); // i/(64/8)

    // calculating the indices that this thread will load into SMEM
    // we'll load 128bit / 32bit = 4 elements per thread at each step
    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);
    constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;
    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);
    constexpr uint rowStrideB = NUM_THREADS / (BN / 4);
    
    // allocate thread-local cache for results in registerfile
    float threadResults[WMITER * TM * WNITER * TN] = {0.0};

    // Move blocktile to beginning of A's row and B's column
    A += C_row * K;
    B += C_col;
    // Move C_ptr to warp's output tile
    C += (C_row + warpRow * WM) * N + C_col + warpCol * WN;

    // outer-most loop over block tiles
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        wt::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
            N, K, A, B, As, Bs, innerRowA, innerColA, innerRowB, innerColB);
        __syncthreads();
        wt::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM,
                            TN>(threadResults, As, Bs, warpRow, warpCol,
                                threadRowInWarp, threadColInWarp);
        A += BK;     // move BK columns to right
        B += BK * N; // move BK rows down
        __syncthreads();
    }

    // write out the results
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        // move C pointer to current warp subtile
        float *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
            // load C vector into registers
            const int i = (wSubRowIdx) * (WNITER * TM) + (wSubColIdx * TM);

              C_interim[(threadRowInWarp * TM) * N + threadColInWarp * TN] = 
              alpha * threadResults[i + 0] + beta * 
              C_interim[(threadRowInWarp * TM) * N + threadColInWarp * TN];

              C_interim[(threadRowInWarp * TM + 1) * N + threadColInWarp * TN] = 
              alpha * threadResults[i + 1] + beta * 
              C_interim[(threadRowInWarp * TM + 1) * N + threadColInWarp * TN];

              C_interim[(threadRowInWarp * TM + 2) * N + threadColInWarp * TN] = 
              alpha * threadResults[i + 2] + beta * 
              C_interim[(threadRowInWarp * TM + 2) * N + threadColInWarp * TN];

              C_interim[(threadRowInWarp * TM + 3) * N + threadColInWarp * TN] = 
              alpha * threadResults[i + 3] + beta * 
              C_interim[(threadRowInWarp * TM + 3) * N + threadColInWarp * TN];
            
        }
    }
}