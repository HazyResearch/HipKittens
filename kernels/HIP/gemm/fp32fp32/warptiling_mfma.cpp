namespace wt {
using float16 = __attribute__( (__vector_size__(16 * sizeof(float)) )) float;
using float_vec4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
template <const int BM, const int BN, const int BK, const int rowStrideA,
          const int rowStrideB>
__device__ void loadFromGmem(int N, int K, const float *A, const float *B,
                             float *As, float *Bs, int innerRowA, int innerColA,
                             int innerRowB, int innerColB) {
  for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
    const float4 tmp = reinterpret_cast<const float4 *>(
        &A[(innerRowA + offset) * K + innerColA * 4])[0];

    As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
  }

  for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
    reinterpret_cast<float4 *>(
        &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] = 
        reinterpret_cast<const float4 *>(
            &B[(innerRowB + offset) * N + innerColB * 4])[0];
  }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
          const int TM, const int TN>
__device__ void
processFromSmem(float *regM, float *regN, float *threadResults, const float *As,
                const float *Bs, const uint warpRow, const uint warpCol,
                const uint threadRowInWarp, const uint threadColInWarp) {
  for (uint dotIdx = 0; dotIdx < BK; dotIdx += 4) {
    // execute warptile matmul
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {

        float A_elem = As[(dotIdx * BM) + (warpRow * WM) + (wSubRowIdx * WSUBM) + (threadRowInWarp * BN) + threadColInWarp];
        float B_elem = Bs[(dotIdx * BN) + (warpCol * WN) + (wSubColIdx * WSUBN) + (threadRowInWarp * BN) + threadColInWarp];
        
        float_vec4 acc = {0.0f};
        acc = __builtin_amdgcn_mfma_f32_16x16x4f32(A_elem, B_elem, acc, 0, 0, 0);

        threadResults[(wSubRowIdx) * (WNITER * TM) + (wSubColIdx * TM)] += acc[0];
        threadResults[(wSubRowIdx) * (WNITER * TM) + (wSubColIdx * TM) + 1] += acc[1];
        threadResults[(wSubRowIdx) * (WNITER * TM) + (wSubColIdx * TM) + 2] += acc[2];
        threadResults[(wSubRowIdx) * (WNITER * TM) + (wSubColIdx * TM) + 3] += acc[3];
      }
    }
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
extern "C" __global__ void matmul_kernel(int M, int N, int K, float *A, float *B, float *C, float alpha, float beta) {

    const int BM = 128;
    const int BN = 128;
    const int BK = 16;
    const int WM = 64;
    const int WN = 64;
    const int WNITER = 4;
    const int TM = 4;
    const int TN = 1;

    const int NUM_THREADS = 256;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    const int C_row = blockIdx.y * BM;
    const int C_col = blockIdx.x * BN;

    // Placement of the warp in the threadblock tile
    const uint warpIdx = threadIdx.x / WARPSIZE; // the warp this thread is in
    const uint warpCol = warpIdx % (BN / WN); 
    const uint warpRow = warpIdx / (BN / WN);

    // size of the warp subtile
    constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr uint WSUBM = WM / WMITER; // 64/4=16
    constexpr uint WSUBN = WN / WNITER; // 64/4=16

    // Placement of the thread in the warp subtile
    const uint threadIdxInWarp = threadIdx.x % WARPSIZE;         // [0, 63]
    const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN); // i%(16/1)
    const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN); // i/(16/1)

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
    // we cache into registers on the warptile level
    float regM[WMITER * TM] = {0.0};
    float regN[WNITER * TN] = {0.0};

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
                            TN>(regM, regN, threadResults, As, Bs, warpRow, warpCol,
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