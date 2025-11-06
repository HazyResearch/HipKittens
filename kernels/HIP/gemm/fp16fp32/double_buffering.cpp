#include <hip_fp16.h>
namespace wt {
template <const int BM, const int BN, const int BK, const int rowStrideA,
          const int rowStrideB>
__device__ void loadFromGmem(int N, int K, const __half *A, const __half *B,
                             __half *As, __half *Bs, int innerRowA, int innerColA,
                             int innerRowB, int innerColB) {
  for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
    const __half2 tmp1 = reinterpret_cast<const __half2 *>(
        &A[(innerRowA + offset) * K + innerColA * 4])[0];
    const __half2 tmp2 = reinterpret_cast<const __half2 *>(
        &A[(innerRowA + offset) * K + innerColA * 4 + 2])[0];
    // float4 tmp;
    // asm("ld.global.nc.v4.f32 {%0, %1, %2, %3}, [%4];"
    //     : "=f"(tmp.x), "=f"(tmp.y), "=f"(tmp.z), "=f"(tmp.w)
    //     : "l"(&A[(innerRowA + offset) * K + innerColA * 4]));
    As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp1.x;
    As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp1.y;
    As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp2.x;
    As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp2.y;
  }

  for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
    reinterpret_cast<__half2 *>(
        &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] = 
        reinterpret_cast<const __half2 *>(
            &B[(innerRowB + offset) * N + innerColB * 4])[0];
    reinterpret_cast<__half2 *>(
        &Bs[(innerRowB + offset) * BN + innerColB * 4 + 2])[0] = 
        reinterpret_cast<const __half2 *>(
            &B[(innerRowB + offset) * N + innerColB * 4 + 2])[0];
    // asm("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
    //     : "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 0]),
    //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 1]),
    //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 2]),
    //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 3])
    //     : "l"(&B[(innerRowB + offset) * N + innerColB * 4]));
  }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
          const int TM, const int TN>
__device__ void
processFromSmem(__half *regM, __half *regN, float *threadResults, const __half *As,
                const __half *Bs, const uint warpRow, const uint warpCol,
                const uint threadRowInWarp, const uint threadColInWarp) {
  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
    // populate registers for whole warptile
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint i = 0; i < TM; ++i) {
        regM[wSubRowIdx * TM + i] =
            As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
               threadRowInWarp * TM + i];
      }
    }
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      for (uint i = 0; i < TN; ++i) {
        regN[wSubColIdx * TN + i] =
            Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
               threadColInWarp * TN + i];
      }
    }

    // execute warptile matmul
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        // calculate per-thread results
        for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
          for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                          (wSubColIdx * TN) + resIdxN] +=
                __half2float(regM[wSubRowIdx * TM + resIdxM]) *
                __half2float(regN[wSubColIdx * TN + resIdxN]);
          }
        }
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
extern "C" __global__ void matmul_kernel(int M, int N, int K, __half *A, __half *B, float *C, float alpha, float beta) {

    const int BM = 128;
    const int BN = 128;
    const int BK = 16;
    const int WM = 64;
    const int WN = 64;
    const int WNITER = 4;
    const int TM = 1;
    const int TN = 4;

    const int NUM_THREADS = 256;

    __shared__ __half As[2 * BM * BK];
    __shared__ __half Bs[2 * BK * BN];

    const int C_row = blockIdx.y * BM;
    const int C_col = blockIdx.x * BN;

    // Placement of the warp in the threadblock tile
    const uint warpIdx = threadIdx.x / WARPSIZE; // the warp this thread is in
    const uint warpCol = warpIdx % (BN / WN); 
    const uint warpRow = warpIdx / (BN / WN);

    // size of the warp subtile
    constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr uint WSUBM = WM / WMITER; // 64/1=64
    constexpr uint WSUBN = WN / WNITER; // 64/1=64

    // Placement of the thread in the warp subtile
    const uint threadIdxInWarp = threadIdx.x % WARPSIZE;         // [0, 63]
    const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN); // i%(64/8)
    const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN); // i/(64/8)

    // setup double buffering split
    bool doubleBufferIdx = threadIdx.x >= (NUM_THREADS / 2);

    // Move blocktile to beginning of A's row and B's column
    A += C_row * K;
    B += C_col;
    // Move C_ptr to warp's output tile
    C += (C_row + warpRow * WM) * N + C_col + warpCol * WN;

    // calculating the indices that this thread will load into SMEM
    // we'll load 128bit / 32bit = 4 elements per thread at each step
    const uint innerRowA = (threadIdx.x % (NUM_THREADS / 2)) / (BK / 4);
    const uint innerColA = (threadIdx.x % (NUM_THREADS / 2)) % (BK / 4);
    constexpr uint rowStrideA = ((NUM_THREADS / 2) * 4) / BK;
    const uint innerRowB = (threadIdx.x % (NUM_THREADS / 2)) / (BN / 4);
    const uint innerColB = (threadIdx.x % (NUM_THREADS / 2)) % (BN / 4);
    constexpr uint rowStrideB = (NUM_THREADS / 2) / (BN / 4);
    
    // allocate thread-local cache for results in registerfile
    float threadResults[WMITER * TM * WNITER * TN] = {0.0};
    // we cache into registers on the warptile level
    __half regM[WMITER * TM] = {0.0};
    __half regN[WNITER * TN] = {0.0};

    if (doubleBufferIdx == 0) {
      // load first (B0)
      wt::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
          N, K, A, B, As, Bs, innerRowA, innerColA, innerRowB, innerColB);
    }
    __syncthreads();

    // outer-most loop over block tiles
    for (uint bkIdx = 0; bkIdx < K; bkIdx += 2 * BK) {
        
        if (doubleBufferIdx == 0) {
          // process current (B0)
          wt::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM,
                              TN>(regM, regN, threadResults, As, Bs, warpRow, warpCol,
                                  threadRowInWarp, threadColInWarp);
          __syncthreads();


          // process current+1 (B1)
          if (bkIdx + BK < K) {
            wt::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN,
                                TM, TN>(regM, regN, threadResults, As + (BM * BK),
                                        Bs + (BK * BN), warpRow, warpCol,
                                        threadRowInWarp, threadColInWarp);
          }
          __syncthreads();

          // load current + 2 (B0)
          if (bkIdx + 2 * BK < K) {
            wt::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
                N, K, A + 2 * BK, B + 2 * BK * N, As, Bs, innerRowA, innerColA,
                innerRowB, innerColB);
          }
        } else {
          // load current + 1 (B1)
          if (bkIdx + BK < K) {
            wt::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
                N, K, A + BK, B + BK * N, As + (BM * BK), Bs + (BK * BN), innerRowA,
                innerColA, innerRowB, innerColB);
          }
          __syncthreads();

          // process current (B0)
          wt::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM,
                              TN>(regM, regN, threadResults, As, Bs, warpRow,
                                  warpCol, threadRowInWarp, threadColInWarp);
          __syncthreads();

          // process current+1 (B1)
          if (bkIdx + BK < K) {
            wt::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN,
                                TM, TN>(regM, regN, threadResults, As + (BM * BK),
                                        Bs + (BK * BN), warpRow, warpCol,
                                        threadRowInWarp, threadColInWarp);
          }
        }

        A += 2 * BK;     // move BK columns to right
        B += 2 * BK * N; // move BK rows down
        __syncthreads();

    }

    // write out the results
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        // move C pointer to current warp subtile
        float *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
        for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
            for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
            // load C vector into registers
            float4 tmp = reinterpret_cast<float4 *>(
                &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                            threadColInWarp * TN + resIdxN])[0];
            // perform GEMM update in reg
            const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                            wSubColIdx * TN + resIdxN;
            tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
            tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
            tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
            tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
            // write back
            reinterpret_cast<float4 *>(
                &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                            threadColInWarp * TN + resIdxN])[0] = tmp;
            }
        }
        }
    }
}