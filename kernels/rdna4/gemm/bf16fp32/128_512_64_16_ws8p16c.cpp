/**
 * @file 128_512_64_16_ws8p16c.cpp
 * @brief Warp-specialised bf16 -> fp32 GEMM for gfx1201 (RX 9070 XT).
 *
 * Tile 128x512, K_STEP 64, k-substep 16. ONE 24-wave workgroup owns an entire
 * WGP: 8 PRODUCER waves own the global->LDS path for A and run no WMMA, and 16
 * CONSUMER waves read A from LDS, pull B straight to registers, and do all the
 * arithmetic. The producer runs one tile AHEAD, so the block barrier orders
 * buffer reuse rather than waiting for data to be made.
 */

// Holds one VGPR live between the WMMA operand loads in kittens::wmma161616_acc
// (see include/rdna4/ops/warp/register/tile/mma.cuh). It changes only register
// assignment; this kernel was measured and verified with it set.
#define WMMA_BANK_PAD 1

namespace gfx1201_gemm {
constexpr int BLOCK_M = 128, BLOCK_N = 512, K_STEP = 64;   // block tile
constexpr int WARPS_M = 2,   WARPS_N = 8;                  // consumer-wave grid
}
#include "common.h"

using namespace kittens;
using namespace gfx1201_gemm;

constexpr int PROD_WAVES     = 8;                          // producer waves
constexpr int LAUNCH_THREADS = (NUM_WARPS + PROD_WAVES) * kittens::WARP_THREADS;
constexpr int KSUB           = 16;                         // consumer k-substep
constexpr int K_STAGGER      = 32;                         // see stagger below


/* ===================== WARP-SPECIALISED GEMM =============================
 *
 * The first PROD_WAVES waves are PRODUCERS: they own the entire global->LDS
 * path for the A tile and run no WMMA. The remaining NUM_WARPS waves are
 * CONSUMERS: they read A from LDS, pull B straight to VGPRs, and do all the
 * arithmetic. Moving the A-load bookkeeping (addresses, staging registers) out
 * of the waves that issue WMMAs is the point of the split.
 *
 * Producers MUST be the low waves. stage()/commit_fast() index off threadIdx.x
 * directly, so a producer group at the top of the block would compute tile
 * offsets past the end of its own cooperative range.
 *
 * SYNCHRONISATION. One ordinary block barrier per K iteration; both groups
 * arrive once and wait once, so the counts match. The producer commits tile
 * kk+1 into the buffer the consumers are not reading, and only rewrites
 * buf[kk&1] after barrier kk+1, by which point the consumers' iteration-kk
 * ds_loads have been consumed by WMMA and are complete.
 */
template <bool FAST>
__global__ __launch_bounds__(LAUNCH_THREADS, 1)
void gemm_ws_kernel(const gemm_globals g, int M, int N, int K)
{
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al(reinterpret_cast<int*>(&__shm[0]));
    A_tile& A_st0 = al.allocate<A_tile>();
    A_tile& A_st1 = al.allocate<A_tile>();
    constexpr int NBUF = 2;

    constexpr int PROD_THREADS = PROD_WAVES * kittens::WARP_THREADS;
    constexpr int NKH  = K_STEP / KSUB;
    constexpr int AH = WARP_M / 16, AW = WARP_N / 16;

    // GROUP_M rasterisation: consecutive workgroups walk a GROUP_M-tall column
    // of output tiles, so they share B (and A) in L2.
    const int num_pid_m = gridDim.x;
    const int num_pid_n = gridDim.y;
    const int pid       = blockIdx.x + blockIdx.y * gridDim.x;
    const int num_pid_in_group = GROUP_M * num_pid_n;
    const int group_id         = pid / num_pid_in_group;
    const int first_pid_m      = group_id * GROUP_M;
    const int group_size_m     = min(num_pid_m - first_pid_m, GROUP_M);
    const int tile_m = first_pid_m + (pid % group_size_m);
    const int tile_n = (pid % num_pid_in_group) / group_size_m;
    const int k_iters = (K + K_STEP - 1) / K_STEP;
    // Each workgroup starts its K walk at a different offset, so resident
    // workgroups do not all read the same K column of A and B at once.
    const int stagger = (pid % K_STAGGER) % k_iters;

    const int wid = warpid();
    int k = stagger;

    if (wid < PROD_WAVES) {
        /* ---------------- PRODUCER: one tile ahead ---------------- */
        // Tile 0 is committed before the loop; iteration kk commits tile kk+1
        // into the OTHER buffer while consumers compute tile kk.
        kittens::gts_staged<PROD_THREADS, kittens::bf16, BLOCK_M, K_STEP> stg;
        kittens::stage<PROD_THREADS>(stg, A_st0, g.a, {0, 0, tile_m, k}, K, K, M);
        {
            if constexpr (FAST) kittens::commit_fast<PROD_THREADS>(A_st0, stg);
            else if (stg.valid) kittens::commit_fast<PROD_THREADS>(A_st0, stg);
            else kittens::load<PROD_THREADS>(A_st0, g.a, {0,0,tile_m,k}, K, K, M);
            kittens::sync::wait_ds();
        }
        for (int kk = 0; kk < k_iters; ++kk) {
            int knext = k + 1; if (knext == k_iters) knext = 0;
            if (kk + 1 < k_iters)
                kittens::stage<PROD_THREADS>(stg, A_st0, g.a,
                                             {0, 0, tile_m, knext}, K, K, M);
            kittens::sync::arrive();
            kittens::sync::wait();
            if (kk + 1 < k_iters) {
                A_tile& A_nx = ((kk + 1) & 1) ? A_st1 : A_st0;
                if constexpr (FAST) kittens::commit_fast<PROD_THREADS>(A_nx, stg);
                else if (stg.valid) kittens::commit_fast<PROD_THREADS>(A_nx, stg);
                else kittens::load<PROD_THREADS>(A_nx, g.a, {0,0,tile_m,knext}, K, K, M);
                kittens::sync::wait_ds();
            }
            k = knext;
        }
        return;                       // producers write no C
    }

    /* ---------------- CONSUMER ---------------- */
    const int cwid   = wid - PROD_WAVES;
    const int warp_r = cwid / WARPS_N;
    const int warp_c = cwid % WARPS_N;

    float2 lacc[AH][AW][4];
    #pragma unroll
    for (int n = 0; n < AH; ++n)
      #pragma unroll
      for (int m = 0; m < AW; ++m)
        #pragma unroll
        for (int i = 0; i < 4; ++i) { lacc[n][m][i].x = 0.f; lacc[n][m][i].y = 0.f; }

    for (int ki = 0; ki < k_iters; ki += NBUF) {
      #pragma unroll
      for (int half = 0; half < NBUF; ++half) {
        const int kk = ki + half;
        if (kk >= k_iters) break;
        int knext = k + 1; if (knext == k_iters) knext = 0;
        A_tile& A_st = (kk & 1) ? A_st1 : A_st0;

        kittens::sync::arrive();
        rt_bf<WARP_N, K_STEP, row_l, rt_16x16_s> B_reg;
        kittens::load_direct<WARP_N, K_STEP, !FAST, FAST>(
            B_reg, g.b, tile_n * BLOCK_N + warp_c * WARP_N, k * K_STEP, N, K);
        kittens::sync::wait();

        #pragma unroll
        for (int kh = 0; kh < NKH; ++kh) {
            rt_bf<WARP_M, KSUB, row_l, rt_16x16_s> A_sub;
            kittens::load(A_sub, A_st, warp_r * WARP_M * K_STEP, kh * KSUB);
            #pragma unroll
            for (int n = 0; n < AH; ++n)
              #pragma unroll
              for (int m = 0; m < AW; ++m)
                #pragma unroll
                for (int kq = 0; kq < KSUB / 16; ++kq)
                  kittens::wmma161616_acc(lacc[n][m],
                                          A_sub.tiles[n][kq].data,
                                          B_reg.tiles[m][kh * (KSUB / 16) + kq].data);
        }
        k = knext;
      }
    }

    bf16* c_base = reinterpret_cast<bf16*>(&g.c[{0, 0, 0, 0}]);
    store_acc_direct<AH, AW>(
        c_base,
        tile_m * BLOCK_M + warp_r * WARP_M,
        tile_n * BLOCK_N + warp_c * WARP_N,
        N, M, lacc);
}

void dispatch(gemm_globals g)
{
    const size_t mem_size = 2 * sizeof(gfx1201_gemm::A_tile);
    hipFuncSetAttribute(reinterpret_cast<const void*>(gemm_ws_kernel<true>),
                        hipFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(mem_size));
    const bool fast = (g.M() % gfx1201_gemm::BLOCK_M == 0) &&
                      (g.K() % gfx1201_gemm::K_STEP  == 0);
    if (fast)
        gemm_ws_kernel<true><<<g.grid(), dim3(LAUNCH_THREADS), mem_size, g.stream>>>(g, g.M(), g.N(), g.K());
    else
        gemm_ws_kernel<false><<<g.grid(), dim3(LAUNCH_THREADS), mem_size, g.stream>>>(g, g.M(), g.N(), g.K());
}

#include "harness.h"
