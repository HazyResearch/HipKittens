/**
 * @file gemm_naive.cpp
 * @brief Rung 1 -- 128x128 naive bf16 -> fp32 GEMM for gfx1250.
 *
 * Same algorithm as the 64x64 gfx1250 naive rung, scaled to a 128x128 block
 * with 4x4 warps (512 threads). Each warp still owns a 32x32 output tile.
 */

#include "common.h"

using namespace kittens;
using namespace gfx1250_gemm_128;

__global__ __launch_bounds__(NUM_THREADS, 1)
void gemm_naive_kernel(const gemm_globals g, int M, int N, int K)
{
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al(reinterpret_cast<int*>(&__shm[0]));

    A_tile& A_st = al.allocate<A_tile>();
    B_tile& B_st = al.allocate<B_tile>();

    rt_fl<WARP_M, WARP_N, col_l, rt_16x16_s> C_acc;
    zero(C_acc);

    const int tile_m  = blockIdx.x;
    const int tile_n  = blockIdx.y;
    const int wid     = warpid();
    const int warp_r  = wid / WARPS_N;
    const int warp_c  = wid % WARPS_N;
    const int k_iters = K / K_STEP;

    for (int k = 0; k < k_iters; ++k) {
        load<NUM_THREADS>(A_st, g.a, {0, 0, tile_m, k}, K);
        load<NUM_THREADS>(B_st, g.b, {0, 0, tile_n, k}, K);

        sync::sync();

        rt_bf<WARP_M, K_STEP, row_l, rt_16x32_s> A_reg;
        rt_bf<WARP_N, K_STEP, row_l, rt_16x32_s> B_reg;
        load(A_reg, A_st, warp_r * WARP_M * K_STEP);
        load(B_reg, B_st, warp_c * WARP_N * K_STEP);

        sync::wait_ds();
        mma_ABt(C_acc, A_reg, B_reg, C_acc);

        sync::sync();
    }

    bf16* c_base = reinterpret_cast<bf16*>(&g.c[{0, 0, 0, 0}]);
    store_acc<WARP_M / 16, WARP_N / 16>(
        c_base,
        tile_m * BLOCK_M + warp_r * WARP_M,
        tile_n * BLOCK_N + warp_c * WARP_N,
        N, C_acc);
}

void dispatch(gemm_globals g)
{
    const size_t mem_size = g.dynamic_shared_memory<1>();
    hipFuncSetAttribute(reinterpret_cast<const void*>(gemm_naive_kernel),
                        hipFuncAttributeMaxDynamicSharedMemorySize,
                        static_cast<int>(mem_size));
    gemm_naive_kernel<<<g.grid(), g.block(), mem_size, g.stream>>>(
        g, g.M(), g.N(), g.K());
}

#include "harness.h"
