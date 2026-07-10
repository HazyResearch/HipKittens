/**
 * @file gemm_segment.cpp
 * @brief Rung 5 -- multi-segment LDS for gfx1250.
 *
 * Diff vs `gemm_split_bar`: place A in LDS segment 0 and B in segment 1 so
 * the hardware's two read ports (256 B/cycle each) can serve A and B reads
 * in parallel. Uses `shared_allocator::allocate_in<segment<IDX>>`.
 */

#include "common.h"

using namespace kittens;
using namespace gfx1250_gemm;

__global__ __launch_bounds__(NUM_THREADS, 1)
void gemm_segment_kernel(const gemm_globals g, int M, int N, int K)
{
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al(reinterpret_cast<int*>(&__shm[0]));

    A_tile(&A_st)[2] = al.allocate_in<segment<0>, A_tile, 2>();
    B_tile(&B_st)[2] = al.allocate_in<segment<1>, B_tile, 2>();

    rt_fl<WARP_M, WARP_N, col_l, rt_16x16_s> C_acc;
    zero(C_acc);

    const int tile_m  = blockIdx.x;
    const int tile_n  = blockIdx.y;
    const int wid     = warpid();
    const int warp_r  = wid / WARPS_N;
    const int warp_c  = wid % WARPS_N;
    const int k_iters = K / K_STEP;

    kittens::load_async<NUM_THREADS>(A_st[0], g.a, {0, 0, tile_m, 0}, K);
    kittens::load_async<NUM_THREADS>(B_st[0], g.b, {0, 0, tile_n, 0}, K);
    kittens::sync::wait_async();
    kittens::sync::arrive(); kittens::sync::wait();

    for (int k = 0; k < k_iters; ++k) {
        const int cur = k & 1, nxt = 1 - cur;

        if (k + 1 < k_iters) {
            kittens::load_async<NUM_THREADS>(A_st[nxt], g.a, {0, 0, tile_m, k + 1}, K);
            kittens::load_async<NUM_THREADS>(B_st[nxt], g.b, {0, 0, tile_n, k + 1}, K);
        }
        kittens::sync::arrive();

        rt_bf<WARP_M, K_STEP, row_l, rt_16x32_s> A_reg;
        rt_bf<WARP_N, K_STEP, row_l, rt_16x32_s> B_reg;
        kittens::load(A_reg, A_st[cur], warp_r * WARP_M * K_STEP);
        kittens::load(B_reg, B_st[cur], warp_c * WARP_N * K_STEP);

        kittens::sync::wait();
        kittens::sync::wait_ds();
        mma_ABt(C_acc, A_reg, B_reg, C_acc);

        kittens::sync::wait_async();
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
    // A occupies segment 0 (0..64KB); B occupies segment 1 (64KB..). We
    // request `MAX_SHARED_MEMORY_PER_SEGMENT + B_size` so the launch covers both segments.
    const size_t mem_size = MAX_SHARED_MEMORY_PER_SEGMENT + 2 * sizeof(B_tile);
    hipFuncSetAttribute(reinterpret_cast<const void*>(gemm_segment_kernel),
                        hipFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(mem_size));
    gemm_segment_kernel<<<g.grid(), g.block(), mem_size, g.stream>>>(g, g.M(), g.N(), g.K());
}

#include "harness.h"
