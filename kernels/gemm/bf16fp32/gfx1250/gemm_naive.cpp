/**
 * @file gemm_naive.cpp
 * @brief Rung 1 -- naive bf16 -> fp32 GEMM for gfx1250.
 *
 * Correctness baseline. Single-buffered LDS, register-mediated global -> LDS
 * copy, narrow `ds_load_b32` shared -> register, plain WMMA via `mma_ABt`.
 * Uses only:
 *   - `kittens::load`          : cooperative register-mediated G -> LDS copy.
 *   - `kittens::sync::sync`    : block-wide split barrier.
 *   - `kittens::sync::wait_ds` : drain LDS reads before WMMA.
 *   - `kittens::load_b32`      : narrow shared -> register load.
 *   - `kittens::mma_ABt`       : 16 x 16 x 32 WMMA via the bf16 builtin.
 */

#include "common.h"

using namespace kittens;
using namespace gfx1250_gemm;

__global__ __launch_bounds__(NUM_THREADS, 1)
void gemm_naive_kernel(const gemm_globals g, int M, int N, int K)
{
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al(reinterpret_cast<int*>(&__shm[0]));

    constexpr int A_ELEMS = BLOCK_M * K_STEP;
    constexpr int B_ELEMS = BLOCK_N * K_STEP;
    bf16(&A_lds)[A_ELEMS] = al.allocate<bf16, A_ELEMS>();
    bf16(&B_lds)[B_ELEMS] = al.allocate<bf16, B_ELEMS>();

    rt_fl<WARP_M, WARP_N, col_l, rt_16x16_s> C_acc;
    zero(C_acc);

    const int tile_m  = blockIdx.x;
    const int tile_n  = blockIdx.y;
    const int wid     = warpid();
    const int warp_r  = wid / WARPS_N;
    const int warp_c  = wid % WARPS_N;
    const int k_iters = K / K_STEP;

    for (int k = 0; k < k_iters; ++k) {
        kittens::load<lds_nopad, BLOCK_M, K_STEP, NUM_THREADS>(
            A_lds, g.a, {0, 0, tile_m, k}, K);
        kittens::load<lds_nopad, BLOCK_N, K_STEP, NUM_THREADS>(
            B_lds, g.b, {0, 0, tile_n, k}, K);

        kittens::sync::sync();

        rt_bf<WARP_M, K_STEP, row_l, rt_16x32_s> A_reg;
        rt_bf<WARP_N, K_STEP, row_l, rt_16x32_s> B_reg;
        kittens::load_b32<WARP_M, K_STEP>(A_reg, A_lds + warp_r * WARP_M * K_STEP);
        kittens::load_b32<WARP_N, K_STEP>(B_reg, B_lds + warp_c * WARP_N * K_STEP);

        kittens::sync::wait_ds();
        mma_ABt(C_acc, A_reg, B_reg, C_acc);

        kittens::sync::sync();
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
    const size_t mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute(reinterpret_cast<const void*>(gemm_naive_kernel),
                        hipFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(mem_size));
    gemm_naive_kernel<<<g.grid(), g.block(), mem_size, g.stream>>>(g, g.M(), g.N(), g.K());
}

#include "harness.h"
