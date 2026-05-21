/**
 * @file gemm_async.cpp
 * @brief Rung 3 -- async global -> LDS for gfx1250.
 *
 * Diff vs `gemm_double_buf`: replace the register-mediated copy with
 * `kittens::async::load`, which emits `global_load_async_to_lds_b128`
 * (or the cluster multicast variant when given a mask). Producers drain
 * via `kittens::async::load_wait`.
 */

#include "common.h"

using namespace kittens;
using namespace gfx1250_gemm;

__global__ __launch_bounds__(NUM_THREADS, 1)
void gemm_async_kernel(const gemm_globals g, int M, int N, int K)
{
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al(reinterpret_cast<int*>(&__shm[0]));

    constexpr int A_ELEMS = BLOCK_M * K_STEP;
    constexpr int B_ELEMS = BLOCK_N * K_STEP;
    bf16(&A_lds)[2][A_ELEMS] = al.allocate<bf16, 2, A_ELEMS>();
    bf16(&B_lds)[2][B_ELEMS] = al.allocate<bf16, 2, B_ELEMS>();

    rt_fl<WARP_M, WARP_N, col_l, rt_16x16_s> C_acc;
    zero(C_acc);

    const int tile_m  = blockIdx.x;
    const int tile_n  = blockIdx.y;
    const int wid     = warpid();
    const int warp_r  = wid / WARPS_N;
    const int warp_c  = wid % WARPS_N;
    const int k_iters = K / K_STEP;

    kittens::async::load<lds_nopad, BLOCK_M, K_STEP, NUM_THREADS>(
        A_lds[0], g.a, {0, 0, tile_m, 0}, K);
    kittens::async::load<lds_nopad, BLOCK_N, K_STEP, NUM_THREADS>(
        B_lds[0], g.b, {0, 0, tile_n, 0}, K);
    kittens::async::load_wait();
    kittens::sync::sync();

    for (int k = 0; k < k_iters; ++k) {
        const int cur = k & 1, nxt = 1 - cur;

        if (k + 1 < k_iters) {
            kittens::async::load<lds_nopad, BLOCK_M, K_STEP, NUM_THREADS>(
                A_lds[nxt], g.a, {0, 0, tile_m, k + 1}, K);
            kittens::async::load<lds_nopad, BLOCK_N, K_STEP, NUM_THREADS>(
                B_lds[nxt], g.b, {0, 0, tile_n, k + 1}, K);
        }

        rt_bf<WARP_M, K_STEP, row_l, rt_16x32_s> A_reg;
        rt_bf<WARP_N, K_STEP, row_l, rt_16x32_s> B_reg;
        kittens::load_b32<WARP_M, K_STEP>(A_reg, A_lds[cur] + warp_r * WARP_M * K_STEP);
        kittens::load_b32<WARP_N, K_STEP>(B_reg, B_lds[cur] + warp_c * WARP_N * K_STEP);

        kittens::sync::wait_ds();
        mma_ABt(C_acc, A_reg, B_reg, C_acc);

        kittens::async::load_wait();
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
    hipFuncSetAttribute(reinterpret_cast<const void*>(gemm_async_kernel),
                        hipFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(mem_size));
    gemm_async_kernel<<<g.grid(), g.block(), mem_size, g.stream>>>(g, g.M(), g.N(), g.K());
}

#include "harness.h"
