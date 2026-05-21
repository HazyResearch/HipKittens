/**
 * @file gemm_expert.cpp
 * @brief Rung 7 -- expert-scheduled bf16 -> fp32 GEMM with reuse-B WMMA burst.
 *
 * Diff vs `gemm_segment`: enable expert SCHED_MODE for the K-loop scope via
 * `kittens::sched::expert` (RAII) and replace `mma_ABt` with
 * `mma_ABt_burst<4>` -- four `__builtin_amdgcn_wmma_f32_16x16x32_bf16` calls
 * with `matrix_b_reuse=true` on the 2nd..4th to engage the hardware's
 * operand-reuse cache. The LLVM `SIInsertHardClauses` pass automatically
 * clauses the WMMA burst.
 */

#include "common.h"

using namespace kittens;
using namespace gfx1250_gemm;

using Pad = lds_pad_default;
constexpr int A_ELEMS_PAD = Pad::padded_elems(BLOCK_M * K_STEP);
constexpr int B_ELEMS_PAD = Pad::padded_elems(BLOCK_N * K_STEP);

__global__ __launch_bounds__(NUM_THREADS, 1)
void gemm_expert_kernel(const gemm_globals g, int M, int N, int K)
{
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al(reinterpret_cast<int*>(&__shm[0]));

    bf16(&A_lds)[2][A_ELEMS_PAD] = al.allocate_in<segment<0>, bf16, 2, A_ELEMS_PAD>();
    bf16(&B_lds)[2][B_ELEMS_PAD] = al.allocate_in<segment<1>, bf16, 2, B_ELEMS_PAD>();

    rt_fl<WARP_M, WARP_N, col_l, rt_16x16_s> C_acc;
    zero(C_acc);

    const int tile_m  = blockIdx.x;
    const int tile_n  = blockIdx.y;
    const int wid     = warpid();
    const int warp_r  = wid / WARPS_N;
    const int warp_c  = wid % WARPS_N;
    const int k_iters = K / K_STEP;

    kittens::sched::expert _sched;     // limited expert mode, off in dtor

    kittens::async::load<Pad, BLOCK_M, K_STEP, NUM_THREADS>(
        A_lds[0], g.a, {0, 0, tile_m, 0}, K);
    kittens::async::load<Pad, BLOCK_N, K_STEP, NUM_THREADS>(
        B_lds[0], g.b, {0, 0, tile_n, 0}, K);
    kittens::async::load_wait();
    kittens::sync::arrive(); kittens::sync::wait();

    for (int k = 0; k < k_iters; ++k) {
        const int cur = k & 1, nxt = 1 - cur;

        if (k + 1 < k_iters) {
            kittens::async::load<Pad, BLOCK_M, K_STEP, NUM_THREADS>(
                A_lds[nxt], g.a, {0, 0, tile_m, k + 1}, K);
            kittens::async::load<Pad, BLOCK_N, K_STEP, NUM_THREADS>(
                B_lds[nxt], g.b, {0, 0, tile_n, k + 1}, K);
        }
        kittens::sync::arrive();

        rt_bf<WARP_M, K_STEP, row_l, rt_16x32_s> A_reg;
        rt_bf<WARP_N, K_STEP, row_l, rt_16x32_s> B_reg;
        kittens::load_b128<Pad, WARP_M, K_STEP>(
            A_reg, A_lds[cur] + Pad::padded(warp_r * WARP_M * K_STEP));
        kittens::load_b128<Pad, WARP_N, K_STEP>(
            B_reg, B_lds[cur] + Pad::padded(warp_c * WARP_N * K_STEP));

        kittens::sync::wait();
        kittens::sync::wait_ds();
        mma_ABt_burst(C_acc, A_reg, B_reg, C_acc);

        kittens::async::load_wait();
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
    const size_t mem_size = LDS_SEGMENT_BYTES + 2 * B_ELEMS_PAD * sizeof(bf16);
    hipFuncSetAttribute(reinterpret_cast<const void*>(gemm_expert_kernel),
                        hipFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(mem_size));
    gemm_expert_kernel<<<g.grid(), g.block(), mem_size, g.stream>>>(g, g.M(), g.N(), g.K());
}

#include "harness.h"
