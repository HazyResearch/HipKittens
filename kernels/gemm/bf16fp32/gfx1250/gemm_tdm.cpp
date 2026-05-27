/**
 * @file gemm_tdm.cpp
 * @brief Rung 8 -- bf16 GEMM using the Tensor Data Mover (TDM) for G->LDS.
 *
 * Diff vs `gemm_expert`: replace cooperative async loads with
 * `tdm::load_async` issued by wave 0 (for A) and wave 1 (for B). Producers
 * drain via `tdm::load_async_wait` against the global `TENSORcnt`.
 *
 * This rung is the "fire-and-drain" TDM path -- no per-transfer semaphore
 * ordering. The companion `gemm_tdm_arrive` rung exercises the explicit
 * `arrive(sem)` + `wait(sem, phase)` pattern.
 *
 * Exercises the N-dimensional TDM API:
 *   - `kittens::tdm::tile<...>`             -- compile-time tile shape.
 *   - `kittens::tdm::load_async<Tile>(...)` -- TDM 2D bulk load.
 *   - `kittens::tdm::load_async_wait<N>()`  -- drain TENSORcnt.
 *
 * Tile: 64x64 output, K_STEP = 32, 4 warps in a 2x2 layout.
 */

#include "common.h"

using namespace kittens;
using namespace gfx1250_gemm;

using Pad = lds_pad_default;
constexpr int A_ELEMS_PAD = Pad::padded_elems(BLOCK_M * K_STEP);
constexpr int B_ELEMS_PAD = Pad::padded_elems(BLOCK_N * K_STEP);

__global__ __launch_bounds__(NUM_THREADS, 1)
void gemm_tdm_kernel(const gemm_globals g, int M, int N, int K)
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

    sched::expert _sched;

    using A_tile = tdm::tile<BLOCK_M, K_STEP>;
    using B_tile = tdm::tile<BLOCK_N, K_STEP>;

    if (wid == 0) {
        tdm::load_async<A_tile, Pad>(
            A_lds[0], g.a, {0, 0, tile_m, 0},
            A_tile::extents{M, K}, A_tile::strides{K});
    }
    if (wid == 1) {
        tdm::load_async<B_tile, Pad>(
            B_lds[0], g.b, {0, 0, tile_n, 0},
            B_tile::extents{N, K}, B_tile::strides{K});
    }
    tdm::load_async_wait();
    sync::sync();

    for (int k = 0; k < k_iters; ++k) {
        const int cur = k & 1, nxt = 1 - cur;

        if (k + 1 < k_iters) {
            if (wid == 0) {
                tdm::load_async<A_tile, Pad>(
                    A_lds[nxt], g.a, {0, 0, tile_m, k + 1},
                    A_tile::extents{M, K}, A_tile::strides{K});
            }
            if (wid == 1) {
                tdm::load_async<B_tile, Pad>(
                    B_lds[nxt], g.b, {0, 0, tile_n, k + 1},
                    B_tile::extents{N, K}, B_tile::strides{K});
            }
        }

        rt_bf<WARP_M, K_STEP, row_l, rt_16x32_s> A_reg;
        rt_bf<WARP_N, K_STEP, row_l, rt_16x32_s> B_reg;
        kittens::load_b128<Pad, WARP_M, K_STEP>(
            A_reg, A_lds[cur] + Pad::padded(warp_r * WARP_M * K_STEP));
        kittens::load_b128<Pad, WARP_N, K_STEP>(
            B_reg, B_lds[cur] + Pad::padded(warp_c * WARP_N * K_STEP));

        sync::wait_ds();
        mma_ABt_burst(C_acc, A_reg, B_reg, C_acc);

        tdm::load_async_wait();
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
    const size_t mem_size = LDS_SEGMENT_BYTES + 2 * B_ELEMS_PAD * sizeof(bf16);
    hipFuncSetAttribute(reinterpret_cast<const void*>(gemm_tdm_kernel),
                        hipFuncAttributeMaxDynamicSharedMemorySize,
                        static_cast<int>(mem_size));
    gemm_tdm_kernel<<<g.grid(), g.block(), mem_size, g.stream>>>(
        g, g.M(), g.N(), g.K());
}

#include "harness.h"
