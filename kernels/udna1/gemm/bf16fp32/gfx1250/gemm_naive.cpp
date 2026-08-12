/**
 * @file gemm_naive.cpp
 * @brief Rung 1 -- the naive baseline the rest of the ladder is built on.
 *
 * Kernel Specification
 *   tile        64x64 macro, 32x32 per warp, 2x2 warps; BLOCK_K 32 = 1 x K_STEP 32
 *   occupancy   4 warps / 128 threads / 1 wave per SIMD; 12 workgroups per CU, register-bound
 *   registers   68 VGPR; 32 are accumulator (WARP_M*WARP_N/32), 32 SGPR
 *   spills      none: 0 VGPR, 0 SGPR, 0 scratch
 *   LDS         1 stage x 8.5 KB = 8.5 KB of 320 KB (2.7%)
 *   sync        per K-block: 2 barriers (full), 2 LDS drains, 2 global-load drains
 *   intensity   32 FLOP per byte of global operand traffic, BM*BN/(BM+BN)
 *
 * One LDS slab and nothing overlapped. A K-block's fill cannot run under the previous block's
 * compute, and every iteration needs two barriers: one to publish the slab, one to establish
 * that every warp has finished reading it before the next fill overwrites it. The correctness
 * baseline: the smallest kernel here that computes the right answer. Uses only:
 *   - `kittens::load(st,gl,idx)`  : register-mediated global -> LDS copy.
 *   - `kittens::sync::fence`      : drain global-load and LDS traffic before each barrier.
 *   - `kittens::sync::sync`       : block-wide barrier (-1). Orders execution, not memory.
 *   - `kittens::load(rt,st,off)`  : shared -> register load (wide `ds_load_b128`).
 *   - `kittens::mma_ABt`          : 16x16x32 WMMA via the bf16 builtin.
 *   - `kittens::store(gl,rt,idx)` : direct column-major epilogue.
 */

#ifndef GFX1250_BLOCK_M
#define GFX1250_BLOCK_M 64
#endif
#ifndef GFX1250_BLOCK_N
#define GFX1250_BLOCK_N 64
#endif
#ifndef GFX1250_BLOCK_K
#define GFX1250_BLOCK_K 32
#endif
#define GFX1250_K_STEP  32
#ifndef GFX1250_WARPS_M
#define GFX1250_WARPS_M 2
#endif
#ifndef GFX1250_WARPS_N
#define GFX1250_WARPS_N 2
#endif

#include "common.h"

using namespace kittens;
using namespace gfx1250_gemm;

__global__ __launch_bounds__(NUM_THREADS, 1)
void gemm_naive_kernel(const gemm_globals g, int M, int N, int K)
{
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al(reinterpret_cast<int*>(&__shm[0]));

    A_tile& A_st = al.allocate<A_tile>();
    B_tile& B_st = al.allocate<B_tile>();

    rt_fl<WARP_M, WARP_N, col_l, rt_16x16_s> C_acc;
    zero(C_acc);

    const int tile_m   = blockIdx.x;
    const int tile_n   = blockIdx.y;
    const int wid      = warpid();
    const int warp_r   = wid / WARPS_N;
    const int warp_c   = wid % WARPS_N;
    const int k_blocks = K / BLOCK_K;
    if (k_blocks <= 0) return;                    // K shorter than one block: nothing to compute

    const int warp_off_a = warp_r * WARP_M * K_STEP;
    const int warp_off_b = warp_c * WARP_N * K_STEP;

    rt_e<WARP_M, K_STEP> A_reg;
    rt_e<WARP_N, K_STEP> B_reg;

    // Main loop: one K-block per iteration.
    for (int kb = 0; kb < k_blocks; ++kb) {
        // Every thread participates: `load` spreads the tile across all NUM_THREADS lanes.
        kittens::load<NUM_THREADS>(A_st, g.a, {0, 0, tile_m, kb}, K);
        kittens::load<NUM_THREADS>(B_st, g.b, {0, 0, tile_n, kb}, K);

        kittens::sync::fence();                   // RAW: publish the slab
        kittens::sync::sync();

        kittens::load(A_reg, A_st, warp_off_a);
        kittens::load(B_reg, B_st, warp_off_b);
        mma_ABt(C_acc, A_reg, B_reg, C_acc);

        kittens::sync::fence();                   // WAR: nobody is still reading the slab
        kittens::sync::sync();
    }

    /* Epilogue: each warp converts its accumulator to bf16 straight into global C. Column-major C
     * makes a lane's run of consecutive rows unit-stride, so store WIDTH is fine; what the direct
     * store gives up is transaction SIZE -- a warp's 32 lanes span 16 columns, so the writes
     * scatter per column. Rung 11 stages through LDS so the block's rows leave as one stream. */
    kittens::store(g.c, C_acc, {0, 0, tile_m * WARPS_M + warp_r, tile_n * WARPS_N + warp_c});
}

void dispatch(gemm_globals g)
{
    const size_t mem_size = g.dynamic_shared_memory<1>();

    /* The direct store writes a lane's run of consecutive rows as one sized buffer store, so the
     * run has to be aligned, which needs the leading dimension to divide it. Kept at 8 rather than
     * loosened to the run length: every tested shape satisfies it and it stays sufficient if a
     * wider accumulator shape lengthens the run. */
    if (g.c.rows() % 8 != 0) {
        std::fprintf(stderr,
            "gemm_naive: column-major C requires M %% 8 == 0 (got M=%d)\n", g.c.rows());
        std::abort();
    }

    gfx1250_gemm::require_k_blocks(g.K(), "gemm_naive");

    hipFuncSetAttribute(reinterpret_cast<const void*>(gemm_naive_kernel),
                        hipFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(mem_size));
    gemm_naive_kernel<<<g.grid(), g.block(), mem_size, g.stream>>>(g, g.M(), g.N(), g.K());
}

#include "harness.h"
