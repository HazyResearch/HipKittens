/**
 * @file 00_gemm_naive.cpp
 * @brief Rung 00 -- the naive baseline (no parent).
 *
 * Kernel Specification
 *   layout      TN -- a is [M, K], b is [N, K], both K-contiguous; c is [M, N] column-major
 *   tile        64x64 macro, 32x32 per warp, 2x2 warps; BLOCK_K 32 = 1 x K_STEP 32
 *   occupancy   4 warps / 128 threads / 1 wave per SIMD; 10 workgroups per CU, register-bound
 *   registers   84 VGPR; 32 are accumulator (WARP_M*WARP_N/32), 40 SGPR
 *   spills      none: 0 VGPR, 0 SGPR, 0 scratch
 *   LDS         1 stage x 8.5 KB = 8.5 KB of 320 KB (2.7%)
 *   sync        per K-block: 2 barriers (full), 2 LDS drains, 1 global-load drain
 *   intensity   32 FLOP per byte of global operand traffic, BM*BN/(BM+BN)
 *
 * One LDS slab and nothing overlapped. A K-block's fill cannot run under the previous block's
 * compute, and every iteration needs two barriers: one to publish the slab, one to establish
 * that every warp has finished reading it before the next fill overwrites it. The correctness
 * baseline: the smallest kernel here that computes the right answer. The existing HK `load`
 * performs global (GL) -> register tile (RT), then `store` performs RT -> shared (ST).
 * Uses only:
 *   - `kittens::load(rt,gl)`      : existing GL -> RT load.
 *   - `kittens::store(st,rt)`     : existing RT -> ST operation.
 *   - `kittens::sync::wait_load`  : drain GL -> RT before consuming the register tile.
 *   - `kittens::sync::wait_ds`    : drain this warp's reads before the slab is refilled.
 *   - `kittens::sync::sync`       : block-wide barrier (-1). Orders execution, not memory.
 *   - `kittens::load(rt,st,off)`  : shared -> register load (wide `ds_load_b128`).
 *   - `kittens::mma_ABt`          : 16x16x32 WMMA via the bf16 builtin.
 *   - `kittens::store(gl,rt,idx)` : direct column-major epilogue.
 */

#include "kittens.cuh"

constexpr int BLOCK_M     = 64;
constexpr int BLOCK_N     = 64;
constexpr int BLOCK_K     = 32;
constexpr int K_STEP      = 32;
constexpr int WARPS_M     = 2;
constexpr int WARPS_N     = 2;
constexpr int WARP_M      = BLOCK_M / WARPS_M;
constexpr int WARP_N      = BLOCK_N / WARPS_N;
constexpr int NUM_WARPS   = WARPS_M * WARPS_N;
constexpr int NUM_THREADS = NUM_WARPS * kittens::WARP_THREADS;
constexpr int K_SUBBLOCKS = BLOCK_K / K_STEP;

#include "common.h"

using namespace kittens;
using namespace gfx1250_gemm;

namespace {
constexpr int FILL_ROWS = 16;
using fill_rt_e = rt_e<FILL_ROWS, K_STEP>;
using fill_st_e = st_e<FILL_ROWS, K_STEP>;
static_assert(BLOCK_M == NUM_WARPS * FILL_ROWS && BLOCK_N == NUM_WARPS * FILL_ROWS);
} // namespace

__global__ __launch_bounds__(NUM_THREADS, 1)
void gemm_00_naive_kernel(const gemm_globals g, int M, int N, int K)
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
        /* One 16x32 slice per warp. The slice reference starts at the parent tile's padded
         * physical origin; the existing store owns every element address within that slice. */
        fill_rt_e fill_rt;
        fill_st_e& A_fill_st = *reinterpret_cast<fill_st_e*>(
            &A_st.data[A_tile::idx(wid * FILL_ROWS, 0)]);
        kittens::load(
            fill_rt, g.a, {0, 0, tile_m * (BLOCK_M / FILL_ROWS) + wid, kb}); // GL -> RT
        kittens::sync::wait_load<0>();
        kittens::store(A_fill_st, fill_rt);                           // RT -> ST

        fill_st_e& B_fill_st = *reinterpret_cast<fill_st_e*>(
            &B_st.data[B_tile::idx(wid * FILL_ROWS, 0)]);
        kittens::load(
            fill_rt, g.b, {0, 0, tile_n * (BLOCK_N / FILL_ROWS) + wid, kb}); // GL -> RT
        kittens::sync::wait_load<0>();
        kittens::store(B_fill_st, fill_rt);                           // RT -> ST

        kittens::sync::wait_ds<0>();              // wait for data (LDS): RT -> ST writes landed
        kittens::sync::sync();                    // wait for everyone (workgroup): the slab is readable

        kittens::load(A_reg, A_st, warp_off_a);
        kittens::load(B_reg, B_st, warp_off_b);
        kittens::sync::wait_ds<0>();              // wait for data (LDS): ST -> RT reads landed
        mma_ABt(C_acc, A_reg, B_reg, C_acc);

        kittens::sync::sync();                    // wait for everyone (workgroup): safe to refill it
    }

    /* Epilogue: each warp converts its accumulator to bf16 straight into global C. Column-major C
     * makes a lane's run of consecutive rows unit-stride, so store WIDTH is fine; what the direct
     * store gives up is transaction SIZE -- a warp's 32 lanes span 16 columns, so the writes
     * scatter per column. Rung 10 stages through LDS so the block's rows leave as one stream. */
    kittens::store(g.c, C_acc, {0, 0, tile_m * WARPS_M + warp_r, tile_n * WARPS_N + warp_c});
}

void dispatch(gemm_globals g, const launch_config& launch)
{
    const size_t mem_size = launch.dynamic_shared_memory<1>();

    /* The direct store writes a lane's run of consecutive rows as one sized buffer store, so the
     * run has to be aligned, which needs the leading dimension to divide it. Kept at 8 rather than
     * loosened to the run length: every tested shape satisfies it and it stays sufficient if a
     * wider accumulator shape lengthens the run. */
    if (g.c.rows() % 8 != 0) {
        std::fprintf(stderr,
            "00_gemm_naive: column-major C requires M %% 8 == 0 (got M=%d)\n", g.c.rows());
        std::abort();
    }

    gfx1250_gemm::require_k_blocks(g.K(), "00_gemm_naive");

    hipFuncSetAttribute(reinterpret_cast<const void*>(gemm_00_naive_kernel),
                        hipFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(mem_size));
    gemm_00_naive_kernel<<<launch.grid, launch.block, mem_size, launch.stream>>>(
        g, g.M(), g.N(), g.K());
}

#include "harness.h"
