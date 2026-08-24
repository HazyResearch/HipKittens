/**
 * @file 01_gemm_double_buf.cpp
 * @brief Rung 01 -- 00_gemm_naive plus a second LDS stage.
 *
 * Kernel Specification
 *   layout      TN -- a is [M, K], b is [N, K], both K-contiguous; c is [M, N] column-major
 *   tile        64x64 macro, 32x32 per warp, 2x2 warps; BLOCK_K 32 = 1 x K_STEP 32
 *   occupancy   4 warps / 128 threads / 1 wave per SIMD; 10 workgroups per CU, register-bound
 *   registers   82 VGPR; 32 are accumulator (WARP_M*WARP_N/32), 38 SGPR
 *   spills      none: 0 VGPR, 0 SGPR, 0 scratch
 *   LDS         2 stages x 8.5 KB = 17 KB of 320 KB (5.3%)
 *   sync        per K-block: 1 barrier (full), 1 LDS drain, 1 global-load drain
 *   intensity   32 FLOP per byte of global operand traffic, BM*BN/(BM+BN)
 *
 * Two interleaved stages in one segment rather than a single slab, so a K-block's fill runs
 * under the previous block's compute and the publish-then-drain barrier pair collapses to one
 * rendezvous. Same 64x64 tile and the same separate GL -> RT, then RT -> ST path as rung 00.
 *
 * The fill still leads the operand reads here. Nothing else covers its global latency, so it
 * must be in flight across the reads and the matrix op below. Uses only:
 *   - `kittens::load(rt,gl)`      : existing GL -> RT load.
 *   - `kittens::store(st,rt)`     : existing RT -> ST operation.
 *   - `kittens::sync::wait_load`  : drain GL -> RT before consuming the register tile.
 *   - `kittens::sync::sync`       : block-wide barrier (-1).
 *   - `kittens::sync::wait_ds`    : drain the wave's LDS reads before the handoff and before exit.
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

static constexpr int S = 2;                    // stages in the LDS ring

namespace {
constexpr int FILL_ROWS = 16;
using fill_rt_e = rt_e<FILL_ROWS, K_STEP>;
using fill_st_e = st_e<FILL_ROWS, K_STEP>;
static_assert(BLOCK_M == NUM_WARPS * FILL_ROWS && BLOCK_N == NUM_WARPS * FILL_ROWS);
} // namespace

// One ring slot, with A and B adjacent.
struct KITTENS_DEFAULT_ALIGN ab_pair { A_tile a; B_tile b; };
static_assert(sizeof(ab_pair) == sizeof(A_tile) + sizeof(B_tile),
              "an interleaved pair must not introduce padding");

__global__ __launch_bounds__(NUM_THREADS, 1)
void gemm_01_double_buf_kernel(const gemm_globals g, int M, int N, int K)
{
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al(reinterpret_cast<int*>(&__shm[0]));

    ab_pair(&ring)[S] = al.allocate_in<segment<0>, ab_pair, S>();

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

    auto issue_fill = [&](int slot, int kblock) {
        fill_rt_e fill_rt;
        fill_st_e& A_fill_st = *reinterpret_cast<fill_st_e*>(
            &ring[slot].a.data[A_tile::idx(wid * FILL_ROWS, 0)]);
        kittens::load(
            fill_rt, g.a,
            {0, 0, tile_m * (BLOCK_M / FILL_ROWS) + wid, kblock}); // GL -> RT
        kittens::sync::wait_load<0>();
        kittens::store(A_fill_st, fill_rt);                              // RT -> ST

        fill_st_e& B_fill_st = *reinterpret_cast<fill_st_e*>(
            &ring[slot].b.data[B_tile::idx(wid * FILL_ROWS, 0)]);
        kittens::load(
            fill_rt, g.b,
            {0, 0, tile_n * (BLOCK_N / FILL_ROWS) + wid, kblock}); // GL -> RT
        kittens::sync::wait_load<0>();
        kittens::store(B_fill_st, fill_rt);                              // RT -> ST
    };

    // Prologue: fill the ring ahead of the loop.
    #pragma unroll
    for (int s = 0; s < S - 1; ++s)
        if (s < k_blocks) issue_fill(s, s);

    kittens::sync::wait_ds<0>();                      // wait for data (LDS): RT -> ST writes landed
    kittens::sync::sync();                            // wait for everyone (workgroup): stage 0 readable

    rt_e<WARP_M, K_STEP> A_reg;
    rt_e<WARP_N, K_STEP> B_reg;

    // Main loop: one K-block per iteration.
    for (int kb = 0; kb < k_blocks; ++kb) {
        const int cur = kb % S, nxt = (kb + 1) % S;

        /* The fill leads the reads so its global latency is covered. The clamped index is a
         * branch-free tail: the last iteration refills a stage nothing reads again. */
        const int fk = (kb + 1 < k_blocks) ? (kb + 1) : (k_blocks - 1);
        issue_fill(nxt, fk);

        kittens::load(A_reg, ring[cur].a, warp_off_a);
        kittens::load(B_reg, ring[cur].b, warp_off_b);
        kittens::sync::wait_ds<0>();    // wait for data (LDS): reads and next-stage writes are done
        mma_ABt(C_acc, A_reg, B_reg, C_acc);

        kittens::sync::sync();          // wait for everyone (workgroup): publish the fill, free the stage
    }

    // Nothing reuses the ring now; this drains the wave's outstanding LDS traffic before exit.
    kittens::sync::wait_ds<0>();     // wait for data (LDS): no read may outlive the workgroup

    // Direct store: warp accumulator to bf16 in global C; rung 00 has the transaction-size cost.
    kittens::store(g.c, C_acc, {0, 0, tile_m * WARPS_M + warp_r, tile_n * WARPS_N + warp_c});
}

void dispatch(gemm_globals g, const launch_config& launch)
{
    const size_t mem_size = launch.dynamic_shared_memory<S>();

    /* The direct store writes a lane's run of consecutive rows as one sized buffer store, so the
     * run has to be aligned, which needs the leading dimension to divide it. Kept at 8 rather than
     * loosened to the run length: every tested shape satisfies it and it stays sufficient if a
     * wider accumulator shape lengthens the run. */
    if (g.c.rows() % 8 != 0) {
        std::fprintf(stderr,
            "01_gemm_double_buf: column-major C requires M %% 8 == 0 (got M=%d)\n", g.c.rows());
        std::abort();
    }

    gfx1250_gemm::require_k_blocks(g.K(), "01_gemm_double_buf");

    hipFuncSetAttribute(reinterpret_cast<const void*>(gemm_01_double_buf_kernel),
                        hipFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(mem_size));
    gemm_01_double_buf_kernel<<<launch.grid, launch.block, mem_size, launch.stream>>>(
        g, g.M(), g.N(), g.K());
}

#include "harness.h"
