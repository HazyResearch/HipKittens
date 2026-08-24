/**
 * @file 02_gemm_async.cpp
 * @brief Rung 02 -- 01_gemm_double_buf plus the cooperative async fill.
 *
 * Kernel Specification
 *   layout      TN -- a is [M, K], b is [N, K], both K-contiguous; c is [M, N] column-major
 *   tile        64x64 macro, 32x32 per warp, 2x2 warps; BLOCK_K 32 = 1 x K_STEP 32
 *   occupancy   4 warps / 128 threads / 1 wave per SIMD; 10 workgroups per CU, register-bound
 *   registers   83 VGPR; 32 are accumulator (WARP_M*WARP_N/32), 30 SGPR
 *   spills      none: 0 VGPR, 0 SGPR, 0 scratch
 *   LDS         2 stages x 8.5 KB = 17 KB of 320 KB (5.3%)
 *   sync        per K-block: 1 barrier (split), 1 LDS drain, 1 async drain
 *   intensity   32 FLOP per byte of global operand traffic, BM*BN/(BM+BN)
 *
 * `global_load_async_to_lds_b128` lands operands in LDS without staging them through VGPRs, so
 * the registers and issue slots the register-mediated copy spent go back to the K-loop and the
 * fill no longer has to lead the operand reads. It is worth about 1.47x, one of the two largest
 * steps on the ladder. What it costs is a second counter to reason about: the fill retires on the
 * async counter, the operand reads on the LDS counter, and the stage handoff has to drain both.
 * Same 64x64 tile, two interleaved stages of a padded 64x32 LDS tile in one segment, plain split
 * barrier, direct column-major epilogue. Uses only:
 *   - `kittens::load`       : cooperative `global_load_async_to_lds_b128` fill.
 *   - `kittens::sync::wait_async` : drain the async fill.
 *   - `kittens::sync::arrive/wait`: split workgroup barrier (-1).
 *   - `kittens::sync::wait_ds`    : drain LDS reads before the matrix op and the handoff.
 *   - `kittens::load(rt,st,off)`  : shared -> register load (wide `ds_load_b128`).
 *   - `kittens::mma_ABt`          : 16x16x32 WMMA via the bf16 builtin.
 *   - `kittens::sched::compiler_fence` : keep the post-wait loads below the barrier.
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

// One ring slot, with A and B adjacent.
struct KITTENS_DEFAULT_ALIGN ab_pair { A_tile a; B_tile b; };
static_assert(sizeof(ab_pair) == sizeof(A_tile) + sizeof(B_tile),
              "an interleaved pair must not introduce padding");

__global__ __launch_bounds__(NUM_THREADS, 1)
void gemm_async_kernel(const gemm_globals g, int M, int N, int K)
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

    // Every thread participates: `load` spreads the tile across all NUM_THREADS lanes.
    auto issue_fill = [&](int slot, int kblock) {
        kittens::load<NUM_THREADS>(ring[slot].a, g.a, {0, 0, tile_m, kblock}, K);
        kittens::load<NUM_THREADS>(ring[slot].b, g.b, {0, 0, tile_n, kblock}, K);
    };

    // Prologue: fill the ring ahead of the loop.
    #pragma unroll
    for (int s = 0; s < S - 1; ++s)
        if (s < k_blocks) issue_fill(s, s);

    kittens::sync::wait_async<0>();                   // wait for data (async copy): stage 0 landed
    kittens::sched::compiler_fence();
    kittens::sync::arrive(); kittens::sync::wait();   // wait for everyone (workgroup): stage 0 ready
    kittens::sched::compiler_fence();

    rt_e<WARP_M, K_STEP> A_reg;
    rt_e<WARP_N, K_STEP> B_reg;

    // Main loop: one K-block per iteration.
    for (int kb = 0; kb < k_blocks; ++kb) {
        const int cur = kb % S, nxt = (kb + 1) % S;

        kittens::sched::compiler_fence();
        // The operand reads lead the fill: an async fill carries its own latency, so
        // reading first gets the matrix op started sooner.
        kittens::load(A_reg, ring[cur].a, warp_off_a);
        kittens::load(B_reg, ring[cur].b, warp_off_b);

        // Branch-free tail: a clamped index rather than an `if`. An async copy has no null
        // form, so the last iteration refills a stage nothing reads again.
        const int fk = (kb + 1 < k_blocks) ? (kb + 1) : (k_blocks - 1);
        issue_fill(nxt, fk);

        /* wait_ds<0> serves both the matrix op's operands and the stage about to be
         * overwritten. The async drain has to be full: that counter does not retire in
         * order, so a partial `wait_async<N>` establishes nothing. */
        kittens::sync::wait_ds<0>();
        mma_ABt(C_acc, A_reg, B_reg, C_acc);
        // wait for data (async copy): all of it -- the engine says "done" before visible.
        kittens::sync::wait_async<0>();
        kittens::sched::compiler_fence();
        kittens::sync::arrive();         // wait for everyone (workgroup): publish new, protect old
        kittens::sync::wait();
        kittens::sched::compiler_fence();
    }

    // Nothing reuses the ring now, but a workgroup must not exit with a fill still writing its LDS.
    kittens::sync::wait_async<0>();      // wait for data (async copy): no fill outlives the workgroup
    kittens::sync::wait_ds<0>();         // wait for data (LDS): nor any read of the ring

    // Direct store: warp accumulator to bf16 in global C; `gemm_naive` has the transaction-size cost.
    kittens::store(g.c, C_acc, {0, 0, tile_m * WARPS_M + warp_r, tile_n * WARPS_N + warp_c});
}

void dispatch(gemm_globals g, const launch_config& launch)
{
    // The C staging tile reuses the ring, so the request is the larger of the two, not the sum.
    const size_t load_lds  = S * sizeof(ab_pair);
    const size_t mem_size  = load_lds;   // no C staging tile to make room for

    /* The direct store writes a lane's run of consecutive rows as one sized buffer store, so the
     * run has to be aligned, which needs the leading dimension to divide it. Kept at 8 rather than
     * loosened to the run length: every tested shape satisfies it and it stays sufficient if a
     * wider accumulator shape lengthens the run. */
    if (g.c.rows() % 8 != 0) {
        std::fprintf(stderr,
            "02_gemm_async: column-major C requires M %% 8 == 0 (got M=%d)\n", g.c.rows());
        std::abort();
    }

    gfx1250_gemm::require_k_blocks(g.K(), "02_gemm_async");

    hipFuncSetAttribute(reinterpret_cast<const void*>(gemm_async_kernel),
                        hipFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(mem_size));
    gemm_async_kernel<<<launch.grid, launch.block, mem_size, launch.stream>>>(
        g, g.M(), g.N(), g.K());
}

#include "harness.h"
