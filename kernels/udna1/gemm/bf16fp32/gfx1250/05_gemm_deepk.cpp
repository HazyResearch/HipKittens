/**
 * @file 05_gemm_deepk.cpp
 * @brief Rung 05 -- 04_gemm_256x256 with the LDS stage deepened, BLOCK_K 32 -> 128.
 *
 * Kernel Specification
 *   layout      TN -- a is [M, K], b is [N, K], both K-contiguous; c is [M, N] column-major
 *   tile        256x256 macro, 64x64 per warp, 4x4 warps; BLOCK_K 128 = 4 x K_STEP 32
 *   occupancy   16 warps / 512 threads / 4 waves per SIMD, one workgroup per CU
 *   registers   226 VGPR against a 256/lane budget (131072 / 512 threads); 128 are accumulator
 *               (WARP_M*WARP_N/32), 26 SGPR
 *   spills      none: 0 VGPR, 0 SGPR, 0 scratch
 *   LDS         2 stages x 4 sub-blocks x 34 KB = 272 KB of 320 KB (85%); a third stage would
 *               need 408 KB, which is what pins the ring at two from here up
 *   sync        per K-block: 1 barrier (split), 4 LDS waits (3 partial, 1 full), 1 async drain
 *   intensity   128 FLOP per byte of global operand traffic, BM*BN/(BM+BN)
 *
 * A stage now holds four matrix sub-steps instead of one, so the per-K-block rendezvous count
 * drops four-fold and a register double-buffer inside the K-block has something to overlap.
 * Deepening the stage also puts the LDS row at 256 B for a 2-byte element, which is eight lanes
 * per cache line and half the fill requests. Together that is worth about 1.17x. What it costs
 * is LDS: a stage is four times the bytes, and that is what fixes the ring at two stages from
 * here up. Same 256x256 tile, 4x4 warps, two async-filled stages interleaved in one segment,
 * plain split barrier, direct column-major epilogue. Uses only:
 *   - `kittens::load`       : cooperative `global_load_async_to_lds_b128` fill.
 *   - `kittens::sync::wait_async` : drain the async fill.
 *   - `kittens::sync::arrive/wait`: split workgroup barrier (-1).
 *   - `kittens::sync::wait_ds`    : partial and full LDS-read drains.
 *   - `kittens::load(rt,st,off)`  : shared -> register load (wide `ds_load_b128`).
 *   - `kittens::mma_ABt`          : 16x16x32 WMMA via the bf16 builtin.
 *   - `kittens::sched::compiler_fence` : keep the post-wait loads below the barrier.
 *   - `kittens::store(gl,rt,idx)` : direct column-major epilogue.
 */

#include "kittens.cuh"

constexpr int BLOCK_M     = 256;
constexpr int BLOCK_N     = 256;
constexpr int BLOCK_K     = 128;
constexpr int K_STEP      = 32;
constexpr int WARPS_M     = 4;
constexpr int WARPS_N     = 4;
constexpr int WARP_M      = BLOCK_M / WARPS_M;
constexpr int WARP_N      = BLOCK_N / WARPS_N;
constexpr int NUM_WARPS   = WARPS_M * WARPS_N;
constexpr int NUM_THREADS = NUM_WARPS * kittens::WARP_THREADS;
constexpr int K_SUBBLOCKS = BLOCK_K / K_STEP;

#include "common.h"

using namespace kittens;
using namespace gfx1250_gemm;

static constexpr int S = 2;                    // stages in the LDS ring

/* One ring slot, with A and B adjacent. Both members are 16-byte aligned and a multiple of 16
 * bytes long, so the pair carries no padding and the ring is exactly the operand bytes. */
struct KITTENS_DEFAULT_ALIGN ab_pair { A_tile a; B_tile b; };
static_assert(sizeof(ab_pair) == sizeof(A_tile) + sizeof(B_tile),
              "an interleaved pair must not introduce padding");

__global__ __launch_bounds__(NUM_THREADS, 1)
void gemm_deepk_kernel(const gemm_globals g, int M, int N, int K)
{
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al(reinterpret_cast<int*>(&__shm[0]));

    constexpr int KS = K_SUBBLOCKS;            // BLOCK_K / K_STEP = 4 sub-steps

    // Interleaved: A(s,j) then B(s,j), then A(s,j+1), so a pair shares one segment.
    ab_pair(&ring)[S][KS] = al.allocate_in<segment<0>, ab_pair, S, KS>();

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

    // One sub-step's ds_load count, so the wait below can retire the previous sub-step's loads
    // while this one's are still in flight.
    constexpr int DS_SUB = ds_loads_per_subblock<WARP_M>()
                         + ds_loads_per_subblock<WARP_N>();

    // Every thread participates: `load` spreads the tile across all NUM_THREADS lanes.
    auto issue_fill = [&](int slot, int kblock) {
        #pragma unroll
        for (int j = 0; j < KS; ++j) {
            kittens::load<NUM_THREADS>(ring[slot][j].a, g.a, {0, 0, tile_m, kblock * KS + j}, K);
            kittens::load<NUM_THREADS>(ring[slot][j].b, g.b, {0, 0, tile_n, kblock * KS + j}, K);
        }
    };

    // Prologue: fill the ring ahead of the loop.
    #pragma unroll
    for (int s = 0; s < S - 1; ++s)
        if (s < k_blocks) issue_fill(s, s);

    kittens::sync::wait_async<0>();                   // wait for data (async copy): stage 0 landed
    kittens::sched::compiler_fence();
    kittens::sync::arrive(); kittens::sync::wait();   // wait for everyone (workgroup): stage 0 ready
    kittens::sched::compiler_fence();

    // Operands are double-buffered in registers: the matrix unit reads one buffer while the
    // next sub-step's loads fill the other.
    rt_e<WARP_M, K_STEP> A_reg[2];
    rt_e<WARP_N, K_STEP> B_reg[2];

    // Main loop: one K-block per iteration.
    for (int kb = 0; kb < k_blocks; ++kb) {
        const int cur = kb % S, nxt = (kb + 1) % S;

        kittens::sched::compiler_fence();
        kittens::load(A_reg[0], ring[cur][0].a, warp_off_a);
        kittens::load(B_reg[0], ring[cur][0].b, warp_off_b);

        // Branch-free tail: a clamped index rather than an `if`. An async copy has no null
        // form, so the last iteration refills a stage nothing reads again.
        const int fk = (kb + 1 < k_blocks) ? (kb + 1) : (k_blocks - 1);
        issue_fill(nxt, fk);

        /* Sub-steps 0 .. KS-2. The wait is partial -- it lets DS_SUB loads stay in flight
         * -- which is sound because the LDS counter retires in order. */
        #pragma unroll
        for (int si = 0; si < KS - 1; ++si) {
            const int c = si & 1, n = 1 - c;
            kittens::load(A_reg[n], ring[cur][si + 1].a, warp_off_a);
            kittens::load(B_reg[n], ring[cur][si + 1].b, warp_off_b);
            // wait for data (LDS): partial -- LDS reads finish in order, so some stay in flight.
            kittens::sync::wait_ds<DS_SUB>();
            mma_ABt(C_acc, A_reg[c], B_reg[c], C_acc);
        }

        /* The stage handoff. The async drain has to be full: that counter does not retire
         * in order, so a partial `wait_async<N>` establishes nothing. */
        constexpr int c_last = (KS - 1) & 1;
        kittens::sync::wait_ds<0>();     // wait for data (LDS): done reading the current stage
        mma_ABt(C_acc, A_reg[c_last], B_reg[c_last], C_acc);
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

    // Direct store: warp accumulator to bf16 in global C; rung 00 has the transaction-size cost.
    kittens::store(g.c, C_acc, {0, 0, tile_m * WARPS_M + warp_r, tile_n * WARPS_N + warp_c});
}

void dispatch(gemm_globals g, const launch_config& launch)
{
    /* The C staging tile reuses the ring, so the request is the larger of the two, not the sum.
     * At 278,528 B of 327,680 B it also holds the kernel to one workgroup per CU. */
    const size_t load_lds  = S * K_SUBBLOCKS * sizeof(ab_pair);
    const size_t mem_size  = load_lds;   // no C staging tile to make room for

    /* The direct store writes a lane's run of consecutive rows as one sized buffer store, so the
     * run has to be aligned, which needs the leading dimension to divide it. Kept at 8 rather than
     * loosened to the run length: every tested shape satisfies it and it stays sufficient if a
     * wider accumulator shape lengthens the run. */
    if (g.c.rows() % 8 != 0) {
        std::fprintf(stderr,
            "05_gemm_deepk: column-major C requires M %% 8 == 0 (got M=%d)\n", g.c.rows());
        std::abort();
    }

    gfx1250_gemm::require_k_blocks(g.K(), "05_gemm_deepk");

    hipFuncSetAttribute(reinterpret_cast<const void*>(gemm_deepk_kernel),
                        hipFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(mem_size));
    gemm_deepk_kernel<<<launch.grid, launch.block, mem_size, launch.stream>>>(
        g, g.M(), g.N(), g.K());
}

#include "harness.h"
