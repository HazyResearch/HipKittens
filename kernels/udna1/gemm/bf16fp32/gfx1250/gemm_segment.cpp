/**
 * @file gemm_segment.cpp
 * @brief Rung 7 -- gemm_deepk with the operand rings separated into different LDS segments.
 *
 * The A ring and the B ring are placed a full 64 KB segment apart rather than interleaved, so
 * the two 256 B/cycle LDS read ports can serve an A read and a B read in the same cycle:
 * corresponding operands go from 17,408 B apart to 139,264 B. Drawing on both ports is what the
 * segmented LDS is there for, and on a kernel that is waiting on them it is worth about 12%.
 *
 * This recipe at 8192^3 is not waiting on them. The tile is large enough that one port already
 * supplies enough bandwidth to keep the matrix units fed, so the step buys nothing here. It is
 * still how the rings should be placed: the win is there as soon as the tile is smaller or the
 * fill slower, and it costs nothing.
 *
 * A K-block is held as K_SUBBLOCKS separate sub-tiles rather than one deep tile, because
 * `load_async` and the shared -> register `load` only agree on layout for a tile one subtile
 * column wide. A deep tile filled this way reads back scrambled. Same 256x256 tile, BLOCK_K=128,
 * 4x4 warps, two async-filled stages, plain split barrier, staged column-major epilogue.
 * Uses only:
 *   - `kittens::load_async`       : cooperative `global_load_async_to_lds_b128` fill.
 *   - `kittens::sync::wait_async` : drain the async fill.
 *   - `kittens::sync::arrive/wait`: split workgroup barrier (-1).
 *   - `kittens::sync::wait_ds`    : partial and full LDS-read drains.
 *   - `kittens::load(rt,st,off)`  : shared -> register load (wide `ds_load_b128`).
 *   - `kittens::mma_ABt`          : 16x16x32 WMMA via the bf16 builtin.
 *   - `kittens::sched::compiler_fence` : keep the post-wait loads below the barrier.
 *   - `kittens::store(gl,rt,st,...)`   : staged column-major epilogue.
 */

#ifndef GFX1250_BLOCK_M
#define GFX1250_BLOCK_M 256
#endif
#ifndef GFX1250_BLOCK_N
#define GFX1250_BLOCK_N 256
#endif
#ifndef GFX1250_BLOCK_K
#define GFX1250_BLOCK_K 128
#endif
#define GFX1250_K_STEP  32
#ifndef GFX1250_WARPS_M
#define GFX1250_WARPS_M 4
#endif
#ifndef GFX1250_WARPS_N
#define GFX1250_WARPS_N 4
#endif

#include "common.h"

using namespace kittens;
using namespace gfx1250_gemm;

/* LDS holds two stages at this tile size, not three -- see `kittens::MAX_SHARED_MEMORY`. */
static constexpr int S = 2;                    // stages in the LDS ring

__global__ __launch_bounds__(NUM_THREADS, 1)
void gemm_segment_kernel(const gemm_globals g, int M, int N, int K)
{
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al(reinterpret_cast<int*>(&__shm[0]));

    constexpr int KS = K_SUBBLOCKS;            // BLOCK_K / K_STEP = 4 sub-steps

    // Blocked: every A sub-tile, then every B sub-tile. That ordering is what puts the two
    // rings a full segment apart.
    A_tile(&A_st)[S][KS] = al.allocate_in<segment<0>, A_tile, S, KS>();
    B_tile(&B_st)[S][KS] = al.allocate_in<segment<0>, B_tile, S, KS>();

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

    // Every thread participates: `load_async` spreads the tile across all NUM_THREADS lanes.
    auto issue_fill = [&](int slot, int kblock) {
        #pragma unroll
        for (int j = 0; j < KS; ++j) {
            kittens::load_async<NUM_THREADS>(A_st[slot][j], g.a, {0, 0, tile_m, kblock * KS + j}, K);
            kittens::load_async<NUM_THREADS>(B_st[slot][j], g.b, {0, 0, tile_n, kblock * KS + j}, K);
        }
    };

    // Prologue: fill the ring ahead of the loop.
    #pragma unroll
    for (int s = 0; s < S - 1; ++s)
        if (s < k_blocks) issue_fill(s, s);

    kittens::sync::wait_async<0>();
    kittens::sched::compiler_fence();
    kittens::sync::arrive(); kittens::sync::wait();   // publish stage 0
    kittens::sched::compiler_fence();

    // Operands are double-buffered in registers: the matrix unit reads one buffer while the
    // next sub-step's loads fill the other.
    rt_e<WARP_M, K_STEP> A_reg[2];
    rt_e<WARP_N, K_STEP> B_reg[2];

    // Main loop: one K-block per iteration.
    for (int kb = 0; kb < k_blocks; ++kb) {
        const int cur = kb % S, nxt = (kb + 1) % S;

        kittens::sched::compiler_fence();
        kittens::load(A_reg[0], A_st[cur][0], warp_off_a);
        kittens::load(B_reg[0], B_st[cur][0], warp_off_b);

        // Branch-free tail: a clamped index rather than an `if`. An async copy has no null
        // form, so the last iteration refills a stage nothing reads again.
        const int fk = (kb + 1 < k_blocks) ? (kb + 1) : (k_blocks - 1);
        issue_fill(nxt, fk);

        /* Sub-steps 0 .. KS-2. The wait is partial -- it lets DS_SUB loads stay in flight
         * -- which is sound because the LDS counter retires in order. */
        #pragma unroll
        for (int si = 0; si < KS - 1; ++si) {
            const int c = si & 1, n = 1 - c;
            kittens::load(A_reg[n], A_st[cur][si + 1], warp_off_a);
            kittens::load(B_reg[n], B_st[cur][si + 1], warp_off_b);
            kittens::sync::wait_ds<DS_SUB>();
            mma_ABt(C_acc, A_reg[c], B_reg[c], C_acc);
        }

        /* The stage handoff. Both counters drain before the signal. The async drain has to
         * be full: unlike the LDS and tensor counters it does not retire in order, so a
         * partial `wait_async<N>` establishes nothing. */
        constexpr int c_last = (KS - 1) & 1;
        kittens::sync::wait_ds<0>();
        mma_ABt(C_acc, A_reg[c_last], B_reg[c_last], C_acc);
        kittens::sync::wait_async<0>();
        kittens::sched::compiler_fence();
        kittens::sync::arrive();                               // --- signal (-1) ---
        kittens::sync::wait();                                 // --- wait (-1) ---
        kittens::sched::compiler_fence();
    }

    // The epilogue reuses the ring, so drain both counters first.
    kittens::sync::wait_async<0>();
    kittens::sync::wait_ds<0>();

    // Epilogue: stage C through LDS so the global store coalesces.
    kittens::store<NUM_THREADS>(
        g.c, C_acc, *reinterpret_cast<C_tile*>(&__shm[0]),
        tile_m * BLOCK_M, tile_n * BLOCK_N, warp_r * WARP_M, warp_c * WARP_N);
}

void dispatch(gemm_globals g)
{
    /* The C staging tile reuses the operand rings, so the request is the larger of the two, not
     * the sum. At 278,528 B of 327,680 B it also holds the kernel to one workgroup per CU. */
    const size_t load_lds  = S * K_SUBBLOCKS * (sizeof(A_tile) + sizeof(B_tile));
    const size_t store_lds = sizeof(C_tile);
    const size_t mem_size  = (load_lds > store_lds ? load_lds : store_lds);

    /* The column-major stream writes 8-element chunks down a column, so it is aligned only if
     * the leading dimension is a multiple of 8. */
    if (g.c.rows() % 8 != 0) {
        std::fprintf(stderr,
            "gemm_segment: column-major C requires M %% 8 == 0 (got M=%d)\n", g.c.rows());
        std::abort();
    }

    gfx1250_gemm::require_k_blocks(g.K(), "gemm_segment");

    hipFuncSetAttribute(reinterpret_cast<const void*>(gemm_segment_kernel),
                        hipFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(mem_size));
    gemm_segment_kernel<<<g.grid(), g.block(), mem_size, g.stream>>>(g, g.M(), g.N(), g.K());
}

#include "harness.h"
