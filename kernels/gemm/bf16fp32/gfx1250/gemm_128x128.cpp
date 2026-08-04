/**
 * @file gemm_128x128.cpp
 * @brief Rung 4 -- gemm_async with the macro tile doubled to 128x128.
 *
 * Arithmetic intensity for a square bf16 tile is M*N/(M+N) FLOP per byte, so it rises from 32 to
 * 64 across this step and the operand traffic per output element halves. That is worth about
 * 1.8x, the largest step on the ladder. The warp grid has to double with the tile, 2x2 -> 4x4, to
 * hold the warp tile at 32x32: a 2x2 grid on a 128x128 tile gives each warp four times the
 * accumulator and spills. Same BLOCK_K=32, two async-filled stages interleaved in one LDS
 * segment, plain split barrier, staged column-major epilogue. Uses only:
 *   - `kittens::load_async`       : cooperative `global_load_async_to_lds_b128` fill.
 *   - `kittens::sync::wait_async` : drain the async fill.
 *   - `kittens::sync::arrive/wait`: split workgroup barrier (-1).
 *   - `kittens::sync::wait_ds`    : drain LDS reads before the matrix op and the handoff.
 *   - `kittens::load(rt,st,off)`  : shared -> register load (wide `ds_load_b128`).
 *   - `kittens::mma_ABt`          : 16x16x32 WMMA via the bf16 builtin.
 *   - `kittens::sched::compiler_fence` : keep the post-wait loads below the barrier.
 *   - `kittens::store(gl,rt,st,...)`   : staged column-major epilogue.
 */

#ifndef GFX1250_BLOCK_M
#define GFX1250_BLOCK_M 128
#endif
#ifndef GFX1250_BLOCK_N
#define GFX1250_BLOCK_N 128
#endif
#ifndef GFX1250_BLOCK_K
#define GFX1250_BLOCK_K 32
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

static constexpr int S = 2;                    // stages in the LDS ring

// One ring slot, with A and B adjacent.
struct KITTENS_DEFAULT_ALIGN ab_pair { A_tile a; B_tile b; };
static_assert(sizeof(ab_pair) == sizeof(A_tile) + sizeof(B_tile),
              "an interleaved pair must not introduce padding");

__global__ __launch_bounds__(NUM_THREADS, 1)
void gemm_128x128_kernel(const gemm_globals g, int M, int N, int K)
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

    // Every thread participates: `load_async` spreads the tile across all NUM_THREADS lanes.
    auto issue_fill = [&](int slot, int kblock) {
        kittens::load_async<NUM_THREADS>(ring[slot].a, g.a, {0, 0, tile_m, kblock}, K);
        kittens::load_async<NUM_THREADS>(ring[slot].b, g.b, {0, 0, tile_n, kblock}, K);
    };

    // Prologue: fill the ring ahead of the loop.
    #pragma unroll
    for (int s = 0; s < S - 1; ++s)
        if (s < k_blocks) issue_fill(s, s);

    kittens::sync::wait_async<0>();
    kittens::sched::compiler_fence();
    kittens::sync::arrive(); kittens::sync::wait();   // publish stage 0
    kittens::sched::compiler_fence();

    rt_e<WARP_M, K_STEP> A_reg;
    rt_e<WARP_N, K_STEP> B_reg;

    // Main loop: one K-block per iteration.
    for (int kb = 0; kb < k_blocks; ++kb) {
        const int cur = kb % S, nxt = (kb + 1) % S;

        kittens::sched::compiler_fence();
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
    // The C staging tile reuses the ring, so the request is the larger of the two, not the sum.
    const size_t load_lds  = S * sizeof(ab_pair);
    const size_t store_lds = sizeof(C_tile);
    const size_t mem_size  = (load_lds > store_lds ? load_lds : store_lds);

    /* The column-major stream writes 8-element chunks down a column, so it is aligned only if
     * the leading dimension is a multiple of 8. */
    if (g.c.rows() % 8 != 0) {
        std::fprintf(stderr,
            "gemm_128x128: column-major C requires M %% 8 == 0 (got M=%d)\n", g.c.rows());
        std::abort();
    }

    gfx1250_gemm::require_k_blocks(g.K(), "gemm_128x128");

    hipFuncSetAttribute(reinterpret_cast<const void*>(gemm_128x128_kernel),
                        hipFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(mem_size));
    gemm_128x128_kernel<<<g.grid(), g.block(), mem_size, g.stream>>>(g, g.M(), g.N(), g.K());
}

#include "harness.h"
