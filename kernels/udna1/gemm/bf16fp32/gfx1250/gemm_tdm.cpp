/**
 * @file gemm_tdm.cpp
 * @brief Rung 8 -- gemm_segment with the fill moved onto the hardware tile-DMA engine.
 *
 * `load_tdm` is a descriptor-driven global -> LDS copy: one wave posts one descriptor where the
 * async path had every lane issue its own transfer, so the waves stop spending issue slots on
 * the fill entirely. That is worth about 1.39x. It also drags two things along, because the API
 * requires them: the stage becomes one deep 256x128 panel instead of four sub-tiles, and the
 * tail skip becomes a count=0 descriptor, since a TDM cannot be EXEC-masked. The barrier is
 * plain: the K-block's last matrix op issues before the signal, so the signal-to-wait window is
 * empty. Same 256x256 tile, BLOCK_K=128, 4x4 warps, two stages, segment-separated LDS, staged
 * column-major epilogue. Uses only:
 *   - `kittens::load_tdm`         : descriptor-driven global -> LDS tile DMA.
 *   - `kittens::sync::wait_tdm`   : drain the tile DMA.
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

using A_deep = st_e<BLOCK_M, BLOCK_K>;
using B_deep = st_e<BLOCK_N, BLOCK_K>;

__global__ __launch_bounds__(NUM_THREADS, 1)
void gemm_tdm_kernel(const gemm_globals g, int M, int N, int K)
{
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al(reinterpret_cast<int*>(&__shm[0]));

    constexpr int KS = K_SUBBLOCKS;            // BLOCK_K / K_STEP = 4 sub-steps

    A_deep(&A_st)[S] = al.allocate_in<segment<0>, A_deep, S>();
    B_deep(&B_st)[S] = al.allocate_in<segment<0>, B_deep, S>();

    rt_fl<WARP_M, WARP_N, col_l, rt_16x16_s> C_acc;
    zero(C_acc);

    const int tile_m   = blockIdx.x;
    const int tile_n   = blockIdx.y;
    const int wid      = warpid();
    const int warp_r   = wid / WARPS_N;
    const int warp_c   = wid % WARPS_N;
    const int k_blocks = K / BLOCK_K;
    if (k_blocks <= 0) return;                    // K shorter than one block: nothing to compute

    const int warp_off_a = warp_r * WARP_M * BLOCK_K;
    const int warp_off_b = warp_c * WARP_N * BLOCK_K;

    // One sub-step's ds_load count, so the wait below can retire the previous sub-step's loads
    // while this one's are still in flight.
    constexpr int DS_SUB = ds_loads_per_subblock<WARP_M>()
                         + ds_loads_per_subblock<WARP_N>();

    /* The two issuers must differ in SIMD parity so both tile-DMA engines are used: the four
     * SIMDs pair as {0,2} and {1,3}, each pair served by one engine. Warps 0 and 1 are on
     * different pairs; warps 0 and 2 would share one and serialize the two fills. */
    auto issue_fill = [&](int slot, int kblock, uint32_t count = 1) {
        if (wid == 0) kittens::load_tdm(A_st[slot], g.a, {0, 0, tile_m, kblock}, M, K, K, 0, count);
        if (wid == 1) kittens::load_tdm(B_st[slot], g.b, {0, 0, tile_n, kblock}, N, K, K, 0, count);
    };

    // Prologue: fill the ring ahead of the loop.
    #pragma unroll
    for (int s = 0; s < S - 1; ++s)
        if (s < k_blocks) issue_fill(s, s);

    kittens::sync::wait_tdm<0>();
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
        kittens::load(A_reg[0], A_st[cur], warp_off_a);
        kittens::load(B_reg[0], B_st[cur], warp_off_b);

        /* Branch-free tail. On the last K-block there is nothing left to fetch, but
         * branching here would diverge the wave. A TDM also cannot be EXEC-masked, so the
         * skip is a count=0 NULL descriptor, which moves no memory. */
        const int      fk  = (kb + 1 < k_blocks) ? (kb + 1) : (k_blocks - 1);
        const uint32_t cnt = (kb + 1 < k_blocks) ? 1u : 0u;
        issue_fill(nxt, fk, cnt);

        /* Sub-steps 0 .. KS-2. The wait is partial -- it lets DS_SUB loads stay in flight
         * -- which is sound because the LDS counter retires in order. */
        #pragma unroll
        for (int si = 0; si < KS - 1; ++si) {
            const int c = si & 1, n = 1 - c;
            kittens::load(A_reg[n], A_st[cur], warp_off_a + (si + 1) * K_STEP);
            kittens::load(B_reg[n], B_st[cur], warp_off_b + (si + 1) * K_STEP);
            kittens::sync::wait_ds<DS_SUB>();
            mma_ABt(C_acc, A_reg[c], B_reg[c], C_acc);
        }

        /* The stage handoff. Both counters drain before the signal: LDS says this wave is
         * done with the current stage, TDM says the next has landed. The final matrix op
         * issues below the wait, so nothing covers the rendezvous. */
        constexpr int c_last = (KS - 1) & 1;
        kittens::sync::wait_ds<0>();
        mma_ABt(C_acc, A_reg[c_last], B_reg[c_last], C_acc);
        kittens::sync::wait_tdm<S - 2>();
        kittens::sched::compiler_fence();
        kittens::sync::arrive();                               // --- signal (-1) ---
        kittens::sync::wait();                                 // --- wait (-1) ---
        kittens::sched::compiler_fence();
    }

    // The epilogue reuses the ring, so drain both counters first.
    kittens::sync::wait_tdm<0>();
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
    const size_t load_lds  = S * (sizeof(A_deep) + sizeof(B_deep));
    const size_t store_lds = sizeof(C_tile);
    const size_t mem_size  = (load_lds > store_lds ? load_lds : store_lds);

    /* The column-major stream writes 8-element chunks down a column, so it is aligned only if
     * the leading dimension is a multiple of 8. */
    if (g.c.rows() % 8 != 0) {
        std::fprintf(stderr,
            "gemm_tdm: column-major C requires M %% 8 == 0 (got M=%d)\n", g.c.rows());
        std::abort();
    }

    gfx1250_gemm::require_k_blocks(g.K(), "gemm_tdm");

    hipFuncSetAttribute(reinterpret_cast<const void*>(gemm_tdm_kernel),
                        hipFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(mem_size));
    gemm_tdm_kernel<<<g.grid(), g.block(), mem_size, g.stream>>>(g, g.M(), g.N(), g.K());
}

#include "harness.h"
