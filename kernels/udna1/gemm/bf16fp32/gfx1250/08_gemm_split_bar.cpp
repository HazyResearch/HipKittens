/**
 * @file 08_gemm_split_bar.cpp
 * @brief Rung 08 -- 07_gemm_tdm with the workgroup barrier actually split.
 *
 * Kernel Specification
 *   layout      TN -- a is [M, K], b is [N, K], both K-contiguous; c is [M, N] column-major
 *   tile        256x256 macro, 64x64 per warp, 4x4 warps; BLOCK_K 128 = 4 x K_STEP 32
 *   occupancy   16 warps / 512 threads / 4 waves per SIMD, one workgroup per CU
 *   registers   234 VGPR against a 256/lane budget (131072 / 512 threads); 128 are accumulator
 *               (WARP_M*WARP_N/32), 46 SGPR
 *   spills      none: 0 VGPR, 0 SGPR, 0 scratch
 *   LDS         2 stages x 136 KB = 272 KB of 320 KB (85%)
 *   sync        per K-block: 1 barrier (split), 4 LDS waits (3 partial, 1 full), 1 TDM drain
 *   window      the K-block's last mma_ABt issues between arrive() and wait()
 *   intensity   128 FLOP per byte of global operand traffic, BM*BN/(BM+BN)
 *
 * Every rung below closes the barrier the instant it opens it -- rungs 00 and 01 through the fused
 * `sync()`, rungs 02 to 07 by issuing `arrive()` and `wait()` back to back, which is the split API
 * spent as a fused barrier. Here the two halves are finally pulled apart and the K-block's last
 * matrix op is issued between them. What makes that legal is that the op reads registers only: a
 * wave is done with the LDS stage the moment its `ds_load`s drain, which is strictly before it is
 * done computing, so it can declare itself finished and keep working. A barrier costs skew rather
 * than instructions -- every wave that arrives early idles until the slowest one shows up -- and
 * this spends that idle time rather than shortening it. Sixteen `v_wmma` sit in the window, worth
 * about 4.3%.
 *
 * It fails silently. Nothing in the language holds the op inside the window: the compiler will
 * hoist it above the signal or sink it below the wait, and when it does the answers stay correct,
 * verification still passes, and the 5% is simply gone. The fences around the matrix op are what
 * pin it. Same 256x256 tile, BLOCK_K=128, 4x4 warps, two-stage TDM ring, segment-separated LDS,
 * direct column-major epilogue. Uses only:
 *   - `kittens::tdm::load_async`         : descriptor-driven global -> LDS tile DMA.
 *   - `kittens::sync::wait_tdm`   : drain the tile DMA.
 *   - `kittens::sync::arrive/wait`: workgroup barrier (-1), in the split form.
 *   - `kittens::sync::wait_ds`    : partial and full LDS-read drains.
 *   - `kittens::load(rt,st,off)`  : shared -> register load (wide `ds_load_b128`).
 *   - `kittens::mma_ABt`          : 16x16x32 WMMA via the bf16 builtin.
 *   - `kittens::sched::compiler_fence` : hold the matrix op inside the signal-to-wait window.
 *   - `kittens::store(gl,rt,idx)` : direct column-major epilogue.
 */

#include "kittens.cuh"

constexpr int BLOCK_M     = 256;
constexpr int BLOCK_N     = 256;
constexpr int BLOCK_K     = 128;
constexpr int K_STEP      = 32;
/* Four waves per SIMD is what the register file allows: at eight it cannot hold the
 * double-buffered operands plus the matrix op held in the window, and the kernel spills. Wave
 * count is co-designed with the split form rather than free to set on its own. */
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

/* LDS holds two stages at this tile size, not three -- see `kittens::MAX_SHARED_MEMORY`. */
static constexpr int S = 2;                    // stages in the LDS ring

using A_deep = st_e<BLOCK_M, BLOCK_K>;
using B_deep = st_e<BLOCK_N, BLOCK_K>;

__global__ __launch_bounds__(NUM_THREADS, 1)
void gemm_split_bar_kernel(const gemm_globals g, int M, int N, int K)
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
        if (wid == 0) kittens::tdm::load_async(A_st[slot], g.a, {0, 0, tile_m, kblock}, M, K, K, 0, count);
        if (wid == 1) kittens::tdm::load_async(B_st[slot], g.b, {0, 0, tile_n, kblock}, N, K, K, 0, count);
    };

    // Split into a signal half and a wait half so real work can be placed between them.
    // Prologue: fill the ring ahead of the loop.
    #pragma unroll
    for (int s = 0; s < S - 1; ++s)
        if (s < k_blocks) issue_fill(s, s);

    kittens::sync::wait_tdm<0>();                     // wait for data (TDM): stage 0 landed
    kittens::sched::compiler_fence();
    kittens::sync::arrive(); kittens::sync::wait();   // wait for everyone (workgroup): no work to fill
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

        /* Sub-steps 0 .. KS-2. The partial wait leaves DS_SUB loads in flight, which is
         * sound because the LDS counter retires in order. The last sub-step is peeled out
         * below because it hands the stage over. */
        #pragma unroll
        for (int si = 0; si < KS - 1; ++si) {
            const int c = si & 1, n = 1 - c;
            kittens::load(A_reg[n], A_st[cur], warp_off_a + (si + 1) * K_STEP);
            kittens::load(B_reg[n], B_st[cur], warp_off_b + (si + 1) * K_STEP);
            // wait for data (LDS): partial -- LDS reads finish in order, so some stay in flight.
            kittens::sync::wait_ds<DS_SUB>();
            mma_ABt(C_acc, A_reg[c], B_reg[c], C_acc);
        }

        /* The stage handoff, in the split form. Both counters drain first: LDS says this wave
         * is done with the current stage, TDM says the next has landed. The final matrix op is
         * then the independent work between signalling and waiting -- independent because it
         * reads registers only. The fences hold it inside the window; without them it is
         * hoisted above the signal or sunk below the wait. */
        constexpr int c_last = (KS - 1) & 1;
        kittens::sync::wait_ds<0>();       // wait for data (LDS): done reading the current stage
        kittens::sync::wait_tdm<S - 2>();  // wait for data (TDM): the next stage's fill has landed
        kittens::sched::compiler_fence();
        kittens::sync::arrive();           // wait for everyone (workgroup): publish next, protect this
        kittens::sched::compiler_fence();
        mma_ABt(C_acc, A_reg[c_last], B_reg[c_last], C_acc);   // 16 v_wmma inside the barrier window
        kittens::sched::compiler_fence();
        kittens::sync::wait();
        kittens::sched::compiler_fence();
    }

    // Nothing reuses the ring now, but a workgroup must not exit with a fill still writing its LDS.
    kittens::sync::wait_tdm<0>();          // wait for data (TDM): no fill outlives the workgroup
    kittens::sync::wait_ds<0>();           // wait for data (LDS): nor any read of the ring

    // Direct store: warp accumulator to bf16 in global C; rung 00 has the transaction-size cost.
    kittens::store(g.c, C_acc, {0, 0, tile_m * WARPS_M + warp_r, tile_n * WARPS_N + warp_c});
}

void dispatch(gemm_globals g, const launch_config& launch)
{
    /* The C staging tile reuses the operand rings, so the request is the larger of the two, not
     * the sum. At 278,528 B of 327,680 B it also holds the kernel to one workgroup per CU. */
    const size_t load_lds  = S * (sizeof(A_deep) + sizeof(B_deep));
    const size_t mem_size  = load_lds;   // no C staging tile to make room for

    /* The direct store writes a lane's run of consecutive rows as one sized buffer store, so the
     * run has to be aligned, which needs the leading dimension to divide it. Kept at 8 rather than
     * loosened to the run length: every tested shape satisfies it and it stays sufficient if a
     * wider accumulator shape lengthens the run. */
    if (g.c.rows() % 8 != 0) {
        std::fprintf(stderr,
            "08_gemm_split_bar: column-major C requires M %% 8 == 0 (got M=%d)\n", g.c.rows());
        std::abort();
    }

    gfx1250_gemm::require_k_blocks(g.K(), "08_gemm_split_bar");

    hipFuncSetAttribute(reinterpret_cast<const void*>(gemm_split_bar_kernel),
                        hipFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(mem_size));
    gemm_split_bar_kernel<<<launch.grid, launch.block, mem_size, launch.stream>>>(
        g, g.M(), g.N(), g.K());
}

#include "harness.h"
