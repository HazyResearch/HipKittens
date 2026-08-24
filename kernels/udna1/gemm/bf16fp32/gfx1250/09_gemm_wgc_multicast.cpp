/**
 * @file 09_gemm_wgc_multicast.cpp
 * @brief Rung 09 -- 08_gemm_split_bar plus a workgroup cluster multicasting both operands.
 *
 * Kernel Specification
 *   layout      TN -- a is [M, K], b is [N, K], both K-contiguous; c is [M, N] column-major
 *   tile        256x256 macro, 64x64 per warp, 4x4 warps; BLOCK_K 128 = 4 x K_STEP 32
 *   occupancy   16 warps / 512 threads / 4 waves per SIMD, one workgroup per CU
 *   cluster     4x4 workgroup cluster (CLUSTER_DIM 4); grid.x and grid.y must both be divisible by 4
 *   registers   220 VGPR against a 256/lane budget (131072 / 512 threads); 128 are accumulator
 *               (WARP_M*WARP_N/32), 50 SGPR
 *   spills      none: 0 VGPR, 0 SGPR, 0 scratch
 *   LDS         2 stages x 136 KB = 272 KB of 320 KB (85%)
 *   sync        per K-block: 2 barriers (workgroup + cluster, split), 4 LDS waits
 *               (3 partial, 1 full), 1 TDM drain
 *   intensity   128 FLOP per byte of global operand traffic, BM*BN/(BM+BN)
 *
 * Sixteen workgroups form a 4x4 cluster and share each fetched panel four ways, which removes
 * 75% of the fill transactions and is worth about 7.2%. What it costs is a second rendezvous: the
 * barrier grows a cluster half alongside the workgroup one, and the launch now has a shape
 * requirement, since a cluster only forms if the grid divides by the cluster dimension.
 * Everything else is `08_gemm_split_bar`: 256x256 macro tile, BLOCK_K=128 walked in four K_STEP=32
 * sub-steps, 4x4 warps, a two-stage TDM-filled LDS ring, the barrier in its split form, and a
 * direct column-major epilogue. Uses only:
 *   - `kittens::tdm::load_async`         : descriptor-driven global -> LDS tile DMA, with multicast.
 *   - `kittens::sync::wait_tdm`   : drain the tile DMA.
 *   - `kittens::sync::arrive/wait`: workgroup barrier (-1), in the split form.
 *   - `kittens::cluster::arrive/wait` : cluster barrier (-3), likewise split.
 *   - `kittens::cluster::sync`    : fused cluster barrier (-3) in the prologue.
 *   - `kittens::sync::wait_ds`    : partial and full LDS-read drains.
 *   - `kittens::load(rt,st,off)`  : shared -> register load (wide `ds_load_b128`).
 *   - `kittens::mma_ABt`          : 16x16x32 WMMA via the bf16 builtin.
 *   - `kittens::sched::compiler_fence` : hold the matrix op inside the signal-to-wait window.
 *   - `kittens::store(gl,rt,idx)` : direct column-major epilogue.
 */

/* Geometry, overridable from the command line. Three constraints bind it:
 *   - VGPRs: a warp tile of WARP_M*WARP_N floats over 32 lanes must leave room for operands.
 *     The 64x64 warp tile puts 128 floats per lane in the accumulator; the kernel totals 220 of 256.
 *   - LDS: S*(A_tile + B_tile) must fit in 327,680 B.
 *   - Tiling: BLOCK_M and BLOCK_N must divide the problem, so on power-of-two shapes they must
 *     be powers of two too, which rules out 320 and 384 whatever the registers allow. */
#include "kittens.cuh"

constexpr int BLOCK_M     = 256;
constexpr int BLOCK_N     = 256;
constexpr int BLOCK_K     = 128;
constexpr int K_STEP      = 32;
/* 4x4 warps is four waves per SIMD, which is what the register file allows: at eight it cannot
 * hold the double-buffered operands plus the matrix op held in the window, and the kernel spills.
 * Wave count is co-designed with the split form rather than free to set on its own. */
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

/* A cluster holds at most 16 workgroups (WG_in_Cluster is 4 bits) and a multicast mask may name
 * at most 5 destinations before the hardware demotes it to an ordinary load, so 4x4 is the
 * largest legal shape whose masks stay inside the demotion limit. It also shares both operands
 * four ways, removing 75% of fill transactions against 50% for 2x2 and 37.5% for 1x4. */
static constexpr int CLUSTER_DIM = 4;

using A_deep = st_e<BLOCK_M, BLOCK_K>;
using B_deep = st_e<BLOCK_N, BLOCK_K>;

__global__
__cluster_dims__(CLUSTER_DIM, CLUSTER_DIM, 1)
__launch_bounds__(NUM_THREADS, 1)
void gemm_wgc_multicast_kernel(const gemm_globals g, int M, int N, int K)
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

    /* Multicast delivery masks. A workgroup's id within the cluster is cx + 4*cy, so the four
     * sharing its A panel have the same cx and the four sharing its B panel the same cy. Bit i
     * of the mask means "deliver to workgroup i"; a wrong ordering is a wrong answer. */
    const uint32_t cx = blockIdx.x & 3u;
    const uint32_t cy = blockIdx.y & 3u;
    const uint32_t a_mask = 0x1111u << cx;
    const uint32_t b_mask = 0xFu << (4u * cy);

    /* The two issuers must differ in SIMD parity so both tile-DMA engines are used: the four
     * SIMDs pair as {0,2} and {1,3}, each pair served by one engine. Warps 0 and 1 are on
     * different pairs; warps 0 and 2 would share one and serialize the two fills. */
    auto issue_fill = [&](int slot, int kblock, uint32_t count = 1) {
        if (wid == 0) kittens::tdm::load_async(A_st[slot], g.a, {0, 0, tile_m, kblock}, M, K, K, a_mask, count);
        if (wid == 1) kittens::tdm::load_async(B_st[slot], g.b, {0, 0, tile_n, kblock}, N, K, K, b_mask, count);
    };

    // Prologue: fill the ring ahead of the loop.
    #pragma unroll
    for (int s = 0; s < S - 1; ++s)
        if (s < k_blocks) issue_fill(s, s);

    kittens::sync::wait_tdm<0>();             // wait for data (TDM): stage 0 landed
    kittens::sched::compiler_fence();
    kittens::cluster::sync();                 // wait for everyone (cluster): no work to fill a window
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

        /* Branch-free tail: a count=0 descriptor rather than an `if`. Branching would let
         * some cluster members skip a rendezvous their peers are still waiting on. */
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

        /* Stage handoff, in order: drain LDS, drain TDM, signal both barriers, the last matrix
         * op, then wait on both. */
        constexpr int c_last = (KS - 1) & 1;
        kittens::sync::wait_ds<0>();               // wait for data (LDS): done reading this stage
        kittens::sync::wait_tdm<S - 2>();          // wait for data (TDM): the next stage has landed
        kittens::sched::compiler_fence();
        kittens::sync::arrive();                   // wait for everyone (workgroup): the 16 warps here
        // wait for everyone (cluster): one signal each -- it counts workgroups, not warps.
        if (wid == 0) kittens::cluster::arrive();
        kittens::sched::compiler_fence();
        mma_ABt(C_acc, A_reg[c_last], B_reg[c_last], C_acc);   // 16 v_wmma inside the barrier window
        kittens::sched::compiler_fence();
        kittens::sync::wait();
        kittens::cluster::wait();
        kittens::sched::compiler_fence();
    }

    // Nothing reuses the ring, but a workgroup must not exit with a fill still writing its LDS.
    kittens::sync::wait_tdm<0>();                  // wait for data (TDM): no fill outlives the group
    kittens::sync::wait_ds<0>();                   // wait for data (LDS): nor any read of the ring

    // Direct store: warp accumulator to bf16 in global C; `gemm_naive` has the transaction-size cost.
    kittens::store(g.c, C_acc, {0, 0, tile_m * WARPS_M + warp_r, tile_n * WARPS_N + warp_c});
}

void dispatch(gemm_globals g, const launch_config& launch)
{
    /* C is stored straight out of registers, so the LDS request is the operand ring alone.
     * At 278,528 B of 327,680 B it also holds the kernel to one workgroup per CU. */
    const size_t load_lds  = S * (sizeof(A_deep) + sizeof(B_deep));
    const size_t mem_size  = load_lds;   // no C staging tile to make room for

    /* The direct store writes a lane's run of consecutive rows as one sized buffer store, so the
     * run has to be aligned, which needs the leading dimension to divide it. Kept at 8 rather than
     * loosened to the run length: every tested shape satisfies it and it stays sufficient if a
     * wider accumulator shape lengthens the run. */
    if (g.c.rows() % 8 != 0) {
        std::fprintf(stderr,
            "09_gemm_wgc_multicast: column-major C requires M %% 8 == 0 (got M=%d)\n", g.c.rows());
        std::abort();
    }

    gfx1250_gemm::require_k_blocks(g.K(), "09_gemm_wgc_multicast");

    hipFuncSetAttribute(reinterpret_cast<const void*>(gemm_wgc_multicast_kernel),
                        hipFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(mem_size));

    const dim3 grid = launch.grid;

    /* Refuse rather than launch without a cluster. The multicast masks name peers by cluster
     * position, so without the cluster they name workgroups that are not co-scheduled. */
    if (grid.x % CLUSTER_DIM != 0 || grid.y % CLUSTER_DIM != 0) {
        printf("!! 09_gemm_wgc_multicast: a %dx%d cluster needs grid.x and grid.y both divisible "
               "by %d; got %ux%u.\n", CLUSTER_DIM, CLUSTER_DIM, CLUSTER_DIM, grid.x, grid.y);
        return;
    }

    hipLaunchKernelGGL(gemm_wgc_multicast_kernel, grid, launch.block, mem_size, launch.stream,
                       g, g.M(), g.N(), g.K());
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess)
        printf("!! 09_gemm_wgc_multicast: cluster launch REJECTED: %s\n", hipGetErrorString(e));
}

#include "harness.h"
