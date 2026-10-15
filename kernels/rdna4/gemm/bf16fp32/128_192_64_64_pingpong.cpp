/**
 * @file 128_192_64_64_pingpong.cpp
 * @brief Symmetric ping-pong bf16 -> fp32 GEMM for gfx1201 (RX 9070 XT).
 *
 * Tile 128x192, K_STEP 64, k-substep 64. Two wave groups of 4; group p computes
 * in phase p and fills LDS in the other, so every wave both computes and loads.
 * Three such workgroups are resident per WGP (6 waves/SIMD). The 192-wide tile
 * spends the register headroom that the LDS-limited occupancy leaves free:
 * 238 VGPRs, two under the 240 boundary where that occupancy would be lost.
 *
 * Kept alongside the warp-specialised kernel because it is the simpler design
 * and the better starting point for extending to new shapes.
 */

namespace gfx1201_gemm {
constexpr int BLOCK_M = 128, BLOCK_N = 192, K_STEP = 64;   // block tile
constexpr int WARPS_M = 2,   WARPS_N = 4;                  // wave grid
}
#include "common.h"

// 8-WAVE PING-PONG -- overlap compute and memory ACROSS waves, not within one.
//
// Two symmetric groups on the same SIMDs: one issues WMMA while the other issues
// memory, and they swap at a whole-block barrier. Every wave computes, so no
// registers are spent on waves that produce no output (the cost of warp
// specialisation on AMD, where registers are allocated per block). There are no
// producers, so no named barriers are needed: both groups hit the same barrier
// and simply run different code between barriers, selected by group id.
//
// THE DECOMPOSITION. The wave grid is 2x4, so warp_r = wid / WARPS_N: waves 0-3
// have warp_r == 0 and waves 4-7 have warp_r == 1. A wave reads A rows
// [warp_r*64, +64), and WARP_M == 64 is exactly half of BLOCK_M == 128. Taking
// GROUP == warp_r, the two groups read DISJOINT halves of A, and each group
// fills its own half. There is no cross-group data dependency; the barrier only
// holds the groups in anti-phase and orders each group's own ds_store before
// its ds_load.
//
// LDS: 2 x st_bf<128,64>, double-buffered across K-steps.
//
// B stays direct-to-VGPR and is streamed KSUB-wide inside the compute phase, so
// the compute group also issues global loads. Only the A path is
// phase-separated.

using namespace kittens;
using namespace gfx1201_gemm;

constexpr int KSUB      = 64;      // k-substep for the register A tile and the B stream
constexpr int K_STAGGER = 32;      // see stagger below

// ONE shared A tile per buffer. The two wave groups take disjoint 64-row BANDS
// of it -- group g owns rows [g*WARP_M, +WARP_M) -- so the group is a runtime
// ROW OFFSET into a compile-time-selected object, never a runtime-selected
// object. Selecting the LDS object at runtime decays it to a generic 64-bit
// pointer (or a scratch array) and costs dozens of address ops per K-step.
using A_full = gfx1201_gemm::A_tile;

constexpr int HALF_THREADS = NUM_THREADS / 2;

template <bool FAST>
__global__ __launch_bounds__(NUM_THREADS, 1)
void gemm_pp_kernel(const gemm_globals g, int M, int N, int K)
{
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al(reinterpret_cast<int*>(&__shm[0]));

    A_full& A0 = al.allocate<A_full>();
    A_full& A1 = al.allocate<A_full>();

    constexpr int AH = WARP_M / 16, AW = WARP_N / 16;
    float2 lacc[AH][AW][4];
    #pragma unroll
    for (int n = 0; n < AH; ++n)
      #pragma unroll
      for (int m = 0; m < AW; ++m)
        #pragma unroll
        for (int i = 0; i < 4; ++i) { lacc[n][m][i].x = 0.f; lacc[n][m][i].y = 0.f; }

    // GROUP_M rasterisation: consecutive workgroups walk a GROUP_M-tall column
    // of output tiles, so they share B (and A) in L2.
    const int num_pid_m = gridDim.x;
    const int num_pid_n = gridDim.y;
    const int pid       = blockIdx.x + blockIdx.y * gridDim.x;
    const int num_pid_in_group = GROUP_M * num_pid_n;
    const int group_id         = pid / num_pid_in_group;
    const int first_pid_m      = group_id * GROUP_M;
    const int group_size_m     = min(num_pid_m - first_pid_m, GROUP_M);
    const int tile_m = first_pid_m + (pid % group_size_m);
    const int tile_n = (pid % num_pid_in_group) / group_size_m;

    const int wid    = warpid();
    const int warp_r = wid / WARPS_N;
    const int warp_c = wid % WARPS_N;
    const int k_iters = (K + K_STEP - 1) / K_STEP;
    // Each workgroup starts its K walk at a different offset, so resident
    // workgroups do not all read the same K column of A and B at once.
    const int stagger = (pid % K_STAGGER) % k_iters;

    // warpid() is `threadIdx.x >> 5`, which LLVM treats as divergent even though
    // it is constant across a wave. readfirstlane asserts what is already true,
    // keeping the group id, LDS row band and A read offset in SGPRs and the
    // phase branches scalar rather than EXEC-masked.
    const int wr        = __builtin_amdgcn_readfirstlane(warp_r);
    const int wc        = __builtin_amdgcn_readfirstlane(warp_c);
    // Rows of the shared A tile each ping-pong group is responsible for FILLING.
    constexpr int ROWS_PER_GRP = BLOCK_M / 2;

    const int grp       = wr;
    const int tid_local = static_cast<int>(threadIdx.x) - grp * HALF_THREADS;
    const int band      = grp * ROWS_PER_GRP;          // row band in the shared tile
    const int a_off     = grp * WARP_M * K_STEP;       // element offset for the read

    auto fill = [&](A_full& dst, int kk) {
        kittens::load<HALF_THREADS, FAST, ROWS_PER_GRP, 0, /*FULL_COLS=*/FAST>(
            dst, g.a, {0, 0, tile_m, kk}, K, K, M, tid_local, band);
    };

    auto compute = [&](A_full& src, int k) {
        // The bounds-checked path uses a 16-wide k-substep.
        constexpr int KS  = FAST ? KSUB : (KSUB > 16 ? 16 : KSUB);
        constexpr int NKH = K_STEP / KS;
        const int nb = tile_n * BLOCK_N + wc * WARP_N;
        // B streamed in KS-wide chunks, double-buffered: chunk kh+1 is in flight
        // while chunk kh is consumed.
        rt_bf<WARP_N, KS, row_l, rt_16x16_s> Br[2];
        kittens::load_direct<WARP_N, KS, !FAST, FAST>(Br[0], g.b, nb, k * K_STEP, N, K);
        #pragma unroll
        for (int kh = 0; kh < NKH; ++kh) {
            if (kh + 1 < NKH)
                kittens::load_direct<WARP_N, KS, !FAST, FAST>(
                    Br[(kh + 1) & 1], g.b, nb, k * K_STEP + (kh + 1) * KS, N, K);
            rt_bf<WARP_M, KS, row_l, rt_16x16_s> A_sub;
            kittens::load(A_sub, src, a_off, kh * KS);
            #pragma unroll
            for (int n = 0; n < AH; ++n)
              #pragma unroll
              for (int m = 0; m < AW; ++m)
                #pragma unroll
                for (int kk = 0; kk < KS / 16; ++kk)
                  kittens::wmma161616_acc(lacc[n][m],
                                          A_sub.tiles[n][kk].data,
                                          Br[kh & 1].tiles[m][kk].data);
        }
    };

    fill(A0, stagger);
    kittens::sync::wait_ds();
    kittens::sync::sync();

    int k = stagger;
    // TWO K-steps per body from a CONSTANT-trip inner loop, so `par` -- and
    // therefore the buffer -- is compile-time in each copy.
    for (int ki = 0; ki < k_iters; ki += 2) {
      #pragma unroll
      for (int par = 0; par < 2; ++par) {
        const int kk = ki + par;
        if (kk >= k_iters) break;
        A_full& src = (par == 0) ? A0 : A1;
        A_full& dst = (par == 0) ? A1 : A0;
        int knext = k + 1; if (knext == k_iters) knext = 0;
        const bool have_next = (kk + 1 < k_iters);

        // THE PING-PONG. Phase p: group p computes, group 1-p issues memory.
        // Both groups execute the same NUMBER of barriers, which is what makes a
        // whole-block barrier legal in wave-uniform divergent control flow.
        for (int phase = 0; phase < 2; ++phase) {
            if (grp == phase) {
                compute(src, k);
            } else {
                if (have_next) fill(dst, knext);
            }
            kittens::sync::wait_ds();
            kittens::sync::sync();
        }
        k = knext;
      }
    }

    bf16* c_base = reinterpret_cast<bf16*>(&g.c[{0, 0, 0, 0}]);
    store_acc_direct<AH, AW>(
        c_base,
        tile_m * BLOCK_M + warp_r * WARP_M,
        tile_n * BLOCK_N + warp_c * WARP_N,
        N, M, lacc);
}

void dispatch(gemm_globals g)
{
    const size_t mem_size = 2 * sizeof(A_full);
    hipFuncSetAttribute(reinterpret_cast<const void*>(gemm_pp_kernel<true>),
                        hipFuncAttributeMaxDynamicSharedMemorySize,
                        static_cast<int>(mem_size));
    hipFuncSetAttribute(reinterpret_cast<const void*>(gemm_pp_kernel<false>),
                        hipFuncAttributeMaxDynamicSharedMemorySize,
                        static_cast<int>(mem_size));
    const bool fast = (g.M() % gfx1201_gemm::BLOCK_M == 0) &&
                      (g.K() % gfx1201_gemm::K_STEP  == 0);
    if (fast)
        gemm_pp_kernel<true><<<g.grid(), g.block(), mem_size, g.stream>>>(g, g.M(), g.N(), g.K());
    else
        gemm_pp_kernel<false><<<g.grid(), g.block(), mem_size, g.stream>>>(g, g.M(), g.N(), g.K());
}

#include "harness.h"
