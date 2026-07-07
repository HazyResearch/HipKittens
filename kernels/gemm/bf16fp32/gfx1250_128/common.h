/**
 * @file common.h
 * @brief Shared boilerplate for the gfx1250 128x128 GEMM ladder.
 *
 * Scales the 64x64 gfx1250 teaching ladder to a 128x128 block tile while
 * keeping the same per-warp micro-tile (32x32 output, rt_16x32 operands,
 * rt_16x16 WMMA accumulators). Each threadblock uses 4x4 = 16 warps
 * (512 threads on wave-32).
 *
 * Roadmap: validate rungs here at 128, then grow BLOCK to 256 with a 2x2
 * LDS half-tile split (MI350-style) and K_STEP=64.
 */

#pragma once

#include "kittens.cuh"

namespace gfx1250_gemm_128 {

/* ----------  TILE CONFIGURATION  ---------- */

constexpr int BLOCK_M     = 128;
constexpr int BLOCK_N     = 128;
constexpr int K_STEP      = 32;
constexpr int WARPS_M     = 4;
constexpr int WARPS_N     = 4;
constexpr int WARP_M      = BLOCK_M / WARPS_M;  // 32
constexpr int WARP_N      = BLOCK_N / WARPS_N;  // 32
constexpr int NUM_WARPS   = WARPS_M * WARPS_N;
constexpr int NUM_THREADS = NUM_WARPS * kittens::WARP_THREADS;

static_assert(WARP_M % 16 == 0 && WARP_N % 16 == 0,
              "warp tile must tile cleanly into 16x16 WMMA outputs");
static_assert(K_STEP % 32 == 0, "gfx1250 WMMA K depth is 32 for bf16");

using gl_bf = kittens::gl<kittens::bf16, -1, -1, -1, -1>;

/* ----------  SHARED TILE TYPES  ---------- */
using A_tile  = kittens::st_bf<BLOCK_M, K_STEP>;
using B_tile  = kittens::st_bf<BLOCK_N, K_STEP>;

/* ----------  GLOBALS  ---------- */

struct gemm_globals {
    gl_bf a, b, c;
    hipStream_t stream;
    int M() const { return a.rows(); }
    int N() const { return c.cols(); }
    int K() const { return a.cols(); }
    dim3   grid()  const { return dim3(M() / BLOCK_M, N() / BLOCK_N); }
    dim3   block() const { return dim3(NUM_THREADS); }
    template <int STAGES = 2>
    size_t dynamic_shared_memory() const {
        return STAGES * (sizeof(A_tile) + sizeof(B_tile));
    }
};

/* ----------  C STORE: WMMA-acc -> global bf16  ---------- */

__device__ static inline void store_acc16(
    kittens::bf16* __restrict__ c_global,
    int gr_base, int gc_base, int N,
    const kittens::rt_base<float, kittens::ducks::rt_layout::col,
                            kittens::ducks::rt_shape::rt_16x16>& tile)
{
    const int L    = kittens::laneid();
    const int half = L / 16;
    const int col  = L % 16;
    const int gc   = gc_base + col;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int gr0 = gr_base + 2 * k     + 8 * half;
        const int gr1 = gr_base + 2 * k + 1 + 8 * half;
        c_global[gr0 * N + gc] =
            kittens::base_types::convertor<kittens::bf16, float>::convert(tile.data[k].x);
        c_global[gr1 * N + gc] =
            kittens::base_types::convertor<kittens::bf16, float>::convert(tile.data[k].y);
    }
}

template<int H, int W>
__device__ static inline void store_acc(
    kittens::bf16* __restrict__ c_global,
    int wgr_base, int wgc_base, int N,
    const kittens::rt_fl<H * 16, W * 16, kittens::ducks::rt_layout::col,
                          kittens::ducks::rt_shape::rt_16x16>& C)
{
    #pragma unroll
    for (int n = 0; n < H; ++n)
        #pragma unroll
        for (int m = 0; m < W; ++m)
            store_acc16(c_global,
                        wgr_base + n * 16, wgc_base + m * 16,
                        N, C.tiles[n][m]);
}

} // namespace gfx1250_gemm_128
