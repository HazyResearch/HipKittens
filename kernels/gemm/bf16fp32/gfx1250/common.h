/**
 * @file common.h
 * @brief Shared boilerplate for the gfx1250 GEMM ladder kernels.
 *
 * Pulls in `kittens.cuh`, sets up the canonical tile constants, defines
 * `gemm_globals`, and provides a single store helper that writes a wave-32
 * WMMA accumulator back to global bf16. Each `gemm_*.cpp` includes this
 * header so the ladder rungs differ only in their compute body.
 */

#pragma once

#include "kittens.cuh"

namespace gfx1250_gemm {

/* ----------  TILE CONFIGURATION  ---------- */

constexpr int BLOCK_M     = 64;
constexpr int BLOCK_N     = 64;
constexpr int K_STEP      = 32;
constexpr int WARPS_M     = 2;
constexpr int WARPS_N     = 2;
constexpr int WARP_M      = BLOCK_M / WARPS_M;
constexpr int WARP_N      = BLOCK_N / WARPS_N;
constexpr int NUM_WARPS   = WARPS_M * WARPS_N;
constexpr int NUM_THREADS = NUM_WARPS * kittens::WARP_THREADS;

using gl_bf = kittens::gl<kittens::bf16, -1, -1, -1, -1>;

/* ----------  GLOBALS  ---------- */

struct gemm_globals {
    gl_bf a, b, c;
    hipStream_t stream;
    int M() const { return a.rows(); }
    int N() const { return c.cols(); }
    int K() const { return a.cols(); }
    dim3   grid()  const { return dim3(M() / BLOCK_M, N() / BLOCK_N); }
    dim3   block() const { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() const { return kittens::MAX_SHARED_MEMORY; }
};

/* ----------  C STORE: WMMA-acc -> global bf16  ---------- */

/**
 * @brief Write a 16x16 wave-32 WMMA accumulator tile out to global bf16.
 *
 * Per the gfx1250 WMMA bf16 ISA layout, the 16x16 D output is held by 32
 * lanes as 8 fp32 each. Lanes 0..15 cover rows 0..7 of the tile, lanes
 * 16..31 cover rows 8..15; in both halves, lane L covers column `L % 16`.
 * Each lane's `data[k]` (`float2`) holds two adjacent rows:
 *   - `data[k].x` -> row `2k     + 8 * (L / 16)`
 *   - `data[k].y` -> row `2k + 1 + 8 * (L / 16)`
 */
__device__ static inline void store_acc16(
    kittens::bf16* __restrict__ c_global_raw,
    int gr_base, int gc_base, int N,
    const kittens::rt_base<float, kittens::ducks::rt_layout::col,
                            kittens::ducks::rt_shape::rt_16x16>& tile)
{
    // Coerce to a uint16_t AS(1) pointer and store the bf16 bit-pattern,
    // forcing the compiler to emit `global_store_b16` instead of a
    // generic flat store. We route through uint16_t because
    // `__hip_bfloat16::operator=` is not address-space-qualified, so
    // direct assignment through an AS(1)-qualified bf16* fails to compile.
    using c_global_t = uint16_t __attribute__((address_space(1)));
    auto* c_global = (c_global_t*)(reinterpret_cast<uintptr_t>(c_global_raw));

    const int L    = kittens::laneid();
    const int half = L / 16;
    const int col  = L % 16;
    const int gc   = gc_base + col;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int gr0 = gr_base + 2 * k     + 8 * half;
        const int gr1 = gr_base + 2 * k + 1 + 8 * half;
        kittens::bf16 v0 =
            kittens::base_types::convertor<kittens::bf16, float>::convert(tile.data[k].x);
        kittens::bf16 v1 =
            kittens::base_types::convertor<kittens::bf16, float>::convert(tile.data[k].y);
        c_global[gr0 * N + gc] = *reinterpret_cast<const uint16_t*>(&v0);
        c_global[gr1 * N + gc] = *reinterpret_cast<const uint16_t*>(&v1);
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

} // namespace gfx1250_gemm
