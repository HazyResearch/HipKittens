/**
 * @file common.h
 * @brief Shared pieces of the gfx1201 bf16 GEMM kernels.
 *
 * The including kernel declares its tile configuration first, inside
 * `namespace gfx1201_gemm`:
 *
 *     namespace gfx1201_gemm {
 *     constexpr int BLOCK_M = 128, BLOCK_N = 192, K_STEP = 64;   // block tile
 *     constexpr int WARPS_M = 2,   WARPS_N = 4;                  // compute-wave grid
 *     }
 *     #include "common.h"
 *
 * This header derives the per-wave tile from that, and defines the LDS tile type,
 * `gemm_globals`, and the epilogue that writes a WMMA accumulator to global bf16.
 */

#pragma once

#include "kittens.cuh"

// Tile-rows per L2 rasterisation group: neighbouring workgroups share A and B in
// L2. 8 ships. 16 is up to 3.8% faster when K is a large power of two (A's rows
// then alias in L2) and within noise elsewhere; build with -DGROUP_M=16 for such
// workloads.
#ifndef GROUP_M
#define GROUP_M 8
#endif

namespace gfx1201_gemm {

/* ----------  TILE CONFIGURATION  ---------- */

constexpr int WARP_M      = BLOCK_M / WARPS_M;
constexpr int WARP_N      = BLOCK_N / WARPS_N;
constexpr int NUM_WARPS   = WARPS_M * WARPS_N;
constexpr int NUM_THREADS = NUM_WARPS * kittens::WARP_THREADS;

using gl_bf = kittens::gl<kittens::bf16, -1, -1, -1, -1>;

/* ----------  SHARED TILE TYPE  ---------- */

// 16 B of padding every two 64-byte subtile rows spreads the WMMA operand reads
// across the LDS banks. The 32-wide subtile must divide K_STEP.
static_assert(K_STEP % 32 == 0, "the A tile's 16x32 subtile must divide K_STEP");
using st_shape_t = kittens::ducks::st_shape::st_16x32_padded<64, 8>;
using A_tile     = kittens::st_bf<BLOCK_M, K_STEP, st_shape_t>;

/* ----------  GLOBALS  ---------- */

struct gemm_globals {
    gl_bf a, b, c;
    hipStream_t stream;
    int M() const { return a.rows(); }
    int N() const { return c.cols(); }
    int K() const { return a.cols(); }
    // ceil: cover partial tiles at the M/N edges. The kernel bounds-checks
    // its loads and stores, so the extra workgroups write nothing out of range.
    dim3   grid()  const { return dim3((M() + BLOCK_M - 1) / BLOCK_M,
                                       (N() + BLOCK_N - 1) / BLOCK_N); }
    dim3   block() const { return dim3(NUM_THREADS); }
};

/* ----------  C STORE: WMMA-acc -> global bf16  ---------- */

/**
 * @brief Write a 16x16 wave-32 WMMA accumulator tile out to global bf16.
 *
 * Per the gfx1201 WMMA bf16 ISA layout, the 16x16 D output is held by 32
 * lanes as 8 fp32 each. Lanes 0..15 cover rows 0..7 of the tile, lanes
 * 16..31 cover rows 8..15; in both halves, lane L covers column `L % 16`.
 * Each lane's `data[k]` (`float2`) holds two adjacent rows:
 *   - `data[k].x` -> row `2k     + 8 * (L / 16)`
 *   - `data[k].y` -> row `2k + 1 + 8 * (L / 16)`
 */
__device__ static inline void store_acc16(
    kittens::bf16* __restrict__ c_global,
    int gr_base, int gc_base, int N, int M_rows,
    const float2 (&data)[4])
{
    const int L    = kittens::laneid();
    const int half = L / 16;
    const int col  = L % 16;
    const int gc   = gc_base + col;
    if (gc >= N) return;                       // column past the N edge
    // Buffer stores: num_records spans the whole C matrix, so a row index past
    // M is discarded in hardware and the per-row branches disappear. The column
    // check must stay -- an out-of-range column is still a valid linear offset
    // (it lands in the next row), so the hardware cannot catch it.
    auto rsrc = __builtin_amdgcn_make_buffer_rsrc(
        (void*)c_global, 0, (int)((long)M_rows * N * sizeof(kittens::bf16)), 0x31014000u);
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int gr0 = gr_base + 2 * k     + 8 * half;
        const int gr1 = gr_base + 2 * k + 1 + 8 * half;
        const kittens::bf16 v0 =
            kittens::base_types::convertor<kittens::bf16, float>::convert(data[k].x);
        const kittens::bf16 v1 =
            kittens::base_types::convertor<kittens::bf16, float>::convert(data[k].y);
        unsigned short b0, b1;
        __builtin_memcpy(&b0, &v0, 2);
        __builtin_memcpy(&b1, &v1, 2);
        // 32-bit offsets: the buffer store's offset operand and num_records are
        // 32-bit, so 64-bit arithmetic here would only add dead high-half work.
        __builtin_amdgcn_raw_buffer_store_b16(b0, rsrc, (gr0 * N + gc) * 2, 0, 0);
        __builtin_amdgcn_raw_buffer_store_b16(b1, rsrc, (gr1 * N + gc) * 2, 0, 0);
    }
}

/**
 * @brief Store a wave's H x W grid of 16x16 accumulators, held as plain float2
 * arrays. (An rt_fl copy of the accumulator would be a second full set of
 * registers that the allocator does not always coalesce away.)
 */
template<int H, int W>
__device__ static inline void store_acc_direct(
    kittens::bf16* __restrict__ c_global,
    int wgr_base, int wgc_base, int N, int M_rows,
    const float2 (&lacc)[H][W][4])
{
    #pragma unroll
    for (int n = 0; n < H; ++n)
        #pragma unroll
        for (int m = 0; m < W; ++m)
            store_acc16(c_global, wgr_base + n * 16, wgc_base + m * 16,
                        N, M_rows, lacc[n][m]);
}

} // namespace gfx1201_gemm
