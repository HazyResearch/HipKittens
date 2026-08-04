/**
 * @file common.h
 * @brief Shared types and tile geometry for the gfx1250 GEMM ladder.
 *
 * Every rung includes this, so the rungs differ only in their compute body. It fixes the
 * operand and output types, derives the tile shape from five macros a rung may override
 * before including it, and carries the two contracts a rung cannot check for itself: the
 * LDS segment placement and the minimum K.
 */

#pragma once

#include <cstdio>
#include <cstdlib>
#include "kittens.cuh"

namespace gfx1250_gemm {

/* ----------  TILE GEOMETRY  ----------
 *
 * The macro tile is BLOCK_M x BLOCK_N of output, split across a WARPS_M x WARPS_N grid of
 * warps. Bigger tiles do more math per byte of operand fetched -- arithmetic intensity for a
 * square tile is M*N/(M+N) FLOP per byte -- which is why the ladder climbs through 64x64,
 * 128x128 and 256x256. 256x256 is the largest the register file allows: at four waves per
 * SIMD the accumulator alone sits at 244 VGPRs of 256.
 *
 * K_STEP is the matrix instruction's K depth, fixed at 32. BLOCK_K is how much K one LDS
 * stage holds, so BLOCK_K / K_STEP sub-steps run per fill. Deepening it amortises the
 * per-block rendezvous over more math and puts the LDS row at 256 B, which is eight lanes
 * per cache line and half the fill requests.
 *
 * Defaults are the smallest rung's shape; the rungs above set what they need.
 */
#ifndef GFX1250_BLOCK_M
#define GFX1250_BLOCK_M 64
#endif
#ifndef GFX1250_BLOCK_N
#define GFX1250_BLOCK_N 64
#endif
#ifndef GFX1250_K_STEP
#define GFX1250_K_STEP 32
#endif
#ifndef GFX1250_BLOCK_K
#define GFX1250_BLOCK_K GFX1250_K_STEP
#endif
#ifndef GFX1250_WARPS_M
#define GFX1250_WARPS_M 2
#endif
#ifndef GFX1250_WARPS_N
#define GFX1250_WARPS_N 2
#endif

constexpr int BLOCK_M     = GFX1250_BLOCK_M;
constexpr int BLOCK_N     = GFX1250_BLOCK_N;
constexpr int K_STEP      = GFX1250_K_STEP;
constexpr int BLOCK_K     = GFX1250_BLOCK_K;
constexpr int WARPS_M     = GFX1250_WARPS_M;
constexpr int WARPS_N     = GFX1250_WARPS_N;
constexpr int WARP_M      = BLOCK_M / WARPS_M;
constexpr int WARP_N      = BLOCK_N / WARPS_N;
constexpr int NUM_WARPS   = WARPS_M * WARPS_N;
constexpr int NUM_THREADS = NUM_WARPS * kittens::WARP_THREADS;
constexpr int K_SUBBLOCKS = BLOCK_K / K_STEP;

/* ----------  TYPES  ----------
 *
 * Operands are bf16 and the accumulator is fp32. `-DGFX1250_ELEM=half` builds the same
 * kernels on `v_wmma_f32_16x16x32_f16` instead; the accumulator stays fp32 either way.
 *
 * Operands are TN: `a` is [M, K] and `b` is [N, K], both K-contiguous, which is the layout
 * the matrix instruction's fragments want, so the kernel computes C = A . B^T.
 *
 * C is column-major, and that lives on the type rather than at the call site so the kernel,
 * the store and the harness's reference all read it from one place and cannot disagree.
 */
#ifndef GFX1250_ELEM
#define GFX1250_ELEM bf16
#endif
using elem_t = kittens::GFX1250_ELEM;
static_assert(sizeof(elem_t) == 2, "GFX1250_ELEM must be a 16-bit float type");

using gl_e = kittens::gl<elem_t, -1, -1, -1, -1>;
using gl_c = kittens::gl<elem_t, -1, -1, -1, -1, kittens::ducks::gl_layout::col_major>;

template<int R, int C> using st_e =
    kittens::st<elem_t, R, C, kittens::ducks::st_shape::st_16x32_padded<>>;

/* The operand fragment the matrix instruction reads, in the 16x32 shape it wants. The
 * accumulator is a separate type, `rt_fl`, because it is fp32 and laid out by column. */
template<int R, int C> using rt_e =
    kittens::rt<elem_t, R, C, kittens::ducks::rt_layout::row, kittens::ducks::rt_shape::rt_16x32>;

using A_tile = st_e<BLOCK_M, K_STEP>;
using B_tile = st_e<BLOCK_N, K_STEP>;

/* The epilogue stages C through LDS before writing it out, reusing the operand rings once
 * the K loop is done. Padding the staged rows every 128 elements keeps the read-back from
 * conflicting on LDS banks. */
using C_tile = kittens::st<elem_t, BLOCK_M, BLOCK_N,
                           kittens::ducks::st_shape::st_16x32_padded<128, 8>>;

/* A rung consumes K in whole blocks of BLOCK_K, so it has no answer for a K shorter than one
 * block. Each kernel returns early in that case, before any barrier or fill, so no peer is left
 * waiting on a workgroup that has gone home. This refuses the launch first, so an
 * out-of-contract shape reads as a sentence rather than as an empty output buffer. */
inline void require_k_blocks(int K, const char* rung)
{
    if (K / BLOCK_K >= 1) return;
    std::fprintf(stderr, "%s: K=%d is shorter than one block of BLOCK_K=%d. Refusing.\n",
                 rung, K, BLOCK_K);
    std::abort();
}

/* `ds_load_b128` count for one warp's WARP_DIM x K_STEP fragment: two wide loads per 16x32
 * subtile. Rungs that double-buffer operands in registers use it to size a partial LDS drain,
 * so the previous sub-step's loads retire while this one's are still in flight. */
template<int WARP_DIM>
__device__ __host__ constexpr int ds_loads_per_subblock() { return (WARP_DIM / 16) * 2; }

struct gemm_globals {
    gl_e a, b;
    gl_c c;
    hipStream_t stream;
    int M() const { return a.rows(); }
    int N() const { return c.cols(); }
    int K() const { return a.cols(); }
    dim3 grid()  const { return dim3(M() / BLOCK_M, N() / BLOCK_N); }
    dim3 block() const { return dim3(NUM_THREADS); }
    template <int STAGES = 2>
    size_t dynamic_shared_memory() const { return STAGES * (sizeof(A_tile) + sizeof(B_tile)); }
};

} // namespace gfx1250_gemm
