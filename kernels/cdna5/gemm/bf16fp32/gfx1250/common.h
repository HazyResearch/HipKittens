/**
 * @file common.h
 * @brief Shared types for the gfx1250 GEMM ladder.
 *
 * Every rung includes this, so the rungs differ only in their compute body. It fixes the
 * operand and output types, builds the tile types from the geometry the rung declares, and
 * carries the two contracts a rung cannot check for itself: the LDS segment placement and
 * the minimum K.
 */

#pragma once

#include <cstdio>
#include <cstdlib>
#include "kittens.cuh"

namespace gfx1250_gemm {

/* Each rung declares its own tile geometry before including this. What follows uses
 * BLOCK_M, BLOCK_N, BLOCK_K, K_STEP and NUM_THREADS from it. See README.md. */

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
    int M() const { return a.rows(); }
    int N() const { return c.cols(); }
    int K() const { return a.cols(); }
};

/* Host-only launch state stays beside, never inside, the object copied into the kernel argument
 * buffer. A dispatch may compute its own LDS byte count when it has a nonstandard allocation. */
struct launch_config {
    hipStream_t stream;
    dim3 grid;
    dim3 block;

    explicit launch_config(const gemm_globals& g, hipStream_t launch_stream = 0)
        : stream(launch_stream),
          grid(g.M() / BLOCK_M, g.N() / BLOCK_N),
          block(NUM_THREADS) {}

    template <int STAGES = 2>
    size_t dynamic_shared_memory() const { return STAGES * (sizeof(A_tile) + sizeof(B_tile)); }
};

} // namespace gfx1250_gemm
