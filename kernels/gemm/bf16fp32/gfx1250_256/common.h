/**
 * @file common.h
 * @brief Shared boilerplate for the gfx1250 256x256 GEMM (MI350X geometry).
 */

#pragma once

#include "kittens.cuh"

namespace gfx1250_gemm_256 {

constexpr int BLOCK_M         = 256;
constexpr int BLOCK_N         = 256;
constexpr int HALF_BLOCK      = BLOCK_M / 2;
constexpr int K_STEP          = 32;

constexpr int WARPS_M         = 2;
constexpr int WARPS_N         = 4;
constexpr int NUM_WAVES       = WARPS_M * WARPS_N;

constexpr int WAVE_M          = BLOCK_M / WARPS_M;
constexpr int WAVE_N          = BLOCK_N / WARPS_N;
constexpr int HALF_WAVE_M     = WAVE_M / 2;
constexpr int HALF_WAVE_N     = WAVE_N / 2;

constexpr int NUM_THREADS     = NUM_WAVES * kittens::WARP_THREADS;
constexpr int LAUNCH_MIN_WAVES_PER_SIMD = 2;

#define GEMM256_KERNEL_ATTR \
    __attribute__((amdgpu_waves_per_eu(LAUNCH_MIN_WAVES_PER_SIMD, LAUNCH_MIN_WAVES_PER_SIMD))) \
    __launch_bounds__(NUM_THREADS, LAUNCH_MIN_WAVES_PER_SIMD)

static_assert(HALF_WAVE_M % 16 == 0 && HALF_WAVE_N % 16 == 0,
              "WMMA accum subtiles must align to 16x16");
static_assert(K_STEP % 32 == 0, "gfx1250 WMMA K depth is 32 for bf16");

using gl_bf = kittens::gl<kittens::bf16, -1, -1, -1, -1>;

using A_quarter = kittens::st_bf<HALF_BLOCK, K_STEP>;
using B_quarter = kittens::st_bf<HALF_BLOCK, K_STEP>;

struct gemm_globals {
    gl_bf a, b, c;
    hipStream_t stream;
    int M() const { return a.rows(); }
    int N() const { return c.cols(); }
    int K() const { return a.cols(); }
    dim3   grid()  const { return dim3(M() / BLOCK_M, N() / BLOCK_N); }
    dim3   block() const { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() const {
        return 4 * sizeof(A_quarter) + 4 * sizeof(B_quarter);
    }
};

} // namespace gfx1250_gemm_256
