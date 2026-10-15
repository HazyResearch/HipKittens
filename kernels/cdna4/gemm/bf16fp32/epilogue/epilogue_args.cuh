#pragma once
#include <stdexcept>
#include <string>
#include "kittens.cuh"
using namespace kittens;

// Tiling of the base GEMM (kernels/cdna4/gemm/bf16fp32/256_256_64_32_with16x32.cpp).
// gemm_base.cuh copies that mainloop, so these values must stay in sync with it.
constexpr int BLOCK_SIZE       = 256;                    // 256 x 256 output tile per block
constexpr int HALF_BLOCK_SIZE  = BLOCK_SIZE / 2;
constexpr int K_STEP           = 64;                     // K consumed per mainloop step
constexpr int WARPS_M          = 2;
constexpr int WARPS_N          = 4;
constexpr int NUM_WARPS        = WARPS_M * WARPS_N;
constexpr int NUM_THREADS      = kittens::WARP_THREADS * NUM_WARPS;
constexpr int REG_BLOCK_M      = BLOCK_SIZE / WARPS_M;   // output rows per warp
constexpr int REG_BLOCK_N      = BLOCK_SIZE / WARPS_N;   // output cols per warp
constexpr int HALF_REG_BLOCK_M = REG_BLOCK_M / 2;
constexpr int HALF_REG_BLOCK_N = REG_BLOCK_N / 2;
// Each warp's accumulator is a fixed 2x2 grid of sub-tiles (rt_fl<...>[2][2]); ops/ indexes it explicitly.
constexpr int SUBTILES_PER_DIM = 2;
// The mainloop consumes K two steps at a time, so K must be a multiple of 2 * K_STEP = 128.
constexpr int K_ALIGN          = 2 * K_STEP;

using _gl_A = gl<bf16, -1, -1, -1, -1>;
using _gl_B = gl<bf16, -1, -1, -1, -1>;
using _gl_C = gl<bf16, -1, -1, -1, -1>;

// Kernel arguments for epilogues with no extra inputs: a = [M,K], b = [N,K], c = [M,N].
// Epilogues with inputs define their own flat struct that starts with {a, b, c}.
struct gemm_args_base {
    _gl_A a;
    _gl_B b;
    _gl_C c;
};

// Throw on a HIP error so Python sees a RuntimeError. (HK's CHECK_CUDA_ERROR only prints to stderr.)
inline void hip_check(hipError_t err, const char* what) {
    if (err != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(err));
}
