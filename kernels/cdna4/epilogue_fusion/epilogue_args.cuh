#pragma once
#include <stdexcept>
#include <string>
#include "kittens.cuh"
using namespace kittens;

// Must match the base GEMM (256_256_64_32_with16x32.cpp).
constexpr int BLOCK_SIZE       = 256;
constexpr int HALF_BLOCK_SIZE  = BLOCK_SIZE / 2;
constexpr int K_STEP           = 64;
constexpr int WARPS_M          = 2;
constexpr int WARPS_N          = 4;
constexpr int NUM_WARPS        = WARPS_M * WARPS_N;
constexpr int NUM_THREADS      = kittens::WARP_THREADS * NUM_WARPS;
constexpr int REG_BLOCK_M      = BLOCK_SIZE / WARPS_M;   // output rows per warp
constexpr int REG_BLOCK_N      = BLOCK_SIZE / WARPS_N;   // output cols per warp
constexpr int HALF_REG_BLOCK_M = REG_BLOCK_M / 2;
constexpr int HALF_REG_BLOCK_N = REG_BLOCK_N / 2;
// Each warp's accumulator is rt_fl<...>[2][2].
constexpr int SUBTILES_PER_DIM = 2;
// The mainloop consumes two K steps per iteration.
constexpr int K_ALIGN          = 2 * K_STEP;

using _gl_A = gl<bf16, -1, -1, -1, -1>;
using _gl_B = gl<bf16, -1, -1, -1, -1>;
using _gl_C = gl<bf16, -1, -1, -1, -1>;

// a = [M,K], b = [N,K], c = [M,N]. Epilogues with inputs define a flat struct starting {a, b, c}.
struct gemm_args_base {
    _gl_A a;
    _gl_B b;
    _gl_C c;
};

// Throws (Python RuntimeError); CHECK_CUDA_ERROR only prints.
inline void hip_check(hipError_t err, const char* what) {
    if (err != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(err));
}
