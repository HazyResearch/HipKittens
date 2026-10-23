/**
 * Runtime-adaptive BLOCK_M MoE dispatcher (gfx1250) -- one binary, host router.
 *
 * Problem: BLOCK_M is a compile-time constant, so a single moe_gemm.cpp build is locked to one tile.
 * When top_k / #experts / #tokens change, routing density (tokens/expert) changes and a fixed BLOCK_M
 * is off-optimal (BM64 over-pads at ~48 tok/expert; BM16/32 waste ILP at high density). Triton adapts
 * its tile_m at dispatch; this TU gives HK the same: ONE binary with the kernel instantiated at
 * BLOCK_M in {16,32,48,64}, and a host router that picks the tile from the routing histogram.
 *
 * Mechanism: include the proven kernel engine (moe_gemm.cpp) once per BLOCK_M inside its own namespace
 * (headers + pybind skipped via -DMOE_AS_ENGINE). Each namespace's device code is byte-identical to
 * the corresponding single-BLOCK_M build, so per-tile correctness is already validated; the only new
 * code is host selection + a thin per-stage dispatch.
 *
 * BLOCK_M ladder = {16, 32, 48, 64}. BM128 is intentionally NOT instantiated: it GPU-page-faults on
 * this kernel (LDS/VGPR overflow at BM128/BN128/double-buffer).
 *
 * Build: pybind module `gfx1250_moe_gemm_adaptive` -- production API: recommend_block_m() +
 * dispatch_bm_raw() / dispatch_gemm2_bm_raw() / dispatch_swiglu_mxfp8_bm_raw().
 *
 * Callers pick BLOCK_M once with recommend_block_m(num_tokens, top_k, E), build the expert permutation
 * padded to that BLOCK_M, then call dispatch_*_bm_raw(bm, ...). The GEMM shapes (N,K) and the routing
 * (expt_data, padded M) must be built for the SAME bm.
 */

#include <stdexcept>
#include <hip/hip_runtime.h>
#include "kittens.cuh"
#include "moe_tiles.cuh"   // MoE-local shared-tile shapes + load_b_preshuffled (global kittens ns)
#include "pyutils/pyutils.cuh"   // pybind11 for the adaptive module
using namespace kittens;

// ---- Instantiate the engine once per BLOCK_M, each in its own namespace. -------------------------
// MOE_AS_ENGINE tells moe_gemm.cpp to skip its own header includes + pybind (provided here once).
// The engine is the typed-tile ("rely on ops") kernel. The compute config (default DECODE, or PREFILL
// via -DMOE_PREFILL) plus BLOCK_N / KTILE / MTILES / PRESHUFFLE come from -D on the build line (before
// the first include) so ALL tiles share one load/compute config; only BLOCK_M differs per namespace.
// (Do NOT define MOE_RAGGED_M -- ragged tiling is ruled out; narrow tiles are the fix.)
#define MOE_AS_ENGINE 1
namespace hk_bm16 {
#define MOE_BLOCK_M 16
#include "moe_gemm.cpp"
#undef MOE_BLOCK_M
}
namespace hk_bm32 {
#define MOE_BLOCK_M 32
#include "moe_gemm.cpp"
#undef MOE_BLOCK_M
}
namespace hk_bm48 {
#define MOE_BLOCK_M 48
#include "moe_gemm.cpp"
#undef MOE_BLOCK_M
}
namespace hk_bm64 {
#define MOE_BLOCK_M 64
#include "moe_gemm.cpp"
#undef MOE_BLOCK_M
}

// ---- Host router: effective-throughput tile selection over {16,32,48,64}. ------------------------
// Picks the BLOCK_M maximizing rate(bm) * useful_fraction, where useful_fraction = d/(ceil(d/bm)*bm)
// (1 - per-expert padding) for density d = ceil(num_tokens*top_k / E) avg tokens/expert, and rate(bm)
// is the measured per-row full-tile rate. Trades padding against per-tile ILP explicitly (HK cannot
// mask padding for free), which beats pure padding-minimization at boundary densities. The model AND
// the rate weights live in one place -- moe_recommend_block_m (moe_compute.cuh), the source of truth;
// this just delegates to it. The pybind `recommend_block_m` binds here.
static inline int moe_bm_pick(long long num_tokens, int top_k, int E) {
    return hk_bm16::moe_recommend_block_m(num_tokens, top_k, E);
}

// Switch helper: run STMT(NS) for the namespace matching bm; throw on an unsupported tile.
#define MOE_BM_SWITCH(bm, STMT)                                                   \
    switch (bm) {                                                                 \
        case 16: { STMT(hk_bm16) } break;                                         \
        case 32: { STMT(hk_bm32) } break;                                         \
        case 48: { STMT(hk_bm48) } break;                                         \
        case 64: { STMT(hk_bm64) } break;                                         \
        default: throw std::invalid_argument(                                     \
            "moe adaptive: BLOCK_M must be one of {16,32,48,64} (BM128 unsupported: LDS/VGPR overflow)"); \
    }

// ---- Adaptive per-stage dispatch (host). Each builds the chosen namespace's globals + launches. ---
// GEMM1 plain (f32 out).
void dispatch_moe_bm(int bm, const uint8_t *a, const uint8_t *b, const uint8_t *sa, const uint8_t *sb,
                     const int32_t *expt, float *c, int M, int N, int K, int E, hipStream_t s = 0) {
#define MK(NS) NS::moe_globals g{a, b, NS::make_scale_a_gl(sa, M, K), NS::make_scale_b_gl(sb, E, N, K), expt, c, M, N, K, E}; NS::dispatch(g, NS::moe_launch_config(g, s));
    MOE_BM_SWITCH(bm, MK)
#undef MK
}
// GEMM2 down-projection (bf16 out, optional residual).
void dispatch_gemm2_bm(int bm, const uint8_t *a, const uint8_t *b, const uint8_t *sa, const uint8_t *sb,
                       const int32_t *expt, kittens::bf16 *c, const kittens::bf16 *residual,
                       int M, int N, int K, int E, hipStream_t s = 0) {
#define MK(NS) NS::moe_gemm2_globals g{a, b, NS::make_scale_a_gl(sa, M, K), NS::make_scale_b_gl(sb, E, N, K), expt, c, residual, M, N, K, E}; NS::dispatch_gemm2(g, NS::moe_launch_config(g, s));
    MOE_BM_SWITCH(bm, MK)
#undef MK
}
// GEMM1 fused SwiGLU + MXFP8 re-quant (FP8 out + e8m0 scale).
void dispatch_swiglu_mxfp8_bm(int bm, const uint8_t *a, const uint8_t *b, const uint8_t *sa,
                              const uint8_t *sb, const int32_t *expt, uint8_t *c, uint8_t *c_scale,
                              const float *bias, const float *gammas, int M, int N, int K, int E,
                              float alpha, float limit, int add_residual, hipStream_t s = 0) {
#define MK(NS) NS::moe_swiglu_mxfp8_globals g{a, b, NS::make_scale_a_gl(sa, M, K), NS::make_scale_b_gl(sb, E, N, K), \
                                              expt, c, c_scale, bias, gammas, \
                                              M, N, K, E, alpha, limit, add_residual};    \
               NS::dispatch_swiglu_mxfp8(g, NS::moe_launch_config(g, s));
    MOE_BM_SWITCH(bm, MK)
#undef MK
}

// ================================= PyBind (production API) =========================================
static void adaptive_check(bool ok, const char *msg) { if (!ok) throw std::invalid_argument(msg); }

PYBIND11_MODULE(gfx1250_moe_gemm_adaptive, m) {
    m.doc() = "gfx1250 fused MoE GEMM with runtime-adaptive BLOCK_M (tiles {16,32,48,64} + host router)";

    m.def("recommend_block_m", &moe_bm_pick,
          "Pick BLOCK_M in {16,32,48,64} from (num_tokens, top_k, E) by maximizing effective useful "
          "throughput = per-tile rate * (1 - per-expert padding fraction).",
          pybind11::arg("num_tokens"), pybind11::arg("top_k"), pybind11::arg("E"));

    m.def("dispatch_bm_raw",
        [](int bm, uintptr_t a, uintptr_t b, uintptr_t sa, uintptr_t sb, uintptr_t expt, uintptr_t c,
           int M, int N, int K, int E) {
            adaptive_check(a && b && sa && sb && expt && c, "dispatch_bm_raw: null pointer");
            adaptive_check(a % 16 == 0 && b % 16 == 0, "dispatch_bm_raw: a,b must be 16B aligned");
            adaptive_check(sa % 4 == 0 && sb % 4 == 0, "dispatch_bm_raw: scales must be 4B aligned");
            dispatch_moe_bm(bm, reinterpret_cast<const uint8_t *>(a), reinterpret_cast<const uint8_t *>(b),
                            reinterpret_cast<const uint8_t *>(sa), reinterpret_cast<const uint8_t *>(sb),
                            reinterpret_cast<const int32_t *>(expt), reinterpret_cast<float *>(c),
                            M, N, K, E);
        },
        "Adaptive plain grouped GEMM (f32 out). bm from recommend_block_m; routing built for that bm.");

    m.def("dispatch_gemm2_bm_raw",
        [](int bm, uintptr_t a, uintptr_t b, uintptr_t sa, uintptr_t sb, uintptr_t expt, uintptr_t c,
           uintptr_t residual, int M, int N, int K, int E) {
            adaptive_check(a && b && sa && sb && expt && c, "dispatch_gemm2_bm_raw: null pointer");
            adaptive_check(a % 16 == 0 && b % 16 == 0, "dispatch_gemm2_bm_raw: a,b must be 16B aligned");
            adaptive_check(sa % 4 == 0 && sb % 4 == 0, "dispatch_gemm2_bm_raw: scales must be 4B aligned");
            dispatch_gemm2_bm(bm, reinterpret_cast<const uint8_t *>(a), reinterpret_cast<const uint8_t *>(b),
                              reinterpret_cast<const uint8_t *>(sa), reinterpret_cast<const uint8_t *>(sb),
                              reinterpret_cast<const int32_t *>(expt), reinterpret_cast<kittens::bf16 *>(c),
                              reinterpret_cast<const kittens::bf16 *>(residual), M, N, K, E);
        },
        "Adaptive GEMM2 down-projection (bf16 out; residual ptr 0 = off).");

    m.def("dispatch_swiglu_mxfp8_bm_raw",
        [](int bm, uintptr_t a, uintptr_t b, uintptr_t sa, uintptr_t sb, uintptr_t expt, uintptr_t c,
           uintptr_t c_scale, int M, int N, int K, int E, float alpha, float limit, int add_residual,
           uintptr_t bias, uintptr_t gammas) {
            adaptive_check(a && b && sa && sb && expt && c && c_scale,
                           "dispatch_swiglu_mxfp8_bm_raw: null pointer");
            adaptive_check(a % 16 == 0 && b % 16 == 0, "dispatch_swiglu_mxfp8_bm_raw: a,b must be 16B aligned");
            adaptive_check(sa % 4 == 0 && sb % 4 == 0, "dispatch_swiglu_mxfp8_bm_raw: scales must be 4B aligned");
            dispatch_swiglu_mxfp8_bm(bm, reinterpret_cast<const uint8_t *>(a),
                                     reinterpret_cast<const uint8_t *>(b), reinterpret_cast<const uint8_t *>(sa),
                                     reinterpret_cast<const uint8_t *>(sb), reinterpret_cast<const int32_t *>(expt),
                                     reinterpret_cast<uint8_t *>(c), reinterpret_cast<uint8_t *>(c_scale),
                                     reinterpret_cast<const float *>(bias), reinterpret_cast<const float *>(gammas),
                                     M, N, K, E, alpha, limit, add_residual);
        },
        "Adaptive GEMM1 fused SwiGLU + MXFP8 re-quant (FP8 out [M,N/2] + e8m0; bias/gammas ptr 0 = off).");
}
