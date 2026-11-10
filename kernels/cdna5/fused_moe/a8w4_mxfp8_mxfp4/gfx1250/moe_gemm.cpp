// moe_gemm.cpp -- entry translation unit for the gfx1250 fused MoE GEMM. Assembles the shared core
// and the per-op fragments, then exports the pybind module. The runtime-adaptive BLOCK_M dispatcher
// (moe_gemm_adaptive.cpp) #includes THIS file once per BLOCK_M inside a namespace; -DMOE_AS_ENGINE
// then skips the library headers (in moe_compute.cuh) and the pybind block below.
#include "moe_compute.cuh"   // config, globals, moe_compute_tile core, load/scale helpers, host router
#include "moe_gemm1.cpp"     // GEMM1 (f32 output) kernel + dispatch
#include "moe_gemm2.cpp"     // GEMM2 down-projection kernel + dispatch
#include "moe_swiglu.cpp"    // SwiGLU + SwiGLU/MXFP8 re-quant kernels + dispatches

// ======================== PyBind ========================
// Skipped in engine mode (the adaptive TU includes this file per-BLOCK_M in namespaces; one module only).
#if !defined(MOE_AS_ENGINE)
PYBIND11_MODULE(gfx1250_moe_gemm, m) {
    m.doc() = "gfx1250 fused MoE GEMM (grouped MXFP8 x MXFP4, per-expert routing). "
              "TDM-load-path CONTRACT: the kernel may read up to 16B past the last K-tile of A and B, so "
              "callers MUST allocate >=16B slack at the END of the A and (whole packed) B buffers. B is a "
              "single packed pointer (fixed E*N*K/2 stride), so slack is at the buffer end, not per expert; "
              "a mid-buffer over-read lands in the next expert's bytes (loaded but unused). Scales are not "
              "over-read.";
    m.def("dispatch_raw",
        [](uintptr_t a_ptr, uintptr_t b_ptr, uintptr_t sa_ptr, uintptr_t sb_ptr,
           uintptr_t expt_ptr, uintptr_t c_ptr,
           int M, int N, int K, int E) {
            moe_check(a_ptr && b_ptr && sa_ptr && sb_ptr && expt_ptr && c_ptr,
                      "dispatch_raw: a/b/scale_a/scale_b/expt_data/c pointers must be non-null");
            moe_check(a_ptr % 16 == 0 && b_ptr % 16 == 0,
                      "dispatch_raw: a and b must be 16-byte aligned (b128 loads)");
            moe_check(sa_ptr % 4 == 0 && sb_ptr % 4 == 0,
                      "dispatch_raw: scale_a and scale_b must be 4-byte aligned (e8m0 uint32 loads)");
            moe_globals g{
                reinterpret_cast<const uint8_t *>(a_ptr),
                reinterpret_cast<const uint8_t *>(b_ptr),
                make_scale_a_gl(reinterpret_cast<const uint8_t *>(sa_ptr), M, K),
                make_scale_b_gl(reinterpret_cast<const uint8_t *>(sb_ptr), E, N, K),
                reinterpret_cast<const int32_t *>(expt_ptr),
                reinterpret_cast<float *>(c_ptr),
                M, N, K, E            };
            dispatch(g, moe_launch_config(g));
        },
        "Raw-pointer dispatch for grouped MoE GEMM (A=MXFP8, B=MXFP4 [E,N,K], expt_data[grid_m]).");
    m.def("dispatch_swiglu_raw",
        [](uintptr_t a_ptr, uintptr_t b_ptr, uintptr_t sa_ptr, uintptr_t sb_ptr,
           uintptr_t expt_ptr, uintptr_t c_ptr,
           int M, int N, int K, int E, float alpha, float limit, int add_residual,
           uintptr_t bias_ptr, uintptr_t gammas_ptr) {
            moe_check(a_ptr && b_ptr && sa_ptr && sb_ptr && expt_ptr && c_ptr,
                      "dispatch_swiglu_raw: a/b/scale_a/scale_b/expt_data/c pointers must be non-null");
            moe_check(a_ptr % 16 == 0 && b_ptr % 16 == 0,
                      "dispatch_swiglu_raw: a and b must be 16-byte aligned (b128 loads)");
            moe_check(sa_ptr % 4 == 0 && sb_ptr % 4 == 0,
                      "dispatch_swiglu_raw: scale_a and scale_b must be 4-byte aligned (e8m0 uint32 loads)");
            moe_swiglu_globals g{
                reinterpret_cast<const uint8_t *>(a_ptr),
                reinterpret_cast<const uint8_t *>(b_ptr),
                make_scale_a_gl(reinterpret_cast<const uint8_t *>(sa_ptr), M, K),
                make_scale_b_gl(reinterpret_cast<const uint8_t *>(sb_ptr), E, N, K),
                reinterpret_cast<const int32_t *>(expt_ptr),
                reinterpret_cast<kittens::bf16 *>(c_ptr),
                reinterpret_cast<const float *>(bias_ptr),
                reinterpret_cast<const float *>(gammas_ptr),
                M, N, K, E, alpha, limit, add_residual            };
            dispatch_swiglu(g, moe_launch_config(g));
        },
        "grouped MoE GEMM + fused SwiGLU (bf16 out [M,N/2]); bias/gammas ptr 0 = off.");
    m.def("dispatch_swiglu_mxfp8_raw",
        [](uintptr_t a_ptr, uintptr_t b_ptr, uintptr_t sa_ptr, uintptr_t sb_ptr,
           uintptr_t expt_ptr, uintptr_t c_ptr, uintptr_t cscale_ptr,
           int M, int N, int K, int E, float alpha, float limit, int add_residual,
           uintptr_t bias_ptr, uintptr_t gammas_ptr) {
            moe_check(a_ptr && b_ptr && sa_ptr && sb_ptr && expt_ptr && c_ptr && cscale_ptr,
                      "dispatch_swiglu_mxfp8_raw: a/b/scale_a/scale_b/expt_data/c/c_scale pointers must be non-null");
            moe_check(a_ptr % 16 == 0 && b_ptr % 16 == 0,
                      "dispatch_swiglu_mxfp8_raw: a and b must be 16-byte aligned (b128 loads)");
            moe_check(sa_ptr % 4 == 0 && sb_ptr % 4 == 0,
                      "dispatch_swiglu_mxfp8_raw: scale_a and scale_b must be 4-byte aligned (e8m0 uint32 loads)");
            moe_swiglu_mxfp8_globals g{
                reinterpret_cast<const uint8_t *>(a_ptr),
                reinterpret_cast<const uint8_t *>(b_ptr),
                make_scale_a_gl(reinterpret_cast<const uint8_t *>(sa_ptr), M, K),
                make_scale_b_gl(reinterpret_cast<const uint8_t *>(sb_ptr), E, N, K),
                reinterpret_cast<const int32_t *>(expt_ptr),
                reinterpret_cast<uint8_t *>(c_ptr),
                reinterpret_cast<uint8_t *>(cscale_ptr),
                reinterpret_cast<const float *>(bias_ptr),
                reinterpret_cast<const float *>(gammas_ptr),
                M, N, K, E, alpha, limit, add_residual            };
            dispatch_swiglu_mxfp8(g, moe_launch_config(g));
        },
        "grouped MoE GEMM + SwiGLU + MXFP8 re-quant (FP8 out [M,N/2] + e8m0 [M,(N/2)/32]).");
    m.def("dispatch_gemm2_raw",
        [](uintptr_t a_ptr, uintptr_t b_ptr, uintptr_t sa_ptr, uintptr_t sb_ptr,
           uintptr_t expt_ptr, uintptr_t c_ptr, uintptr_t residual_ptr,
           int M, int N, int K, int E) {
            moe_check(a_ptr && b_ptr && sa_ptr && sb_ptr && expt_ptr && c_ptr,
                      "dispatch_gemm2_raw: a/b/scale_a/scale_b/expt_data/c pointers must be non-null");
            moe_check(a_ptr % 16 == 0 && b_ptr % 16 == 0,
                      "dispatch_gemm2_raw: a and b must be 16-byte aligned (b128 loads)");
            moe_check(sa_ptr % 4 == 0 && sb_ptr % 4 == 0,
                      "dispatch_gemm2_raw: scale_a and scale_b must be 4-byte aligned (e8m0 uint32 loads)");
            moe_gemm2_globals g{
                reinterpret_cast<const uint8_t *>(a_ptr),
                reinterpret_cast<const uint8_t *>(b_ptr),
                make_scale_a_gl(reinterpret_cast<const uint8_t *>(sa_ptr), M, K),
                make_scale_b_gl(reinterpret_cast<const uint8_t *>(sb_ptr), E, N, K),
                reinterpret_cast<const int32_t *>(expt_ptr),
                reinterpret_cast<kittens::bf16 *>(c_ptr),
                reinterpret_cast<const kittens::bf16 *>(residual_ptr),
                M, N, K, E            };
            dispatch_gemm2(g, moe_launch_config(g));
        },
        "GEMM2 down-projection: grouped MXFP8 x MXFP4 -> bf16 [M,N] (residual ptr 0 = off).");
}
#endif
