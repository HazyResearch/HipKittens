#pragma once
#include "base.cuh"
#include "vec.cuh"

// Output RMSNorm: out = (A @ B^T) * r[:, None] * gamma[None, :]. It normalizes the GEMM's OUTPUT: r is the
// precomputed per-row 1/rms of A @ B^T (hk.inv_rms(x, w)) and gamma is that norm's weight over the N output
// columns (hk calls it out_gamma). Not the pre-norm gamma of A, which has K entries. r first, then gamma.
struct RMSNormScaleGlobals {
    _gl_A a; _gl_B b; _gl_C c;
    gl<float, -1, -1, -1, -1> r;      // [M] fp32; as a 4-D gl: [1, 1, 1, M]
    gl<bf16, -1, -1, -1, -1> gamma;   // [N] bf16; as a 4-D gl: [1, 1, 1, N]
};
struct RMSNormScaleEpilogue {
    template<typename Globals, typename Accum>
    static __device__ inline void apply(const Globals& g, Accum& C, int row, int col, int wr, int wc) {
        apply_inv_rms(g, C, row, col, wr, wc);
        apply_gamma(g, C, row, col, wr, wc);
        store_C(g, C, row, col, wr, wc);
    }
};
