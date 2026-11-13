#pragma once
#include "base.cuh"
#include "vec.cuh"

// Output RMSNorm: out = (A @ B^T) * r[:, None] * gamma[None, :]. r = 1/rms of A @ B^T; gamma has N entries
// (out_gamma), not the K-entry pre-norm gamma.
struct RMSNormScaleGlobals {
    _gl_A a; _gl_B b; _gl_C c;
    gl<float, -1, -1, -1, -1> r;      // [M] fp32
    gl<bf16, -1, -1, -1, -1> gamma;   // [N] bf16
};
struct RMSNormScaleEpilogue {
    template<typename Globals, typename Accum>
    static __device__ inline void apply(const Globals& g, Accum& C, int row, int col, int wr, int wc) {
        apply_inv_rms(g, C, row, col, wr, wc);
        apply_gamma(g, C, row, col, wr, wc);
        store_C(g, C, row, col, wr, wc);
    }
};
