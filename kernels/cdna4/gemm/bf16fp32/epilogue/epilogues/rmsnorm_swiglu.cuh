#pragma once
#include "base.cuh"
#include "vec.cuh"
#include "activations.cuh"

// RMSNorm + SwiGLU (dim-reducing): out = silu(gate) * value, where [gate | value] = r[:, None] * (A @ B^T).
// r is the precomputed per-row 1/rms. The caller folds the norm's per-feature gamma into B's rows
// (rmsnorm(x, gamma) @ W == r * (x @ (gamma-folded W))) and permutes B's rows exactly as for
// SwiGLU (epilogues/swiglu.cuh), so gate and value are register-co-resident.
struct RmsnormSwigluGlobals {
    _gl_A a; _gl_B b; _gl_C c;
    gl<float, -1, -1, -1, -1> r;  // [M] fp32; as a 4-D gl: [1, 1, 1, M]
};
struct RmsnormSwigluEpilogue {
    static constexpr int out_cols(int n) { return n / 2; }
    template<typename Globals, typename Accum>
    static __device__ inline void apply(const Globals& g, Accum& C, int row, int col, int wr, int wc) {
        apply_inv_rms(g, C, row, col, wr, wc);   // r * (A @ B^T)
        silu_op(C[0][0]); silu_op(C[1][0]);      // silu on the gate pieces
        mul(C[0][0], C[0][0], C[0][1]);          // gate * value, register-only
        mul(C[1][0], C[1][0], C[1][1]);
        store_swiglu(g, C, row, col, wr, wc);    // half-width store
    }
};
