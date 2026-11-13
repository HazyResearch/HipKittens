#pragma once
#include "base.cuh"
#include "vec.cuh"
#include "rotary.cuh"

// out = RoPE(r[:, None] * (A @ B^T)). A = hg (input times gamma), B the plain weight, r per-row 1/rms
// (commutes with the rotation).
struct RmsnormRopeGlobals {
    _gl_A a; _gl_B b; _gl_C c;
    gl<float, -1, -1, -1, -1> r;        // [M] fp32
    gl<bf16, -1, -1, -1, -1> cos_sin;   // [M, Dh] bf16, one head
    int rope_cols;                      // rotated: [0, rope_cols)
};
struct RmsnormRopeEpilogue {
    static constexpr bool split_rope_pairs = true;
    template<typename Globals, typename Accum>
    static __device__ inline void apply(const Globals& g, Accum& C, int row, int col, int wr, int wc) {
        apply_inv_rms(g, C, row, col, wr, wc);
        apply_rope_store_natural(g, C, row, col, wr, wc);
    }
};
