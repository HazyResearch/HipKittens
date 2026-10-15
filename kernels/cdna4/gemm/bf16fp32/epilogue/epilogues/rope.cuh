#pragma once
#include "base.cuh"
#include "rotary.cuh"

// out = RoPE(A @ B^T) with interleaved pairs (2k, 2k+1), written in natural column order. The caller
// permutes B's rows and cos_sin with rope_perm (see ops/rotary.cuh).
struct RopeGlobals {
    _gl_A a; _gl_B b; _gl_C c;
    gl<bf16, -1, -1, -1, -1> cos_sin;   // [M, N] bf16: interleaved [cos, sin], rope_perm'd columns
};
struct RopeEpilogue {
    template<typename Globals, typename Accum>
    static __device__ inline void apply(const Globals& g, Accum& C, int row, int col, int wr, int wc) {
        apply_rope_store_natural(g, C, row, col, wr, wc);
    }
};
