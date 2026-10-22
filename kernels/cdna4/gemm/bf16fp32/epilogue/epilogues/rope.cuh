#pragma once
#include "base.cuh"
#include "rotary.cuh"

// out = RoPE(A @ B^T) with interleaved pairs (2k, 2k+1), written in natural column order. B is the natural
// weight; split_rope_pairs pairs its rows in registers (gemm_base.cuh). cos_sin is one head's natural
// [M, Dh] table, shared by all heads (see ops/rotary.cuh).
struct RopeGlobals {
    _gl_A a; _gl_B b; _gl_C c;
    gl<bf16, -1, -1, -1, -1> cos_sin;   // [M, Dh] bf16: natural interleaved [cos, sin] of one head
    int rope_cols;                      // columns [0, rope_cols) are rotated, the rest stored as is
};
struct RopeEpilogue {
    static constexpr bool split_rope_pairs = true;
    template<typename Globals, typename Accum>
    static __device__ inline void apply(const Globals& g, Accum& C, int row, int col, int wr, int wc) {
        apply_rope_store_natural(g, C, row, col, wr, wc);
    }
};
