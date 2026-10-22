#pragma once
#include "base.cuh"
#include "vec.cuh"
#include "rotary.cuh"

// RMSNorm + RoPE: out = RoPE(r[:, None] * (A @ B^T)), interleaved pairs, natural column order.
// r is the precomputed per-row 1/rms. The caller folds the norm's gamma into B's rows; B is otherwise natural,
// paired in registers by split_rope_pairs (gemm_base.cuh). cos_sin is one head's natural [M, Dh] table, shared
// by all heads; columns >= rope_cols stay unrotated (ops/rotary.cuh). r is per-row, so it commutes with the
// rotation.
struct RmsnormRopeGlobals {
    _gl_A a; _gl_B b; _gl_C c;
    gl<float, -1, -1, -1, -1> r;        // [M] fp32; as a 4-D gl: [1, 1, 1, M]
    gl<bf16, -1, -1, -1, -1> cos_sin;   // [M, Dh] bf16: natural interleaved [cos, sin] of one head
    int rope_cols;                      // columns [0, rope_cols) are rotated, the rest stored as is
};
struct RmsnormRopeEpilogue {
    static constexpr bool split_rope_pairs = true;
    template<typename Globals, typename Accum>
    static __device__ inline void apply(const Globals& g, Accum& C, int row, int col, int wr, int wc) {
        apply_inv_rms(g, C, row, col, wr, wc);
        apply_rope_store_natural(g, C, row, col, wr, wc);
    }
};
