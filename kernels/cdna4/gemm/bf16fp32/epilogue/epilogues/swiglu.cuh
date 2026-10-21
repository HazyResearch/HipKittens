#pragma once
#include "base.cuh"
#include "activations.cuh"

// SwiGLU: out = silu(gate) * value, [M, 2*d_ff] -> [M, d_ff].
// b is the natural W_gate_up operand ([N, K], rows [0, N/2) = gate, [N/2, N) = value). split_gate_up makes
// gemm_kernel load, for block col, gate rows [col*128, col*128+128) as B half 0 and the matching value rows
// as B half 1 (gemm_base.cuh), so each thread holds gate[j] in C[*][0] and value[j] in C[*][1] and the
// product needs no data movement.
struct SwigluEpilogue {
    static constexpr bool split_gate_up = true;
    static constexpr int out_cols(int n) { return n / 2; }
    template<typename Globals, typename Accum>
    static __device__ inline void apply(const Globals& g, Accum& C, int row, int col, int wr, int wc) {
        silu_op(C[0][0]); silu_op(C[1][0]);
        mul(C[0][0], C[0][0], C[0][1]);
        mul(C[1][0], C[1][0], C[1][1]);
        store_swiglu(g, C, row, col, wr, wc);
    }
};
