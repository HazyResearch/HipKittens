#pragma once
#include "base.cuh"
#include "activations.cuh"

// SwiGLU: out = silu(gate) * value, [M, 2*d_ff] -> [M, d_ff].
// Requires b's rows (the columns of W_gate_up) permuted so that, in every 256-column block, columns
// 0..127 are gate[j] and columns 128..255 are the matching value[j]. Then each thread holds a gate
// value in C[*][0] and its value partner in C[*][1], and the product needs no data movement.
// Permutation for d_ff = N/2: gate j -> column (j/128)*256 + j%128, value j -> that column + 128.
struct SwigluEpilogue {
    static constexpr int out_cols(int n) { return n / 2; }
    template<typename Globals, typename Accum>
    static __device__ inline void apply(const Globals& g, Accum& C, int row, int col, int wr, int wc) {
        silu_op(C[0][0]); silu_op(C[1][0]);
        mul(C[0][0], C[0][0], C[0][1]);
        mul(C[1][0], C[1][0], C[1][1]);
        store_swiglu(g, C, row, col, wr, wc);
    }
};
