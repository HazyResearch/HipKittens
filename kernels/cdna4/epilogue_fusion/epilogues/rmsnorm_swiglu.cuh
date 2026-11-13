#pragma once
#include "base.cuh"
#include "vec.cuh"
#include "activations.cuh"

// out = silu(gate) * value, [gate | value] = r[:, None] * (A @ B^T). A = hg (input times gamma), B the plain
// W_gate_up; rmsnorm(x, gamma) @ W.t() == r * ((x * gamma) @ W.t()).
struct RmsnormSwigluGlobals {
    _gl_A a; _gl_B b; _gl_C c;
    gl<float, -1, -1, -1, -1> r;  // [M] fp32
};
struct RmsnormSwigluEpilogue {
    static constexpr bool split_gate_up = true;
    static constexpr int out_cols(int n) { return n / 2; }
    template<typename Globals, typename Accum>
    static __device__ inline void apply(const Globals& g, Accum& C, int row, int col, int wr, int wc) {
        apply_inv_rms(g, C, row, col, wr, wc);
        silu_op(C[0][0]); silu_op(C[1][0]);      // gate
        mul(C[0][0], C[0][0], C[0][1]);          // gate * value
        mul(C[1][0], C[1][0], C[1][1]);
        store_swiglu(g, C, row, col, wr, wc);
    }
};
