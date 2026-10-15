#pragma once
#include "base.cuh"
#include "activations.cuh"

// out = silu(A @ B^T). No extra inputs, so it uses gemm_args_base.
struct SiluEpilogue {
    template<typename Globals, typename Accum>
    static __device__ inline void apply(const Globals& g, Accum& C, int row, int col, int wr, int wc) {
        silu_op(C[0][0]); silu_op(C[0][1]);
        silu_op(C[1][0]); silu_op(C[1][1]);
        store_C(g, C, row, col, wr, wc);
    }
};
