#pragma once
#include "base.cuh"
#include "activations.cuh"

// out = silu(A @ B^T)
struct SiluEpilogue {
    template<typename Globals, typename Accum>
    static __device__ inline void apply(const Globals& g, Accum& C, int row, int col, int wr, int wc) {
        subtile_coords co = block_coords(row, col, wr, wc);
        #pragma unroll
        for (int i = 0; i < SUBTILES_PER_DIM; i++)
            #pragma unroll
            for (int j = 0; j < SUBTILES_PER_DIM; j++) {
                silu_op(C[i][j]);
                store(g.c, C[i][j], {0, 0, co.m[i], co.n[j]});
            }
    }
};
