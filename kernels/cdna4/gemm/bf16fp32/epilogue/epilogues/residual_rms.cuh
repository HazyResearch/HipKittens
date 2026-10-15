#pragma once
#include <type_traits>
#include "base.cuh"
#include "reductions.cuh"

// h = (A @ B^T) + residual; stores h (bf16 [M, N], the next residual stream) and per-row partial sums
// of squares of h (fp32, before rounding to bf16), which rms_reduce turns into r = 1/rms(h).
struct ResidualRMSGlobals {
    _gl_A a; _gl_B b; _gl_C c;
    gl<bf16, -1, -1, -1, -1> residual;    // [M, N] bf16
    gl<float, -1, -1, -1, -1> partials;   // [N/64, M] fp32; as a 4-D gl: [1, 1, N/64, M] (row on the last axis)
};
struct ResidualRMSEpilogue {
    template<typename Globals, typename Accum>
    static __device__ inline void apply(const Globals& g, Accum& C, int row, int col, int wr, int wc) {
        using Tile = std::remove_all_extents_t<Accum>;
        subtile_coords co = block_coords(row, col, wr, wc);
        #pragma unroll
        for (int i = 0; i < SUBTILES_PER_DIM; ++i) {
            #pragma unroll
            for (int j = 0; j < SUBTILES_PER_DIM; ++j) {
                Tile t;
                load(t, g.residual, {0, 0, co.m[i], co.n[j]});
                add(C[i][j], C[i][j], t);
            }
        }
        partial_row_sum_sq(g, C, row, col, wr, wc);
        store_C(g, C, row, col, wr, wc);
    }
};
