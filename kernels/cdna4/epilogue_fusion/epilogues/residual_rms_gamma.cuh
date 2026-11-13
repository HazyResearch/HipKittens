#pragma once
#include <type_traits>
#include "base.cuh"
#include "reductions.cuh"
#include "vec.cuh"

// residual_rms plus hg = h * gamma: the next RMSNorm's gain is applied here in the producer, so the
// consuming GEMM uses an unfolded weight.
struct ResidualRMSGammaGlobals {
    _gl_A a; _gl_B b; _gl_C c;
    gl<bf16, -1, -1, -1, -1> residual;    // [M, N] bf16
    gl<float, -1, -1, -1, -1> partials;   // [N/64, M] fp32
    gl<bf16, -1, -1, -1, -1> gamma;       // [N] bf16
    gl<bf16, -1, -1, -1, -1> hg;          // [M, N] bf16
};
struct ResidualRMSGammaEpilogue {
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
        partial_row_sum_sq(g, C, row, col, wr, wc);   // unscaled h
        store_C(g, C, row, col, wr, wc);              // h
        apply_gamma(g, C, row, col, wr, wc);          // fp32
        #pragma unroll
        for (int i = 0; i < SUBTILES_PER_DIM; ++i) {
            #pragma unroll
            for (int j = 0; j < SUBTILES_PER_DIM; ++j)
                store(g.hg, C[i][j], {0, 0, co.m[i], co.n[j]});
        }
    }
};
