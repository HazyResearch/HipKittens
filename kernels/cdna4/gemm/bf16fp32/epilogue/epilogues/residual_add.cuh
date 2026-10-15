#pragma once
#include <type_traits>
#include "base.cuh"

// out = (A @ B^T) + residual, with residual a bf16 [M, N] skip connection.
struct ResidualAddGlobals {
    _gl_A a; _gl_B b; _gl_C c;
    gl<bf16, -1, -1, -1, -1> residual;   // [M, N] bf16
};
struct ResidualAddEpilogue {
    // One sub-tile at a time: load its residual, add, store, then move on. Storing each sub-tile as
    // soon as it is done ends its live range early.
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
                store(g.c, C[i][j], {0, 0, co.m[i], co.n[j]});
            }
        }
    }
};
