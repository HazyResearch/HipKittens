#pragma once
#include <type_traits>
#include "base.cuh"

// Broadcast a vector over the accumulator. A per-ROW value (one per output row, e.g. 1/rms) lives in
// a col_vec and is applied with mul_row; a per-COLUMN value (one per output feature, e.g. gamma)
// lives in a row_vec and is applied with mul_col. The vector kind names its shape, not what it scales.

// C[m, :] *= r[m]. r is fp32 [1, 1, 1, M] (M on the last axis); a col_vec loads by its last-axis index.
template<typename Globals, typename Accum>
__device__ inline void apply_inv_rms(const Globals& g, Accum& C, int row, int col, int wr, int wc) {
    using Tile = std::remove_all_extents_t<Accum>;
    using CV   = typename Tile::col_vec;
    subtile_coords co = block_coords(row, col, wr, wc);
    CV r0, r1;
    load(r0, g.r, {co.m[0]});                            // rows of C[0][*]
    load(r1, g.r, {co.m[1]});                            // rows of C[1][*]
    mul_row(C[0][0], C[0][0], r0); mul_row(C[0][1], C[0][1], r0);
    mul_row(C[1][0], C[1][0], r1); mul_row(C[1][1], C[1][1], r1);
}

// C[:, n] *= gamma[n]. gamma is bf16 [1, 1, 1, N]; a row_vec loads by full coordinates.
template<typename Globals, typename Accum>
__device__ inline void apply_gamma(const Globals& g, Accum& C, int row, int col, int wr, int wc) {
    using Tile = std::remove_all_extents_t<Accum>;
    using RV   = typename Tile::row_vec;
    subtile_coords co = block_coords(row, col, wr, wc);
    RV g0, g1;
    load(g0, g.gamma, {0, 0, 0, co.n[0]});               // columns of C[*][0]
    load(g1, g.gamma, {0, 0, 0, co.n[1]});               // columns of C[*][1]
    mul_col(C[0][0], C[0][0], g0); mul_col(C[1][0], C[1][0], g0);
    mul_col(C[0][1], C[0][1], g1); mul_col(C[1][1], C[1][1], g1);
}

// Multiply every accumulator value by the scalar alpha, read on the GPU from a 1-element fp32
// tensor (g.alpha). Because it is read at run time, alpha can change between launches (or inside
// a captured graph) without recompiling.
template<typename Globals, typename Accum>
__device__ inline void apply_scale(const Globals& g, Accum& C) {
    const float a = g.alpha[{0, 0, 0, 0}];
    mul(C[0][0], C[0][0], a); mul(C[0][1], C[0][1], a);
    mul(C[1][0], C[1][0], a); mul(C[1][1], C[1][1], a);
}
