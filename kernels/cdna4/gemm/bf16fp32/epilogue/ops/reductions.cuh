#pragma once
#include <type_traits>
#include "base.cuh"

// Per-row partial sum of squares, one partial per (row, warp column group). Used by RMSNorm,
// which needs sum(x^2) over all N columns of a row; a separate reduction adds up the N/64 partials.
//
// Group grp = col*WARPS_N + wc covers the 64 columns this warp holds in block column `col`:
// two 32-column chunks, wc and wc+4, of that 256-column block (not adjacent; see block_coords).
// partials is [1, 1, N/64, M]: group index on axis 2, row on the last axis (a per-row col_vec
// stores along the last axis).
template<typename Globals, typename Accum>
__device__ inline void partial_row_sum_sq(const Globals& g, const Accum& C, int row, int col, int wr, int wc) {
    using Tile = std::remove_all_extents_t<Accum>;
    using CV   = typename Tile::col_vec;                   // one value per row
    Tile sq; CV p0, p1;
    mul(sq, C[0][0], C[0][0]); row_sum(p0, sq);            // rows of C[0][*]: first chunk
    mul(sq, C[0][1], C[0][1]); row_sum(p0, sq, p0);        //                  + second chunk
    mul(sq, C[1][0], C[1][0]); row_sum(p1, sq);            // rows of C[1][*]
    mul(sq, C[1][1], C[1][1]); row_sum(p1, sq, p1);
    const int grp = col * WARPS_N + wc;
    subtile_coords co = block_coords(row, col, wr, wc);
    store(g.partials, p0, {0, 0, grp, co.m[0]});
    store(g.partials, p1, {0, 0, grp, co.m[1]});
}
