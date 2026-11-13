#pragma once
#include <type_traits>
#include "base.cuh"

// partials[grp, m] = sum of squares of row m over this warp's 64 columns, grp = col*WARPS_N + wc
// (chunks wc and wc+4 of the block, not adjacent). partials is [1, 1, N/64, M].
template<typename Globals, typename Accum>
__device__ inline void partial_row_sum_sq(const Globals& g, const Accum& C, int row, int col, int wr, int wc) {
    using Tile = std::remove_all_extents_t<Accum>;
    using CV   = typename Tile::col_vec;
    Tile sq; CV p0, p1;
    mul(sq, C[0][0], C[0][0]); row_sum(p0, sq);
    mul(sq, C[0][1], C[0][1]); row_sum(p0, sq, p0);
    mul(sq, C[1][0], C[1][0]); row_sum(p1, sq);
    mul(sq, C[1][1], C[1][1]); row_sum(p1, sq, p1);
    const int grp = col * WARPS_N + wc;
    subtile_coords co = block_coords(row, col, wr, wc);
    store(g.partials, p0, {0, 0, grp, co.m[0]});
    store(g.partials, p1, {0, 0, grp, co.m[1]});
}
