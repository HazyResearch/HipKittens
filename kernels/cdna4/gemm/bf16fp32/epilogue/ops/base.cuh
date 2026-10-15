#pragma once
#include "epilogue_args.cuh"

// Where a warp's 2x2 accumulator sub-tiles land in the output, in sub-tile units
// (HALF_REG_BLOCK_M = 64 rows, HALF_REG_BLOCK_N = 32 cols). Same mapping as the base GEMM's
// final stores. Every op in ops/ takes its coordinates from here.
//   row, col : this block's 256 x 256 output tile
//   wr, wc   : this warp's position in the block (WARPS_M x WARPS_N = 2 x 4)
struct subtile_coords { int m[SUBTILES_PER_DIM]; int n[SUBTILES_PER_DIM]; };
__device__ inline subtile_coords block_coords(int row, int col, int wr, int wc) {
    return { { row*SUBTILES_PER_DIM*WARPS_M + wr, row*SUBTILES_PER_DIM*WARPS_M + WARPS_M + wr },
             { col*SUBTILES_PER_DIM*WARPS_N + wc, col*SUBTILES_PER_DIM*WARPS_N + WARPS_N + wc } };
}

// Store all four accumulator sub-tiles to g.c (output has the GEMM's [M, N] shape).
template<typename Globals, typename Accum>
__device__ inline void store_C(const Globals& g, const Accum& C, int row, int col, int wr, int wc) {
    subtile_coords co = block_coords(row, col, wr, wc);
    store(g.c, C[0][0], {0, 0, co.m[0], co.n[0]});
    store(g.c, C[0][1], {0, 0, co.m[0], co.n[1]});
    store(g.c, C[1][0], {0, 0, co.m[1], co.n[0]});
    store(g.c, C[1][1], {0, 0, co.m[1], co.n[1]});
}

// Half-width store for SwiGLU. After the epilogue, the result sits in the gate sub-tiles C[*][0];
// C[*][1] held the matching value columns and is no longer needed. The output c is [M, N/2]: the
// gate sub-tile at column-chunk n0 = col*8 + wc (wc < 4, 32-col chunks) goes to output chunk col*4 + wc.
template<typename Globals, typename Accum>
__device__ inline void store_swiglu(const Globals& g, const Accum& C, int row, int col, int wr, int wc) {
    constexpr int CHUNKS_PER_BLOCK = BLOCK_SIZE / HALF_REG_BLOCK_N;        // 8 x 32-col chunks per 256 cols
    constexpr int CHUNKS_PER_HALF  = HALF_BLOCK_SIZE / HALF_REG_BLOCK_N;   // 4 chunks in the 128-col gate half
    subtile_coords co = block_coords(row, col, wr, wc);
    const int out_n = (co.n[0] / CHUNKS_PER_BLOCK) * CHUNKS_PER_HALF + (co.n[0] % CHUNKS_PER_BLOCK);
    store(g.c, C[0][0], {0, 0, co.m[0], out_n});
    store(g.c, C[1][0], {0, 0, co.m[1], out_n});
}
