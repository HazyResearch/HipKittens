#pragma once
#include "epilogue_args.cuh"

// Output coordinates of a warp's 2x2 sub-tiles in 64x32 sub-tile units, as in the base GEMM's stores.
// row, col: block tile; wr, wc: warp position (2 x 4).
struct subtile_coords { int m[SUBTILES_PER_DIM]; int n[SUBTILES_PER_DIM]; };
__device__ inline subtile_coords block_coords(int row, int col, int wr, int wc) {
    return { { row*SUBTILES_PER_DIM*WARPS_M + wr, row*SUBTILES_PER_DIM*WARPS_M + WARPS_M + wr },
             { col*SUBTILES_PER_DIM*WARPS_N + wc, col*SUBTILES_PER_DIM*WARPS_N + WARPS_N + wc } };
}

template<typename Globals, typename Accum>
__device__ inline void store_C(const Globals& g, const Accum& C, int row, int col, int wr, int wc) {
    subtile_coords co = block_coords(row, col, wr, wc);
    store(g.c, C[0][0], {0, 0, co.m[0], co.n[0]});
    store(g.c, C[0][1], {0, 0, co.m[0], co.n[1]});
    store(g.c, C[1][0], {0, 0, co.m[1], co.n[0]});
    store(g.c, C[1][1], {0, 0, co.m[1], co.n[1]});
}

// SwiGLU store of C[*][0] into c = [M, N/2]: 32-col chunk col*8 + wc goes to col*4 + wc.
template<typename Globals, typename Accum>
__device__ inline void store_swiglu(const Globals& g, const Accum& C, int row, int col, int wr, int wc) {
    constexpr int CHUNKS_PER_BLOCK = BLOCK_SIZE / HALF_REG_BLOCK_N;
    constexpr int CHUNKS_PER_HALF  = HALF_BLOCK_SIZE / HALF_REG_BLOCK_N;
    subtile_coords co = block_coords(row, col, wr, wc);
    const int out_n = (co.n[0] / CHUNKS_PER_BLOCK) * CHUNKS_PER_HALF + (co.n[0] % CHUNKS_PER_BLOCK);
    store(g.c, C[0][0], {0, 0, co.m[0], out_n});
    store(g.c, C[1][0], {0, 0, co.m[1], out_n});
}
