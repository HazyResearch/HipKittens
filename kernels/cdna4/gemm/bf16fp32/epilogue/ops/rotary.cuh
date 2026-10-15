#pragma once
#include <cstdint>
#include <type_traits>
#include "base.cuh"

// Interleaved RoPE: out[2k] = x*cos + y*sin, out[2k+1] = y*cos - x*sin, stored in natural column order.
// The caller permutes B's rows and cos_sin with rope_perm (pair k -> columns k%128 and k%128 + 128 of
// 256-column block k/128), so x is in C[i][0] and y in C[i][1] of the same thread.
// Each rotated pair is written as one round-to-nearest bf16x2 word at columns (2k, 2k+1); c must be
// 4-byte aligned. Lane L of a 16x16 base tile owns column L%16, so its pair is
// k = col*128 + wc*32 + j*16 + L%16.
template<typename Globals, typename Accum>
__device__ inline void apply_rope_store_natural(const Globals& g, Accum& C, int row, int col, int wr, int wc) {
    using Tile = std::remove_all_extents_t<Accum>;
    static_assert(std::is_same_v<typename Tile::layout, ducks::rt_layout::col>, "derived for the col_l accumulator");
    static_assert(std::is_same_v<typename Tile::shape, rt_16x16_s>, "derived for rt_16x16_s");
    static_assert(Tile::rows == HALF_REG_BLOCK_M && Tile::cols == HALF_REG_BLOCK_N, "derived for the GEMM sub-tile shape");
    static_assert(Tile::base_tile_rows == 16 && Tile::base_tile_cols == 16 &&
                  Tile::base_tile_stride == 4 && Tile::base_tile_num_strides == 1,
                  "rt_16x16_s lane/register mapping changed");

    subtile_coords co = block_coords(row, col, wr, wc);
    Tile cos_t, sin_t, t0, t1;
    const int lane = kittens::laneid();
    const int lane_col = lane % Tile::base_tile_cols;
    const int lane_row = Tile::base_tile_stride * (lane / Tile::base_tile_cols);
    const int pair_base = col * HALF_BLOCK_SIZE + wc * Tile::cols;

    #pragma unroll
    for (int i = 0; i < SUBTILES_PER_DIM; ++i) {
        load(cos_t, g.cos_sin, {0, 0, co.m[i], co.n[0]});   // cos_k, aligned with C[i][0]
        load(sin_t, g.cos_sin, {0, 0, co.m[i], co.n[1]});   // sin_k, aligned with C[i][1]
        mul(t0, C[i][1], sin_t);                             // y * sin
        mul(t1, C[i][1], cos_t);                             // y * cos (y no longer needed)
        mul(C[i][1], C[i][0], sin_t);
        sub(C[i][1], t1, C[i][1]);                           // odd  = y * cos - x * sin
        mul(C[i][0], C[i][0], cos_t);
        add(C[i][0], C[i][0], t0);                           // even = x * cos + y * sin

        #pragma unroll
        for (int tile_row = 0; tile_row < Tile::height; ++tile_row) {
            #pragma unroll
            for (int tile_col = 0; tile_col < Tile::width; ++tile_col) {
                const int out_col = 2 * (pair_base + tile_col * Tile::base_tile_cols + lane_col);
                #pragma unroll
                for (int reg = 0; reg < Tile::base_tile_stride / 2; ++reg) {
                    const float2 even = C[i][0].tiles[tile_row][tile_col].data[reg];
                    const float2 odd  = C[i][1].tiles[tile_row][tile_col].data[reg];
                    const int out_row = co.m[i] * Tile::rows + tile_row * Tile::base_tile_rows + lane_row + 2 * reg;
                    bf16* row0 = g.c.raw_ptr + static_cast<int64_t>(out_row) * g.c.cols() + out_col;
                    bf16* row1 = row0 + g.c.cols();
                    *reinterpret_cast<bf16_2*>(row0) = base_types::convertor<bf16_2, float2>::convert({even.x, odd.x});
                    *reinterpret_cast<bf16_2*>(row1) = base_types::convertor<bf16_2, float2>::convert({even.y, odd.y});
                }
            }
        }
    }
}
