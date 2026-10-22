#pragma once
#include <cstdint>
#include <type_traits>
#include "base.cuh"

// Interleaved RoPE: out[2k] = x*cos + y*sin, out[2k+1] = y*cos - x*sin, stored in natural column order.
// split_rope_pairs (gemm_base.cuh) loads pair k's two weight rows as B halves 0 and 1, so x is in C[i][0] and
// y in C[i][1] of the same thread (pair k -> columns k%128 and k%128 + 128 of 256-column block k/128).
// g.cos_sin is ONE head's natural interleaved table [M, Dh] ([m, 2j] = cos, [m, 2j+1] = sin of head pair j),
// shared by every head: pair k uses head pair j = k % (Dh/2). Columns >= g.rope_cols (a multiple of 256, so a
// whole block either side) are stored unrotated (cos 1, sin 0): fused QKV leaves V unrotated this way.
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
    const bool rotate = col * BLOCK_SIZE < g.rope_cols;     // uniform per block (rope_cols % 256 == 0)
    const int dh = g.cos_sin.cols();
    const int half_dh = dh / 2;

    #pragma unroll
    for (int i = 0; i < SUBTILES_PER_DIM; ++i) {
        // cos_t / sin_t in C[i]'s register layout: element (tile_row, tile_col, reg) is pair
        // pair_base + tile_col*16 + lane_col at rows out_row (.x) and out_row + 1 (.y), as in the store below.
        #pragma unroll
        for (int tile_row = 0; tile_row < Tile::height; ++tile_row) {
            #pragma unroll
            for (int tile_col = 0; tile_col < Tile::width; ++tile_col) {
                const int j = (pair_base + tile_col * Tile::base_tile_cols + lane_col) % half_dh;
                #pragma unroll
                for (int reg = 0; reg < Tile::base_tile_stride / 2; ++reg) {
                    float2 cos_v = {1.f, 1.f}, sin_v = {0.f, 0.f};
                    if (rotate) {
                        const int r0 = co.m[i] * Tile::rows + tile_row * Tile::base_tile_rows + lane_row + 2 * reg;
                        const bf16* p0 = g.cos_sin.raw_ptr + static_cast<int64_t>(r0) * dh + 2 * j;
                        const float2 cs0 = base_types::convertor<float2, bf16_2>::convert(*reinterpret_cast<const bf16_2*>(p0));
                        const float2 cs1 = base_types::convertor<float2, bf16_2>::convert(*reinterpret_cast<const bf16_2*>(p0 + dh));
                        cos_v = {cs0.x, cs1.x};
                        sin_v = {cs0.y, cs1.y};
                    }
                    cos_t.tiles[tile_row][tile_col].data[reg] = cos_v;
                    sin_t.tiles[tile_row][tile_col].data[reg] = sin_v;
                }
            }
        }
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
