/**
 * @file
 * @brief Global 4-bit weight codes -> a dequantized bf16 register tile.
 *
 * The layout is hk's repack of an AWQ or GPTQ checkpoint: for each row n of
 * W (an output feature), its K codes packed eight to an int32 in K order --
 * nibble t of qw[n][w] is column 8w + t -- and per group of G columns a
 * scale s[n][k/G] and the zero point times the scale, zs[n][k/G]. Element
 * (n, k) of W is then q * s - zs, one FMA.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

/**
 * @brief Load tile rows n0.. and columns k0.. of W from its 4-bit codes,
 * dequantized into a bf16 row-layout tile.
 *
 * qw is int32 [N, K/8]; s and zs are bf16 [N, K/G]. A lane holds a base
 * tile's 16 columns of its row, two words of codes, read as one 8-byte load.
 * G is a multiple of 16 and k0 of 16, so each base tile lies in one group;
 * when G is also a multiple of the tile's width and k0 of that width -- the
 * caller's contract then -- a row's whole tile shares one group and its
 * scale is read once.
 */
template<int G, ducks::rt::all RT, ducks::gl::all QL, ducks::gl::all SL>
__device__ inline static void load_w4(RT &dst, const QL &qw, const SL &s, const SL &zs,
                                      int n0, int k0) {
    using T = typename base_types::packing<typename RT::dtype>::unpacked_type;
    using L = typename RT::layout;
    using base = rt_base<T, L>;
    static_assert(std::is_same_v<L, ducks::rt_layout::row>, "W4 codes load into row-layout tiles");
    static_assert(std::is_same_v<T, bf16>, "W4 codes dequantize into bf16 tiles");
    static_assert(base::element_stride == 1 && base::elements_per_thread == 16,
                  "a lane holds a base tile's 16 columns of its row");
    static_assert(std::is_same_v<typename QL::dtype, int>, "codes are packed in int32");
    static_assert(std::is_same_v<typename SL::dtype, bf16>, "scales are bf16");
    static_assert(G % 16 == 0, "a group spans whole base tiles");
    constexpr bool one_group = G % (16 * RT::width) == 0;

    const int r = laneid() & 15;
    const int words = qw.cols();
    const int groups = s.cols();
    const int *qp = reinterpret_cast<const int *>(qw.raw_ptr);
    const bf16 *sp = s.raw_ptr;
    const bf16 *zp = zs.raw_ptr;
    #pragma unroll
    for (int i = 0; i < dst.height; i++) {
        const int n = n0 + i * dst.tile_size_row + r;
        const int *row = qp + n * words + k0 / 8;
        float sc = 0.f, zsc = 0.f;
        if constexpr (one_group) {
            sc = base_types::convertor<float, bf16>::convert(sp[n * groups + k0 / G]);
            zsc = base_types::convertor<float, bf16>::convert(zp[n * groups + k0 / G]);
        }
        #pragma unroll
        for (int j = 0; j < dst.width; j++) {
            if constexpr (!one_group) {
                const int g = (k0 + j * dst.tile_size_col) / G;
                sc = base_types::convertor<float, bf16>::convert(sp[n * groups + g]);
                zsc = base_types::convertor<float, bf16>::convert(zp[n * groups + g]);
            }
            const int2 w = *reinterpret_cast<const int2 *>(row + 2 * j);
            #pragma unroll
            for (int h = 0; h < 2; h++) {
                const uint32_t word = uint32_t(h ? w.y : w.x);
                #pragma unroll
                for (int t = 0; t < 4; t++) {
                    const float a = float((word >> (8 * t)) & 0xFu) * sc - zsc;
                    const float b = float((word >> (8 * t + 4)) & 0xFu) * sc - zsc;
                    dst.tiles[i][j].data[4 * h + t] =
                        base_types::convertor<bf16_2, float2>::convert(float2{a, b});
                }
            }
        }
    }
}

} // namespace kittens
