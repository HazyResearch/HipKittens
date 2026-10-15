/**
 * @file
 * @brief Global 4-bit weight codes -> a dequantized bf16 or half register tile.
 *
 * The layout is hk's repack of an AWQ or GPTQ checkpoint. Codes are packed
 * eight to an int32 in K order -- nibble t of word w of row n is column
 * 8w + t -- and the words are interleaved by 16-row block and 32-column
 * chunk: [N/16][K/32][16 rows][4 words]. A load instruction then reads 16
 * rows' 16 bytes as 256 contiguous bytes, where a row-major layout made it
 * 16 lines 2 KB apart, 16 bytes of each. Per group of G columns there is a
 * scale s[n][k/G] and the zero point times the scale, zs[n][k/G], so
 * element (n, k) of W is q * s - zs, one FMA.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

namespace detail {

/// Eight codes of one word -> four 16-bit pairs of T (bf16 or half),
/// q * s - zs: columns (2t, 2t+1) in out[t]. The codes are spread one to a
/// byte by two masks and converted a byte at a time (v_cvt_f32_ubyte*).
/// bf16: each float is rounded half up and its top half taken, one add each
/// and one v_perm a pair. half: v_cvt_pkrtz, one instruction a pair. A
/// round-to-nearest-even conversion was four times the instructions, and the
/// dequantize, not the codes' bandwidth, was what bounded the kernel.
template<typename T>
__device__ inline void w4_word(uint32_t word, float sc, float zsc, uint32_t out[4]) {
    const uint32_t lo = word & 0x0F0F0F0Fu;          // columns 0, 2, 4, 6
    const uint32_t hi = (word >> 4) & 0x0F0F0F0Fu;   // columns 1, 3, 5, 7
    #pragma unroll
    for (int t = 0; t < 4; t++) {
        const float a = float((lo >> (8 * t)) & 0xFFu) * sc - zsc;
        const float b = float((hi >> (8 * t)) & 0xFFu) * sc - zsc;
        if constexpr (std::is_same_v<T, bf16>) {
            const uint32_t ua = __float_as_uint(a) + 0x8000u;
            const uint32_t ub = __float_as_uint(b) + 0x8000u;
            out[t] = __builtin_amdgcn_perm(ub, ua, 0x07060302u);
        } else {
            const auto p = __builtin_amdgcn_cvt_pkrtz(a, b);
            __builtin_memcpy(&out[t], &p, 4);
        }
    }
}

} // namespace detail

/**
 * @brief Load tile rows n0.. and columns k0.. of W from its 4-bit codes,
 * dequantized into a bf16 or half row-layout tile.
 *
 * qw is int32 [N, K/8] in the interleaved order above; s and zs are the tile's type,
 * [N, K/G]. A lane holds a base tile's 16 columns of its row, two words of
 * codes; it reads two base tiles' four words as one 16-byte load, so the
 * tile is a whole number of 32-column chunks and k0 a multiple of 32. G is a
 * multiple of 16, so each base tile lies in one group; when G is also a
 * multiple of the tile's width and k0 of that width -- the caller's contract
 * then -- a row's whole tile shares one group and its scale is read once.
 *
 * A 16-bit row tile holds each row twice, in lanes l and l + 16. Each half
 * dequantizes its own eight of a base tile's sixteen columns and takes the
 * other eight from its partner with v_permlanex16, so no code is converted
 * twice.
 */
template<int G, ducks::rt::all RT, ducks::gl::all QL, ducks::gl::all SL>
__device__ inline static void load_w4(RT &dst, const QL &qw, const SL &s, const SL &zs,
                                      int n0, int k0) {
    using T = typename base_types::packing<typename RT::dtype>::unpacked_type;
    using L = typename RT::layout;
    using base = rt_base<T, L>;
    static_assert(std::is_same_v<L, ducks::rt_layout::row>, "W4 codes load into row-layout tiles");
    static_assert(std::is_same_v<T, bf16> || std::is_same_v<T, half>,
                  "W4 codes dequantize into bf16 or half tiles");
    static_assert(base::element_stride == 1 && base::elements_per_thread == 16,
                  "a lane holds a base tile's 16 columns of its row");
    static_assert(std::is_same_v<typename QL::dtype, int>, "codes are packed in int32");
    static_assert(std::is_same_v<typename SL::dtype, T>, "scales are the tile's type");
    static_assert(G % 16 == 0, "a group spans whole base tiles");
    static_assert(RT::width % 2 == 0, "the tile is a whole number of 32-column chunks");
    constexpr bool one_group = G % (16 * RT::width) == 0;

    const int r = laneid() & 15;
    const bool upper = laneid() >= 16;           // this half's columns: 8..15
    const int chunks = qw.cols() / 4;            // 32-column chunks per row
    const int groups = s.cols();
    const int *qp = reinterpret_cast<const int *>(qw.raw_ptr);
    const T *sp = s.raw_ptr;
    const T *zp = zs.raw_ptr;
    #pragma unroll
    for (int i = 0; i < dst.height; i++) {
        const int n = n0 + i * dst.tile_size_row + r;
        // This row's words in chunk c: block (n / 16), chunk c, row r, 4 words.
        const int *blk = qp + (n >> 4) * chunks * 64 + r * 4;
        float sc = 0.f, zsc = 0.f;
        if constexpr (one_group) {
            sc = base_types::convertor<float, T>::convert(sp[n * groups + k0 / G]);
            zsc = base_types::convertor<float, T>::convert(zp[n * groups + k0 / G]);
        }
        #pragma unroll
        for (int j = 0; j < dst.width; j += 2) {
            const int4 w = *reinterpret_cast<const int4 *>(
                blk + ((k0 + j * dst.tile_size_col) / 32) * 64);
            #pragma unroll
            for (int jj = 0; jj < 2; jj++) {
                if constexpr (!one_group) {
                    const int g = (k0 + (j + jj) * dst.tile_size_col) / G;
                    sc = base_types::convertor<float, T>::convert(sp[n * groups + g]);
                    zsc = base_types::convertor<float, T>::convert(zp[n * groups + g]);
                }
                // Word 0 of a base tile is its columns 0..7, word 1 8..15.
                const uint32_t mine = uint32_t(jj ? (upper ? w.w : w.z) : (upper ? w.y : w.x));
                uint32_t own[4], other[4];
                detail::w4_word<T>(mine, sc, zsc, own);
                #pragma unroll
                for (int t = 0; t < 4; t++) {
                    other[t] = uint32_t(__builtin_amdgcn_permlanex16(
                        int(own[t]), int(own[t]), 0x76543210u, 0xfedcba98u, false, false));
                }
                #pragma unroll
                for (int t = 0; t < 4; t++) {
                    const uint32_t c0 = upper ? other[t] : own[t];
                    const uint32_t c1 = upper ? own[t] : other[t];
                    __builtin_memcpy(&dst.tiles[i][j + jj].data[t], &c0, 4);
                    __builtin_memcpy(&dst.tiles[i][j + jj].data[4 + t], &c1, 4);
                }
            }
        }
    }
}

} // namespace kittens
