/**
 * @file
 * @brief Conversions on vectors stored in registers.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

template<ducks::rv::all RV2, ducks::rv::all RV1>
__device__ static inline void copy(RV2 &dst, const RV1 &src) {
    static_assert(RV1::length == RV2::length, "Register vectors must be the same length.");
    using D1 = RV1::dtype;
    using D2 = RV2::dtype;

    using D1_1 = base_types::packing<D1>::unpacked_type;
    using D1_2 = base_types::packing<D1_1>::packed_type;

    using D2_1 = base_types::packing<D2>::unpacked_type;
    using D2_2 = base_types::packing<D2_1>::packed_type;

    if constexpr (std::is_same_v<typename RV1::layout, typename RV2::layout>) {
        /*
         * naive and ortho give a lane one entry per outer step whatever the dtype,
         * so for them a typecast really is pointwise.
         *
         * align does not. Its outer step is a tile's element axis, and gfx11 spreads
         * that axis differently depending on replication -- see the copy() for
         * rt_base, which this mirrors. An f32 lane holds the 8 positions whose
         * parity matches its wave half; a bf16/half lane holds all 16, duplicated
         * across the halves. So the two have different inner_dims and a pointwise
         * loop would both read the wrong slots and leave half the destination
         * undefined.
         */
        constexpr bool is_align = std::is_same_v<typename RV1::layout, ducks::rv_layout::align>;
        if constexpr (!is_align || RV1::inner_dim == RV2::inner_dim) {
            #pragma unroll
            for(int i = 0; i < RV1::outer_dim; i++) {
                #pragma unroll
                for(int j = 0; j < RV1::inner_dim; j++) {
                    dst[i][j] = base_types::convertor<D2, D1>::convert(src[i][j]);
                }
            }
        }
        else if constexpr (kittens::WMMA_REPLICATION<D1_1> == 1) {
            // accumulator -> operand. Entry p of this lane is position 2p+h; the
            // mirror lane holds 2p+(1-h). One exchange per entry completes the pair,
            // and h decides which way round they interleave.
            const int h = laneid() >> 4;
            #pragma unroll
            for(int i = 0; i < RV1::outer_dim; i++) {
                #pragma unroll
                for(int p = 0; p < RV1::inner_dim*2; p++) {
                    const D1_1 own   = (p & 1) ? src[i][p>>1].y : src[i][p>>1].x;
                    const D1_1 other = lane_xor<HALF_WAVE_SWAP>(own);
                    dst[i][p].x = base_types::convertor<D2_1, D1_1>::convert(h ? other : own);
                    dst[i][p].y = base_types::convertor<D2_1, D1_1>::convert(h ? own   : other);
                }
            }
        }
        else {
            // operand -> accumulator. Pure discard: this lane keeps the positions
            // whose parity is its own. Convert before selecting -- see the rt_base
            // copy for why a select on bf16/half is what pushes the tile to scratch.
            const int h = laneid() >> 4;
            #pragma unroll
            for(int i = 0; i < RV1::outer_dim; i++) {
                #pragma unroll
                for(int p = 0; p < RV2::inner_dim*2; p++) {
                    const D2_1 lo = base_types::convertor<D2_1, D1_1>::convert(src[i][p].x);
                    const D2_1 hi = base_types::convertor<D2_1, D1_1>::convert(src[i][p].y);
                    const D2_1 c  = h ? hi : lo;
                    if(p & 1) dst[i][p>>1].y = c;
                    else      dst[i][p>>1].x = c;
                }
            }
        }
    } else if constexpr (std::is_same_v<typename RV1::layout, ducks::rv_layout::naive> && std::is_same_v<typename RV2::layout, ducks::rv_layout::align>) { 
        static_assert(false, "Unsupported layout conversion");
    } else if constexpr (std::is_same_v<typename RV1::layout, ducks::rv_layout::align> && std::is_same_v<typename RV2::layout, ducks::rv_layout::naive>) {
        static_assert(false, "Unsupported layout conversion");
    } else if constexpr (std::is_same_v<typename RV1::layout, ducks::rv_layout::align> && std::is_same_v<typename RV2::layout, ducks::rv_layout::ortho>) {
        static_assert(false, "Unsupported layout conversion");
    } else if constexpr (std::is_same_v<typename RV1::layout, ducks::rv_layout::ortho> && std::is_same_v<typename RV2::layout, ducks::rv_layout::align>) {
        static_assert(false, "Unsupported layout conversion");
    } else if constexpr (std::is_same_v<typename RV1::layout, ducks::rv_layout::ortho> && std::is_same_v<typename RV2::layout, ducks::rv_layout::naive>) {
        static_assert(false, "Unsupported layout conversion");
    } else if constexpr (std::is_same_v<typename RV1::layout, ducks::rv_layout::naive> && std::is_same_v<typename RV2::layout, ducks::rv_layout::ortho>) {
        static_assert(false, "Unsupported layout conversion");
    } else {
        static_assert(false, "Unsupported layout conversion");
    }
}

} // namespace kittens