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

    if constexpr (std::is_same_v<typename RV1::layout, typename RV2::layout>) { // just a simple copy /typecast
        #pragma unroll
        for(int i = 0; i < RV1::outer_dim; i++) {
            #pragma unroll
            for(int j = 0; j < RV1::inner_dim; j++) {
                dst[i][j] = base_types::convertor<D2, D1>::convert(src[i][j]);
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