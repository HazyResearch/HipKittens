/**
 * @file
 * @brief Functions for transferring data directly between shared memory and registers and back.
 */

#pragma once

#include <type_traits>

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"
#include "../util/util.cuh"

namespace kittens {

/*
 * gfx11 register-vector layouts. All three follow from the tile layouts in
 * types/register/rt_base.cuh -- a register vector is just one axis of a tile.
 *
 *   naive : entry i lives in lane i%32, outer step i/32. No replication; the
 *           layout for plain vector work that never touches a tile.
 *   ortho : indexed along the *lane* axis (l%16), so entry 16*w + l%16 sits in
 *           data[w][0], replicated across the two wave halves.
 *   align : indexed along the *element* axis, so the vector is replicated across
 *           the 16 lanes of a half. Which entries a lane holds depends on the
 *           element type -- a bf16/half operand's lane covers all 16, while an
 *           f32 accumulator's two halves split the entries even/odd. That is
 *           what rv_align_elem() resolves.
 *
 * None of these are bandwidth-critical (a vector is 1/16 of a tile), so they are
 * all plain elementwise accesses. Lanes that map to the same entry issue the
 * same address and LDS broadcasts them.
 *
 * They do go through an address_space(3) pointer rather than src.data: a
 * shared vector reached through shared_allocator is a generic pointer (see
 * util.cuh), and generic accesses compile to flat_load/flat_store, which count
 * against vmcnt as well as lgkmcnt -- every wait on one becomes a wait on all
 * outstanding global traffic too. LDS completes in order, so the compiler's
 * own waits for these stay correct beside the library's inline-asm LDS ops.
 */
namespace detail {
// Moved as raw bits: bf16 and half are class types, and neither a reference
// nor `this` can bind to an address_space(3) object.
template<typename U>
using lds_bits = std::conditional_t<sizeof(U) == 2, uint16_t,
                 std::conditional_t<sizeof(U) == 4, uint32_t, uint64_t>>;
template<typename U>
__device__ inline U lds_read(const U *p) {
    static_assert(sizeof(U) == 2 || sizeof(U) == 4 || sizeof(U) == 8);
    using B = lds_bits<U>;
    const B b = *(__attribute__((address_space(3))) const B*)(uint32_t)(uintptr_t)p;
    U u;
    __builtin_memcpy(&u, &b, sizeof(U));
    return u;
}
template<typename U>
__device__ inline void lds_write(U *p, const U &u) {
    static_assert(sizeof(U) == 2 || sizeof(U) == 4 || sizeof(U) == 8);
    using B = lds_bits<U>;
    B b;
    __builtin_memcpy(&b, &u, sizeof(U));
    *(__attribute__((address_space(3))) B*)(uint32_t)(uintptr_t)p = b;
}
} // namespace detail

/**
 * @brief Load data from a shared vector into a register vector.
 *
 * @tparam RV The register vector type
 * @tparam SV The shared vector type
 * @param dst[out] The destination register vector.
 * @param src[in]  The source shared vector.
 */
template<ducks::rv::all RV, ducks::sv::all SV>
__device__ inline static void load(RV &dst, const SV &src) {
    using T2 = typename RV::dtype;
    using U  = typename SV::dtype;
    using T  = typename base_types::packing<T2>::unpacked_type;

    static_assert(SV::length == RV::length);

    const int lane = ::kittens::laneid();

    if constexpr (std::is_same_v<typename RV::layout, align_l>) {
        #pragma unroll
        for(int w = 0; w < dst.outer_dim; w++) {
            #pragma unroll
            for(int e = 0; e < rv_align_held<T>(); e++) {
                const int p = rv_align_entry<T>(e, lane);
                const T val = base_types::convertor<T, U>::convert(
                    detail::lds_read(&src.data[w*kittens::TILE_ROW_DIM<T> + p]));
                if(e & 1) dst.data[w][e>>1].y = val;
                else      dst.data[w][e>>1].x = val;
            }
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
        #pragma unroll
        for(int w = 0; w < dst.outer_dim; w++) {
            dst[w][0] = base_types::convertor<T, U>::convert(
                detail::lds_read(&src.data[w*kittens::TILE_ROW_DIM<T> + (lane & 15)]));
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {
        #pragma unroll
        for(int w = 0; w < dst.outer_dim; w++) {
            const int idx = w*WARP_THREADS + lane;
            if(idx < dst.length) {
                dst[w][0] = base_types::convertor<T, U>::convert(detail::lds_read(&src.data[idx]));
            }
        }
    }
}

/**
 * @brief Store data into a shared vector from a register vector.
 *
 * @tparam RV The register vector type
 * @tparam SV The shared vector type
 * @param dst[out] The destination shared vector.
 * @param src[in]  The source register vector.
 */
template<ducks::sv::all SV, ducks::rv::all RV>
__device__ inline static void store(SV &dst, const RV &src) {
    using T2 = typename RV::dtype;
    using U  = typename SV::dtype;
    using T  = typename base_types::packing<T2>::unpacked_type;

    static_assert(SV::length == RV::length);

    const int lane = ::kittens::laneid();

    // In the replicated layouts several lanes hold the same entry and all write
    // it. The values agree by construction, so the duplicate writes are benign.
    if constexpr (std::is_same_v<typename RV::layout, align_l>) {
        #pragma unroll
        for(int w = 0; w < src.outer_dim; w++) {
            #pragma unroll
            for(int e = 0; e < rv_align_held<T>(); e++) {
                const int p = rv_align_entry<T>(e, lane);
                const T val = (e & 1) ? src.data[w][e>>1].y : src.data[w][e>>1].x;
                detail::lds_write(&dst.data[w*kittens::TILE_ROW_DIM<T> + p],
                                  base_types::convertor<U, T>::convert(val));
            }
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
        #pragma unroll
        for(int w = 0; w < src.outer_dim; w++) {
            detail::lds_write(&dst.data[w*kittens::TILE_ROW_DIM<T> + (lane & 15)],
                              base_types::convertor<U, T>::convert(src[w][0]));
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {
        #pragma unroll
        for(int w = 0; w < src.outer_dim; w++) {
            const int idx = w*WARP_THREADS + lane;
            if(idx < src.length) {
                detail::lds_write(&dst.data[idx], base_types::convertor<U, T>::convert(src[w][0]));
            }
        }
    }
}

}
