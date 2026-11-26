/**
 * @file
 * @brief Reduction operations mapping tiles to vectors.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

/*
 * On gfx11 a tile reduction is one of exactly two shapes, and which one you get
 * depends only on whether the reduced axis is the tile's *element* axis or its
 * *lane* axis. See rt_base_coord() for those axes.
 *
 *   element axis  ->  the values are already in one lane, so the reduction is
 *                     register-local. Accumulators additionally need a single
 *                     cross-half exchange, because their two wave halves hold
 *                     alternating positions rather than mirrors. The result is
 *                     one value per lane: an rv_layout::ortho vector.
 *   lane axis     ->  the values are spread over the 16 lanes of a wave half,
 *                     so it is a 4-step butterfly. Both halves compute their own
 *                     copy, which is exactly what rv_layout::align wants. The
 *                     result keeps the tile's per-lane element count.
 *
 * The four (reduction, layout) combinations land on these two shapes as:
 *
 *            row layout          col layout
 *   row_red  element (ortho)     lane    (align)
 *   col_red  lane    (align)     element (ortho)
 *
 * and in every case the shape's output layout is the one rt_base already
 * declares as row_vec_layout / col_vec_layout, so the static_asserts below are
 * checking a derivation rather than a coincidence.
 *
 * Note there is no broadcast step anywhere. The CDNA versions end with a
 * packed_shfl from a leader lane; a butterfly already leaves the answer in every
 * lane, and the element-axis case never left one.
 */

namespace detail {

/// Reduce along the element axis (within each lane). Produces an ortho vector.
template<typename op, ducks::rv::all V, ducks::rt::all T, bool reset, bool by_row>
__device__ static inline void elem_axis_reduce(V &dst, const T &src, const V &src_accum) {
    using dtype = typename T::dtype;                                  // packed
    using RT    = typename base_types::packing<typename V::dtype>::unpacked_type;
    using base  = rt_base<typename T::T, typename T::layout>;

    static_assert(std::is_same_v<typename V::layout, ducks::rv_layout::ortho>);
    static_assert(V::outer_dim == (by_row ? T::height : T::width));

    constexpr int OUTER = by_row ? T::height : T::width;
    constexpr int INNER = by_row ? T::width  : T::height;

    #pragma unroll
    for(int a = 0; a < OUTER; a++) {
        dtype accum = by_row ? src.tiles[a][0].data[0] : src.tiles[0][a].data[0];
        #pragma unroll
        for(int b = 0; b < INNER; b++) {
            #pragma unroll
            for(int k = 0; k < T::packed_per_tile; k++) {
                if(b == 0 && k == 0) continue;            // already seeded
                const dtype v = by_row ? src.tiles[a][b].data[k] : src.tiles[b][a].data[k];
                accum = op::template op<dtype>(accum, v);
            }
        }
        RT single = op::template op<RT>(accum.x, accum.y);
        // The accumulator's halves hold alternating positions along this axis,
        // so each half has only half the values; operands mirror and need no
        // exchange at all.
        if constexpr (base::halves_interleave) {
            single = op::template op<RT>(single, lane_xor<HALF_WAVE_SWAP>(single));
        }
        dst[a][0] = reset ? single : op::template op<RT>(src_accum[a][0], single);
    }
}

/// Reduce along the lane axis (across a wave half). Produces an align vector.
template<typename op, ducks::rv::all V, ducks::rt::all T, bool reset, bool by_row>
__device__ static inline void lane_axis_reduce(V &dst, const T &src, const V &src_accum) {
    using RT2 = typename V::dtype;                                    // packed

    static_assert(std::is_same_v<typename V::layout, ducks::rv_layout::align>);
    static_assert(std::is_same_v<RT2, typename T::dtype>);
    static_assert(V::outer_dim == (by_row ? T::height : T::width));
    static_assert(V::inner_dim == T::packed_per_tile);

    constexpr int OUTER = by_row ? T::height : T::width;
    constexpr int INNER = by_row ? T::width  : T::height;

    #pragma unroll
    for(int a = 0; a < OUTER; a++) {
        RT2 accum[T::packed_per_tile];
        #pragma unroll
        for(int k = 0; k < T::packed_per_tile; k++) {
            accum[k] = by_row ? src.tiles[a][0].data[k] : src.tiles[0][a].data[k];
        }
        #pragma unroll
        for(int b = 1; b < INNER; b++) {
            #pragma unroll
            for(int k = 0; k < T::packed_per_tile; k++) {
                const RT2 v = by_row ? src.tiles[a][b].data[k] : src.tiles[b][a].data[k];
                accum[k] = op::template op<RT2>(accum[k], v);
            }
        }
        // Butterfly over the 16 lanes of a half. Masks stay below 16, so the
        // halves never mix -- each computes its own copy of the answer, which is
        // what align's replication means.
        #pragma unroll
        for(int k = 0; k < T::packed_per_tile; k++) {
            const RT2 r = half_wave_butterfly<op, RT2>(accum[k]);
            dst[a][k] = reset ? r : op::template op<RT2>(src_accum[a][k], r);
        }
    }
}

} // namespace detail

/**
 * @brief Reduce each row of a tile to a single value, over the columns.
 *
 * @tparam op The operation to be applied for reduction.
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @tparam reset Whether to ignore src_accum and start fresh.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when reset is false.
 */
template<typename op, ducks::rv::all V, ducks::rt::row_layout T, bool reset>
__device__ static inline void row_reduce(V &row_accum, const T &src, const V &src_accum) {
    static_assert(std::is_same_v<typename V::layout,
                  typename rt_base<typename T::T, typename T::layout>::col_vec_layout>);
    detail::elem_axis_reduce<op, V, T, reset, true>(row_accum, src, src_accum);
}
template<typename op, ducks::rv::all V, ducks::rt::col_layout T, bool reset>
__device__ static inline void row_reduce(V &row_accum, const T &src, const V &src_accum) {
    static_assert(std::is_same_v<typename V::layout,
                  typename rt_base<typename T::T, typename T::layout>::col_vec_layout>);
    detail::lane_axis_reduce<op, V, T, reset, true>(row_accum, src, src_accum);
}

/**
 * @brief Reduce each column of a tile to a single value, over the rows.
 *
 * @tparam op The operation to be applied for reduction.
 * @tparam V The vector type for the column accumulator.
 * @tparam T The matrix type.
 * @tparam reset Whether to ignore src_accum and start fresh.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when reset is false.
 */
template<typename op, ducks::rv::all V, ducks::rt::row_layout T, bool reset>
__device__ static inline void col_reduce(V &col_accum, const T &src, const V &src_accum) {
    static_assert(std::is_same_v<typename V::layout,
                  typename rt_base<typename T::T, typename T::layout>::row_vec_layout>);
    detail::lane_axis_reduce<op, V, T, reset, false>(col_accum, src, src_accum);
}
template<typename op, ducks::rv::all V, ducks::rt::col_layout T, bool reset>
__device__ static inline void col_reduce(V &col_accum, const T &src, const V &src_accum) {
    static_assert(std::is_same_v<typename V::layout,
                  typename rt_base<typename T::T, typename T::layout>::row_vec_layout>);
    detail::elem_axis_reduce<op, V, T, reset, false>(col_accum, src, src_accum);
}

/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

// two-operand row reductions. (Accumulate and REPLACE.)
/**
 * @brief Store the maximum of each row of the src register tile in the row_accum column vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void row_max(V &row_accum, const T &src)  {
    row_reduce<base_ops::max, V, T, true>(row_accum, src, row_accum);
}
/**
 * @brief Store the minimum of each row of the src register tile in the row_accum column vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void row_min(V &row_accum, const T &src)  {
    row_reduce<base_ops::min, V, T, true>(row_accum, src, row_accum);
}
/**
 * @brief Store the sum of each row of the src register tile in the row_accum column vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void row_sum(V &row_accum, const T &src)  {
    row_reduce<base_ops::sum, V, T, true>(row_accum, src, row_accum);
}
/**
 * @brief Store the product of each row of the src register tile in the row_accum column vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void row_prod(V &row_accum, const T &src) {
    row_reduce<base_ops::mul, V, T, true>(row_accum, src, row_accum);
}
// three-operand row reductions. (Accumulate ONTO.)
/**
 * @brief Store the maximum of each row of the src register tile, as well as the src_accum column vector, in the row_accum column vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void row_max(V &row_accum, const T &src, const V &src_accum)  {
    row_reduce<base_ops::max, V, T, false>(row_accum, src, src_accum);
}
/**
 * @brief Store the minimum of each row of the src register tile, as well as the src_accum column vector, in the row_accum column vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void row_min(V &row_accum, const T &src, const V &src_accum)  {
    row_reduce<base_ops::min, V, T, false>(row_accum, src, src_accum);
}
/**
 * @brief Store the sum of each row of the src register tile, as well as the src_accum column vector, in the row_accum column vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void row_sum(V &row_accum, const T &src, const V &src_accum)  {
    row_reduce<base_ops::sum, V, T, false>(row_accum, src, src_accum);
}
/**
 * @brief Store the product of each row of the src register tile, as well as the src_accum column vector, in the row_accum column vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void row_prod(V &row_accum, const T &src, const V &src_accum) {
    row_reduce<base_ops::mul, V, T, false>(row_accum, src, src_accum);
}

// two-operand col reductions. (Accumulate and REPLACE.)

/**
 * @brief Store the maximum of each column of the src register tile in the col_accum row vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void col_max(V &col_accum, const T &src)  {
    col_reduce<base_ops::max, V, T, true>(col_accum, src, col_accum);
}
/**
 * @brief Store the minimum of each column of the src register tile in the col_accum row vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void col_min(V &col_accum, const T &src)  {
    col_reduce<base_ops::min, V, T, true>(col_accum, src, col_accum);
}
/**
 * @brief Store the sum of each column of the src register tile in the col_accum row vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void col_sum(V &col_accum, const T &src)  {
    col_reduce<base_ops::sum, V, T, true>(col_accum, src, col_accum);
}
/**
 * @brief Store the product of each column of the src register tile in the col_accum row vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void col_prod(V &col_accum, const T &src) {
    col_reduce<base_ops::mul, V, T, true>(col_accum, src, col_accum);
}
// three-operand col reductions. (Accumulate ONTO.)
/**
 * @brief Store the maximum of each column of the src register tile, as well as the src_accum row vector, in the col_accum row vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void col_max(V &col_accum, const T &src, const V &src_accum)  {
    col_reduce<base_ops::max, V, T, false>(col_accum, src, src_accum);
}
/**
 * @brief Store the minimum of each column of the src register tile, as well as the src_accum row vector, in the col_accum row vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void col_min(V &col_accum, const T &src, const V &src_accum)  {
    col_reduce<base_ops::min, V, T, false>(col_accum, src, src_accum);
}
/**
 * @brief Store the sum of each column of the src register tile, as well as the src_accum row vector, in the col_accum row vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void col_sum(V &col_accum, const T &src, const V &src_accum)  {
    col_reduce<base_ops::sum, V, T, false>(col_accum, src, src_accum);
}
/**
 * @brief Store the product of each column of the src register tile, as well as the src_accum row vector, in the col_accum row vector.
 *
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when accumulating onto an existing value.
 */
template<ducks::rv::all V, ducks::rt::all T>
__device__ static inline void col_prod(V &col_accum, const T &src, const V &src_accum) {
    col_reduce<base_ops::mul, V, T, false>(col_accum, src, src_accum);
}

}