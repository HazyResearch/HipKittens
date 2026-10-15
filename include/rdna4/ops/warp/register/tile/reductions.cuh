/**
 * @file
 * @brief Reduction operations mapping tiles to vectors.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

/* ---------- DPP butterfly cross-lane reduction ----------
 *
 * The shfl_down ladder below used `packed_shfl_down`, which lowers to
 * `ds_bpermute_b32` -- an LDS-PIPE instruction, even in a kernel with no LDS.
 * Every one costs an `s_waitcnt lgkmcnt(0)`. ATT on the gfx1201 attention
 * kernel measured those waits at **64.7% of ALL stall**: 32,896 hits
 * at 752 cycles each, 24.7M cycles, far ahead of the next term. The kernel has
 * zero LDS and zero barriers; this was the entire remaining bottleneck.
 *
 * A shfl_down ladder followed by a broadcast from the leader is equivalent, for
 * an associative and commutative op (max, sum, min, prod -- all of them here),
 * to a BUTTERFLY: xor-exchange at 1, 2, 4, 8. The butterfly leaves EVERY lane
 * holding the full result, so the broadcast disappears too.
 *
 * DPP's `row_xmask:N` XORs the lane index with N inside a 16-lane row and runs
 * on the VALU, so it needs no lgkmcnt wait at all. The whole reduction here is
 * within one 16-lane row (max_shift = base_tile_cols/2 = 8, leader =
 * (lane/16)*16), so row_xmask alone is sufficient -- no cross-half exchange.
 *
 * Verified present on gfx1201 against the assembler. Falls back to the original
 * ladder for any geometry where the reduction is not a 16-lane row.
 */
namespace detail {
template<int XOR_N> __device__ __forceinline__ uint32_t dpp_xor_u32(uint32_t v) {
    // row_xmask:N is dpp_ctrl 0x160 + N
    return (uint32_t)__builtin_amdgcn_update_dpp((int)0, (int)v, 0x160 + XOR_N,
                                                 0xf /*row_mask*/, 0xf /*bank_mask*/,
                                                 true /*bound_ctrl*/);
}
template<int XOR_N, typename T> __device__ __forceinline__ T dpp_xor(T x) {
    if constexpr (sizeof(T) == 4) {
        uint32_t v; __builtin_memcpy(&v, &x, 4);
        v = dpp_xor_u32<XOR_N>(v);
        T o; __builtin_memcpy(&o, &v, 4); return o;
    } else {
        static_assert(sizeof(T) == 8, "dpp_xor supports 4- and 8-byte packed types");
        uint32_t v[2]; __builtin_memcpy(v, &x, 8);
        v[0] = dpp_xor_u32<XOR_N>(v[0]);
        v[1] = dpp_xor_u32<XOR_N>(v[1]);
        T o; __builtin_memcpy(&o, v, 8); return o;
    }
}
} // namespace detail


/**
 * @brief Perform a row-wise reduction on a matrix in row-major layout.
 *
 * This function template performs a parallel reduction across the rows of a matrix using a specified operation.
 * It leverages warp shuffle functions for efficient intra-warp communication.
 *
 * @tparam op The operation to be applied for reduction.
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type with row layout.
 * @tparam reset A boolean flag indicating whether to reset the accumulator (ignore src_accum) or not.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when reset is false.
 */
template<typename op, ducks::rv::all V, ducks::rt::row_layout T, bool reset>
__device__ static inline void row_reduce(V &row_accum, const T &src, const V &src_accum) {
    // I actually like these static asserts because they give more verbose errors when things go wrong.
    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout, typename T::shape>::col_vec_layout>); // compatible layout
    static_assert(std::is_same_v<typename V::T2, typename T::dtype>); // compatible type
    static_assert(V::outer_dim == T::height); // compatible size

    using dtype = T::dtype;
    using RT = V::dtype;
    using RT2 = base_types::packing<RT>::packed_type;

    static_assert(!std::is_same_v<RT, fp8e4m3>, "Unsupported type for reduction");

    const int leader = laneid() % T::base_tile_rows;
    const int max_shift = T::base_tile_threads_per_reduction / 2;

    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        dtype accum_packed = src.tiles[i][0].data[0];
        for (int k = 1; k < src.packed_per_base_tile; k++) {
            accum_packed = op::template op<dtype>(accum_packed, src.tiles[i][0].data[k]);
        }

        #pragma unroll
        for(int j = 1; j < src.width; j++) {
            #pragma unroll
            for (int k = 0; k < src.packed_per_base_tile; k++) {
                accum_packed = op::template op<dtype>(accum_packed, src.tiles[i][j].data[k]);
            }
        }
        RT accum_single = op::template op<RT>(accum_packed.x, accum_packed.y);

        if constexpr (sizeof(RT) == 4 && T::base_tile_rows == 16) {
            // RDNA4 rt_16x16: the reduction spans exactly TWO lanes, `lane` and
            // `lane ^ 16` (8 elements each of a 16-row column), so max_shift is
            // always 1 and the ladder degenerates to one __shfl_down(x, 16)
            // followed by a broadcast from the leader.
            //
            // Both lower to `ds_bpermute_b32`, an LDS-PIPE instruction, each
            // costing an `s_waitcnt lgkmcnt(0)` -- in a kernel with no LDS at
            // all. ATT on the gfx1201 attention kernel measured those
            // waits at **64.7% of ALL stall**: 32,896 hits, 752 cycles each,
            // 24.7M cycles, more than twice the next term.
            //
            // A shfl_down followed by a broadcast is, for an associative and
            // commutative op (max/sum/min/prod -- all of them here), exactly a
            // BUTTERFLY. v_permlanex16_b32 exchanges the two 16-lane halves in
            // the VALU with no lgkmcnt wait, and leaves EVERY lane holding the
            // full result, so the broadcast disappears as well: two bpermutes
            // and one wait become one VALU op.
            //
            // sel0/sel1 = 0x76543210 / 0xFEDCBA98 is the identity permutation
            // within each half, i.e. a plain lane <-> lane^16 swap.
            uint32_t v_; __builtin_memcpy(&v_, &accum_single, 4);
            uint32_t s_ = (uint32_t)__builtin_amdgcn_permlanex16(
                              (int)v_, (int)v_, 0x76543210u, 0xFEDCBA98u, false, false);
            RT other_; __builtin_memcpy(&other_, &s_, 4);
            accum_single = op::template op<RT>(accum_single, other_);
        } else {
            for (int shift = max_shift; shift > 0; shift--) {
                accum_single = op::template op<RT>(accum_single, __shfl_down(accum_single, shift * T::base_tile_rows));
            }

            accum_single = __shfl(accum_single, leader);
        }

        if(reset) {
            row_accum[i][0] = accum_single;
        }
        else {
            row_accum[i][0] = op::template op<RT>(src_accum[i][0], accum_single);
        }
    }
}

/**
 * @brief Perform a row-wise reduction on a matrix in column-major layout.
 *
 * This function template performs a parallel reduction across the rows of a matrix using a specified operation.
 * It leverages warp shuffle functions for efficient intra-warp communication and is optimized for column-major matrices.
 *
 * @tparam op The operation to be applied for reduction.
 * @tparam V The vector type for the row accumulator.
 * @tparam T The matrix type with column layout.
 * @tparam reset A boolean flag indicating whether to reset the accumulator (ignore src_accum) or not.
 * @param[out] row_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when reset is false.
 */

template<typename op, ducks::rv::all V, ducks::rt::col_layout T, bool reset>
__device__ static inline void row_reduce(V &row_accum, const T &src, const V &src_accum) {
    // I actually like these static asserts because they give more verbose errors when things go wrong.
    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout, typename T::shape>::col_vec_layout>); // compatible layout
    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>); // compatible type
    static_assert(V::outer_dim == T::height); // compatible size

    using RT2 = V::dtype;
    using RT = base_types::packing<RT2>::unpacked_type;

    static_assert(!std::is_same_v<RT, fp8e4m3>, "Unsupported type for reduction");

    const int leader = (laneid() / T::base_tile_cols) * T::base_tile_cols;
    const int packed_per_tile = src.packed_per_base_tile;
    const int max_shift = T::base_tile_cols / 2;
    constexpr bool max_shift_is_row16 = (T::base_tile_cols == 16);

    RT2 accum[packed_per_tile];

    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        #pragma unroll
        for(int k = 0; k < packed_per_tile; k++) {
            accum[k] = src.tiles[i][0].data[k];
        }
        #pragma unroll
        for(int j = 1; j < src.width; j++) {
            #pragma unroll
            for(int k = 0; k < packed_per_tile; k++) {
                accum[k] = op::template op<RT2>(accum[k], src.tiles[i][j].data[k]);
            }
        }

        #pragma unroll
        for(int k = 0; k < packed_per_tile; k++) {
            if constexpr (max_shift_is_row16) {
                // Butterfly: leaves every lane with the full result, so the
                // broadcast below is a no-op. VALU only, no lgkmcnt wait.
                accum[k] = op::template op<RT2>(accum[k], detail::dpp_xor<1>(accum[k]));
                accum[k] = op::template op<RT2>(accum[k], detail::dpp_xor<2>(accum[k]));
                accum[k] = op::template op<RT2>(accum[k], detail::dpp_xor<4>(accum[k]));
                accum[k] = op::template op<RT2>(accum[k], detail::dpp_xor<8>(accum[k]));
            } else {
                for (int shift = max_shift; shift > 0; shift /= 2) {
                    accum[k] = op::template op<RT2>(accum[k], packed_shfl_down(MASK_ALL, accum[k], shift));
                }
            }
        }

        if constexpr (reset) {
            #pragma unroll
            for(int k = 0; k < packed_per_tile; k++) {
                row_accum[i][k] = accum[k];
            }
        }
        else {
            #pragma unroll
            for(int k = 0; k < packed_per_tile; k++) {
                row_accum[i][k] = op::template op<RT2>(src_accum[i][k], accum[k]);
            }
        }

        if constexpr (!max_shift_is_row16) {
            #pragma unroll
            for(int k = 0; k < packed_per_tile; k++) {
                row_accum[i][k] = packed_shfl(MASK_ALL, row_accum[i][k], leader);
            }
        }
    }
}


// Col reduction.
/**
 * @brief Perform a column-wise reduction on a matrix in row-major layout.
 *
 * This function template performs a parallel reduction across the columns of a matrix using a specified operation.
 * It leverages warp shuffle functions for efficient intra-warp communication and is optimized for row-major matrices.
 *
 * @tparam op The operation to be applied for reduction.
 * @tparam V The vector type for the column accumulator.
 * @tparam T The matrix type with row layout.
 * @tparam reset A boolean flag indicating whether to reset the accumulator (ignore src_accum) or not.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when reset is false.
 */

template<typename op, ducks::rv::all V, ducks::rt::row_layout T, bool reset>
__device__ static inline void col_reduce(V &col_accum, const T &src, const V &src_accum) {
    // I actually like these static asserts because they give more verbose errors when things go wrong.
    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout, typename T::shape>::row_vec_layout>); // compatible layout
    static_assert(std::is_same_v<typename V::dtype, typename T::dtype>); // compatible type
    static_assert(V::outer_dim == T::width); // compatible size

    using RT2 = V::dtype;
    using RT = base_types::packing<RT2>::unpacked_type;

    static_assert(!std::is_same_v<RT, fp8e4m3>, "Unsupported type for reduction");

    const int leader = (laneid() / T::base_tile_rows) * T::base_tile_rows;
    constexpr int packed_per_tile = T::packed_per_base_tile;
    constexpr int max_shift = T::base_tile_rows / 2;
    constexpr bool max_shift_is_row16 = (T::base_tile_rows == 16);

    RT2 accum[packed_per_tile];

    #pragma unroll
    for(int j = 0; j < src.width; j++) {
        #pragma unroll
        for(int k = 0; k < packed_per_tile; k++) {
            accum[k] = src.tiles[0][j].data[k];
        }
        #pragma unroll
        for(int i = 1; i < src.height; i++) {
            #pragma unroll
            for(int k = 0; k < packed_per_tile; k++) {
                accum[k] = op::template op<RT2>(accum[k], src.tiles[i][j].data[k]);
            }
        }

        #pragma unroll
        for(int k = 0; k < packed_per_tile; k++) {
            if constexpr (max_shift_is_row16) {
                // Butterfly: leaves every lane with the full result, so the
                // broadcast below is a no-op. VALU only, no lgkmcnt wait.
                accum[k] = op::template op<RT2>(accum[k], detail::dpp_xor<1>(accum[k]));
                accum[k] = op::template op<RT2>(accum[k], detail::dpp_xor<2>(accum[k]));
                accum[k] = op::template op<RT2>(accum[k], detail::dpp_xor<4>(accum[k]));
                accum[k] = op::template op<RT2>(accum[k], detail::dpp_xor<8>(accum[k]));
            } else {
                for (int shift = max_shift; shift > 0; shift /= 2) {
                    accum[k] = op::template op<RT2>(accum[k], packed_shfl_down(MASK_ALL, accum[k], shift));
                }
            }
        }

        if constexpr (reset) {
            #pragma unroll
            for(int k = 0; k < packed_per_tile; k++) {
                col_accum[j][k] = accum[k];
            }
        }
        else {
            #pragma unroll
            for(int k = 0; k < packed_per_tile; k++) {
                col_accum[j][k] = op::template op<RT2>(src_accum[j][k], accum[k]);
            }
        }

        #pragma unroll
        for(int k = 0; k < packed_per_tile; k++) {
            col_accum[j][k] = packed_shfl(MASK_ALL, col_accum[j][k], leader);
        }
    }
}
/**
 * @brief Perform a column-wise reduction on a matrix in column-major layout.
 *
 * This function template performs a parallel reduction across the columns of a matrix using a specified operation.
 * It leverages warp shuffle functions for efficient intra-warp communication and is optimized for column-major matrices.
 *
 * @tparam op The operation to be applied for reduction.
 * @tparam V The vector type for the column accumulator.
 * @tparam T The matrix type with column layout.
 * @tparam reset A boolean flag indicating whether to reset the accumulator (ignore src_accum) or not.
 * @param[out] col_accum The accumulator where the result of the reduction is stored.
 * @param[in] src The source matrix on which to perform the reduction.
 * @param[in] src_accum The initial value of the accumulator, used when reset is false.
 */
template<typename op, ducks::rv::all V, ducks::rt::col_layout T, bool reset>
__device__ static inline void col_reduce(V &col_accum, const T &src, const V &src_accum) {
    using RT = V::dtype;
    using RT2 = base_types::packing<RT>::packed_type;

    // I actually like these static asserts because they give more verbose errors when things go wrong.
    static_assert(std::is_same_v<typename V::layout, typename rt_base<typename T::T, typename T::layout, typename T::shape>::row_vec_layout>); // compatible layout
    static_assert(std::is_same_v<RT2, typename T::dtype>); // compatible type
    static_assert(V::outer_dim == T::width); // compatible size

    static_assert(!std::is_same_v<RT, fp8e4m3>, "Unsupported type for reduction");

    const int leader = laneid() % T::base_tile_cols;
    const int max_shift = T::base_tile_threads_per_reduction / 2;

    #pragma unroll
    for(int j = 0; j < src.width; j++) { // note now width is the outer loop
        RT2 accum_packed = src.tiles[0][j].data[0];
        #pragma unroll
        for(int k = 1; k < src.packed_per_base_tile; k++) {
            accum_packed = op::template op<RT2>(accum_packed, src.tiles[0][j].data[k]);
        }
        #pragma unroll
        for(int i = 1; i < src.height; i++) { // and height is the inner loop
            #pragma unroll
            for(int k = 0; k < src.packed_per_base_tile; k++) {
                accum_packed = op::template op<RT2>(accum_packed, src.tiles[i][j].data[k]);
            }
        }

        RT accum_single = op::template op<RT>(accum_packed.x, accum_packed.y);
        
        if constexpr (sizeof(RT) == 4 && T::base_tile_cols == 16) {
            // RDNA4 rt_16x16. The reduction spans exactly two lanes, `lane` and `lane ^ 16`, so
            // the ladder is one __shfl_down(x, 16) plus a leader broadcast --
            // two `ds_bpermute_b32`, an LDS-PIPE instruction, in a kernel with
            // no LDS. ATT on the gfx1201 attention kernel put the
            // resulting `s_waitcnt lgkmcnt(0)` at 64.7% of ALL stall: 32,896
            // hits at 752 cycles, 24.7M cycles, twice the next term.
            //
            // shfl_down + broadcast == a BUTTERFLY for an associative and
            // commutative op. v_permlanex16_b32 does the lane^16 exchange in
            // the VALU with no lgkmcnt wait and leaves EVERY lane holding the
            // full result, so the broadcast disappears too.
            // sel0/sel1 = 0x76543210 / 0xFEDCBA98 is the identity within each
            // half, i.e. a plain lane <-> lane^16 swap.
            uint32_t v_; __builtin_memcpy(&v_, &accum_single, 4);
            uint32_t s_ = (uint32_t)__builtin_amdgcn_permlanex16(
                              (int)v_, (int)v_, 0x76543210u, 0xFEDCBA98u, false, false);
            RT other_; __builtin_memcpy(&other_, &s_, 4);
            accum_single = op::template op<RT>(accum_single, other_);
        } else {
            for (int shift = max_shift; shift > 0; shift--) {
                accum_single = op::template op<RT>(accum_single, __shfl_down(accum_single, shift * T::base_tile_cols));
            }

            accum_single = __shfl(accum_single, leader);
        }

        if(reset) {
            col_accum[j][0] = accum_single;
        }
        else {
            col_accum[j][0] = op::template op<RT>(src_accum[j][0], accum_single);
        }
    }
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