/**
 * @file
 * @brief The basic 16x16 register tile on which larger register tiles are built.
 */
 
#pragma once

#include <type_traits>

#include "../../common/common.cuh"
#include "rt_layout.cuh"
#include "rv_layout.cuh"

namespace kittens {

/* ----------  BASE 16x16 SUBTILE STRUCT  ---------- */

namespace ducks {
/**
 * @namespace rt_base
 * 
 * @brief The namespace where concepts and abstract types for register base (16x16) tiles live.
 */
namespace rt_base {
/**
 * @brief A dummy type used to identify register base tiles.
 * 
 * For a type to quack like an rt_base, it should define its identifier as ducks::rt_base::identifier.
 * If a type quacks like ducks::rt_base::identifier, it will be treated as an rt_base by compiler checks.
 */
struct identifier {};
}
} // namespace ducks

/**
 * @brief Basic tile structure for computation in registers.
 *
 * @tparam T2 The packed data type used for the matrix elements.
 * @tparam _layout The layout of the base tile, either row-major or column-major.
 *
 * This type is a primarily utility for building larger inline templates
 * out of PTX primitives and managing layouts.
 *
 * In general, you probably want a row-major tile, unless you specifically want to call mma
 *
 * The four shapes that actually occur on gfx1100, with `l` the lane id and `e`
 * the element index within a lane (data[] is these packed in pairs):
 *
 *   rt_base<bf16|half, row>  A operand   e -> (row l%16, col e),      e in 0..15
 *   rt_base<bf16|half, col>  B operand   e -> (row e,     col l%16),  e in 0..15
 *   rt_base<float, col>      accumulator e -> (row 2e + l/16, col l%16), e in 0..7
 *   rt_base<float, row>      transposed  e -> (row l%16, col 2e + l/16), e in 0..7
 *
 * The convention matches CDNA's: `row` means the lane picks the row and the
 * elements run along the columns, `col` is the transpose.  What differs is that
 * the operand shapes cover all 16 of k in a single lane and are duplicated
 * across the two wave halves, and that the accumulator's element axis is
 * strided by 2 rather than contiguous.
 */
template<typename _T, ducks::rt_layout::all _layout> struct rt_base {
    using identifier = ducks::rt_base::identifier; ///< Type identifier for the rt_base structure.
    using layout = _layout; ///< Layout of the matrix tile.
    static_assert(kittens::ducks::base_types::T1<_T>); // confirm it's a supported type
    using T = kittens::base_types::packing<_T>::unpacked_type;
    using T2 = kittens::base_types::packing<_T>::packed_type;
    using dtype = T2; ///< Data type of the matrix elements

    static_assert(
        std::is_same_v<dtype, bf16_2> || std::is_same_v<dtype, float2> || std::is_same_v<dtype, half_2>,
        "rt_base was provided an unsupported type. (RDNA3 has no fp8 WMMA.)"
    );

    static constexpr int tile_size_row        = kittens::TILE_ROW_DIM<T, layout>;
    static constexpr int tile_size_col        = kittens::TILE_COL_DIM<T, layout>;
    static constexpr int rows                 = tile_size_row; ///< Number of rows.
    static constexpr int cols                 = tile_size_col; ///< Number of cols.
    static constexpr int num_elements         = rows*cols;

    /// How many times over the wave holds this tile. 2 for WMMA operands (the
    /// two 16-lane halves mirror each other), 1 for the f32 accumulator.
    /// See WMMA_REPLICATION in common/util.cuh for the measured layouts.
    static constexpr int replication          = kittens::WMMA_REPLICATION<T>;
    static constexpr int elements_per_thread  = replication * num_elements / kittens::WARP_THREADS;

    static constexpr int packed_per_thread    = (elements_per_thread / base_types::packing<dtype>::num()) ;
    static constexpr int registers_per_thread = packed_per_thread * sizeof(dtype) / 4; // registers are 32-bit words

    // Both shapes come out to 8 VGPRs per lane per 16x16 tile.
    static_assert(registers_per_thread == 8, "unexpected gfx11 WMMA fragment size");

    /// Distance along the element axis between element `e` and element `e+1` of
    /// a single lane. 1 for a WMMA operand (a lane holds a contiguous K vector);
    /// 2 for the f32 accumulator (the two wave halves interleave along the axis).
    static constexpr int element_stride = 2 / replication;
    /// Whether the wave half shifts a lane's elements along the element axis.
    /// True only for the accumulator, where lane l and l^16 are *not* mirrors.
    static constexpr bool halves_interleave = (replication == 1);

    using row_vec_layout = std::conditional_t<std::is_same_v<layout, ducks::rt_layout::row>, ducks::rv_layout::align, ducks::rv_layout::ortho>; // for holding column reductions
    using col_vec_layout = std::conditional_t<std::is_same_v<layout, ducks::rt_layout::row>, ducks::rv_layout::ortho, ducks::rv_layout::align>; // for holding row reductions

    dtype data[packed_per_thread]; ///< The actual storage for the base tile
};

/**
 * @brief Where element `e` of lane `l` sits inside a 16x16 base tile, as {row, col}.
 *
 * `layout::row` means the lane selects the row and the elements run along the
 * columns; `col` is the transpose.  The position along that element axis is
 * `element_stride*e`, shifted by the wave half when the halves interleave.
 *
 * This is the single place the gfx11 fragment layouts are written down as
 * arithmetic.  Memory ops, reductions and layout conversions all derive their
 * indices from here rather than open-coding `laneid%16` the way the CDNA tree
 * does, because on RDNA3 the mapping differs between operands and accumulators.
 */
template<typename T, ducks::rt_layout::all L>
__device__ static inline int2 rt_base_coord(int e, int lane) {
    using base = rt_base<T, L>;
    const int pos = base::element_stride * e + (base::halves_interleave ? (lane >> 4) : 0);
    const int l16 = lane & 15;
    if constexpr (std::is_same_v<L, ducks::rt_layout::row>) { return int2{l16, pos}; }
    else                                                    { return int2{pos, l16}; }
}

/**
 * @brief Where entry `p` of an rv_layout::align vector lives within a lane.
 *
 * An `align` vector is indexed along a tile's element axis, so it inherits that
 * axis' lane mapping: the vector is replicated across the 16 lanes of a wave
 * half, and -- for the f32 accumulator, whose halves interleave -- each half
 * holds only every other entry.
 *
 * Returns the element index `e` in 0..elements_per_thread-1 that this lane uses
 * for entry `p` in 0..15, or -1 if this lane's half does not hold `p` at all.
 * It is the inverse of the `pos` computed by rt_base_coord().
 */
template<typename T>
__device__ static inline int rv_align_elem(int p, int lane) {
    using base = rt_base<T, ducks::rt_layout::row>; // stride/interleave depend only on T
    if constexpr (base::halves_interleave) return ((p & 1) == (lane >> 4)) ? (p >> 1) : -1;
    else                                   return p; // replication 2: every lane holds every entry
}

/**
 * @brief The other direction: how many entries of a 16-entry align_l block a
 * lane holds, and which entry its `e`-th slot is.
 *
 * Accesses are written as a loop over these slots, not over all sixteen
 * entries with rv_align_elem's -1 skipped: the skip depends on the lane, so
 * that form compiles to one divergent branch -- and one serialized memory
 * round trip -- per entry, where this one is a straight run of loads whose
 * address merely differs by half.
 */
template<typename T>
__device__ static inline constexpr int rv_align_held() {
    using base = rt_base<T, ducks::rt_layout::row>;
    return base::halves_interleave ? TILE_ROW_DIM<T> / 2 : TILE_ROW_DIM<T>;
}
template<typename T>
__device__ static inline int rv_align_entry(int e, int lane) {
    using base = rt_base<T, ducks::rt_layout::row>;
    if constexpr (base::halves_interleave) return 2 * e + (lane >> 4);
    else                                   return e;
}

/* ----------  CONCEPTS  ---------- */

namespace ducks {
namespace rt_base {
/**
* @brief Concept for all register base tiles.
* @tparam T The type to check against the concept requirements.
*
* Requires:
* - T has a nested type identifier that is the same as rt_base::identifier.
*/
template<typename T> concept all = requires {
    typename T::identifier; // Checks if T::identifier exists
} && std::is_same_v<typename T::identifier, identifier>; // Checks if T::identifier is ducks::rt::identifier
} // namespace rt
} // namespace ducks

/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

template<ducks::rt_layout::all L=ducks::rt_layout::row> using rt_base_fl = rt_base<float, L>;
template<ducks::rt_layout::all L=ducks::rt_layout::row> using rt_base_bf = rt_base<bf16, L>;
template<ducks::rt_layout::all L=ducks::rt_layout::row> using rt_base_hf = rt_base<half, L>;
}
