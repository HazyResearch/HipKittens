/**
 * @file
 * @brief The ThunderKittens shared tile struct.
 */

#pragma once

#include "../../common/common.cuh"
#include "sv.cuh"
#include "st_shape.cuh"
#include "st_layout.cuh"

/* ----------  MAIN TILE STRUCT  ---------- */

// these are helper structs for type inference
namespace kittens {
namespace ducks {
/**
 * @namespace st
 * 
 * @brief The namespace where concepts and abstract types for shared tiles live.
 */
namespace st {
/**
 * @brief A dummy type used to identify shared tiles.
 * 
 * For a type to quack like an st, it should define its identifier as ducks::st::identifier.
 * If a type quacks like ducks::st::identifier, it will be treated as an st by compiler checks.
 * This is particularly useful for subtiles.
 */
struct identifier {};
}
} // namespace ducks

// Forward declaration of subtile
template<
    typename ST,
    int _subtile_height,
    int _subtile_width
>
struct st_subtile;

/**
 * @brief Shared memory tile structure for various data types and layouts.
 *
 * @tparam T The data type of the elements in the tile. Not packed!
 * @tparam _rows The height of the tile.
 * @tparam _cols The width of the tile.
 * @tparam _shape How addresses are permuted within the tile (swizzle or padding).
 * @tparam _layout Which axis is contiguous. Orthogonal to `_shape`, and defaults to `row`, which
 *                 is what every operand tile is -- only a C tile staged for a column-major
 *                 destination asks for `col`.
 */
template<typename _T, int _rows, int _cols, ducks::st_shape::all _shape,
         ducks::st_layout::all _layout = ducks::st_layout::row>
struct KITTENS_DEFAULT_ALIGN st {
    using identifier = ducks::st::identifier; ///< Type identifier for shared memory tile.
    using T = base_types::packing<_T>::unpacked_type;
    using T2 = base_types::packing<_T>::packed_type;
    using dtype = T; ///< Data type of the elements in the tile.
    using shape = _shape;
    using layout = _layout;

    // define underlying data as same as that projected, to make clear that this is *not* a subtile.
    static constexpr int underlying_rows              = _rows;
    static constexpr int underlying_cols              = _cols;
    static constexpr int underlying_num_elements      = underlying_rows * underlying_cols;

    static constexpr int underlying_subtile_rows      = shape::rows;
    static constexpr int underlying_subtile_cols      = shape::cols;
    static constexpr int underlying_subtile_row_bytes = shape::cols * sizeof(T);
    static constexpr int underlying_subtile_elements  = underlying_subtile_rows * underlying_subtile_cols;
    static constexpr int underlying_subtile_bytes     = underlying_subtile_elements * sizeof(T);
    static constexpr int underlying_subtile_bytes_per_thread = shape::template bytes_per_thread<T>();

    static constexpr int underlying_subtiles_per_row  = underlying_cols / underlying_subtile_cols;
    static constexpr int underlying_subtiles_per_col  = underlying_rows / underlying_subtile_rows;

    static constexpr int rows                = _rows; ///< Total number of rows in the tile.
    static constexpr int cols                = _cols; ///< Total number of cols in the tile.
    static constexpr int num_elements        = rows * cols; ///< Total number of elements in the tile.

    static constexpr int subtiles_per_row    = cols / underlying_subtile_cols;
    static constexpr int subtiles_per_col    = rows / underlying_subtile_rows;

    static_assert(base_types::packing<dtype>::num() == 1); // must be a 1-packed type (e.g. float, bf16, etc)

    // Storage is shape-driven: a size-preserving shape (XOR swizzle, the CDNA /
    // NVIDIA default) reports `rows*cols`, while a size-increasing shape (the
    // gfx1250 `st_16x32_padded` bank-conflict padding) reports the inflated
    // element count via `storage_elems()`. For every non-padded shape this is
    // exactly `rows*cols`, so non-gfx1250 layouts are byte-identical.
    template<typename S> static constexpr int storage_for() {
        if constexpr (ducks::st_shape::padded<S>) return S::storage_elems(rows*cols);
        else return rows*cols;
    }
    static constexpr int storage_elements = storage_for<shape>();

    dtype data[storage_elements]; ///< Raw data storage for the tile.

    /* (row, col) -> physical element offset. The shape decides the arrangement:
     *   padded shapes    `layout`-ordered with the shape's periodic padding. The TDM engine
     *                    deposits this and cannot swizzle, which is why padding exists.
     *   swizzled shapes  subtile-major, with the shape's XOR swizzle inside each subtile.
     * Both directions of a transfer must address through here, or writer and reader disagree. */
    __device__ __host__ __forceinline__ static constexpr int idx(int r, int c) {
        if constexpr (ducks::st_shape::padded<shape>) {
            return shape::padded(layout_flat(r, c));
        } else {
            constexpr int SR = shape::rows, SC = shape::cols;
            const int sub_id = (r / SR) * (cols / SC) + (c / SC);
            return sub_id * (SR * SC)
                 + int(shape::template swizzle<T>({r % SR, c % SC})) / int(sizeof(T));
        }
    }
    __device__ __forceinline__ static T* idx(T *ptr, int2 coord) {
        return &ptr[idx(coord.x, coord.y)];
    }
    /// @brief LDS byte address, for the ops that address in bytes rather than elements.
    __device__ __forceinline__ static uint32_t idx(uint32_t ptr, int2 coord) {
        return ptr + sizeof(T) * idx(coord.x, coord.y);
    }

    __device__ __forceinline__ dtype& operator[](const int2 &rowcol) {
        return data[idx(rowcol.x, rowcol.y)];
    }
    __device__ __forceinline__ const dtype& operator[](const int2 &rowcol) const {
        return data[idx(rowcol.x, rowcol.y)];
    }
    __device__ __forceinline__ dtype& operator[](int i)             { return data[i]; }
    __device__ __forceinline__ const dtype& operator[](int i) const { return data[i]; }

    /// @brief Subtile-local XOR swizzle, for bodies that decompose into subtiles themselves.
    __device__ __forceinline__ static const uint32_t swizzle(int2 coord) {
        return shape::template swizzle<T>(coord);
    }

private:
    /// @brief (row, col) -> `layout`-ordered flat index, before the shape's padding.
    __device__ __host__ __forceinline__ static constexpr int layout_flat(int r, int c) {
        if constexpr (std::is_same_v<layout, ducks::st_layout::col>) return c * rows + r;
        else                                                        return r * cols + c;
    }

public:
    // vector types
    using col_vec = sv<dtype, rows>; ///< Column vector type for this tile
    using row_vec = sv<dtype, cols>; ///< Row vector type for this tile

    template<int subtile_rows, int subtile_cols> using subtile = st_subtile<st<_T, _rows, _cols, _shape, _layout>, subtile_rows, subtile_cols>;
};


/**
 * @brief A reference into a chunk of shared tile memory.
 *
 * The st_subtile is a drop-in replacement for an st which internally
 * references the appropriate memory while performing minimal address
 * calculations. You should never create this directly, but instead
 * have subtile_inplace return it for you instead. (`auto` is nice.)
 *
 * You can generally just pretend this is an st. But not for wgmma's.
 */
template<
    typename _ST,
    int _subtile_rows,
    int _subtile_cols
>
struct st_subtile {
    using identifier = ducks::st::identifier; // i quack like an st, gcc will never know the difference
    using ST = _ST;
    using T = ST::T;
    using T2 = ST::T2;
    using dtype = T; ///< Data type of the elements in the tile.
    using shape = ST::shape;
    using layout = ST::layout;

    static constexpr int underlying_rows              = ST::underlying_rows;
    static constexpr int underlying_cols              = ST::underlying_cols;
    static constexpr int underlying_num_elements      = ST::underlying_num_elements;

    static constexpr int underlying_subtile_cols      = ST::underlying_subtile_cols;
    static constexpr int underlying_subtile_row_bytes = ST::underlying_subtile_row_bytes;
    static constexpr int underlying_subtile_rows      = ST::underlying_subtile_rows;
    static constexpr int underlying_subtile_elements  = ST::underlying_subtile_elements;
    static constexpr int underlying_subtile_bytes     = ST::underlying_subtile_bytes;
    static constexpr int underlying_subtile_bytes_per_thread = ST::underlying_subtile_bytes_per_thread;
    
    static constexpr int underlying_subtiles_per_row  = ST::underlying_subtiles_per_row;
    static constexpr int underlying_subtiles_per_col  = ST::underlying_subtiles_per_col;

    static constexpr int rows                = _subtile_rows;
    static constexpr int cols                = _subtile_cols;
    static constexpr int num_elements        = rows * cols;

    static constexpr int subtiles_per_row    = cols / underlying_subtile_cols;
    static constexpr int subtiles_per_col    = rows / underlying_subtile_rows;

    dtype *data;
    int row_offset, col_offset;

    __device__ st_subtile(ST &src, int2 rowcol) {
        row_offset = rowcol.x * rows;
        col_offset = rowcol.y * cols;
        const int subtile_row_offset = row_offset / underlying_subtile_rows;
        const int subtile_col_offset = col_offset / underlying_subtile_cols;
        const int subtile_id = subtile_row_offset * underlying_subtiles_per_row + subtile_col_offset;
        const int subtile_offset = subtile_id * underlying_subtile_elements;
        data = &src.data[subtile_offset];
    }

    __device__ __forceinline__ static const uint32_t swizzle(int2 coord) {
        return ST::swizzle(coord);
    }

    // vector types
    using col_vec = sv<dtype, rows>;
    using row_vec = sv<dtype, cols>;
};

/* ----------  CONCEPTS  ---------- */

namespace ducks {
namespace st {

/**
* @brief Concept for all shared tiles.
* @tparam T The type to check against the concept requirements.
*
* Requires:
* - T has a nested type identifier that is the same as st::identifier.
*/
template<typename T> concept all = requires {
    typename T::identifier; // Checks if T::identifier exists
} && std::is_same_v<typename T::identifier, identifier>; // Checks if T::identifier is ducks::st::identifier

/**
 * @brief Shared tiles by layout, mirroring `ducks::rt::` and `ducks::gl::`.
 *
 * Spelled as a conjunction with `all` so that an overload constrained on a layout is more
 * constrained than one on `all`, and wins partial ordering. That is how a column-major destination
 * picks up its own path instead of the blanket refusal.
 */
template<typename T> concept row_layout = all<T> && std::is_same_v<typename T::layout, st_layout::row>;
template<typename T> concept col_layout = all<T> && std::is_same_v<typename T::layout, st_layout::col>;

} // namespace st
} // namespace ducks


/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

// gfx1250 defaults the shared-tile shape to `st_16x32_padded` so the 2-arg
// `st_bf<R, C>` form resolves without naming a shape, and the layout to `row` so the 3-arg form
// resolves without naming a layout.
template<int _height, int _width, ducks::st_shape::all _shape = ducks::st_shape::st_16x32_padded<>, ducks::st_layout::all _layout = ducks::st_layout::row> using st_bf = st<bf16,  _height, _width, _shape, _layout>;
template<int _height, int _width, ducks::st_shape::all _shape = ducks::st_shape::st_16x32_padded<>, ducks::st_layout::all _layout = ducks::st_layout::row> using st_hf = st<half,  _height, _width, _shape, _layout>;
template<int _height, int _width, ducks::st_shape::all _shape = ducks::st_shape::st_16x32_padded<>, ducks::st_layout::all _layout = ducks::st_layout::row> using st_fl = st<float, _height, _width, _shape, _layout>;
template<int _height, int _width, ducks::st_shape::all _shape = ducks::st_shape::st_16x32_padded<>, ducks::st_layout::all _layout = ducks::st_layout::row> using st_fp8e4m3 = st<fp8e4m3, _height, _width, _shape, _layout>;
}
