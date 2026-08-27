/**
 * @file
 * @brief Templated layouts for global memory.
 */
 
#pragma once

#include "../../common/common.cuh"
#include "../shared/shared.cuh"
#include "util.cuh"
#include "gl_layout.cuh"

namespace kittens {

/* ----------  Global layout descriptor  ---------- */

namespace ducks {
namespace gl {
struct identifier {};
}
}

template<typename _T, int b, int d, int r, int c,
         ducks::gl_layout::all _Layout = ducks::gl_layout::row_major>
struct gl {
    using identifier = ducks::gl::identifier;

    using layout = _Layout; ///< Memory layout of the descriptor.

    using T     = base_types::packing<_T>::unpacked_type;
    using T2    = base_types::packing<_T>::packed_type;
    using dtype = T;

    T* raw_ptr;

    static constexpr int __b__ = b, __d__ = d, __r__ = r, __c__ = c; // Not to be touched by the user.

    ducks::gl::make_dim_t<b> batch_internal;
    ducks::gl::make_dim_t<d> depth_internal;
    ducks::gl::make_dim_t<r> rows_internal;
    ducks::gl::make_dim_t<c> cols_internal;

    template <int B=__b__> __device__ __host__ static constexpr std::enable_if_t<(B > 0), int> batch() { return B; }
    template <int B=__b__> __device__ __host__ std::enable_if_t<(B == -1), int> batch() const { return batch_internal; }
    template <int D=__d__> __device__ __host__ static constexpr std::enable_if_t<(D > 0), int> depth() { return D; }
    template <int D=__d__> __device__ __host__ std::enable_if_t<(D == -1), int> depth() const { return depth_internal; }
    template <int R=__r__> __device__ __host__ static constexpr std::enable_if_t<(R > 0), int> rows() { return R; }
    template <int R=__r__> __device__ __host__ std::enable_if_t<(R == -1), int> rows() const { return rows_internal; }
    template <int C=__c__> __device__ __host__ static constexpr std::enable_if_t<(C > 0), int> cols() { return C; }
    template <int C=__c__> __device__ __host__ std::enable_if_t<(C == -1), int> cols() const { return cols_internal; }

    __host__ inline gl(T *_data,
                        ducks::gl::make_arg_t<b> _batch,
                        ducks::gl::make_arg_t<d> _depth,
                        ducks::gl::make_arg_t<r> _rows,
                        ducks::gl::make_arg_t<c> _cols) :
            raw_ptr(_data), batch_internal(_batch), depth_internal(_depth), rows_internal(_rows), cols_internal(_cols) {}
    __host__ __device__ inline gl(const gl &other) :
            raw_ptr(other.raw_ptr), batch_internal(other.batch_internal), depth_internal(other.depth_internal), rows_internal(other.rows_internal), cols_internal(other.cols_internal) {}
    __device__ inline T& operator[](const coord<ducks::default_type> &idx) const {
        return raw_ptr[this->idx(idx)];
    }
    /* One constrained overload per layout, so selection is by resolution rather than by a
     * branch. A layout with no overload here is not a silent fall-through: it is "no viable
     * member" at the call site. */
    __device__ inline int64_t idx(const coord<ducks::default_type> &idx) const
        requires std::is_same_v<layout, ducks::gl_layout::row_major> {
        return ((int64_t(idx.b)*depth() + idx.d)*rows() + idx.r)*cols() + idx.c;
    }
    __device__ inline int64_t idx(const coord<ducks::default_type> &idx) const
        requires std::is_same_v<layout, ducks::gl_layout::col_major> {
        return ((int64_t(idx.b)*depth() + idx.d)*cols() + idx.c)*rows() + idx.r;
    }
    template<int axis> __device__ inline size_t shape() const {
        static_assert(axis==0 || axis==1 || axis==2 || axis==3, "Axis must be 0, 1, 2, or 3.");
        if constexpr (axis==0) { return size_t(batch()); }
        else if constexpr (axis==1) { return size_t(depth()); }
        else if constexpr (axis==2) { return size_t(rows()); }
        else if constexpr (axis==3) { return size_t(cols()); }
    }
    /* Strides follow the layout. Under col_major the ROW axis is the unit-stride one, which is
     * the whole point: it is what lets a store whose register-contiguous axis is M emit a wide
     * store instead of a scatter. */
    /* Likewise per layout. The `axis` branching stays a branch -- axis is an int, not a layout,
     * and there is nothing for resolution to select on. */
    template<int axis> __device__ inline size_t stride() const
        requires std::is_same_v<layout, ducks::gl_layout::row_major> {
        static_assert(axis==0 || axis==1 || axis==2 || axis==3, "Axis must be 0, 1, 2, or 3.");
        if      constexpr (axis==0) { return depth()*rows()*cols(); }
        else if constexpr (axis==1) { return rows()*cols(); }
        else if constexpr (axis==2) { return cols(); }
        else                        { return 1; }
    }
    template<int axis> __device__ inline size_t stride() const
        requires std::is_same_v<layout, ducks::gl_layout::col_major> {
        static_assert(axis==0 || axis==1 || axis==2 || axis==3, "Axis must be 0, 1, 2, or 3.");
        if      constexpr (axis==0) { return depth()*rows()*cols(); }
        else if constexpr (axis==1) { return rows()*cols(); }
        else if constexpr (axis==2) { return 1; }
        else                        { return rows(); }
    }

    /* Element offset of (row, col) from a tile base, in the units `raw_ptr` is indexed in. The one
     * place a layout becomes an address: the contiguous axis contributes its coordinate unscaled and
     * the other is scaled by its stride. `axis` names the tile's row axis, as it does for `stride`.
     *
     * The multiply stays 32-bit on purpose. Both coordinates are bounded by the tile and the stride by
     * the tensor, so the product cannot overflow, while widening it makes the backend give up on
     * proving the base wave-uniform and materialise it into a VGPR pair. */
    template<int axis> __device__ inline int64_t offset(int row, int col) const
        requires std::is_same_v<layout, ducks::gl_layout::row_major> {
        return (int64_t)row * (int)stride<axis>() + col;
    }
    template<int axis> __device__ inline int64_t offset(int row, int col) const
        requires std::is_same_v<layout, ducks::gl_layout::col_major> {
        return (int64_t)col * (int)stride<3>() + row;
    }
};

namespace ducks {
namespace gl {
/**
* @brief Concept for all global layouts.
* @tparam T The type to check against the concept requirements.
*
* Requires:
* - T has a nested type identifier that is the same as ducks::gl::identifier.
*/
template<typename T> concept all = requires {
    typename T::identifier; // Checks if T::identifier exists
} && std::is_same_v<typename T::identifier, identifier>; // Checks if T::identifier is ducks::gl::identifier
/**
* @brief Concept for global layouts with row-major layout.
* @tparam T The type to check against the concept requirements.
*
* Requires:
* - T is a global layout.
* - T has an internal type layout that is ducks::gl_layout::row_major.
*/
template<typename T>
concept row_layout = all<T> && std::is_same_v<typename T::layout, ducks::gl_layout::row_major>;
/**
* @brief Concept for global layouts with column-major layout.
* @tparam T The type to check against the concept requirements.
*
* Requires:
* - T is a global layout.
* - T has an internal type layout that is ducks::gl_layout::col_major.
*/
template<typename T>
concept col_layout = all<T> && std::is_same_v<typename T::layout, ducks::gl_layout::col_major>;
}
}

// Structs for initializing global layouts automatically.
// struct unsafe_gl {
//     uint64_t data;
//     int b, d, r, c;
//     unsafe_gl(uint64_t data, int b, int d, int r, int c) : data(data), b(b), d(d), r(r), c(c) {}
// };
template<int N> auto make_unsafe_gl_arg(int param) { // typename std::conditional_t<(N < 0), std::nullptr_t, int>
    if constexpr (N > 0) { return nullptr; }
    else                 { return param;   }
}
template<ducks::gl::all GL, bool safe=true> __host__ inline GL make_gl(uint64_t data, int b, int d, int r, int c) {
    if constexpr (safe) {
        if(GL::__b__ > 0 && b != GL::__b__) {
            throw std::runtime_error("Batch dimension mismatch.");
        }
        if(GL::__d__ > 0 && d != GL::__d__) {
            throw std::runtime_error("Depth dimension mismatch.");
        }
        if(GL::__r__ > 0 && r != GL::__r__) {
            throw std::runtime_error("Row dimension mismatch.");
        }
        if(GL::__c__ > 0 && c != GL::__c__) {
            throw std::runtime_error("Column dimension mismatch.");
        }
    }
    return GL(
        reinterpret_cast<typename GL::dtype*>(data),
        make_unsafe_gl_arg<GL::__b__>(b),
        make_unsafe_gl_arg<GL::__d__>(d),
        make_unsafe_gl_arg<GL::__r__>(r),
        make_unsafe_gl_arg<GL::__c__>(c)
    );
}

} // namespace kittens
