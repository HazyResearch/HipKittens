/**
 * @file
 * @brief Functions for transferring data directly between global memory and registers and back.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"
#include "../util/util.cuh"
#include "../../sync/barrier.cuh"

namespace kittens {

namespace detail {

/* A lane owns a run of `RT::base_tile_stride` elements: along COLUMNS for a row-layout tile,
 * along ROWS for a col-layout one. The run is contiguous in memory exactly when that axis is the
 * global layout's unit-stride axis -- the diagonal below -- and that is the only thing deciding
 * between one wide transfer and an element-by-element walk.
 *
 *                     row-major GL      col-major GL
 *   row-layout RT     contiguous        strided
 *   col-layout RT     strided           contiguous
 */
template<ducks::rt::all RT, ducks::gl::all GL>
inline constexpr bool run_is_contiguous =
    (ducks::rt::row_layout<RT> && ducks::gl::row_layout<GL>) ||
    (ducks::rt::col_layout<RT> && ducks::gl::col_layout<GL>);

/// Flat element offset of (row, col). The single place a global layout is interpreted.
template<ducks::gl::all GL>
__device__ __forceinline__ static int64_t gl_offset(const GL &g, int row, int col, int row_stride) {
    if constexpr (ducks::gl::row_layout<GL>)
        return (int64_t)row * row_stride + col;
    else if constexpr (ducks::gl::col_layout<GL>)
        return (int64_t)col * (int)g.template stride<3>() + row;
    else
        static_assert(ducks::gl_layout::unhandled<typename GL::layout>,
            "gl_offset is implemented for row_major and col_major only.");
}

/// Element step between consecutive members of a lane's run; 1 when the run is contiguous.
template<ducks::rt::all RT, ducks::gl::all GL>
__device__ __forceinline__ static int64_t run_step(const GL &g, int row_stride) {
    if constexpr (run_is_contiguous<RT, GL>)      return 1;
    else if constexpr (ducks::rt::row_layout<RT>) return (int)g.template stride<3>();
    else                                          return row_stride;
}

/// Move one contiguous run of `E` elements through the widest buffer ops that fit it.
template<int E, typename U, typename U2>
__device__ __forceinline__ static void buffer_load_run(U2 *tmp, const buffer_resource &br,
                                                       int64_t elem_off) {
    constexpr int BYTES = E * (int)sizeof(U);
    const uint32_t b = (uint32_t)(elem_off * (int64_t)sizeof(U));
    if constexpr (BYTES == 8) {
        float2 v = std::bit_cast<float2>(llvm_amdgcn_raw_buffer_load_b64(std::bit_cast<i32x4>(br), b, 0, 0));
        __builtin_memcpy(tmp, &v, 8);
    } else if constexpr (BYTES == 16) {
        float4 v = std::bit_cast<float4>(llvm_amdgcn_raw_buffer_load_b128(std::bit_cast<i32x4>(br), b, 0, 0));
        __builtin_memcpy(tmp, &v, 16);
    } else if constexpr (BYTES == 32) {
        float4 v[2];
        v[0] = std::bit_cast<float4>(llvm_amdgcn_raw_buffer_load_b128(std::bit_cast<i32x4>(br), b,      0, 0));
        v[1] = std::bit_cast<float4>(llvm_amdgcn_raw_buffer_load_b128(std::bit_cast<i32x4>(br), b + 16, 0, 0));
        __builtin_memcpy(tmp, v, 32);
    } else {
        static_assert(BYTES == 8 || BYTES == 16 || BYTES == 32, "Unsupported run width");
    }
}

template<int E, typename U, typename U2>
__device__ __forceinline__ static void buffer_store_run(const U2 *tmp, const buffer_resource &br,
                                                        int64_t elem_off) {
    constexpr int BYTES = E * (int)sizeof(U);
    const uint32_t b = (uint32_t)(elem_off * (int64_t)sizeof(U));
    if constexpr (BYTES == 8) {
        uint64_t v; __builtin_memcpy(&v, tmp, 8);
        llvm_amdgcn_raw_buffer_store_b64(v, std::bit_cast<i32x4>(br), b, 0, 0);
    } else if constexpr (BYTES == 16) {
        __uint128_t v; __builtin_memcpy(&v, tmp, 16);
        llvm_amdgcn_raw_buffer_store_b128(v, std::bit_cast<i32x4>(br), b, 0, 0);
    } else if constexpr (BYTES == 32) {
        __uint128_t v0, v1;
        __builtin_memcpy(&v0, tmp, 16);
        __builtin_memcpy(&v1, (const char*)tmp + 16, 16);
        llvm_amdgcn_raw_buffer_store_b128(v0, std::bit_cast<i32x4>(br), b,      0, 0);
        llvm_amdgcn_raw_buffer_store_b128(v1, std::bit_cast<i32x4>(br), b + 16, 0, 0);
    } else {
        static_assert(BYTES == 8 || BYTES == 16 || BYTES == 32, "Unsupported run width");
    }
}

/// Origin of the run a lane owns for subtile (i,j), stride group k.
template<ducks::rt::all RT, typename TILE>
__device__ __forceinline__ static void run_origin(const TILE &t, int laneid, int i, int j, int k,
                                                  int &row, int &col) {
    if constexpr (ducks::rt::row_layout<RT>) {
        row = t.base_tile_rows*i + laneid % t.base_tile_rows;
        col = t.base_tile_cols*j + t.base_tile_stride * (laneid / t.base_tile_rows)
            + k * t.base_tile_elements_per_stride_group;
    } else {
        constexpr int rows_per_lane = RT::base_tile_num_strides * RT::base_tile_stride;
        row = t.base_tile_rows*i + rows_per_lane * (laneid / t.base_tile_cols)
            + k * t.base_tile_stride;
        col = t.base_tile_cols*j + laneid % t.base_tile_cols;
    }
}

} // namespace detail

/**
 * @brief Load a global tile into a register tile, for either layout of either.
 *
 * @tparam RT The destination register tile type; its layout sets the run axis.
 * @param dst[out] The destination tile to load data into.
 * @param src[in] The source array to load data from.
 * @param idx[in] The index of the tile to load data from.
 */
template<int axis, ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void load(RT &dst, const GL &src, const COORD &idx) {
    using T2 = typename RT::dtype;
    using U  = typename GL::dtype;
    using U2 = typename base_types::packing<U>::packed_type;
    constexpr int packing = base_types::packing<typename RT::dtype>::num();
    constexpr int E = RT::base_tile_stride;

    static_assert(!std::is_same_v<typename base_types::packing<typename RT::dtype>::unpacked_type,
                                  fp8e4m3>, "Unsupported type for load");

    U *src_ptr = (U*)&src[(idx.template unit_coord<axis, 3>())];
    const int row_stride = src.template stride<axis>();
    const int laneid = kittens::laneid();
    const int64_t step = detail::run_step<RT, GL>(src, row_stride);

    const uint32_t buffer_size = src.batch() * src.depth() * src.rows() * src.cols() * sizeof(U);
    const buffer_resource br = make_buffer_resource(
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src_ptr)), buffer_size, 0x00020000);

    #pragma unroll
    for (int i = 0; i < dst.height; i++) {
        #pragma unroll
        for (int j = 0; j < dst.width; j++) {
            #pragma unroll
            for (int k = 0; k < dst.base_tile_num_strides; k++) {
                int row, col;
                detail::run_origin<RT>(dst, laneid, i, j, k, row, col);
                const int64_t off = detail::gl_offset(src, row, col, row_stride);

                U2 tmp[E / packing];
                if constexpr (detail::run_is_contiguous<RT, GL>) {
                    detail::buffer_load_run<E, U>(tmp, br, off);
                } else {
                    #pragma unroll
                    for (int l = 0; l < E / packing; l++) {
                        tmp[l].x = src_ptr[off + (int64_t)(l*2)     * step];
                        tmp[l].y = src_ptr[off + (int64_t)(l*2 + 1) * step];
                    }
                }

                #pragma unroll
                for (int l = 0; l < E / packing; l++)
                    dst.tiles[i][j].data[l + k * E / packing] =
                        base_types::convertor<T2, U2>::convert(tmp[l]);
            }
        }
    }
}

template<ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void load(RT &dst, const GL &src, const COORD &idx) {
    load<2, RT, GL>(dst, src, idx);
}

/**
 * @brief Store a register tile to a global tile, for either layout of either.
 *
 * @tparam RT The source register tile type; its layout sets the run axis.
 * @param[out] dst The destination array in global memory to store data into.
 * @param[in] src The source register tile to store data from.
 * @param idx[in] The tile coordinate in the destination array.
 */
template<int axis, ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void store(const GL &dst, const RT &src, const COORD &idx) {
    using T2 = typename RT::dtype;
    using U  = typename GL::dtype;
    using U2 = typename base_types::packing<U>::packed_type;
    constexpr int packing = base_types::packing<typename RT::dtype>::num();
    constexpr int E = RT::base_tile_stride;

    static_assert(!std::is_same_v<typename base_types::packing<typename RT::dtype>::unpacked_type,
                                  fp8e4m3>, "Unsupported type for store");

    U *dst_ptr = (U*)&dst[(idx.template unit_coord<axis, 3>())];
    const int row_stride = dst.template stride<axis>();
    const int laneid = kittens::laneid();
    const int64_t step = detail::run_step<RT, GL>(dst, row_stride);

    const uint32_t buffer_size = dst.batch() * dst.depth() * dst.rows() * dst.cols() * sizeof(U);
    const buffer_resource br = make_buffer_resource(
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dst_ptr)), buffer_size, 0x00020000);

    #pragma unroll
    for (int i = 0; i < src.height; i++) {
        #pragma unroll
        for (int j = 0; j < src.width; j++) {
            #pragma unroll
            for (int k = 0; k < src.base_tile_num_strides; k++) {
                int row, col;
                detail::run_origin<RT>(src, laneid, i, j, k, row, col);
                const int64_t off = detail::gl_offset(dst, row, col, row_stride);

                U2 tmp[E / packing];
                #pragma unroll
                for (int l = 0; l < E / packing; l++)
                    tmp[l] = base_types::convertor<U2, T2>::convert(
                                 src.tiles[i][j].data[l + k * E / packing]);

                if constexpr (detail::run_is_contiguous<RT, GL>) {
                    detail::buffer_store_run<E, U>(tmp, br, off);
                } else {
                    #pragma unroll
                    for (int l = 0; l < E / packing; l++) {
                        dst_ptr[off + (int64_t)(l*2)     * step] = tmp[l].x;
                        dst_ptr[off + (int64_t)(l*2 + 1) * step] = tmp[l].y;
                    }
                }
            }
        }
    }
}

template<ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void store(const GL &dst, const RT &src, const COORD &idx) {
    store<2, RT, GL, COORD>(dst, src, idx);
}

/**
 * @brief Store a WMMA accumulator to global memory through an LDS staging tile.
 *
 * A wave32's 32 lanes span 16 columns, so a direct store lands as 16 scattered 32-byte
 * transactions however well packed in registers; reaching a 512-byte contiguous stream needs the
 * rows held by four warps to meet, and LDS is the only place they can. The layout decides
 * whether the stage transposes: row_major C needs it, col_major C does not and gets
 * `ds_store_b128` instead of 128 narrow stores.
 *
 * ⚠ The caller must certify `scratch` is dead -- it is normally the operand ring reinterpreted,
 * safe only once every warp has finished reading its operands, which is what the entry barrier
 * establishes.
 *
 * @tparam N_THREADS  threads per block (drives the streaming distribution).
 * @param dst      global C; its type carries the layout and the leading dimension.
 * @param acc      this warp's col-layout WMMA accumulator.
 * @param scratch  dead LDS to stage through.
 * @param block_row,block_col   global element origin of the block.
 * @param warp_rbase,warp_cbase warp's element origin within the block.
 */
namespace detail {

/// Stage into LDS for a row-major destination: the accumulator is M-contiguous and the tile is
/// N-contiguous, so this transposes, one element at a time.
template<ducks::gl::row_layout GL, ducks::rt::col_layout RT,
         typename T, int BM, int BN, ducks::st_shape::all Shape>
__device__ static __attribute__((always_inline)) inline void stage_write(
    const GL&, const RT& acc, st<T, BM, BN, Shape>& scratch,
    int warp_rbase, int warp_cbase, int half, int col)
{
    #pragma unroll
    for (int n = 0; n < RT::height; ++n) {
        #pragma unroll
        for (int m = 0; m < RT::width; ++m) {
            const auto& tile = acc.tiles[n][m];
            const int rb = warp_rbase + n * 16;
            const int cb = warp_cbase + m * 16 + col;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int r0 = rb + 2 * k     + 8 * half;
                const int r1 = rb + 2 * k + 1 + 8 * half;
                scratch.data[scratch.padded(r0 * BN + cb)] =
                    base_types::convertor<T, float>::convert(tile.data[k].x);
                scratch.data[scratch.padded(r1 * BN + cb)] =
                    base_types::convertor<T, float>::convert(tile.data[k].y);
            }
        }
    }
}
/// Stage into LDS for a column-major destination: M-contiguous too, so nothing to transpose --
/// a lane's eight values are eight consecutive rows, one `ds_store_b128`.
template<ducks::gl::col_layout GL, ducks::rt::col_layout RT,
         typename T, int BM, int BN, ducks::st_shape::all Shape>
__device__ static __attribute__((always_inline)) inline void stage_write(
    const GL&, const RT& acc, st<T, BM, BN, Shape>& scratch,
    int warp_rbase, int warp_cbase, int half, int col)
{
    #pragma unroll
    for (int n = 0; n < RT::height; ++n) {
        #pragma unroll
        for (int m = 0; m < RT::width; ++m) {
            const auto& tile = acc.tiles[n][m];
            const int r0 = warp_rbase + n * 16 + 8 * half;
            const int c  = warp_cbase + m * 16 + col;
            T buf[8];
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                buf[2 * k]     = base_types::convertor<T, float>::convert(tile.data[k].x);
                buf[2 * k + 1] = base_types::convertor<T, float>::convert(tile.data[k].y);
            }
            *reinterpret_cast<__uint128_t*>(&scratch.data[scratch.padded(c * BM + r0)]) =
                *reinterpret_cast<const __uint128_t*>(buf);
        }
    }
}

/**
* @brief Stream the staged tile out to a row-major global destination.
*
* Walks the tile in row-major linear order, so each thread moves a contiguous `b128` and the
* wave covers 512 contiguous bytes.
*/
template<int N_THREADS, ducks::gl::row_layout GL, typename T, int BM, int BN,
         ducks::st_shape::all Shape>
__device__ static __attribute__((always_inline)) inline void read_back(
    const GL& dst, const st<T, BM, BN, Shape>& scratch, int block_row, int block_col)
{
    const int tid = threadIdx.x;
    const int N   = dst.cols();
    #pragma unroll
    for (int idx = tid * 8; idx < BM * BN; idx += N_THREADS * 8) {
        const int r = idx / BN, c = idx % BN;
        const __uint128_t v = *reinterpret_cast<const __uint128_t*>(
            &scratch.data[scratch.padded(idx)]);
        *reinterpret_cast<__uint128_t*>(
            &dst.raw_ptr[(block_row + r) * N + block_col + c]) = v;
    }
}
/**
* @brief Stream the staged tile out to a column-major global destination.
*
* Walks the tile in column-major linear order. The leading dimension is derived from the
* descriptor rather than passed, because the descriptor already knows it.
*/
template<int N_THREADS, ducks::gl::col_layout GL, typename T, int BM, int BN,
         ducks::st_shape::all Shape>
__device__ static __attribute__((always_inline)) inline void read_back(
    const GL& dst, const st<T, BM, BN, Shape>& scratch, int block_row, int block_col)
{
    const int tid = threadIdx.x;
    const int ldm = (int)dst.rows();
    #pragma unroll
    for (int idx = tid * 8; idx < BM * BN; idx += N_THREADS * 8) {
        const int c = idx / BM, r = idx % BM;
        const __uint128_t v = *reinterpret_cast<const __uint128_t*>(
            &scratch.data[scratch.padded(idx)]);
        *reinterpret_cast<__uint128_t*>(
            &dst.raw_ptr[(int64_t)(block_col + c) * ldm + block_row + r]) = v;
    }
}

} // namespace detail

template<int N_THREADS, ducks::gl::all GL, ducks::rt::col_layout RT,
         typename T, int BM, int BN, ducks::st_shape::all Shape>
__device__ static inline void store(
    const GL& dst, const RT& acc, st<T, BM, BN, Shape>& scratch,
    int block_row, int block_col, int warp_rbase, int warp_cbase)
{
    /* Layout-dependent work lives in the `detail` helpers, one concept-constrained overload
     * per layout. What is left here is the lane decomposition, the barriers, and the order. */
    const int L    = kittens::laneid();
    const int half = L / 16;
    const int col  = L % 16;

    /* The barrier orders execution, not LDS traffic, so each one needs its own drain. Omitting
     * either produces silent wrong output: the operand tiles share this LDS. */
    ::kittens::sync::wait_ds<0>();   // WAR: operand ds_reads have landed
    ::kittens::sync::sync();

    detail::stage_write(dst, acc, scratch, warp_rbase, warp_cbase, half, col);

    ::kittens::sync::wait_ds<0>();   // RAW: the staged tile is published
    ::kittens::sync::sync();

    detail::read_back<N_THREADS>(dst, scratch, block_row, block_col);
}


}