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
    const int laneid = kittens::laneid();
    /* Step between consecutive elements of a lane's run: 1 when the run is contiguous, otherwise the
     * stride along the axis it walks -- columns for a row-layout register tile, rows for a col-layout
     * one. Only the choice of axis is the register tile's; the stride itself is the global tile's. */
    int64_t step = 1;
    if constexpr (!run_is_contiguous<RT, GL>) {
        if constexpr (ducks::rt::row_layout<RT>) step = (int)src.template stride<3>();
        else                                     step = (int)src.template stride<axis>();
    }

    const uint32_t buffer_size = src.batch() * src.depth() * src.rows() * src.cols() * sizeof(U);
    const buffer_resource br = make_buffer_resource(
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src_ptr)), buffer_size, 0x00020000);

    #pragma unroll
    for (int i = 0; i < dst.height; i++) {
        #pragma unroll
        for (int j = 0; j < dst.width; j++) {
            #pragma unroll
            for (int k = 0; k < dst.base_tile_num_strides; k++) {
                /* Origin of this lane's run for subtile (i, j), stride group k. A row-layout
                 * tile splits a base tile by rows and runs along columns; a col-layout tile splits
                 * it by columns and runs along rows. */
                int row, col;
                if constexpr (ducks::rt::row_layout<RT>) {
                    row = RT::base_tile_rows * i + laneid % RT::base_tile_rows;
                    col = RT::base_tile_cols * j
                        + RT::base_tile_stride * (laneid / RT::base_tile_rows)
                        + k * RT::base_tile_elements_per_stride_group;
                } else {
                    constexpr int rows_per_lane = RT::base_tile_num_strides * RT::base_tile_stride;
                    row = RT::base_tile_rows * i + rows_per_lane * (laneid / RT::base_tile_cols)
                        + k * RT::base_tile_stride;
                    col = RT::base_tile_cols * j + laneid % RT::base_tile_cols;
                }
                const int64_t off = src.template offset<axis>(row, col);

                U2 tmp[E / packing];
                if constexpr (run_is_contiguous<RT, GL>) {
                    // One contiguous run through the widest buffer op that fits it.
                    constexpr int BYTES = E * (int)sizeof(U);
                    const uint32_t boff = (uint32_t)(off * (int64_t)sizeof(U));
                    if constexpr (BYTES == 8) {
                        float2 v = std::bit_cast<float2>(llvm_amdgcn_raw_buffer_load_b64(std::bit_cast<i32x4>(br), boff, 0, 0));
                        __builtin_memcpy(tmp, &v, 8);
                    } else if constexpr (BYTES == 16) {
                        float4 v = std::bit_cast<float4>(llvm_amdgcn_raw_buffer_load_b128(std::bit_cast<i32x4>(br), boff, 0, 0));
                        __builtin_memcpy(tmp, &v, 16);
                    } else if constexpr (BYTES == 32) {
                        float4 v[2];
                        v[0] = std::bit_cast<float4>(llvm_amdgcn_raw_buffer_load_b128(std::bit_cast<i32x4>(br), boff,      0, 0));
                        v[1] = std::bit_cast<float4>(llvm_amdgcn_raw_buffer_load_b128(std::bit_cast<i32x4>(br), boff + 16, 0, 0));
                        __builtin_memcpy(tmp, v, 32);
                    } else {
                        static_assert(BYTES == 8 || BYTES == 16 || BYTES == 32, "Unsupported run width");
                    }
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
    const int laneid = kittens::laneid();
    /* Step between consecutive elements of a lane's run: 1 when the run is contiguous, otherwise the
     * stride along the axis it walks -- columns for a row-layout register tile, rows for a col-layout
     * one. Only the choice of axis is the register tile's; the stride itself is the global tile's. */
    int64_t step = 1;
    if constexpr (!run_is_contiguous<RT, GL>) {
        if constexpr (ducks::rt::row_layout<RT>) step = (int)dst.template stride<3>();
        else                                     step = (int)dst.template stride<axis>();
    }

    const uint32_t buffer_size = dst.batch() * dst.depth() * dst.rows() * dst.cols() * sizeof(U);
    const buffer_resource br = make_buffer_resource(
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dst_ptr)), buffer_size, 0x00020000);

    #pragma unroll
    for (int i = 0; i < src.height; i++) {
        #pragma unroll
        for (int j = 0; j < src.width; j++) {
            #pragma unroll
            for (int k = 0; k < src.base_tile_num_strides; k++) {
                /* Origin of this lane's run for subtile (i, j), stride group k. A row-layout
                 * tile splits a base tile by rows and runs along columns; a col-layout tile splits
                 * it by columns and runs along rows. */
                int row, col;
                if constexpr (ducks::rt::row_layout<RT>) {
                    row = RT::base_tile_rows * i + laneid % RT::base_tile_rows;
                    col = RT::base_tile_cols * j
                        + RT::base_tile_stride * (laneid / RT::base_tile_rows)
                        + k * RT::base_tile_elements_per_stride_group;
                } else {
                    constexpr int rows_per_lane = RT::base_tile_num_strides * RT::base_tile_stride;
                    row = RT::base_tile_rows * i + rows_per_lane * (laneid / RT::base_tile_cols)
                        + k * RT::base_tile_stride;
                    col = RT::base_tile_cols * j + laneid % RT::base_tile_cols;
                }
                const int64_t off = dst.template offset<axis>(row, col);

                U2 tmp[E / packing];
                #pragma unroll
                for (int l = 0; l < E / packing; l++)
                    tmp[l] = base_types::convertor<U2, T2>::convert(
                                 src.tiles[i][j].data[l + k * E / packing]);

                if constexpr (run_is_contiguous<RT, GL>) {
                    // One contiguous run through the widest buffer op that fits it.
                    constexpr int BYTES = E * (int)sizeof(U);
                    const uint32_t boff = (uint32_t)(off * (int64_t)sizeof(U));
                    if constexpr (BYTES == 8) {
                        uint64_t v; __builtin_memcpy(&v, tmp, 8);
                        llvm_amdgcn_raw_buffer_store_b64(v, std::bit_cast<i32x4>(br), boff, 0, 0);
                    } else if constexpr (BYTES == 16) {
                        __uint128_t v; __builtin_memcpy(&v, tmp, 16);
                        llvm_amdgcn_raw_buffer_store_b128(v, std::bit_cast<i32x4>(br), boff, 0, 0);
                    } else if constexpr (BYTES == 32) {
                        __uint128_t v0, v1;
                        __builtin_memcpy(&v0, tmp, 16);
                        __builtin_memcpy(&v1, (const char*)tmp + 16, 16);
                        llvm_amdgcn_raw_buffer_store_b128(v0, std::bit_cast<i32x4>(br), boff,      0, 0);
                        llvm_amdgcn_raw_buffer_store_b128(v1, std::bit_cast<i32x4>(br), boff + 16, 0, 0);
                    } else {
                        static_assert(BYTES == 8 || BYTES == 16 || BYTES == 32, "Unsupported run width");
                    }
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



}