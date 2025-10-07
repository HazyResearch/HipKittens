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
// These probably need to be redone to reduce bank conflicts.
// They currently work fine with xor layout but it should be
// possible to reduce their bank conflicts with other layouts too.

/**
 * @brief Load data from a shared tile into a register tile.
 *
 * @tparam RT The register tile type
 * @tparam ST The shared tile type
 * @param dst[out] The destination register tile.
 * @param src[in]  The source shared tile.
 */
template<ducks::rt::row_layout RT, ducks::st::all ST>
__device__ inline static void load(RT &dst, const ST &src) {

    static_assert(RT::rows == ST::rows, "register tile and shared tile must match rows");
    static_assert(RT::cols == ST::cols,  "register tile and shared tile must match cols");

    using T2 = RT::dtype;
    using T  = base_types::packing<T2>::unpacked_type;
    using U  = ST::dtype;
    using U2 = base_types::packing<U >::packed_type;
    constexpr int packing = base_types::packing<typename RT::dtype>::num();

    const int laneid = kittens::laneid();

    const int row_offset = laneid % dst.base_tile_rows;
    const int col_offset = dst.base_tile_stride * (laneid / dst.base_tile_rows);

    const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&src.data[0]);

    #pragma unroll
    for (int i = 0; i < dst.height; i++) {
        const int row = i * dst.base_tile_rows + row_offset;

        #pragma unroll
        for (int j = 0; j < dst.width; j++) {

            #pragma unroll
            for (int k = 0; k < dst.base_tile_num_strides; k++) {
                const int col = j * dst.base_tile_cols + col_offset + k * dst.base_tile_elements_per_stride_group;
                const uint32_t addr = src.idx(src_ptr, {row, col});

                int idx = k * dst.base_tile_stride / packing;

                if constexpr (std::is_same_v<U2, bf16_2>) {

                    // Use ds_read_b128 for stride == 8, dtype == bf16
                    if constexpr (RT::base_tile_stride == 8) {
                        asm volatile(
                            "ds_read_b128 %0, %1 offset:0\n"
                            : "=v"(*reinterpret_cast<float4*>(&dst.tiles[i][j].data[idx]))
                            : "v"(addr)
                            : "memory"
                        );
                    // Use ds_read_b64 for stride == 4, dtype == bf16
                    } else if constexpr (RT::base_tile_stride == 4) {
                        asm volatile(
                            "ds_read_b64 %0, %1 offset:0\n"
                            : "=v"(*reinterpret_cast<float2*>(&dst.tiles[i][j].data[idx]))
                            : "v"(addr)
                            : "memory"
                        );
                    } else {
                        static_assert(false, "Unsupported stride");
                    }
                } else {
                    static_assert(false, "Unsupported type");
                }
            }
        }
    }
}

template<ducks::rt::col_layout RT, ducks::st::all ST>
__device__ inline static void load(RT &dst, const ST &src) {

    static_assert(RT::rows == ST::rows, "register tile and shared tile must match rows");
    static_assert(RT::cols == ST::cols,  "register tile and shared tile must match cols");

    using T2 = RT::dtype;
    using T  = base_types::packing<T2>::unpacked_type;
    using U  = ST::dtype;
    using U2 = base_types::packing<U >::packed_type;
    constexpr int packing = base_types::packing<typename RT::dtype>::num();

    const int laneid = kittens::laneid();

    const int row_offset = ((laneid % 16) / 4) + ((laneid / dst.base_tile_cols) * dst.base_tile_stride);
    const int col_offset = ((laneid % 4) * 4) + (16 * ((laneid % dst.base_tile_cols) / 16));

    const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&src.data[0]);
    
    #pragma unroll
    for (int i = 0; i < dst.height; i++) {
        #pragma unroll
        for (int j = 0; j < dst.width; j++) {
            const int col = j * dst.base_tile_cols + col_offset;
            #pragma unroll
            for (int k = 0; k < dst.base_tile_num_strides; k++) {
                const int row = i * dst.base_tile_rows + row_offset + k * dst.base_tile_elements_per_stride_group;
                const uint32_t addr = src.idx(src_ptr, {row, col});

                int idx = k * dst.base_tile_stride / packing;

                if constexpr (std::is_same_v<U2, bf16_2>) {

                    // Use two ds_read_b64_tr_b16 for stride == 8, dtype == bf16
                    if constexpr (RT::base_tile_stride == 8) {
                        const int next_row = row + 4;
                        const uint32_t next_addr = src.idx(src_ptr, {next_row, col});
                        asm volatile(
                            "ds_read_b64_tr_b16 %0, %2 offset:0\n"
                            "ds_read_b64_tr_b16 %1, %3 offset:0\n"
                            : "=v"(*reinterpret_cast<float2*>(&dst.tiles[i][j].data[idx])), 
                            "=v"(*reinterpret_cast<float2*>(&dst.tiles[i][j].data[idx + 2]))
                            : "v"(addr), "v"(next_addr)
                            : "memory"
                        );
                    // Use one ds_read_b64_tr_b16 for stride == 4, dtype == bf16
                    } else if constexpr (RT::base_tile_stride == 4) {
                        asm volatile(
                            "ds_read_b64_tr_b16 %0, %1 offset:0\n"
                            : "=v"(*reinterpret_cast<float2*>(&dst.tiles[i][j].data[idx]))
                            : "v"(addr)
                            : "memory"
                        );
                    } else {
                        static_assert(false, "Unsupported stride");
                    }
                } else {
                    static_assert(false, "Unsupported type");
                }
            }
        }
    }
}

/**
 * @brief Store data into a shared tile from a register tile.
 *
 * @tparam RT The register tile type
 * @tparam ST The shared tile type
 * @param dst[out] The destination shared tile.
 * @param src[in]  The source register tile.
 */
template<ducks::rt::row_layout RT, ducks::st::all ST>
__device__ inline static void store(ST &dst, const RT &src) {

    static_assert(RT::rows == ST::rows, "register tile and shared tile must match rows");
    static_assert(RT::cols == ST::cols,  "register tile and shared tile must match cols");

    using T2 = RT::dtype;
    using T  = base_types::packing<T2>::unpacked_type;
    using U  = ST::dtype;
    using U2 = base_types::packing<U >::packed_type;
    constexpr int packing = base_types::packing<typename RT::dtype>::num();

    const int laneid = kittens::laneid();

    const int row_offset = laneid % dst.base_tile_rows;
    const int col_offset = dst.base_tile_stride * (laneid / dst.base_tile_rows);

    const uint32_t dst_ptr = reinterpret_cast<uintptr_t>(&dst.data[0]);

    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        const int row = i * dst.base_tile_rows + row_offset;

        #pragma unroll
        for(int j = 0; j < dst.width; j++) {

            #pragma unroll
            for (int k = 0; k < dst.base_tile_num_strides; k++) {
                const int col = j * dst.base_tile_cols + col_offset + k * dst.base_tile_elements_per_stride_group;
                const uint32_t addr = dst.idx(dst_ptr, {row, col});

                int idx = k * dst.base_tile_stride / packing;

                if constexpr (std::is_same_v<U2, bf16_2>) {

                    // Use ds_write_b128 for stride == 8, dtype == bf16
                    if constexpr (RT::base_tile_stride == 8) {
                        asm volatile(
                            "ds_write_b128 %0, %1 offset:%0\n"
                            : 
                            : "v"(addr), "v"(*reinterpret_cast<const float2*>(&src.tiles[i][j].data[idx]))
                        );
                    // Use ds_write_b64 for stride == 4, dtype == bf16
                    } else if constexpr (RT::base_tile_stride == 4) {
                        asm volatile(
                            "ds_write_b64 %0, %1 offset:0\n"
                            : 
                            : "v"(addr), "v"(*reinterpret_cast<const float2*>(&src.tiles[i][j].data[idx]))
                        );
                    } else {
                        static_assert(false, "Unsupported stride");
                    }
                } else {
                    static_assert(false, "Unsupported type");
                }
            }
        }
    }
}

template<ducks::rt::col_layout RT, ducks::st::all ST>
__device__ inline static void store(ST &dst, const RT &src) {

    static_assert(RT::rows == ST::rows, "register tile and shared tile must match rows");
    static_assert(RT::cols == ST::cols,  "register tile and shared tile must match cols");

    using T2 = RT::dtype;
    using T  = base_types::packing<T2>::unpacked_type;
    using U  = ST::dtype;
    using U2 = base_types::packing<U >::packed_type;
    constexpr int packing = base_types::packing<typename RT::dtype>::num();

    const int laneid = kittens::laneid();

    const int row_offset = dst.base_tile_stride * (laneid / dst.base_tile_cols);
    const int col_offset = laneid % dst.base_tile_cols;

    #pragma unroll
    for(int i = 0; i < dst.height; i++) {

        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            const int col = j * dst.base_tile_cols + col_offset;
        
            #pragma unroll
            for (int k = 0; k < dst.base_tile_num_strides; k++) {
                const int row = i * dst.base_tile_rows + row_offset + k * dst.base_tile_elements_per_stride_group;

                #pragma unroll
                for (int l = 0; l < dst.base_tile_stride / packing; l++) {
                    const int idx = l + k * dst.base_tile_stride / packing;
                    dst[{row + l * 2, col}] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[idx].x);
                    dst[{row + l * 2 + 1, col}] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[idx].y);
                }
            }
        }
    }
}

} // namespace kittens