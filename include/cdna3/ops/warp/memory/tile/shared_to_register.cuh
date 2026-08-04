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
template<ducks::rt::all RT, ducks::st::all ST>
__device__ inline static void load(RT &dst, const ST &src) {

    static_assert(RT::height == ST::height, "register tile and shared tile must match height");
    static_assert(RT::width  == ST::width,  "register tile and shared tile must match width");

    using T2 = RT::dtype;
    using T  = base_types::packing<T2>::unpacked_type;
    using U  = ST::dtype;
    using U2 = base_types::packing<U >::packed_type;

    const int laneid = kittens::laneid() % kittens::WARP_THREADS;
    const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&src.data[0]);

    int row_offset, col_offset;
    if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
        row_offset = laneid%16;
        col_offset = std::is_same_v<T, fp8e4m3> ? 8*(laneid/16) : 4*(laneid/16);
    }
    else {
        row_offset = std::is_same_v<T, fp8e4m3> ? 8*(laneid/16) : 4*(laneid/16);
        col_offset = laneid%16;
    }


    #pragma unroll
    for(int j = 0; j < dst.width; j++) {
        const int col = j*dst.tile_size_col + col_offset;
        uint32_t addr = src.idx(src_ptr, {row_offset, col});
        #pragma unroll
        for(int i = 0; i < dst.height; i++) {
            const int row = i*dst.tile_size_row + row_offset;
            if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) { // handle the row-major layout

                if constexpr (sizeof(typename ST::dtype) == 4) {
                    // handle float32
                    float2 loaded0 = load_shared_vec(src.idx(src_ptr, {row, col}));
                    float2 loaded1 = load_shared_vec(src.idx(src_ptr, {row, col+2}));
                    dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(loaded0);
                    dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(loaded1);
                } else {
                    // handle fp16 and bf16
                    if constexpr (sizeof(T) == sizeof(U)) {
                        // Same size: no conversion needed (e.g., bf16->bf16 or half->half)
                        // float2 loaded = load_shared_vec(src.idx(src_ptr, {row, col}));
                        // U2* tmp = reinterpret_cast<U2*>(&loaded);
                        // dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(tmp[0]);
                        // dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(tmp[1]);
                        
                        // avoid v_bfi_b32
                        asm volatile(
                            "ds_read_b64 %0, %1 offset:%2\n"
                            : "=v"(*reinterpret_cast<uint64_t*>(&dst.tiles[i][j].data[0]))
                            : "v"(addr), "i"(i * ST::underlying_cols * kittens::TILE_ROW_DIM<U, typename ST::layout> * sizeof(U))
                            : "memory"
                        );
                    } else {
                        // Different size: converting bf16/half -> float
                        // Need to load and convert each pair separately
                        U2 loaded0 = *reinterpret_cast<const U2*>(&src[{row, col}]);
                        U2 loaded1 = *reinterpret_cast<const U2*>(&src[{row, col+2}]);
                        dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(loaded0);
                        dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(loaded1);
                    }
                }
            }
            else { // handle the column-major layout
                if constexpr (std::is_same_v<T, fp8e4m3>) {
                    dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(
                        base_types::make_fp8e4m3_4(
                            src[{row, col}], src[{row+1, col}], src[{row+2, col}], src[{row+3, col}]
                        )
                    );
                    dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(
                        base_types::make_fp8e4m3_4(
                            src[{row+4, col}], src[{row+5 , col}], src[{row+6, col}], src[{row+7, col}]
                        )
                    );
                }
                else {
                    dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(U2{src[{row, col}], src[{row+1, col}]});
                    dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(U2{src[{row+2, col}], src[{row+3, col}]});
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
template<ducks::rt::all RT, ducks::st::all ST>
__device__ inline static void store(ST &dst, const RT &src) {

    static_assert(RT::height == ST::height, "register tile and shared tile must match height");
    static_assert(RT::width  == ST::width,  "register tile and shared tile must match width");

    using T2 = RT::dtype;
    using T  = base_types::packing<T2>::unpacked_type;
    using U  = ST::dtype;
    using U2 = base_types::packing<U >::packed_type;

    int laneid = kittens::laneid() % kittens::WARP_THREADS;
    uint32_t dst_ptr = reinterpret_cast<uintptr_t>(&dst.data[0]);

    int row_offset, col_offset;
    if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
        row_offset = laneid%16;
        col_offset = 4*(laneid/16);
    }
    else {
        row_offset = 4*(laneid/16);
        col_offset = laneid%16;
    }
    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        int row = i*src.tile_size_row + row_offset;
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = j*src.tile_size_col + col_offset;

            if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) { // handle the row-major layout
                // *(U2*)(&dst[{row, col+0}]) = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[0]);
                // *(U2*)(&dst[{row, col+2}]) = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[1]);
                if constexpr (sizeof(typename ST::dtype) == 4) {
                    // handle float32
                    store_shared_vec(dst.idx(dst_ptr, {row, col}), base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[0]));
                    store_shared_vec(dst.idx(dst_ptr, {row, col+2}), base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[1]));
                } else {
                    // handle fp16 and bf16
                    if constexpr (sizeof(T) == sizeof(U)) {
                        // Same size: no conversion needed (e.g., bf16->bf16 or half->half)
                        float2 loaded = *reinterpret_cast<const float2*>(src.tiles[i][j].data);
                        store_shared_vec(dst.idx(dst_ptr, {row, col}), loaded);
                    } else {
                        // Different size: converting float -> bf16/half
                        // Need to convert and store each pair separately
                        U2 converted0 = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[0]);
                        U2 converted1 = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[1]);
                        *reinterpret_cast<U2*>(&dst[{row, col}]) = converted0;
                        *reinterpret_cast<U2*>(&dst[{row, col+2}]) = converted1;
                    }
                }
            }
            else { // handle the column-major layout
                U2 tmp[2];
                tmp[0] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[0]);
                tmp[1] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[1]);
            
                dst[{row+0, col}] = std::bit_cast<U>(tmp[0].x);
                dst[{row+1, col}] = std::bit_cast<U>(tmp[0].y);
                dst[{row+2, col}] = std::bit_cast<U>(tmp[1].x);
                dst[{row+3, col}] = std::bit_cast<U>(tmp[1].y);
            }            
        }
    }
}
}