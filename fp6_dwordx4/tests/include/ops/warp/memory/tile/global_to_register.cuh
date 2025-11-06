/**
 * @file
 * @brief Functions for transferring data directly between global memory and registers and back.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"
#include "../util/util.cuh"

namespace kittens {


__device__ inline i32x4 make_srsrc(const void* ptr, uint32_t range_bytes, uint32_t row_stride_bytes = 0) {
    std::uintptr_t as_int = reinterpret_cast<std::uintptr_t>(ptr);   // width = sizeof(void*)
    std::uint64_t  as_u64 = static_cast<std::uint64_t>(as_int);    // widen if host is 32-bit
    buffer_resource rsrc = make_buffer_resource(as_u64, range_bytes, 0x110000);

    row_stride_bytes &= 0x3FFF;
    if (row_stride_bytes) {
        // - The swizzle stride lives in bits 13:0 of word2.
        //   Max value = 0x3FFF (8 KiB – one cache line per bank).
        uint64_t stride_field = row_stride_bytes;
        stride_field = stride_field | 0x4000;         // Cache swizzle
        stride_field = stride_field | 0x8000;         // Swizzle enable
        rsrc.ptr |= stride_field << 48;
    }

    return *reinterpret_cast<const i32x4*>(&rsrc);
}


template<int axis, ducks::rt::row_layout RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void load(RT &dst, const GL &src, const COORD &idx) {
    using T2 = RT::dtype;
    using U = typename GL::dtype;
    using U2 = base_types::packing<U>::packed_type;

    U *src_ptr = (U*)&src[(idx.template unit_coord<axis, 3>())];
    const int row_stride = src.template stride<axis>();
    int laneid = kittens::laneid();

    #ifdef KITTENS_CDNA4
    int packed_size;
    int row_offset;
    int col_offset;
    int REPEAT;
    if (sizeof(U2) == 32) {
        packed_size = sizeof(U2) / sizeof(U);
        row_offset = laneid%32, col_offset = packed_size *(laneid/32);
        REPEAT = 1;
    } else {
        packed_size = sizeof(U2) / sizeof(U);
        row_offset = laneid%32, col_offset = 4 * packed_size * (laneid/32);
        REPEAT = 2;
    }
    #else
    int row_offset = laneid%16, col_offset = 4*(laneid/16);
    int REPEAT = 1;
    #endif

    uint32_t buffer_size;
    if (sizeof(U2) == 32) {
        buffer_size = (src.batch() * src.depth() * src.rows() * src.cols() * 6 + 7) / 8;
    } else {
        buffer_size = src.batch() * src.depth() * src.rows() * src.cols() * sizeof(U);
    }
    std::uintptr_t as_int = reinterpret_cast<std::uintptr_t>(src_ptr);
    std::uint64_t  as_u64 = static_cast<std::uint64_t>(as_int);    // widen if host is 32-bit
    buffer_resource br = make_buffer_resource(as_u64, buffer_size, 0x00020000);

    #pragma unroll
    for (int z = 0; z < REPEAT; z++) {

        #pragma unroll
        for(int i = 0; i < dst.height; i++) {
            int row = dst.tile_size_row*i + row_offset;


            #pragma unroll
            for(int j = 0; j < dst.width; j++) {
                int col = dst.tile_size_col*j + col_offset + z*8*packed_size;

                if constexpr (sizeof(U2) == 4) { 

                    U2* tmp;
                    #ifdef KITTENS_CDNA4
                    if constexpr (sizeof(U2) == 4) {  // bf16_2 case and fp6 case
                        float4 loaded = std::bit_cast<float4>(llvm_amdgcn_raw_buffer_load_b128(
                            std::bit_cast<i32x4>(br),
                            (row*row_stride + col) * sizeof(U),
                            0,
                            0
                        ));
                        tmp = reinterpret_cast<U2*>(&loaded);
                        #pragma unroll
                        for(int k = 0; k < 4; k++) {
                            dst.tiles[i][j].data[k + z*4] = base_types::convertor<T2, U2>::convert(tmp[k]);
                        }
                    }
                    #else // bf16_2 case
                    float2 loaded = std::bit_cast<float2>(llvm_amdgcn_raw_buffer_load_b64(
                        std::bit_cast<i32x4>(br),
                        (row*row_stride + col) * sizeof(U),
                        0,
                        0
                    ));
                    tmp = reinterpret_cast<U2*>(&loaded);
                    #pragma unroll
                    for(int k = 0; k < 2; k++) {
                        dst.tiles[i][j].data[k] = base_types::convertor<T2, U2>::convert(tmp[k]);
                    }
                    #endif

                } else if constexpr (sizeof(U2) == 32) {

                    int element_offset = row * row_stride + col;
                     int byte_offset = (element_offset * 6) / 8;
                     
                     __uint128_t loaded1 = llvm_amdgcn_raw_buffer_load_b128(
                         std::bit_cast<i32x4>(br),
                         byte_offset,
                         0,
                         0
                     );
                     
                     uint64_t loaded2 = llvm_amdgcn_raw_buffer_load_b64(
                         std::bit_cast<i32x4>(br),
                         byte_offset + 16,  // shift 16 bytes
                         0,
                         0
                     );
                     
                     U2 tmp;
                     *reinterpret_cast<__uint128_t*>(&tmp) = loaded1;  // bytes 0-15
                     *reinterpret_cast<uint64_t*>(reinterpret_cast<char*>(&tmp) + 16) = loaded2;  // bytes 16-23
                     dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(tmp);

                }
                else { // float2 case
                    #ifdef KITTENS_CDNA4
                    float4 loaded1 = std::bit_cast<float4>(llvm_amdgcn_raw_buffer_load_b128(
                        std::bit_cast<i32x4>(br),
                        (row*row_stride + col) * sizeof(U),
                        0,
                        0
                    ));
                    float4 loaded2 = std::bit_cast<float4>(llvm_amdgcn_raw_buffer_load_b128(
                        std::bit_cast<i32x4>(br),
                        (row*row_stride + col + 4) * sizeof(U),
                        0,
                        0
                    ));
                    U2* tmp1 = reinterpret_cast<U2*>(&loaded1);
                    U2* tmp2 = reinterpret_cast<U2*>(&loaded2);
                    #pragma unroll
                    for(int k = 0; k < 2; k++) {
                        dst.tiles[i][j].data[k + z*4] = base_types::convertor<T2, U2>::convert(tmp1[k]);
                    }
                    #pragma unroll
                    for(int k = 0; k < 2; k++) {
                        dst.tiles[i][j].data[k + 2 + z*4] = base_types::convertor<T2, U2>::convert(tmp2[k]);
                    }
                    #else
                    U2* tmp;
                    float4 loaded = std::bit_cast<float4>(llvm_amdgcn_raw_buffer_load_b128(
                        std::bit_cast<i32x4>(br),
                        (row*row_stride + col) * sizeof(U),
                        0,
                        0
                    ));
                    tmp = reinterpret_cast<U2*>(&loaded);

                    #pragma unroll
                    for(int k = 0; k < 2; k++) {
                        dst.tiles[i][j].data[k] = base_types::convertor<T2, U2>::convert(tmp[k]);
                    }
                    #endif
                }
            }
        }
    }
}


/**
 * @brief Load data from a source array into a column-major layout tile.
 *
 * @tparam RT The column-major layout tile type.
 * @tparam U The data type of the source array.
 * @param dst[out] The destination tile to load data into.
 * @param src[in] The source array to load data from.
 * @param row_stride[in] The stride in elements between rows in the source array.
 */
template<int axis, ducks::rt::col_layout RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void load(RT &dst, const GL &src, const COORD &idx) {
    using T = base_types::packing<typename RT::dtype>::unpacked_type;
    using U = typename GL::dtype;
    
    U *src_ptr = (U*)&src[(idx.template unit_coord<axis, 3>())];
    const int row_stride = src.template stride<axis>();
    int laneid = kittens::laneid();

    #ifdef KITTENS_CDNA4
    const int row_offset = 8*(laneid/32), col_offset = laneid%32;
    int REPEAT = 2;
    #else:
    const int row_offset = 4*(laneid/16), col_offset = laneid%16;
    int REPEAT = 1;
    #endif

    #pragma unroll
    for (int z = 0; z < REPEAT; z++) {

        #pragma unroll
        for(int i = 0; i < dst.height; i++) {
            int row = i*dst.tile_size_row + row_offset + z*16;

            #pragma unroll
            for(int j = 0; j < dst.width; j++) {
                int col = j*dst.tile_size_col + col_offset;

                #ifdef KITTENS_CDNA4
                dst.tiles[i][j].data[0+z*4].x = base_types::convertor<T, U>::convert(src_ptr[(row+0)*row_stride + col]);
                dst.tiles[i][j].data[0+z*4].y = base_types::convertor<T, U>::convert(src_ptr[(row+1)*row_stride + col]);
                dst.tiles[i][j].data[1+z*4].x = base_types::convertor<T, U>::convert(src_ptr[(row+2)*row_stride + col]);
                dst.tiles[i][j].data[1+z*4].y = base_types::convertor<T, U>::convert(src_ptr[(row+3)*row_stride + col]);

                dst.tiles[i][j].data[2+z*4].x = base_types::convertor<T, U>::convert(src_ptr[(row+4)*row_stride + col]);
                dst.tiles[i][j].data[2+z*4].y = base_types::convertor<T, U>::convert(src_ptr[(row+5)*row_stride + col]);
                dst.tiles[i][j].data[3+z*4].x = base_types::convertor<T, U>::convert(src_ptr[(row+6)*row_stride + col]);
                dst.tiles[i][j].data[3+z*4].y = base_types::convertor<T, U>::convert(src_ptr[(row+7)*row_stride + col]);
                #else
                dst.tiles[i][j].data[0].x = base_types::convertor<T, U>::convert(src_ptr[(row+0)*row_stride + col]);
                dst.tiles[i][j].data[0].y = base_types::convertor<T, U>::convert(src_ptr[(row+1)*row_stride + col]);
                dst.tiles[i][j].data[1].x = base_types::convertor<T, U>::convert(src_ptr[(row+2)*row_stride + col]);
                dst.tiles[i][j].data[1].y = base_types::convertor<T, U>::convert(src_ptr[(row+3)*row_stride + col]);
                #endif

            }
        }
    }

}

#ifdef KITTENS_CDNA4
template<int axis, ducks::rt::accumulator_layout RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void load(RT &dst, const GL &src, const COORD &idx) {
    using T = base_types::packing<typename RT::dtype>::unpacked_type;
    using U = typename GL::dtype;

    U *src_ptr = (U*)&src[(idx.template unit_coord<axis, 3>())];
    const int row_stride = src.template stride<axis>();
    int laneid = kittens::laneid();

    int col_offset = laneid%32, row_offset = laneid/32;

    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            int col = dst.tile_size_col*j + col_offset;
            #pragma unroll
            for (int ii = 0; ii < 4; ii++) {
                int row = dst.tile_size_row*i + ii * 8 + row_offset * 4;

                dst.tiles[i][j].data[ii * 2].x = base_types::convertor<T, U>::convert(src_ptr[(row+0)*row_stride + col]);
                dst.tiles[i][j].data[ii * 2].y = base_types::convertor<T, U>::convert(src_ptr[(row+1)*row_stride + col]);
                dst.tiles[i][j].data[ii * 2 + 1].x = base_types::convertor<T, U>::convert(src_ptr[(row+2)*row_stride + col]);
                dst.tiles[i][j].data[ii * 2 + 1].y = base_types::convertor<T, U>::convert(src_ptr[(row+3)*row_stride + col]);
            }
        }
    }

}
#endif
template<ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void load(RT &dst, const GL &src, const COORD &idx) {
    load<2, RT, GL>(dst, src, idx);
}

/**
 * @brief Store data from a register tile to a destination array in global memory with a row-major layout.
 *
 * @tparam RT The register tile type with a row-major layout.
 * @tparam U The data type of the destination array.
 * @param[out] dst The destination array in global memory to store data into.
 * @param[in] src The source register tile to store data from.
 * @param row_stride[in] The stride in elements between rows in the destination array.
 */
template<int axis, ducks::rt::row_layout RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void store(const GL &dst, const RT &src, const COORD &idx) {
    using T2 = RT::dtype;
    using U = typename GL::dtype;
    using U2 = base_types::packing<U>::packed_type;

    U *dst_ptr = (U*)&dst[(idx.template unit_coord<axis, 3>())];
    const int row_stride = dst.template stride<axis>();
    int laneid = kittens::laneid();

    #ifdef KITTENS_CDNA4
    int packed_size;
    int row_offset; 
    int col_offset;
    int REPEAT;
    if (sizeof(U2) == 32) {
        packed_size = sizeof(U2) / sizeof(U);
        row_offset = laneid%32, col_offset = packed_size *(laneid/32);
        REPEAT = 1;
    } else {
        packed_size = sizeof(U2) / sizeof(U);
        row_offset = laneid%32, col_offset = 4 * packed_size * (laneid/32);
        REPEAT = 2;
    }
    #else
    int row_offset = laneid%16, col_offset = 4*(laneid/16);
    int REPEAT = 1;
    #endif

    for (int z = 0; z < REPEAT; z++) {

        #pragma unroll
        for(int i = 0; i < src.height; i++) {
            int row = src.tile_size_row*i + row_offset;
            
            #pragma unroll
            for(int j = 0; j < src.width; j++) {
                int col = src.tile_size_col*j + col_offset + z*8*packed_size;
                #ifdef KITTENS_CDNA4

                if constexpr (sizeof(U2) == 4) { // bf16_2
                    U2 tmp[4];
                    #pragma unroll
                    for(int k = 0; k < 4; k++) {
                        tmp[k] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[k + z*4]);
                    }
                    *(bytes_16*)&dst_ptr[row*row_stride + col] = *(bytes_16*)&tmp[0]; // 
                } else if constexpr (sizeof(U2) == 32) {  // fp6_e2m3_32 (32 bytes; 24 real bytes)

                    U2 tmp;
                     tmp = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[0]);
 
                     // uint32_t buffer_size = dst.batch() * dst.depth() * dst.rows() * dst.cols() * sizeof(U);
                     uint32_t buffer_size = (dst.batch() * dst.depth() * dst.rows() * dst.cols() * 6 + 7) / 8;
                     std::uintptr_t as_int = reinterpret_cast<std::uintptr_t>(dst_ptr);
                     std::uint64_t as_u64 = static_cast<std::uint64_t>(as_int);
                     buffer_resource br = make_buffer_resource(as_u64, buffer_size, 0x00020000);
 
                     int element_offset = row * row_stride + col;
                     int byte_offset = (element_offset * 6) / 8;
 
                     llvm_amdgcn_raw_buffer_store_b128(
                         *reinterpret_cast<__uint128_t*>(&tmp),
                         std::bit_cast<i32x4>(br),
                         byte_offset,  // Use the correctly calculated offset
                         0,
                         0
                     );
 
                     llvm_amdgcn_raw_buffer_store_b64(
                         *reinterpret_cast<uint64_t*>(reinterpret_cast<char*>(&tmp) + 16),
                         std::bit_cast<i32x4>(br),
                         byte_offset + 16,  // Just add 16 to byte_offset
                         0,
                         0
                     );
                }
                else { // float2
                    U2 tmp[4];
                    #pragma unroll
                    for(int k = 0; k < 4; k++) {
                        tmp[k] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[k + z*4]);
                    }
                    *(bytes_16*)&dst_ptr[row*row_stride + col] = *(bytes_16*)&tmp[0];
                    *(bytes_16*)&dst_ptr[row*row_stride + col + 4] = *(bytes_16*)&tmp[2];
                }

                #else

                U2 tmp[2];
                #pragma unroll
                for(int k = 0; k < 2; k++) {
                    tmp[k] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[k]);
                }
                if constexpr (sizeof(U2) == 4) { // bf16_2
                    *(bytes_8*)&dst_ptr[row*row_stride + col] = *(bytes_8*)tmp;
                }
                else { // float2
                    *(bytes_16*)&dst_ptr[row*row_stride + col] = *(bytes_16*)tmp;
                }
                #endif
            }
        }
    }
}

/**
 * @brief Store data from a register tile to a destination array in global memory with a column-major layout.
 *
 * @tparam RT The register tile type with a column-major layout.
 * @tparam U The data type of the destination array.
 * @param[out] dst The destination array in global memory to store data into.
 * @param[in] src The source register tile to store data from.
 * @param row_stride[in] The stride in elements between rows in the destination array.
 */
template<int axis, ducks::rt::col_layout RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void store(const GL &dst, const RT &src, const COORD &idx) {
    using T = base_types::packing<typename RT::dtype>::unpacked_type;
    using U = typename GL::dtype;

    U *dst_ptr = (U*)&dst[(idx.template unit_coord<axis, 3>())];
    const int row_stride = dst.template stride<axis>();
    const int laneid = kittens::laneid();

    #ifdef KITTENS_CDNA4
    const int row_offset = 8*(laneid/32), col_offset = laneid%32;
    int REPEAT = 2;
    #else
    const int row_offset = 4*(laneid/16), col_offset = laneid%16;
    int REPEAT = 1;
    #endif

    #pragma unroll
    for (int z = 0; z < REPEAT; z++) {

        #pragma unroll
        for(int i = 0; i < src.height; i++) {
            const int row = i*src.tile_size_row + row_offset + z*16;

            #ifdef KITTENS_CDNA4
            #pragma unroll
            for(int j = 0; j < src.width; j++) {

                const int col = j*src.tile_size_col + col_offset;
                dst_ptr[(row+0)*row_stride + col] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[0+z*4].x);
                dst_ptr[(row+1)*row_stride + col] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[0+z*4].y);
                dst_ptr[(row+2)*row_stride + col] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[1+z*4].x);
                dst_ptr[(row+3)*row_stride + col] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[1+z*4].y);

                dst_ptr[(row+4)*row_stride + col] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[2+z*4].x);
                dst_ptr[(row+5)*row_stride + col] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[2+z*4].y);
                dst_ptr[(row+6)*row_stride + col] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[3+z*4].x);
                dst_ptr[(row+7)*row_stride + col] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[3+z*4].y);
            }

            #else
            #pragma unroll
            for(int j = 0; j < src.width; j++) {
                const int col = j*src.tile_size_col + col_offset;
                dst_ptr[(row+0)*row_stride + col] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[0].x);
                dst_ptr[(row+1)*row_stride + col] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[0].y);
                dst_ptr[(row+2)*row_stride + col] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[1].x);
                dst_ptr[(row+3)*row_stride + col] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[1].y);
            }
            #endif 

        }
    }
}

#ifdef KITTENS_CDNA4
template<int axis, ducks::rt::accumulator_layout RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void store(const GL &dst, const RT &src, const COORD &idx) {
    using T = base_types::packing<typename RT::dtype>::unpacked_type;
    using U = typename GL::dtype;

    U *dst_ptr = (U*)&dst[(idx.template unit_coord<axis, 3>())];
    const int row_stride = dst.template stride<axis>();
    int laneid = kittens::laneid();

    if constexpr (sizeof(T) == 4) {
        const int row_offset = 4*(laneid/16), col_offset = laneid%16;
    
        #pragma unroll
        for(int i = 0; i < src.height; i++) {
            const int row = i*src.tile_size_row + row_offset;

            #pragma unroll
            for(int j = 0; j < src.width; j++) {
                const int col = j*src.tile_size_col + col_offset;
                dst_ptr[(row+0)*row_stride + col] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[0].x);
                dst_ptr[(row+1)*row_stride + col] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[0].y);
                dst_ptr[(row+2)*row_stride + col] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[1].x);
                dst_ptr[(row+3)*row_stride + col] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[1].y);
            }

        }
    } else {

        int col_offset = laneid%32;
        int row_offset = laneid/32;

        #pragma unroll
        for(int i = 0; i < src.height; i++) {
            #pragma unroll
            for(int j = 0; j < src.width; j++) {
                int col = src.tile_size_col*j + col_offset;
                #pragma unroll
                for (int ii = 0; ii < 4; ii++) {
                    int row = src.tile_size_row*i + ii * 8 + row_offset * 4;

                    dst_ptr[(row+0)*row_stride + col] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[ii * 2].x);
                    dst_ptr[(row+1)*row_stride + col] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[ii * 2].y);
                    dst_ptr[(row+2)*row_stride + col] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[ii * 2 + 1].x);
                    dst_ptr[(row+3)*row_stride + col] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[ii * 2 + 1].y);
                }
            }
        }
    }
}
#endif

template<ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void store(const GL &dst, const RT &src, const COORD &idx) {
    store<2, RT, GL, COORD>(dst, src, idx);
}

}