/**
 * @file
 * @brief Functions for transferring data directly between global memory and registers and back.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

template<int axis, ducks::rt::row_layout RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void load(RT &dst, const GL &src, const COORD &idx) {
    using T2 = RT::dtype;
    using U = typename GL::dtype;
    using U2 = base_types::packing<U>::packed_type;

    U *src_ptr = (U*)&src[(idx.template unit_coord<axis, 3>())];
    const int row_stride = src.template stride<axis>();
    int laneid = kittens::laneid();
    int row_offset = laneid%16, col_offset = 4*(laneid/16);

    uint32_t buffer_size = src.batch * src.depth * src.rows * src.cols * sizeof(U);
    buffer_resource br = make_buffer_resource(src_ptr, buffer_size, 0x00020000);

    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        int row = dst.tile_size_row*i + row_offset;
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            int col = dst.tile_size_col*j + col_offset;
            U2* tmp;
            if constexpr (sizeof(U2) == 4) { // bf16_2
                float2 loaded = std::bit_cast<float2>(llvm_amdgcn_raw_buffer_load_b64(
                    std::bit_cast<i32x4>(br),
                    (row*row_stride + col) * sizeof(U),
                    0,
                    0
                ));
                tmp = reinterpret_cast<U2*>(&loaded);
            }
            else { // float2
                float4 loaded = std::bit_cast<float4>(llvm_amdgcn_raw_buffer_load_b128(
                    std::bit_cast<i32x4>(br),
                    (row*row_stride + col) * sizeof(U),
                    0,
                    0
                ));
                tmp = reinterpret_cast<U2*>(&loaded);
            }
            #pragma unroll
            for(int k = 0; k < 2; k++) {
                dst.tiles[i][j].data[k] = base_types::convertor<T2, U2>::convert(tmp[k]);
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
    int row_offset = 4*(laneid/16), col_offset = laneid%16;
    
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        int row = i*dst.tile_size_row + row_offset;
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            int col = j*dst.tile_size_col + col_offset;
            dst.tiles[i][j].data[0].x = base_types::convertor<T, U>::convert(src_ptr[(row+0)*row_stride + col]);
            dst.tiles[i][j].data[0].y = base_types::convertor<T, U>::convert(src_ptr[(row+1)*row_stride + col]);
            dst.tiles[i][j].data[1].x = base_types::convertor<T, U>::convert(src_ptr[(row+2)*row_stride + col]);
            dst.tiles[i][j].data[1].y = base_types::convertor<T, U>::convert(src_ptr[(row+3)*row_stride + col]);
        }
    }

}
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
    int row_offset = laneid%16, col_offset = 4*(laneid/16);

    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        int row = src.tile_size_row*i + row_offset;
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = src.tile_size_col*j + col_offset;
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
        }
    }
}

using v2u32 = uint32_t __attribute__((ext_vector_type(2)));

__device__ inline uint16_t to_bf16(float x) {
    return static_cast<uint16_t>(__float_as_uint(x) >> 16);
}

__device__ inline uint32_t pack_bf16x2(float a, float b) {
    return (static_cast<uint32_t>(to_bf16(b)) << 16) | to_bf16(a);
}

__device__ void llvm_amdgcn_raw_buffer_store_b64(
    v2u32 vdata, i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.store.i64");

__device__ inline i32x4 make_srsrc(const void* ptr, uint32_t range_bytes) {
    buffer_resource rsrc = make_buffer_resource(ptr, range_bytes, 0x110000);
    return *reinterpret_cast<const i32x4*>(&rsrc);
}

// template<int axis, ducks::rt::row_layout RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
// __device__ inline void store(const GL& dst, const RT& src, const COORD& idx) {
//     static_assert(std::is_same_v<typename RT::dtype, float2>, "Expected RT dtype = float2");
//     static_assert(std::is_same_v<typename GL::dtype, bf16>, "Expected GL dtype = bf16");

//     const int row_stride = dst.template stride<axis>();
//     const int laneid     = kittens::laneid();
//     const int row_offset = laneid % 16;
//     const int col_offset = 4 * (laneid / 16);

//     coord<> unit = idx.template unit_coord<axis, 3>();
//     bf16* base_ptr = (bf16*)&dst[unit];
//     int range_bytes = row_stride * RT::rows * sizeof(bf16);
//     i32x4 srsrc = make_srsrc(base_ptr, range_bytes);

//     #pragma unroll
//     for (int i = 0; i < RT::height; ++i) {
//         int row = i * RT::tile_size_row + row_offset;
//         #pragma unroll
//         for (int j = 0; j < RT::width; ++j) {
//             int col = j * RT::tile_size_col + col_offset;

//             float2 d0 = src.tiles[i][j].data[0]; // {a, b}
//             float2 d1 = src.tiles[i][j].data[1]; // {c, d}

//             v2u32 packed = {
//                 pack_bf16x2(d0.x, d0.y),
//                 pack_bf16x2(d1.x, d1.y)
//             };

//             uint32_t offset_bytes = (row * row_stride + col) * sizeof(bf16);
//             llvm_amdgcn_raw_buffer_store_b64(packed, srsrc, offset_bytes, 0, 0);
//         }
//     }
// }


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
    const int row_offset = 4*(laneid/16), col_offset = laneid%16;

    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        const int row = i*src.tile_size_row + row_offset;

        const int row0 = (row+0)*row_stride;
        const int row1 = (row+1)*row_stride;
        const int row2 = (row+2)*row_stride;
        const int row3 = (row+3)*row_stride;

        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            const int col = j*src.tile_size_col + col_offset;
            dst_ptr[row0 + col] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[0].x);
            dst_ptr[row1 + col] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[0].y);
            dst_ptr[row2 + col] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[1].x);
            dst_ptr[row3 + col] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[1].y);
        }
    }
}
template<ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void store(const GL &dst, const RT &src, const COORD &idx) {
    store<2, RT, GL, COORD>(dst, src, idx);
}

/**
 * @brief Load data from a source array into a row-major layout tile.
 *
 * @tparam RT The row-major layout tile type.
 * @tparam U The data type of the source array.
 * @param dst[out] The destination tile to load data into.
 * @param src[in] The source array to load data from.
 * @param row_stride[in] The stride in elements between rows in the source array.
 */
template<int axis, ducks::rt::row_layout RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void load_async(RT &dst, const GL &src, const COORD &idx) {
    using T2 = RT::dtype;
    using U = typename GL::dtype;
    using U2 = base_types::packing<U>::packed_type;

    U *src_ptr = (U*)&src[(idx.template unit_coord<axis, 3>())];
    const int row_stride = src.template stride<axis>();
    int laneid = kittens::laneid();
    int row_offset = laneid%16, col_offset = 4*(laneid/16);

    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        int row = dst.tile_size_row*i + row_offset;
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            int col = dst.tile_size_col*j + col_offset;
            U2* tmp;
            if constexpr (sizeof(U2) == 4) { // bf16_2
                float2 loaded = load_global_vec2_async((float2*) (src_ptr + (row*row_stride + col)));
                tmp = reinterpret_cast<U2*>(&loaded);
            }
            else { // float2
                float4 loaded = load_global_vec4_async((float4*) (src_ptr + (row*row_stride + col)));
                tmp = reinterpret_cast<U2*>(&loaded);
            }
            #pragma unroll
            for(int k = 0; k < 2; k++) {
                dst.tiles[i][j].data[k] = base_types::convertor<T2, U2>::convert(tmp[k]);
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
__device__ inline static void load_async(RT &dst, const GL &src, const COORD &idx) {
    using T = base_types::packing<typename RT::dtype>::unpacked_type;
    using U = typename GL::dtype;
    
    U *src_ptr = (U*)&src[(idx.template unit_coord<axis, 3>())];
    const int row_stride = src.template stride<axis>();
    int laneid = kittens::laneid();
    int row_offset = 4*(laneid/16), col_offset = laneid%16;
    
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        int row = i*dst.tile_size_row + row_offset;
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            int col = j*dst.tile_size_col + col_offset;
            dst.tiles[i][j].data[0].x = base_types::convertor<T, U>::convert(src_ptr[(row+0)*row_stride + col]);
            dst.tiles[i][j].data[0].y = base_types::convertor<T, U>::convert(src_ptr[(row+1)*row_stride + col]);
            dst.tiles[i][j].data[1].x = base_types::convertor<T, U>::convert(src_ptr[(row+2)*row_stride + col]);
            dst.tiles[i][j].data[1].y = base_types::convertor<T, U>::convert(src_ptr[(row+3)*row_stride + col]);
        }
    }

}
template<ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void load_async(RT &dst, const GL &src, const COORD &idx) {
    load_async<2, RT, GL>(dst, src, idx);
}

}