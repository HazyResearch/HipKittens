#include "kittens.cuh"
using namespace kittens;

template<int axis, ducks::rt::accumulator_layout RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void store_transposed_split(const GL &dst, const RT &src, const COORD &idx, int jj) {
    using T2 = RT::dtype;
    using T = base_types::packing<typename RT::dtype>::unpacked_type;
    using U = typename GL::dtype;
    using U2 = base_types::packing<U>::packed_type;

    const int row_stride = dst.template stride<axis>();
    U *dst_ptr = (U*)&dst[(idx.template unit_coord<axis, 3>())];
    i32x4 srsrc = make_srsrc(dst_ptr, row_stride * RT::rows * sizeof(U));

    int laneid = kittens::laneid();
    int col_offset = (laneid/32) * 4;
    int row_offset = laneid%32;

    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        int row = src.tile_size_row * i + row_offset;
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = src.tile_size_col * j + col_offset;

            // U2 data[2];

            // data[0] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[jj * 2]);
            // data[1] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[jj * 2 + 1]);
            // buffer_store_vec2(*(float2*)&data, srsrc, (row*row_stride + col + (jj * 8)) * sizeof(U));
            dst_ptr[row*row_stride + col + (jj * 8) + 0] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[jj * 2].x);
            dst_ptr[row*row_stride + col + (jj * 8) + 1] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[jj * 2].y);
            dst_ptr[row*row_stride + col + (jj * 8) + 2] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[jj * 2 + 1].x);
            dst_ptr[row*row_stride + col + (jj * 8) + 3] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[jj * 2 + 1].y);
        }
    }
}

template<ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void store_transposed_split(const GL &dst, const RT &src, const COORD &idx, int jj) {
    store_transposed_split<1, RT, GL, COORD>(dst, src, idx, jj);
}

template<int axis, ducks::rt::accumulator_layout RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void store_transposed(const GL &dst, const RT &src, const COORD &idx) {
    using T2 = RT::dtype;
    using T = base_types::packing<typename RT::dtype>::unpacked_type;
    using U = typename GL::dtype;
    using U2 = base_types::packing<U>::packed_type;

    const int row_stride = dst.template stride<axis>();
    U *dst_ptr = (U*)&dst[(idx.template unit_coord<axis, 3>())];
    i32x4 srsrc = make_srsrc(dst_ptr, row_stride * RT::rows * sizeof(U));

    int laneid = kittens::laneid();
    int col_offset = (laneid/32) * 4;
    int row_offset = laneid%32;

    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        int row = src.tile_size_row * i + row_offset;
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = src.tile_size_col * j + col_offset;

            #pragma unroll
            for (int jj = 0; jj < 4; jj++) {
                U2 data[2];

                data[0] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[jj * 2]);
                data[1] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[jj * 2 + 1]);
                kittens::llvm_amdgcn_raw_buffer_store_b64(*(uint64_t*)&data, srsrc, (row*row_stride + col + (jj * 8)) * sizeof(U), 0, static_cast<int>(kittens::coherency::cache_all));
            }
        }
    }
}

template<ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void store_transposed(const GL &dst, const RT &src, const COORD &idx) {
    store_transposed<1, RT, GL, COORD>(dst, src, idx);
}



