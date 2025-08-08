/**
 * @file
 * @brief Functions for transferring data directly between global and shared memory and back.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

#ifdef KITTENS_CDNA4
__device__ inline int get_accum_thread_row_offset(int row) {
    // x in [0..15]
    return ((row & 4) << 1)   // bit-2 → bit-3
            | ((row & 8) >> 1)   // bit-3 → bit-2
            | ( row & 3 );       // bits-0..1 unchanged
}

enum class coherency {
	cache_all = 0,
	cache_global = 1,
	cache_stream = 2,
	non_temporal = 3
};

using as3_uint32_ptr = uint32_t __attribute__((address_space(3)))*;
using index_t = int;
using int32x4_t = int32_t __attribute__((ext_vector_type(4)));

extern "C" __device__ void 
llvm_amdgcn_raw_buffer_load_lds(int32x4_t rsrc, // does not change (buffer resource; scalar array?)
                                as3_uint32_ptr lds_ptr, // does not change
                                index_t size, // does not change (16 bytes)
                                index_t voffset, 
                                index_t soffset, 
                                index_t offset,  // does not change (0); instruction offset
                                index_t aux) __asm("llvm.amdgcn.raw.buffer.load.lds"); // cache coherency
#endif

#ifdef KITTENS_CDNA4
template<int axis, bool assume_aligned,
         ducks::rt_layout::all RT_layout, ducks::st::all ST, ducks::gl::all GL,
         ducks::coord::tile COORD = coord<ST>,
         int N_THREADS = WARP_THREADS>
__device__ inline void load(ST& dst, const GL& src, const COORD& idx)
{

    using T = typename ST::dtype;
    static_assert(sizeof(T) == 2, "only supporting 16-bit dtypes");
    constexpr int memcpy_per_tile =  ST::rows * ST::cols * sizeof(T) / (16 * N_THREADS); // 16 --> 32
    static_assert(memcpy_per_tile > 0, "memcpy_per_tile must be greater than 0. Please decrease the number of threads.");
    
    constexpr int elem_per_thread = 16 / sizeof(T);  // 8
    constexpr int elem_per_warp = elem_per_thread * kittens::WARP_THREADS; // 512
    const int laneid = kittens::laneid() % N_THREADS;
    const int warp_id = laneid / N_THREADS;
    const int row_stride = src.template stride<axis>();

    constexpr int num_warps = N_THREADS / 64;
    constexpr int num_register_subtiles = kittens::TILE_ROW_DIM<T> * kittens::TILE_COL_DIM<T> / elem_per_warp;
    constexpr int num_register_tiles_per_row = ST::cols / kittens::TILE_ROW_DIM<T>;

    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    T* global_ptr = (T*)&src[unit_coord];
    i32x4 srsrc = make_srsrc(global_ptr, row_stride * ST::rows * sizeof(T));

    const T* lds_base = &dst.data[0] + (warp_id * elem_per_warp);

    #pragma unroll
    for (int i = 0; i < memcpy_per_tile; i++) {

        const int register_tile_id = (warp_id + i * num_warps) / num_register_subtiles;
        const int register_subtile_id = (warp_id + i * num_warps) % num_register_subtiles;

        int row_offset, col_offset; 

        if constexpr (std::is_same_v<RT_layout, ducks::rt_layout::row>) {
            const int register_subtile_cols = kittens::TILE_COL_DIM<T> / num_register_subtiles;
            const int num_register_subtiles_per_row = num_register_tiles_per_row * num_register_subtiles;
            const int warp_col_offset = ((register_tile_id % num_register_tiles_per_row) * num_register_subtiles + register_subtile_id) * register_subtile_cols;
            const int warp_row_offset = (register_tile_id / num_register_tiles_per_row) * kittens::TILE_ROW_DIM<T>;

            col_offset = warp_col_offset + (laneid / 32) * elem_per_thread;
            row_offset = warp_row_offset + (laneid % 32);
        } else if constexpr (std::is_same_v<RT_layout, ducks::rt_layout::col>) {
            const int register_subtile_rows = kittens::TILE_ROW_DIM<T> / num_register_subtiles;
            const int num_register_subtiles_per_row = num_register_tiles_per_row;
            const int warp_col_offset = (register_tile_id % num_register_subtiles_per_row) * kittens::TILE_COL_DIM<T>;
            const int warp_row_offset = ((register_tile_id / num_register_subtiles_per_row) * num_register_subtiles + register_subtile_id) * register_subtile_rows;
    
            col_offset = warp_col_offset + (laneid % 4) * elem_per_thread;
            row_offset = warp_row_offset + (laneid / 4);
        } else {
            const int register_subtile_rows = kittens::TILE_ROW_DIM<T> / num_register_subtiles;
            const int num_register_subtiles_per_row = num_register_tiles_per_row;
            const int warp_col_offset = (register_tile_id % num_register_subtiles_per_row) * kittens::TILE_COL_DIM<T>;
            const int warp_row_offset = ((register_tile_id / num_register_subtiles_per_row) * num_register_subtiles + register_subtile_id) * register_subtile_rows;
    
            col_offset = warp_col_offset + (laneid % 4) * elem_per_thread;
            row_offset = warp_row_offset + get_accum_thread_row_offset(laneid / 4);
        }

        const int offset_in_global = (row_offset * row_stride + col_offset) * sizeof(T);

        const T* lds_elem_ptr = lds_base + (i * N_THREADS * elem_per_thread);
        uintptr_t lds_addr = reinterpret_cast<uintptr_t>(lds_elem_ptr);
        as3_uint32_ptr lds_ptr = (as3_uint32_ptr)(lds_addr);

        llvm_amdgcn_raw_buffer_load_lds(
            srsrc, // buffer resource
            lds_ptr,
            16, // 16 bytes
            offset_in_global,
            0, 
            0, // instruction offset
            static_cast<index_t>(coherency::cache_all)); // cache coherency
    }
}

template<ducks::rt_layout::all RT_layout, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load(ST &dst, const GL &src, const COORD &idx) {
    load<2, false, RT_layout, ST, GL, COORD, WARP_THREADS>(dst, src, idx);
}
#else
template< int  axis, bool assume_aligned,
          ducks::st::all ST, ducks::gl::all GL,
          ducks::coord::tile COORD = coord<ST>,
          int  N_THREADS = WARP_THREADS >
__device__ inline void load(ST& dst, const GL& src, const COORD& idx)
{
    using T = typename ST::dtype;
    const int row_stride = src.template stride<axis>();
    // we can handle this many rows each time we run a memcpy_async
    constexpr int elem_per_memcpy = sizeof(float4)/sizeof(typename ST::dtype); // if bf16, then 16/2 = 8 
    constexpr int elem_per_half_memcpy = sizeof(float2)/sizeof(typename ST::dtype); // if bf16, then 8/2 = 4
    constexpr int memcpy_per_row = ST::cols / elem_per_memcpy; // if 64 columns, then 64/8 = 8
    constexpr int total_calls = (ST::cols * ST::rows + N_THREADS*elem_per_memcpy-1) / (N_THREADS*elem_per_memcpy); // round up

    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    typename GL::dtype *src_ptr = (typename GL::dtype*)&src[unit_coord];

    uint32_t dst_ptr = reinterpret_cast<uintptr_t>(&dst.data[0]);
    const int laneid = threadIdx.x % N_THREADS;

    // TODO: This is a hack to avoid the issue of too many VGPRs.
    // We should find a better way to do this.
    const int small_calls = 16;
    const int big_calls = (total_calls + small_calls - 1) / small_calls;
    float4    buf[small_calls];

    for (int i = 0; i < big_calls; i++) {
        const int offset = i * small_calls;
        #pragma unroll
        for(int j = 0; j < small_calls; j++) {
            int load_idx = (offset + j) * N_THREADS + laneid;
            int row = load_idx / memcpy_per_row;
            int col = (load_idx % memcpy_per_row) * elem_per_memcpy;

            if (row < dst.rows) {
                buf[j] = load_global_vec4_async((float4*) (src_ptr + (row * row_stride + col))); // thread loads 128-bits, 16-bytes
            }
        }

        #ifdef BUILTINS_ONLY
        __builtin_amdgcn_s_waitcnt(0);
        #else
        asm volatile("s_waitcnt vmcnt(0)"); 
        #endif

        #pragma unroll
        for(int j = 0; j < small_calls; j++) {
            int load_idx = (offset + j) * N_THREADS + laneid;
            int row = load_idx / memcpy_per_row;
            int col = (load_idx % memcpy_per_row) * elem_per_memcpy;

            if (row < dst.rows) {
                store_shared_vec(dst.idx(dst_ptr, {row, col}), {buf[j].x, buf[j].y});
                store_shared_vec(dst.idx(dst_ptr, {row, col + elem_per_half_memcpy}), {buf[j].z, buf[j].w});
            }
        }

        #ifdef BUILTINS_ONLY
        __builtin_amdgcn_s_waitcnt(0);
        #else
        asm volatile("s_waitcnt lgkmcnt(0)");
        #endif
    } 
}

template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load(ST &dst, const GL &src, const COORD &idx) {
    load<2, false, ST, GL, COORD, WARP_THREADS>(dst, src, idx);
}
#endif

/**
 * @brief Stores data from a shared memory tile into global memory.
 *
 * @tparam ST The type of the shared tile.
 * @param[out] dst The destination global memory array.
 * @param[in] src The source shared memory tile.
 * @param row_stride[in] The stride between rows in the destination array.
 */

#ifdef KITTENS_CDNA4
template<int axis, bool assume_aligned, 
        ducks::rt_layout::all RT_layout, ducks::st::all ST, ducks::gl::all GL, 
        ducks::coord::tile COORD=coord<ST>, int N_THREADS=WARP_THREADS>
__device__ static inline void store(const GL &dst, const ST &src, const COORD &idx) {

    using U = typename GL::dtype;
    using T = typename ST::dtype;
    static_assert(sizeof(T) == 2, "only supporting 16-bit dtypes");
    constexpr int memcpy_per_tile =  ST::rows * ST::cols * sizeof(T) / (16 * N_THREADS); // 16 --> 32
    static_assert(memcpy_per_tile > 0, "memcpy_per_tile must be greater than 0. Please decrease the number of threads.");

    const int laneid = kittens::laneid() % N_THREADS;
    const int elem_per_thread = 16 / sizeof(T); // 8 
    constexpr int elem_per_warp = elem_per_thread * kittens::WARP_THREADS; // 512
    const int warp_id = laneid / N_THREADS;
    const int row_stride = dst.template stride<axis>();

    constexpr int num_warps = N_THREADS / 64;
    constexpr int num_register_subtiles = kittens::TILE_ROW_DIM<T> * kittens::TILE_COL_DIM<T> / elem_per_warp;
    constexpr int num_register_tiles_per_row = ST::cols / kittens::TILE_ROW_DIM<T>;

    const T* lds_base = &src.data[0] + (warp_id * elem_per_warp);

    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    U* global_ptr = (U*)&dst[unit_coord];

    #pragma unroll
    for (int i = 0; i < memcpy_per_tile; i++) {

        const int register_tile_id = (warp_id + i * num_warps) / num_register_subtiles;
        const int register_subtile_id = (warp_id + i * num_warps) % num_register_subtiles;

        int row_offset, col_offset;

        if constexpr (std::is_same_v<RT_layout, ducks::rt_layout::row>) {
            const int register_subtile_cols = kittens::TILE_COL_DIM<T> / num_register_subtiles;
            const int num_register_subtiles_per_row = num_register_tiles_per_row * num_register_subtiles;
            const int warp_col_offset = ((register_tile_id % num_register_tiles_per_row) * num_register_subtiles + register_subtile_id) * register_subtile_cols;
            const int warp_row_offset = (register_tile_id / num_register_tiles_per_row) * kittens::TILE_ROW_DIM<T>;

            col_offset = warp_col_offset + (laneid / 32) * elem_per_thread;
            row_offset = warp_row_offset + (laneid % 32);
        } else if constexpr (std::is_same_v<RT_layout, ducks::rt_layout::col>) {
            const int register_subtile_rows = kittens::TILE_ROW_DIM<T> / num_register_subtiles;
            const int num_register_subtiles_per_row = num_register_tiles_per_row;
            const int warp_col_offset = (register_tile_id % num_register_subtiles_per_row) * kittens::TILE_COL_DIM<T>;
            const int warp_row_offset = ((register_tile_id / num_register_subtiles_per_row) * num_register_subtiles + register_subtile_id) * register_subtile_rows;
    
            col_offset = warp_col_offset + (laneid % 4) * elem_per_thread;
            row_offset = warp_row_offset + (laneid / 4);
        } else {
            const int register_subtile_rows = kittens::TILE_ROW_DIM<T> / num_register_subtiles;
            const int num_register_subtiles_per_row = num_register_tiles_per_row;
            const int warp_col_offset = (register_tile_id % num_register_subtiles_per_row) * kittens::TILE_COL_DIM<T>;
            const int warp_row_offset = ((register_tile_id / num_register_subtiles_per_row) * num_register_subtiles + register_subtile_id) * register_subtile_rows;
    
            col_offset = warp_col_offset + (laneid % 4) * elem_per_thread;
            row_offset = warp_row_offset + get_accum_thread_row_offset(laneid / 4);
        }

        const int offset_in_global = (row_offset * row_stride + col_offset);
        const T* lds_elem_ptr = lds_base + (i * N_THREADS * elem_per_thread);

        global_ptr[offset_in_global] = lds_elem_ptr[laneid * elem_per_thread];
        global_ptr[offset_in_global + 1] = lds_elem_ptr[laneid * elem_per_thread + 1];
        global_ptr[offset_in_global + 2] = lds_elem_ptr[laneid * elem_per_thread + 2];
        global_ptr[offset_in_global + 3] = lds_elem_ptr[laneid * elem_per_thread + 3];
        global_ptr[offset_in_global + 4] = lds_elem_ptr[laneid * elem_per_thread + 4];
        global_ptr[offset_in_global + 5] = lds_elem_ptr[laneid * elem_per_thread + 5];
        global_ptr[offset_in_global + 6] = lds_elem_ptr[laneid * elem_per_thread + 6];
        global_ptr[offset_in_global + 7] = lds_elem_ptr[laneid * elem_per_thread + 7];
    }
}
template<ducks::rt_layout::all RT_layout, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store(const GL &dst, const ST &src, const COORD &idx) {
    store<2, false, RT_layout, ST, GL, COORD, WARP_THREADS>(dst, src, idx);
}
#else
template<int axis, bool assume_aligned, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>, int N_THREADS=WARP_THREADS>
__device__ static inline void store(const GL &dst, const ST &src, const COORD &idx) {
    using T = typename ST::dtype;
    const int row_stride = dst.template stride<axis>();
    // we can handle this many rows each time we run a memcpy_async
    constexpr int elem_per_memcpy = sizeof(float4)/sizeof(typename ST::dtype);
    constexpr int elem_per_float = sizeof(float)/sizeof(typename ST::dtype);
    constexpr int memcpy_per_row = ST::cols / elem_per_memcpy;
    constexpr int total_calls = (ST::cols * ST::rows + N_THREADS*elem_per_memcpy-1) / (N_THREADS*elem_per_memcpy); // round up

    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    typename GL::dtype *dst_ptr = (typename GL::dtype*)&dst[unit_coord];

    uint32_t src_ptr = reinterpret_cast<uintptr_t>(&src.data[0]);
    int laneid = threadIdx.x % N_THREADS;

    #pragma unroll
    for(int i = 0; i < total_calls; i++) {

        int load_idx = i * N_THREADS + laneid;
        int row = load_idx / memcpy_per_row;
        int col = (load_idx*elem_per_memcpy) % src.cols;

        if (row < src.rows) {
            *(float*) &dst_ptr[row * row_stride + col] = *(float*)(&src[{row, col}]);
            *(float*) &dst_ptr[row * row_stride + col + elem_per_float] = *(float*)(&src[{row, col + elem_per_float}]);
            *(float*) &dst_ptr[row * row_stride + col + elem_per_float * 2] = *(float*)(&src[{row, col + elem_per_float * 2}]);
            *(float*) &dst_ptr[row * row_stride + col + elem_per_float * 3] = *(float*)(&src[{row, col + elem_per_float * 3}]);
        }
    }
}
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store(const GL &dst, const ST &src, const COORD &idx) {
    store<2, false, ST, GL, COORD, WARP_THREADS>(dst, src, idx);
}
#endif
}