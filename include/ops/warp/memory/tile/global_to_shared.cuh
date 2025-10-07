/**
 * @file
 * @brief Functions for transferring data directly between global and shared memory and back.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

template<int axis, bool assume_aligned,
         ducks::st::all ST, ducks::gl::all GL,
         ducks::coord::tile COORD = coord<ST>,
         int N_THREADS = WARP_THREADS>
__device__ inline void load(ST& dst, const GL& src, const COORD& idx)
{
    using T = typename ST::dtype;

    constexpr int bytes_per_thread = 16;
    constexpr int memcpy_per_tile = ST::rows * ST::cols * sizeof(T) / (bytes_per_thread * N_THREADS);
    static_assert(memcpy_per_tile > 0, "memcpy_per_tile must be greater than 0. Please decrease the number of threads.");
    
    constexpr int elem_per_thread = bytes_per_thread / sizeof(T);
    constexpr int elem_per_warp = elem_per_thread * kittens::WARP_THREADS;
    constexpr int bytes_per_warp = elem_per_warp * sizeof(T);
    constexpr int num_warps = N_THREADS / kittens::WARP_THREADS;
    constexpr int bytes_per_subtile_row = ST::underlying_subtile_cols * sizeof(T);
    constexpr int bytes_per_subtile = bytes_per_subtile_row * ST::underlying_rows;
    const int laneid = kittens::laneid();
    const int warpid = kittens::warpid() % num_warps;

    const int row_stride = src.template stride<axis>();

    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    T* global_ptr = (T*)&src[unit_coord];
    i32x4 srsrc = make_srsrc(global_ptr, row_stride * ST::rows * sizeof(T));

    const T* lds_base = &dst.data[0] + (warpid * elem_per_warp);

    #pragma unroll
    for (int i = 0; i < memcpy_per_tile; i++) {

        int lane_byte_offset = (laneid * bytes_per_thread) + (warpid * bytes_per_warp) + (i * num_warps * bytes_per_warp);
        int subtile_id = lane_byte_offset / bytes_per_subtile;
        lane_byte_offset = lane_byte_offset % bytes_per_subtile;

        int row = lane_byte_offset / bytes_per_subtile_row;
        int col = (lane_byte_offset % bytes_per_subtile_row) / sizeof(T);
        uint32_t swizzled_shared_byte_offset = dst.idx((uint32_t)0, {row, col});

        int swizzled_global_row = swizzled_shared_byte_offset / bytes_per_subtile_row;
        int swizzled_global_col = (swizzled_shared_byte_offset % bytes_per_subtile_row) / sizeof(T) + subtile_id * ST::underlying_subtile_cols;
        uint32_t swizzled_global_byte_offset = (swizzled_global_row * row_stride + swizzled_global_col) * sizeof(T);

        const T* lds_elem_ptr = lds_base + (i * N_THREADS * elem_per_thread);
        uintptr_t lds_addr = reinterpret_cast<uintptr_t>(lds_elem_ptr);
        as3_uint32_ptr lds_ptr = (as3_uint32_ptr)(lds_addr);

        llvm_amdgcn_raw_buffer_load_lds(
            srsrc, // buffer resource
            lds_ptr,
            bytes_per_thread,
            swizzled_global_byte_offset,
            0, 
            0, // instruction offset
            static_cast<int>(coherency::cache_all)); // cache coherency
    }
}

template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load(ST &dst, const GL &src, const COORD &idx) {
    load<2, false, ST, GL, COORD, WARP_THREADS>(dst, src, idx);
}

template<int axis, bool assume_aligned,
         ducks::st::all ST, ducks::gl::all GL,
         int N_THREADS = WARP_THREADS>
__device__ inline void prefill_swizzled_offsets(
    ST& dst, const GL& src, uint32_t* swizzled_offsets)
{

    using T = typename ST::dtype;
    constexpr int bytes_per_thread = 16;
    constexpr int memcpy_per_tile =  ST::rows * ST::cols * sizeof(T) / (bytes_per_thread * N_THREADS);
    static_assert(memcpy_per_tile > 0, "memcpy_per_tile must be greater than 0. Please decrease the number of threads.");
    
    constexpr int elem_per_thread = bytes_per_thread / sizeof(T);
    constexpr int elem_per_warp = elem_per_thread * kittens::WARP_THREADS;
    constexpr int bytes_per_warp = elem_per_warp * sizeof(T);
    constexpr int num_warps = N_THREADS / kittens::WARP_THREADS;
    constexpr int bytes_per_subtile_row = ST::underlying_subtile_cols * sizeof(T);
    constexpr int bytes_per_subtile = bytes_per_subtile_row * ST::underlying_rows;
    const int laneid = kittens::laneid();
    const int warpid = kittens::warpid() % num_warps;

    const int row_stride = src.template stride<axis>();

    #pragma unroll
    for (int i = 0; i < memcpy_per_tile; i++) {
        int lane_byte_offset = (laneid * bytes_per_thread) + (warpid * bytes_per_warp) + (i * num_warps * bytes_per_warp);
        int subtile_id = lane_byte_offset / bytes_per_subtile;
        lane_byte_offset = lane_byte_offset % bytes_per_subtile;

        int row = lane_byte_offset / bytes_per_subtile_row;
        int col = (lane_byte_offset % bytes_per_subtile_row) / sizeof(T);

        uint32_t swizzled_shared_byte_offset = dst.idx((uint32_t)0, {row, col});
        int swizzled_global_row = swizzled_shared_byte_offset / bytes_per_subtile_row;
        int swizzled_global_col = (swizzled_shared_byte_offset % bytes_per_subtile_row) / sizeof(T) + subtile_id * ST::underlying_subtile_cols;
        uint32_t swizzled_global_byte_offset = (swizzled_global_row * row_stride + swizzled_global_col) * sizeof(T);
        swizzled_offsets[i] = swizzled_global_byte_offset;
    }
}

template<int axis, bool assume_aligned,
         ducks::st::all ST, ducks::gl::all GL,
         ducks::coord::tile COORD = coord<ST>,
         int N_THREADS = WARP_THREADS>
__device__ inline void load(ST& dst, const GL& src, const COORD& idx, const uint32_t* swizzled_offsets)
{
    using T = typename ST::dtype;

    constexpr int bytes_per_thread = 16;
    constexpr int memcpy_per_tile = ST::rows * ST::cols * sizeof(T) / (bytes_per_thread * N_THREADS);
    static_assert(memcpy_per_tile > 0, "memcpy_per_tile must be greater than 0. Please decrease the number of threads.");
    
    constexpr int elem_per_thread = bytes_per_thread / sizeof(T);
    constexpr int elem_per_warp = elem_per_thread * kittens::WARP_THREADS;
    constexpr int num_warps = N_THREADS / kittens::WARP_THREADS;
    const int laneid = kittens::laneid();
    const int warpid = kittens::warpid() % num_warps;

    const int row_stride = src.template stride<axis>();
    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    T* global_ptr = (T*)&src[unit_coord];
    i32x4 srsrc = make_srsrc(global_ptr, row_stride * ST::rows * sizeof(T));

    const T* lds_base = &dst.data[0] + (warpid * elem_per_warp);

    #pragma unroll
    for (int i = 0; i < memcpy_per_tile; i++) {

        const T* lds_elem_ptr = lds_base + (i * N_THREADS * elem_per_thread);
        uintptr_t lds_addr = reinterpret_cast<uintptr_t>(lds_elem_ptr);
        as3_uint32_ptr lds_ptr = (as3_uint32_ptr)(lds_addr);

        llvm_amdgcn_raw_buffer_load_lds(
            srsrc, // buffer resource
            lds_ptr,
            bytes_per_thread,
            swizzled_offsets[i],
            0, 
            0, // instruction offset
            static_cast<int>(coherency::cache_all)); // cache coherency
    }
}

template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load(ST &dst, const GL &src, const COORD &idx, const uint32_t* swizzled_offsets) {
    load<2, false, ST, GL, COORD, WARP_THREADS>(dst, src, idx, swizzled_offsets);
}

/**
 * @brief Stores data from a shared memory tile into global memory.
 *
 * @tparam ST The type of the shared tile.
 * @param[out] dst The destination global memory array.
 * @param[in] src The source shared memory tile.
 * @param row_stride[in] The stride between rows in the destination array.
 */

template<int axis, bool assume_aligned, 
        ducks::st::all ST, ducks::gl::all GL, 
        ducks::coord::tile COORD=coord<ST>, int N_THREADS=WARP_THREADS>
__device__ static inline void store(const GL &dst, const ST &src, const COORD &idx) {

    using T = typename ST::dtype;
    using U = typename GL::dtype;

    constexpr int bytes_per_thread = 16;
    constexpr int memcpy_per_tile =  ST::rows * ST::cols * sizeof(T) / (bytes_per_thread * N_THREADS);
    static_assert(memcpy_per_tile > 0, "memcpy_per_tile must be greater than 0. Please decrease the number of threads.");

    const int elem_per_thread = bytes_per_thread / sizeof(T);
    constexpr int elem_per_warp = elem_per_thread * kittens::WARP_THREADS;
    constexpr int bytes_per_warp = elem_per_warp * sizeof(T);
    constexpr int num_warps = N_THREADS / kittens::WARP_THREADS;
    constexpr int bytes_per_subtile_row = ST::underlying_subtile_cols * sizeof(T);
    constexpr int bytes_per_subtile = bytes_per_subtile_row * ST::underlying_rows;
    const int laneid = kittens::laneid();
    const int warpid = kittens::warpid() % num_warps;

    const int row_stride = dst.template stride<axis>();

    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    uintptr_t dst_ptr = reinterpret_cast<uintptr_t>(&dst[unit_coord]);
    uintptr_t src_ptr = reinterpret_cast<uintptr_t>(&src.data[0]);

    #pragma unroll
    for (int i = 0; i < memcpy_per_tile; i++) {
        int lane_byte_offset = (laneid * bytes_per_thread) + (warpid * bytes_per_warp) + (i * num_warps * bytes_per_warp);
        int subtile_id = lane_byte_offset / bytes_per_subtile;
        lane_byte_offset = lane_byte_offset % bytes_per_subtile;

        int row = lane_byte_offset / bytes_per_subtile_row;
        int col = (lane_byte_offset % bytes_per_subtile_row) / sizeof(T) + subtile_id * ST::underlying_subtile_cols;

        int global_byte_offset = (row * row_stride + col) * sizeof(T);

        U* dst_elem_ptr = (U*)(dst_ptr + global_byte_offset);

        #pragma unroll
        for (int j = 0; j < bytes_per_thread / sizeof(T); j++) {
            dst_elem_ptr[j] = src[{row, col + j}];
        }
    }
}
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store(const GL &dst, const ST &src, const COORD &idx) {
    store<2, false, ST, GL, COORD, WARP_THREADS>(dst, src, idx);
}
}