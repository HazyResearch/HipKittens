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

    constexpr int bytes_per_thread = ST::underlying_subtile_bytes_per_thread;
    constexpr int bytes_per_warp = bytes_per_thread * kittens::WARP_THREADS;
    constexpr int memcpy_per_tile = ST::rows * ST::cols * sizeof(T) / (bytes_per_thread * N_THREADS);
    static_assert(ST::rows * ST::cols * sizeof(T) >= bytes_per_warp, "shared tile must be at least 1024 bytes");
    
    constexpr int num_warps = N_THREADS / kittens::WARP_THREADS;
    const int laneid = kittens::laneid();
    const int warpid = kittens::warpid() % num_warps;

    const int row_stride = src.template stride<axis>();

    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    T* global_ptr = (T*)&src[unit_coord];
    auto srsrc = make_srsrc(global_ptr, row_stride * ST::rows * sizeof(T));

    const uintptr_t lds_base = reinterpret_cast<uintptr_t>(&dst.data[0]) + (warpid * bytes_per_warp);

    if constexpr (memcpy_per_tile > 0) {

        #pragma unroll
        for (int i = 0; i < memcpy_per_tile; i++) {

            const int lane_byte_offset = (laneid * bytes_per_thread) + (warpid * bytes_per_warp) + (i * num_warps * bytes_per_warp);
            const int subtile_id = lane_byte_offset / ST::underlying_subtile_bytes;
            const int subtile_row = subtile_id / ST::underlying_subtiles_per_row;
            const int subtile_col = subtile_id % ST::underlying_subtiles_per_row;
            const int subtile_lane_byte_offset = lane_byte_offset % ST::underlying_subtile_bytes;

            const int row = subtile_lane_byte_offset / ST::underlying_subtile_row_bytes;
            const int col = (subtile_lane_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T);

            const uint32_t swizzled_shared_byte_offset = dst.swizzle({row, col});

            const int swizzled_global_row = (swizzled_shared_byte_offset / ST::underlying_subtile_row_bytes) + subtile_row * ST::underlying_subtile_rows;
            const int swizzled_global_col = (swizzled_shared_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T) + subtile_col * ST::underlying_subtile_cols;
            const uint32_t swizzled_global_byte_offset = (swizzled_global_row * row_stride + swizzled_global_col) * sizeof(T);

            uintptr_t lds_addr = lds_base + (i * num_warps * bytes_per_warp);
            __attribute__((address_space(3))) void* lds_ptr = (__attribute__((address_space(3))) void*)(lds_addr);

#ifdef __HIP_DEVICE_COMPILE__
            __builtin_amdgcn_raw_ptr_buffer_load_lds(
                srsrc,
                lds_ptr,
                bytes_per_thread,
                swizzled_global_byte_offset,
                0, 
                0,
                static_cast<int>(coherency::cache_all));
#endif
        }
    }
    // there are leftover loads that need to be handled here
    if constexpr (memcpy_per_tile * (bytes_per_thread * N_THREADS) != ST::rows * ST::cols * sizeof(T)) {

        constexpr int leftover_bytes = ST::rows * ST::cols * sizeof(T) - memcpy_per_tile * (bytes_per_thread * N_THREADS);
        constexpr int leftover_threads = leftover_bytes / bytes_per_thread;
        constexpr int leftover_warps = leftover_threads / kittens::WARP_THREADS;

        if (warpid < leftover_warps) {
            const int lane_byte_offset = (laneid * bytes_per_thread) + (warpid * bytes_per_warp) + (memcpy_per_tile * num_warps * bytes_per_warp);
            const int subtile_id = lane_byte_offset / ST::underlying_subtile_bytes;
            const int subtile_row = subtile_id / ST::underlying_subtiles_per_row;
            const int subtile_col = subtile_id % ST::underlying_subtiles_per_row;
            const int subtile_lane_byte_offset = lane_byte_offset % ST::underlying_subtile_bytes;

            const int row = subtile_lane_byte_offset / ST::underlying_subtile_row_bytes;
            const int col = (subtile_lane_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T);

            const uint32_t swizzled_shared_byte_offset = dst.swizzle({row, col});

            const int swizzled_global_row = (swizzled_shared_byte_offset / ST::underlying_subtile_row_bytes) + subtile_row * ST::underlying_subtile_rows;
            const int swizzled_global_col = (swizzled_shared_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T) + subtile_col * ST::underlying_subtile_cols;
            const uint32_t swizzled_global_byte_offset = (swizzled_global_row * row_stride + swizzled_global_col) * sizeof(T);

            uintptr_t lds_addr = lds_base + (memcpy_per_tile * num_warps * bytes_per_warp);
            __attribute__((address_space(3))) void* lds_ptr = (__attribute__((address_space(3))) void*)(lds_addr);

#ifdef __HIP_DEVICE_COMPILE__
            __builtin_amdgcn_raw_ptr_buffer_load_lds(
                srsrc,
                lds_ptr,
                bytes_per_thread,
                swizzled_global_byte_offset,
                0, 
                0,
                static_cast<int>(coherency::cache_all));
#endif
        }
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
 
    constexpr int bytes_per_thread = ST::underlying_subtile_bytes_per_thread;
    constexpr int bytes_per_warp = bytes_per_thread * kittens::WARP_THREADS;
    constexpr int memcpy_per_tile =  ST::rows * ST::cols * sizeof(T) / (bytes_per_thread * N_THREADS);
    static_assert(ST::rows * ST::cols * sizeof(T) >= bytes_per_warp, "shared tile must be at least 1024 bytes");

    constexpr int num_warps = N_THREADS / kittens::WARP_THREADS;
    const int laneid = kittens::laneid();
    const int warpid = kittens::warpid() % num_warps;

    const int row_stride = src.template stride<axis>();

    #pragma unroll
    for (int i = 0; i < memcpy_per_tile; i++) {
        const int lane_byte_offset = (laneid * bytes_per_thread) + (warpid * bytes_per_warp) + (i * num_warps * bytes_per_warp);
        const int subtile_id = lane_byte_offset / ST::underlying_subtile_bytes;
        const int subtile_row = subtile_id / ST::underlying_subtiles_per_row;
        const int subtile_col = subtile_id % ST::underlying_subtiles_per_row;
        const int subtile_lane_byte_offset = lane_byte_offset % ST::underlying_subtile_bytes;

        int row = subtile_lane_byte_offset / ST::underlying_subtile_row_bytes;
        int col = (subtile_lane_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T);
        const uint32_t swizzled_shared_byte_offset = dst.swizzle({row, col});

        const int swizzled_global_row = (swizzled_shared_byte_offset / ST::underlying_subtile_row_bytes) + subtile_row * ST::underlying_subtile_rows;
        const int swizzled_global_col = (swizzled_shared_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T) + subtile_col * ST::underlying_subtile_cols;
        const uint32_t swizzled_global_byte_offset = (swizzled_global_row * row_stride + swizzled_global_col) * sizeof(T);
        swizzled_offsets[i] = swizzled_global_byte_offset;
    }

    // there are leftover loads that need to be handled here
    if constexpr (memcpy_per_tile * (bytes_per_thread * N_THREADS) != ST::rows * ST::cols * sizeof(T)) {

        constexpr int leftover_bytes = ST::rows * ST::cols * sizeof(T) - memcpy_per_tile * (bytes_per_thread * N_THREADS);
        constexpr int leftover_threads = leftover_bytes / bytes_per_thread;
        constexpr int leftover_warps = leftover_threads / kittens::WARP_THREADS;

        if (warpid < leftover_warps) {
            const int lane_byte_offset = (laneid * bytes_per_thread) + (warpid * bytes_per_warp) + (memcpy_per_tile * num_warps * bytes_per_warp);
            const int subtile_id = lane_byte_offset / ST::underlying_subtile_bytes;
            const int subtile_row = subtile_id / ST::underlying_subtiles_per_row;
            const int subtile_col = subtile_id % ST::underlying_subtiles_per_row;
            const int subtile_lane_byte_offset = lane_byte_offset % ST::underlying_subtile_bytes;

            const int row = subtile_lane_byte_offset / ST::underlying_subtile_row_bytes;
            const int col = (subtile_lane_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T);

            const uint32_t swizzled_shared_byte_offset = dst.swizzle({row, col});

            const int swizzled_global_row = (swizzled_shared_byte_offset / ST::underlying_subtile_row_bytes) + subtile_row * ST::underlying_subtile_rows;
            const int swizzled_global_col = (swizzled_shared_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T) + subtile_col * ST::underlying_subtile_cols;
            const uint32_t swizzled_global_byte_offset = (swizzled_global_row * row_stride + swizzled_global_col) * sizeof(T);

            swizzled_offsets[memcpy_per_tile] = swizzled_global_byte_offset;
        }
    }
}

template<int axis, bool assume_aligned,
         ducks::st::all ST, ducks::gl::all GL,
         ducks::coord::tile COORD = coord<ST>,
         int N_THREADS = WARP_THREADS>
__device__ inline void load(ST& dst, const GL& src, const COORD& idx, const uint32_t* swizzled_offsets)
{
    using T = typename ST::dtype;

    constexpr int bytes_per_thread = ST::underlying_subtile_bytes_per_thread;
    constexpr int bytes_per_warp = bytes_per_thread * kittens::WARP_THREADS;
    constexpr int elements_per_warp = bytes_per_warp / sizeof(T);
    constexpr int memcpy_per_tile = ST::rows * ST::cols * sizeof(T) / (bytes_per_thread * N_THREADS);
    static_assert(ST::rows * ST::cols * sizeof(T) >= bytes_per_warp, "shared tile must be at least 1024 bytes");
    
    constexpr int num_warps = N_THREADS / kittens::WARP_THREADS;
    const int warpid = kittens::warpid() % num_warps;

    const int row_stride = src.template stride<axis>();
    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    T* global_ptr = (T*)&src[unit_coord];
    auto srsrc = make_srsrc(global_ptr, row_stride * ST::rows * sizeof(T));

    const T* lds_base = &dst.data[0] + (warpid * elements_per_warp);

    #pragma unroll
    for (int i = 0; i < memcpy_per_tile; i++) {

        const T* lds_elem_ptr = lds_base + (i * num_warps * elements_per_warp);
        uintptr_t lds_addr = reinterpret_cast<uintptr_t>(lds_elem_ptr);
        __attribute__((address_space(3))) void* lds_ptr = (__attribute__((address_space(3))) void*)(lds_addr);

#ifdef __HIP_DEVICE_COMPILE__
        __builtin_amdgcn_raw_ptr_buffer_load_lds(
            srsrc,
            lds_ptr,
            bytes_per_thread,
            swizzled_offsets[i],
            0, 
            0,
            static_cast<int>(coherency::cache_all));
#endif
    }

    // there are leftover loads that need to be handled here
    if constexpr (memcpy_per_tile * (bytes_per_thread * N_THREADS) != ST::rows * ST::cols * sizeof(T)) {

        constexpr int leftover_bytes = ST::rows * ST::cols * sizeof(T) - memcpy_per_tile * (bytes_per_thread * N_THREADS);
        constexpr int leftover_threads = leftover_bytes / bytes_per_thread;
        constexpr int leftover_warps = leftover_threads / kittens::WARP_THREADS;

        if (warpid < leftover_warps) {

            const T* lds_elem_ptr = lds_base + (memcpy_per_tile * num_warps * elements_per_warp);
            uintptr_t lds_addr = reinterpret_cast<uintptr_t>(lds_elem_ptr);
            __attribute__((address_space(3))) void* lds_ptr = (__attribute__((address_space(3))) void*)(lds_addr);

#ifdef __HIP_DEVICE_COMPILE__
            __builtin_amdgcn_raw_ptr_buffer_load_lds(
                srsrc,
                lds_ptr,
                bytes_per_thread,
                swizzled_offsets[memcpy_per_tile],
                0, 
                0,
                static_cast<int>(coherency::cache_all));
#endif
        }
    }
}

template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load(ST &dst, const GL &src, const COORD &idx, const uint32_t* swizzled_offsets) {
    load<2, false, ST, GL, COORD, WARP_THREADS>(dst, src, idx, swizzled_offsets);
}

/**
 * @brief Async variant of load (global→shared via buffer DMA).
 *
 * Uses __builtin_amdgcn_raw_ptr_buffer_load_async_lds so that subsequent
 * ds_read instructions are NOT blocked by a conservative s_waitcnt.
 * Caller must place __builtin_amdgcn_asyncmark() after the call, and
 * __builtin_amdgcn_wait_asyncmark(N) before reading the destination tile.
 */
template<int axis, bool assume_aligned,
         ducks::st::all ST, ducks::gl::all GL,
         ducks::coord::tile COORD = coord<ST>,
         int N_THREADS = WARP_THREADS>
__device__ inline void load_async(ST& dst, const GL& src, const COORD& idx, const uint32_t* swizzled_offsets)
{
    using T = typename ST::dtype;

    constexpr int bytes_per_thread = ST::underlying_subtile_bytes_per_thread;
    constexpr int bytes_per_warp = bytes_per_thread * kittens::WARP_THREADS;
    constexpr int elements_per_warp = bytes_per_warp / sizeof(T);
    constexpr int memcpy_per_tile = ST::rows * ST::cols * sizeof(T) / (bytes_per_thread * N_THREADS);
    static_assert(ST::rows * ST::cols * sizeof(T) >= bytes_per_warp, "shared tile must be at least 1024 bytes");

    constexpr int num_warps = N_THREADS / kittens::WARP_THREADS;
    const int warpid = kittens::warpid() % num_warps;

    const int row_stride = src.template stride<axis>();
    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    T* global_ptr = (T*)&src[unit_coord];
    auto srsrc = make_srsrc(global_ptr, row_stride * ST::rows * sizeof(T));

    const T* lds_base = &dst.data[0] + (warpid * elements_per_warp);

    #pragma unroll
    for (int i = 0; i < memcpy_per_tile; i++) {

        const T* lds_elem_ptr = lds_base + (i * num_warps * elements_per_warp);
        uintptr_t lds_addr = reinterpret_cast<uintptr_t>(lds_elem_ptr);
        __attribute__((address_space(3))) void* lds_ptr = (__attribute__((address_space(3))) void*)(lds_addr);

#ifdef __HIP_DEVICE_COMPILE__
        __builtin_amdgcn_raw_ptr_buffer_load_async_lds(
            srsrc,
            lds_ptr,
            bytes_per_thread,
            swizzled_offsets[i],
            0,
            0,
            static_cast<int>(coherency::cache_all));
#endif
    }

    if constexpr (memcpy_per_tile * (bytes_per_thread * N_THREADS) != ST::rows * ST::cols * sizeof(T)) {

        constexpr int leftover_bytes = ST::rows * ST::cols * sizeof(T) - memcpy_per_tile * (bytes_per_thread * N_THREADS);
        constexpr int leftover_threads = leftover_bytes / bytes_per_thread;
        constexpr int leftover_warps = leftover_threads / kittens::WARP_THREADS;

        if (warpid < leftover_warps) {

            const T* lds_elem_ptr = lds_base + (memcpy_per_tile * num_warps * elements_per_warp);
            uintptr_t lds_addr = reinterpret_cast<uintptr_t>(lds_elem_ptr);
            __attribute__((address_space(3))) void* lds_ptr = (__attribute__((address_space(3))) void*)(lds_addr);

#ifdef __HIP_DEVICE_COMPILE__
            __builtin_amdgcn_raw_ptr_buffer_load_async_lds(
                srsrc,
                lds_ptr,
                bytes_per_thread,
                swizzled_offsets[memcpy_per_tile],
                0,
                0,
                static_cast<int>(coherency::cache_all));
#endif
        }
    }
}

template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx, const uint32_t* swizzled_offsets) {
    load_async<2, false, ST, GL, COORD, WARP_THREADS>(dst, src, idx, swizzled_offsets);
}

template<int axis, bool assume_aligned, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD = coord<ST>, int N_THREADS = WARP_THREADS>
__attribute__((always_inline)) 
__device__ __forceinline__ void load(ST& dst, const GL& src, const COORD& idx,
                                const uint32_t* __restrict__ swizzled_offsets,
                                __amdgpu_buffer_rsrc_t SRD,
                                const void* base_ptr, const uint32_t lds_base)
{
    using T = typename ST::dtype;
    static_assert(sizeof(T) == 2 || sizeof(T) == 1, "only supporting 16 and 8-bit dtypes");

    constexpr int bytes_per_thread = 16;
    constexpr int bytes_per_memcpy = bytes_per_thread * N_THREADS;
    constexpr int memcpy_per_tile  = (ST::rows * ST::cols * sizeof(T)) / bytes_per_memcpy;
    static_assert(bytes_per_memcpy % 16 == 0, "LDS bump must be 16-aligned");

    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    T* __restrict__ gptr = (T*)&src[unit_coord];

    int SOFF = static_cast<int>(
        reinterpret_cast<const char*>(gptr) - reinterpret_cast<const char*>(base_ptr)
    );

    __attribute__((address_space(3))) void* lds_ptr =
        (__attribute__((address_space(3))) void*)(uintptr_t)lds_base;

    #pragma unroll
    for (int i = 0; i < memcpy_per_tile; ++i) {
#ifdef __HIP_DEVICE_COMPILE__
        __builtin_amdgcn_raw_ptr_buffer_load_lds(
            SRD,
            lds_ptr,
            16,
            swizzled_offsets[i],
            SOFF,
            0,
            static_cast<int>(coherency::cache_all)
        );
#endif

        lds_ptr = (__attribute__((address_space(3))) void*)(
            (uintptr_t)lds_ptr + bytes_per_memcpy);
    }
}
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load(ST &dst, const GL &src, const COORD &idx, const uint32_t* __restrict__ swizzled_offsets, __amdgpu_buffer_rsrc_t srd, const void* base_ptr, uint32_t lds_base) {
    load<2, false, ST, GL, COORD, WARP_THREADS>(dst, src, idx, swizzled_offsets, srd, base_ptr, lds_base);
}

/**
 * @brief Async variant of SRD-based load (global→shared via buffer DMA).
 *
 * Uses __builtin_amdgcn_raw_ptr_buffer_load_async_lds so that subsequent
 * ds_read instructions are NOT blocked by a conservative s_waitcnt.
 * Caller must place __builtin_amdgcn_asyncmark() after the call, and
 * __builtin_amdgcn_wait_asyncmark(N) before reading the destination tile.
 */
template<int axis, bool assume_aligned, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD = coord<ST>, int N_THREADS = WARP_THREADS>
__attribute__((always_inline))
__device__ __forceinline__ void load_async(ST& dst, const GL& src, const COORD& idx,
                                const uint32_t* __restrict__ swizzled_offsets,
                                __amdgpu_buffer_rsrc_t SRD,
                                const void* base_ptr, const uint32_t lds_base)
{
    using T = typename ST::dtype;
    static_assert(sizeof(T) == 2 || sizeof(T) == 1, "only supporting 16 and 8-bit dtypes");

    constexpr int bytes_per_thread = 16;
    constexpr int bytes_per_memcpy = bytes_per_thread * N_THREADS;
    constexpr int memcpy_per_tile  = (ST::rows * ST::cols * sizeof(T)) / bytes_per_memcpy;
    static_assert(bytes_per_memcpy % 16 == 0, "LDS bump must be 16-aligned");

    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    T* __restrict__ gptr = (T*)&src[unit_coord];

    int SOFF = static_cast<int>(
        reinterpret_cast<const char*>(gptr) - reinterpret_cast<const char*>(base_ptr)
    );

    __attribute__((address_space(3))) void* lds_ptr =
        (__attribute__((address_space(3))) void*)(uintptr_t)lds_base;

    #pragma unroll
    for (int i = 0; i < memcpy_per_tile; ++i) {
#ifdef __HIP_DEVICE_COMPILE__
        __builtin_amdgcn_raw_ptr_buffer_load_async_lds(
            SRD,
            lds_ptr,
            16,
            swizzled_offsets[i],
            SOFF,
            0,
            static_cast<int>(coherency::cache_all)
        );
#endif

        lds_ptr = (__attribute__((address_space(3))) void*)(
            (uintptr_t)lds_ptr + bytes_per_memcpy);
    }
}
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx, const uint32_t* __restrict__ swizzled_offsets, __amdgpu_buffer_rsrc_t srd, const void* base_ptr, uint32_t lds_base) {
    load_async<2, false, ST, GL, COORD, WARP_THREADS>(dst, src, idx, swizzled_offsets, srd, base_ptr, lds_base);
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

    static_assert(std::is_same_v<T, U>, "T and U must be the same type");
    static_assert(!std::is_same_v<T, fp8e4m3>, "Unsupported type for store");

    constexpr int bytes_per_thread = ST::underlying_subtile_bytes_per_thread;
    constexpr int elems_per_thread = bytes_per_thread / sizeof(T);
    constexpr int bytes_per_warp = bytes_per_thread * kittens::WARP_THREADS;
    constexpr int memcpy_per_tile =  ST::rows * ST::cols * sizeof(T) / (bytes_per_thread * N_THREADS);

    constexpr int num_warps = N_THREADS / kittens::WARP_THREADS;
    const int laneid = kittens::laneid();
    const int warpid = kittens::warpid() % num_warps;

    const int row_stride = dst.template stride<axis>();

    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    uintptr_t dst_ptr = reinterpret_cast<uintptr_t>(&dst[unit_coord]);
    uintptr_t src_ptr = reinterpret_cast<uintptr_t>(&src.data[0]);

    if constexpr (memcpy_per_tile > 0) {

        #pragma unroll
        for (int i = 0; i < memcpy_per_tile; i++) {
            const int lane_byte_offset = (laneid * bytes_per_thread) + (warpid * bytes_per_warp) + (i * num_warps * bytes_per_warp);
            const int subtile_id = lane_byte_offset / ST::underlying_subtile_bytes;
            const int subtile_row = subtile_id / ST::underlying_subtiles_per_row;
            const int subtile_col = subtile_id % ST::underlying_subtiles_per_row;
            const int subtile_lane_byte_offset = lane_byte_offset % ST::underlying_subtile_bytes;

            const int row = subtile_lane_byte_offset / ST::underlying_subtile_row_bytes;
            const int col = (subtile_lane_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T);
            const uint32_t swizzled_shared_byte_offset = src.swizzle({row, col});

            const int swizzled_global_row = (swizzled_shared_byte_offset / ST::underlying_subtile_row_bytes) + subtile_row * ST::underlying_subtile_rows;
            const int swizzled_global_col = (swizzled_shared_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T) + subtile_col * ST::underlying_subtile_cols;
            const uint32_t swizzled_global_byte_offset = (swizzled_global_row * row_stride + swizzled_global_col) * sizeof(T);

            U* dst_elem_ptr = (U*)(dst_ptr + swizzled_global_byte_offset);
            T* src_elem_ptr = (T*)(src_ptr + lane_byte_offset);

            #pragma unroll
            for (int j = 0; j < elems_per_thread; j++) {
                dst_elem_ptr[j] = kittens::base_types::convertor<U, T>::convert(src_elem_ptr[j]);
            }
        }
    }

    if constexpr (memcpy_per_tile * (bytes_per_thread * N_THREADS) != ST::rows * ST::cols * sizeof(T)) {

        constexpr int leftover_bytes = ST::rows * ST::cols * sizeof(T) - memcpy_per_tile * (bytes_per_thread * N_THREADS);
        constexpr int leftover_threads = leftover_bytes / bytes_per_thread;
        constexpr int leftover_warps = leftover_threads / kittens::WARP_THREADS;

        if (warpid < leftover_warps) {
            const int lane_byte_offset = (laneid * bytes_per_thread) + (warpid * bytes_per_warp) + (memcpy_per_tile * num_warps * bytes_per_warp);
            const int subtile_id = lane_byte_offset / ST::underlying_subtile_bytes;
            const int subtile_row = subtile_id / ST::underlying_subtiles_per_row;
            const int subtile_col = subtile_id % ST::underlying_subtiles_per_row;
            const int subtile_lane_byte_offset = lane_byte_offset % ST::underlying_subtile_bytes;

            const int row = subtile_lane_byte_offset / ST::underlying_subtile_row_bytes;
            const int col = (subtile_lane_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T);
            const uint32_t swizzled_shared_byte_offset = src.swizzle({row, col});

            const int swizzled_global_row = (swizzled_shared_byte_offset / ST::underlying_subtile_row_bytes) + subtile_row * ST::underlying_subtile_rows;
            const int swizzled_global_col = (swizzled_shared_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T) + subtile_col * ST::underlying_subtile_cols;
            const uint32_t swizzled_global_byte_offset = (swizzled_global_row * row_stride + swizzled_global_col) * sizeof(T);

            U* dst_elem_ptr = (U*)(dst_ptr + swizzled_global_byte_offset);
            T* src_elem_ptr = (T*)(src_ptr + lane_byte_offset);

            #pragma unroll
            for (int j = 0; j < elems_per_thread; j++) {
                dst_elem_ptr[j] = kittens::base_types::convertor<U, T>::convert(src_elem_ptr[j]);
            }
        }
    }
}
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store(const GL &dst, const ST &src, const COORD &idx) {
    store<2, false, ST, GL, COORD, WARP_THREADS>(dst, src, idx);
}
}
