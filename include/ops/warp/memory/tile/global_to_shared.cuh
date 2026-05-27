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
    i32x4 srsrc = make_srsrc(global_ptr, row_stride * ST::rows * sizeof(T));

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
    i32x4 srsrc = make_srsrc(global_ptr, row_stride * ST::rows * sizeof(T));

    const T* lds_base = &dst.data[0] + (warpid * elements_per_warp);

    #pragma unroll
    for (int i = 0; i < memcpy_per_tile; i++) {

        const T* lds_elem_ptr = lds_base + (i * num_warps * elements_per_warp);
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

    // there are leftover loads that need to be handled here
    if constexpr (memcpy_per_tile * (bytes_per_thread * N_THREADS) != ST::rows * ST::cols * sizeof(T)) {

        constexpr int leftover_bytes = ST::rows * ST::cols * sizeof(T) - memcpy_per_tile * (bytes_per_thread * N_THREADS);
        constexpr int leftover_threads = leftover_bytes / bytes_per_thread;
        constexpr int leftover_warps = leftover_threads / kittens::WARP_THREADS;

        if (warpid < leftover_warps) {

            const T* lds_elem_ptr = lds_base + (memcpy_per_tile * num_warps * elements_per_warp);
            uintptr_t lds_addr = reinterpret_cast<uintptr_t>(lds_elem_ptr);
            as3_uint32_ptr lds_ptr = (as3_uint32_ptr)(lds_addr);

            llvm_amdgcn_raw_buffer_load_lds(
                srsrc, // buffer resource
                lds_ptr,
                bytes_per_thread,
                swizzled_offsets[memcpy_per_tile],
                0, 
                0, // instruction offset
                static_cast<int>(coherency::cache_all)); // cache coherency
        }
    }
}

template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load(ST &dst, const GL &src, const COORD &idx, const uint32_t* swizzled_offsets) {
    load<2, false, ST, GL, COORD, WARP_THREADS>(dst, src, idx, swizzled_offsets);
}


using as3_uint32_ptr = __attribute__((address_space(3))) unsigned int*;
inline __device__ __forceinline__ uint32_t to_sgpr_u32(uint32_t x) {
    x = __builtin_amdgcn_readfirstlane(x); // make uniform
    asm volatile("" : "+s"(x));            // keep in SGPR class
    return x;
}

template<int axis, bool assume_aligned, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD = coord<ST>, int N_THREADS = WARP_THREADS>
__attribute__((always_inline)) 
__device__ __forceinline__ void load(ST& dst, const GL& src, const COORD& idx,
                                const uint32_t* __restrict__ swizzled_offsets,
                                i32x4 SRD,
                                const void* base_ptr, const uint32_t lds_base)
{
    using T = typename ST::dtype;
    static_assert(sizeof(T) == 2 || sizeof(T) == 1, "only supporting 16 and 8-bit dtypes");

    constexpr int bytes_per_thread = 16;
    constexpr int bytes_per_memcpy = bytes_per_thread * N_THREADS;
    constexpr int memcpy_per_tile  = (ST::rows * ST::cols * sizeof(T)) / bytes_per_memcpy;
    static_assert(bytes_per_memcpy % 16 == 0, "LDS bump must be 16-aligned");

    constexpr int elem_per_thread = bytes_per_thread / sizeof(T);
    constexpr int elem_per_warp   = elem_per_thread * kittens::WARP_THREADS;

    // ---- compute per-tile base pointer and scalar offset (SOFF) ----
    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    T* __restrict__ gptr = (T*)&src[unit_coord];

    uint32_t SOFF = to_sgpr_u32(static_cast<uint32_t>(
    reinterpret_cast<const char*>(gptr) - reinterpret_cast<const char*>(base_ptr)
    ));

    // // ---- LDS base (byte address) as SGPR (wave-uniform) ----
    // const int num_warps = N_THREADS / kittens::WARP_THREADS;
    // const int wid = warpid() % num_warps;
    // uint32_t lds_base = to_sgpr_u32(static_cast<uint32_t>(
    // reinterpret_cast<uintptr_t>(&dst.data[0]) + wid * elem_per_warp * sizeof(T)
    // ));

    // ---- SGPR cursor we bump each iteration (no new readfirstlane) ----
    uint32_t lds_cur = lds_base;
    asm volatile("" : "+s"(lds_cur)); 

    #pragma unroll
    for (int i = 0; i < memcpy_per_tile; ++i) {
        int32_t lds_byte = lds_cur;                 // still SGPR
        asm volatile("" : "+s"(lds_byte));           // keep it SGPR at the use

        asm volatile("s_mov_b32 m0, %0" :: "s"(lds_byte));
        llvm_amdgcn_raw_buffer_load_lds(
            SRD, 
            (as3_uint32_ptr)0, 
            16, 
            swizzled_offsets[i], 
            SOFF, 
            0,
            static_cast<int>(coherency::cache_all)
        );

        // SGPR bump (compiler emits s_add_u32)
        lds_cur += bytes_per_memcpy;
    }
}
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load(ST &dst, const GL &src, const COORD &idx, const uint32_t* __restrict__ swizzled_offsets, i32x4 srd, const void* base_ptr, uint32_t lds_base) {
    load<2, false, ST, GL, COORD, WARP_THREADS>(dst, src, idx, swizzled_offsets, srd, base_ptr, lds_base);
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

/* ========================================================================== *
 * gfx1250 raw-pointer global -> LDS cooperative copy
 *
 * The gfx1250 family relies on `global_load_async_to_lds_*`,
 * `cluster_load_async_to_lds_*`, and `tensor_load_to_lds` (TDM) which all take
 * raw 16 B-aligned LDS pointers and emit single 16 B requests per lane (or per
 * warp, for TDM). Padded LDS layouts further complicate offset math, so the
 * gfx1250 path takes raw `T*` LDS slabs plus a compile-time padding descriptor.
 * Kernels allocate the slab via `shared_allocator::allocate_in<segment<I>>`
 * and call the helpers in this file (the cooperative `load` below) or in the
 * sibling headers (`async.cuh`, `tdm.cuh`) to populate it.
 * ========================================================================== */
#ifdef KITTENS_UDNA1

/**
 * @brief Compile-time LDS padding descriptor.
 *
 * Encodes the "insert `AMOUNT` pad elements every `INTERVAL` elements" rule
 * used to break bank-conflict hotspots on gfx1250 LDS. The defaults
 * (interval = 128 bf16 = 256 B, amount = 8 bf16 = 16 B) match the
 * recommended layout for 16-bit operands.
 */
template<int INTERVAL = 128, int AMOUNT = 8>
struct lds_padded {
    static constexpr int interval = INTERVAL;
    static constexpr int amount   = AMOUNT;
    __device__ __host__ __forceinline__ static constexpr int padded(int flat) {
        return flat + (flat / INTERVAL) * AMOUNT;
    }
    static constexpr int padded_elems(int total) {
        return total + (total / INTERVAL) * AMOUNT;
    }
};

/// @brief Padding descriptor for unpadded LDS layouts.
struct lds_nopad {
    static constexpr int interval = 0;
    static constexpr int amount   = 0;
    __device__ __host__ __forceinline__ static constexpr int padded(int flat) { return flat; }
    static constexpr int padded_elems(int total) { return total; }
};

/// @brief Default LDS padding for bf16 GEMMs.
using lds_pad_default = lds_padded<128, 8>;

namespace detail {
/**
 * @brief Subtile-major flat index helper for gfx1250 LDS layouts.
 *
 * Mirrors the indexing used by the gfx1250 path in
 * `ops/warp/memory/tile/shared_to_register.cuh`: rows/cols within a subtile
 * are stored row-major, subtiles are laid out row-major across the tile.
 * Used by both the cooperative `kittens::load` below and the per-lane async
 * load in `async.cuh`.
 */
template<int ROWS, int COLS, int SUB_ROWS, int SUB_COLS>
__device__ __forceinline__ int subtile_flat(int flat) {
    constexpr int sub_elems    = SUB_ROWS * SUB_COLS;
    constexpr int subs_per_row = COLS / SUB_COLS;
    const int subtile_id = flat / sub_elems;
    const int local_idx  = flat % sub_elems;
    const int local_row  = local_idx / SUB_COLS;
    const int local_col  = local_idx % SUB_COLS;
    const int sub_r      = subtile_id / subs_per_row;
    const int sub_c      = subtile_id % subs_per_row;
    return sub_r * SUB_ROWS * COLS
         + sub_c * SUB_COLS
         + local_row * COLS
         + local_col;
}
} // namespace detail

/**
 * @brief Cooperative register-mediated global -> LDS tile copy.
 *
 * Plain `global_load` -> VGPR -> `ds_store` path. Use this for correctness
 * baselines and for kernels that do not yet exercise the async or TDM paths.
 * The `Pad` parameter controls the per-element LDS placement; pass
 * `lds_nopad` for flat layouts and `lds_padded<INTERVAL, AMOUNT>` to insert
 * bank-conflict-avoiding pads at the LDS side.
 *
 * The `T*` first-argument signature distinguishes this from the CDNA
 * `load(ST&, ...)` overload above; the two coexist via overload resolution
 * but the gfx1250 path is the one selected when LDS storage is a raw `T*`
 * slab (the layout used by all gfx1250 async/TDM helpers).
 */
template<typename Pad = lds_nopad, int ROWS = 0, int COLS = 0, int N_THREADS = WARP_THREADS,
         typename T, ducks::gl::all GL, ducks::coord::tile COORD = coord<>>
__device__ inline void load(T* __restrict__ lds_dst_raw, const GL& src, const COORD& idx,
                            int row_stride)
{
    static_assert(ROWS > 0 && COLS > 0, "ROWS and COLS must be specified");
    constexpr int total_elems = ROWS * COLS;
    const int tid = threadIdx.x;
    // The COORD is interpreted as tile-index coordinates `{b, d, tile_row, tile_col}`
    // -- convert to element coordinates by multiplying the trailing two by ROWS/COLS.
    const int gr_base = idx.r * ROWS;
    const int gc_base = idx.c * COLS;
    const T* base = src.raw_ptr
                  + (((int64_t(idx.b) * src.depth() + idx.d) * src.rows() + gr_base)
                     * src.cols() + gc_base);

    // Coerce the LDS destination to a sizeof(T)-wide AS(3) integer pointer
    // and write the source bit-pattern, forcing the compiler to emit
    // `ds_store_b{8,16,32,64}` instead of a generic flat store. We round-trip
    // through an integer of matching width because element types like
    // `__hip_bfloat16` do not provide address-space-qualified `operator=`,
    // so direct assignment through an AS(3)-qualified `T*` fails to compile.
    static_assert(sizeof(T) == 1 || sizeof(T) == 2 ||
                  sizeof(T) == 4 || sizeof(T) == 8,
                  "cooperative load expects sizeof(T) in {1, 2, 4, 8}");
    using lds_int = std::conditional_t<sizeof(T) == 1, uint8_t,
                    std::conditional_t<sizeof(T) == 2, uint16_t,
                    std::conditional_t<sizeof(T) == 4, uint32_t,
                                       uint64_t>>>;
    using lds_int_as3 = lds_int __attribute__((address_space(3)));
    auto* lds_dst = (lds_int_as3*)(reinterpret_cast<uintptr_t>(lds_dst_raw));

    #pragma unroll
    for (int i = tid; i < total_elems; i += N_THREADS) {
        const int row = i / COLS;
        const int col = i % COLS;
        // Subtile-major LDS layout (rows of 16 x 32 subtiles).
        const int lds_flat = detail::subtile_flat<ROWS, COLS, 16, 32>(i);
        const T v = base[row * row_stride + col];
        lds_dst[Pad::padded(lds_flat)] = *reinterpret_cast<const lds_int*>(&v);
    }
}

#endif // KITTENS_UDNA1
}
