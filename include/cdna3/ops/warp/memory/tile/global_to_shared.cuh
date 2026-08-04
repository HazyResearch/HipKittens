/**
 * @file
 * @brief Functions for transferring data directly between global and shared memory and back.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

template< int  axis, bool assume_aligned,
          ducks::st::all ST, ducks::gl::all GL,
          ducks::coord::tile COORD = coord<ST>,
          int  N_THREADS = WARP_THREADS >
__device__ inline void load(ST& dst, const GL& src, const COORD& idx)
{
    using T = typename ST::dtype;
    const int row_stride = src.template stride<axis>();
    // we can handle this many rows each time we run a memcpy_async
    constexpr int elem_per_memcpy = sizeof(float4)/sizeof(typename ST::dtype); // if bf16, then 16/2 = 8. if fp8, then 16/1 = 16.
    constexpr int elem_per_half_memcpy = sizeof(float2)/sizeof(typename ST::dtype); // if bf16, then 8/2 = 4. if fp8, then 8/1 = 8.
    constexpr int memcpy_per_row = ST::cols / elem_per_memcpy; // if 64 columns, then 64/8 = 8 or 64/16 = 4
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


/********************************************* Register Pipelining ************************************************** */

/**
 * @brief Load from global memory to registers with proper batching for cache locality
 *
 * @tparam reg_buffer The register buffer to store data into.
 * @tparam U The data type of the destination array.
 * @param[out] reg_buffer The register buffer to store data into.
 * @param[in] buffer_size The size of the register buffer.
 * @param[in] src The source global memory array to store data from.
 * @param[in] idx The index into the source global memory array.
 * @param[in] dst_template The template of the ultimate shared tile that will be loaded into.
 */
template<int axis=2, bool assume_aligned=false,
        int N_THREADS = WARP_THREADS,
        ducks::st::all ST, 
        ducks::gl::all GL,
        ducks::coord::tile COORD = coord<ST>
>
__device__ inline void load_global_to_register_buffer(float4* reg_buffer, const int buffer_size, const GL& src, const COORD& idx, const ST& dst_template) {
    using T = typename ST::dtype;
    constexpr int elem_per_memcpy = sizeof(float4)/sizeof(T);
    constexpr int memcpy_per_row = ST::cols / elem_per_memcpy;
    constexpr int total_chunks = (ST::rows * ST::cols) / elem_per_memcpy;
    constexpr int total_calls = (total_chunks + N_THREADS - 1) / N_THREADS;
    constexpr int small_calls = 16;
    const int big_calls = (total_calls + small_calls - 1) / small_calls;

    const int row_stride = src.template stride<axis>();
    const int row_stride_bytes = row_stride * sizeof(T);
    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    T* base_ptr = (T*)&src[unit_coord];  // global memory pointer
    const int laneid = threadIdx.x % N_THREADS;

    // buffer resource
    const int total_bytes = row_stride * ST::rows * sizeof(T);
    i32x4 srsrc = make_srsrc(base_ptr, total_bytes, row_stride_bytes);

    int buf_idx = 0;
    for (int i = 0; i < big_calls && buf_idx < buffer_size; ++i) {
        const int offset = i * small_calls;
        #pragma unroll
        for (int j = 0; j < small_calls; ++j) {
            const int chunk_idx = (offset + j) * N_THREADS + laneid;
            if (chunk_idx < total_chunks && buf_idx < buffer_size) {
                int row = chunk_idx / memcpy_per_row;
                int col = (chunk_idx % memcpy_per_row) * elem_per_memcpy;
                int flat_offset = row * row_stride + col;
                int byte_offset = flat_offset * sizeof(T);
                __uint128_t raw = llvm_amdgcn_raw_buffer_load_b128(srsrc, byte_offset, 0, 0);
                reg_buffer[buf_idx] = *reinterpret_cast<float4*>(&raw);
                buf_idx++;
            }
        }
    }
}

/**
 * @brief Store from registers to shared memory (preserving the batched pattern)
 *
 * @tparam reg_buffer The register buffer to store data into.
 * @tparam ST The type of the destination shared tile.
 * @param[out] dst The destination shared tile to store data into.
 * @param[in] reg_buffer The register buffer to store data from.
 */
template<int N_THREADS = WARP_THREADS, ducks::st::all ST>
__device__ inline void store_register_buffer_to_shared(ST& dst, const float4* reg_buffer) {
    using T = typename ST::dtype;
    constexpr int elem_per_memcpy = sizeof(float4)/sizeof(T);
    constexpr int elem_per_half_memcpy = sizeof(float2)/sizeof(T);
    constexpr int memcpy_per_row = ST::cols / elem_per_memcpy;
    
    uint32_t dst_ptr = reinterpret_cast<uintptr_t>(&dst.data[0]);
    const int laneid = threadIdx.x % N_THREADS;
    
    constexpr int total_chunks = (ST::rows * ST::cols) / elem_per_memcpy;
    constexpr int total_calls = (total_chunks + N_THREADS - 1) / N_THREADS;
    constexpr int small_calls = 16;
    const int big_calls = (total_calls + small_calls - 1) / small_calls;

    int buf_idx = 0;
    // Store in the same batched pattern to maintain locality
    #pragma unroll
    for (int i = 0; i < big_calls; i++) {
        const int offset = i * small_calls;
        #pragma unroll
        for(int j = 0; j < small_calls; j++) {
            int load_idx = (offset + j) * N_THREADS + laneid;
            int row = load_idx / memcpy_per_row;
            int col = (load_idx % memcpy_per_row) * elem_per_memcpy;
            if (row < dst.rows && buf_idx < 64) { // Safety check - use fixed limit
                const float4& buf_val = reg_buffer[buf_idx];
                store_shared_vec(dst.idx(dst_ptr, {row, col}), {buf_val.x, buf_val.y});
                store_shared_vec(dst.idx(dst_ptr, {row, col + elem_per_half_memcpy}), {buf_val.z, buf_val.w});
                buf_idx++;
            }
        } // Wait for this batch of stores to complete
    }
}



/******************************************************************************************************************** */



/**
 * @brief Stores data from a shared memory tile into global memory.
 *
 * @tparam ST The type of the shared tile.
 * @param[out] dst The destination global memory array.
 * @param[in] src The source shared memory tile.
 * @param row_stride[in] The stride between rows in the destination array.
 */
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
}