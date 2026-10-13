/**
 * @file
 * @brief Functions for transferring data directly between global and shared memory and back.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

/**
 * @brief How many 16-byte chunks each lane moves for one shared tile.
 *
 * This is the size a caller's register staging buffer must have for the
 * load_global_to_register_buffer / store_register_buffer_to_shared pair below.
 * RDNA has no global->LDS DMA, so that pair is the only asynchronous path there
 * is, and getting this number right is the caller's side of the contract.
 */
template<ducks::st::all ST, int N_THREADS = WARP_THREADS>
static constexpr int stage_calls =
    ((ST::rows * ST::cols) / (sizeof(float4)/sizeof(typename ST::dtype)) + N_THREADS - 1) / N_THREADS;

/// Loads issued (and registers held) per batch. See the note in load() below.
static constexpr int stage_batch = 8;

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
    constexpr int memcpy_per_row = ST::cols / elem_per_memcpy; // if 64 columns, then 64/8 = 8 or 64/16 = 4
    constexpr int total_calls = (ST::cols * ST::rows + N_THREADS*elem_per_memcpy-1) / (N_THREADS*elem_per_memcpy); // round up

    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    typename GL::dtype *src_ptr = (typename GL::dtype*)&src[unit_coord];
    // Rows past the tensor's end are written to LDS as zeros, not skipped: the
    // tile's previous contents would otherwise be read as data.
    const int valid_rows = rows_in_bounds<axis, ST::rows>(src, unit_coord);

    uint32_t dst_ptr = reinterpret_cast<uintptr_t>(&dst.data[0]);
    const int laneid = threadIdx.x % N_THREADS;

    // Loads are issued in batches of `small_calls` so that only that many are in
    // flight -- and staged in registers -- at once. Each entry is a float4, i.e.
    // 4 VGPRs, so a batch of 8 costs 32 VGPRs out of the 256 a gfx11 wave gets.
    // CDNA uses 16 here, but a wave32 lane covers twice as many chunks as a
    // wave64 one for the same tile, so the same number would double the live
    // range. Eight is still enough outstanding loads to cover memory latency.
    constexpr int small_calls = stage_batch;
    const int big_calls = (total_calls + small_calls - 1) / small_calls;
    float4    buf[small_calls];

    for (int i = 0; i < big_calls; i++) {
        const int offset = i * small_calls;
        #pragma unroll
        for(int j = 0; j < small_calls; j++) {
            int load_idx = (offset + j) * N_THREADS + laneid;
            int row = load_idx / memcpy_per_row;
            int col = (load_idx % memcpy_per_row) * elem_per_memcpy;

            if (row < valid_rows) {
                buf[j] = load_global_vec4_async((float4*) (src_ptr + (row * row_stride + col))); // thread loads 128-bits, 16-bytes
            }
            else buf[j] = float4{0.f, 0.f, 0.f, 0.f};
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
                // One ds_write_b128: `col` is a whole number of float4s, so the
                // 16 bytes land inside a single swizzle granule and stay
                // contiguous. See the swizzle note in types/shared/st.cuh.
                store_shared_vec4(dst.idx(dst_ptr, {row, col}), buf[j]);
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
    constexpr int total_calls = stage_calls<ST, N_THREADS>;

    const int row_stride = src.template stride<axis>();
    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    // The buffer starts at the tile's first row, column 0; the tile's column
    // offset is the instruction's soffset. In a K loop the descriptor is then
    // loop-invariant and the one moving part is a scalar, where building it
    // at the tile's own column rebuilt base and descriptor every K-tile.
    coord<> row0 = unit_coord;
    row0.c = 0;
    T* row_ptr = (T*)&src[row0];
    const uint32_t col0 = uint32_t(unit_coord.c) * sizeof(T);
    const int laneid = threadIdx.x % N_THREADS;

    // buffer resource: a plain raw buffer spanning the tile's rows that are
    // inside the tensor, so reads past the tensor's last row clamp to zero
    // instead of walking off the allocation -- with exactly the loads a whole
    // tile issues, so the caller's vmcnt is unchanged. The config word is the
    // gfx11 one -- see make_srsrc; the gfx9 word the CDNA tree uses silently
    // reads back zeros here.
    const int valid_rows = rows_in_bounds<axis, ST::rows>(src, unit_coord);
    const int total_bytes = row_stride * valid_rows * sizeof(T);
    i32x4 srsrc = make_srsrc(row_ptr, total_bytes);

    // Buffer slot c holds chunk c*N_THREADS + laneid. store_register_buffer_to_shared
    // below walks the identical indexing, which is what keeps the two halves of
    // this pair in agreement; do not make one of them skip slots.
    #pragma unroll
    for (int c = 0; c < total_calls; ++c) {
        const int chunk_idx = c * N_THREADS + laneid;
        if (c < buffer_size && chunk_idx < total_chunks) {
            int row = chunk_idx / memcpy_per_row;
            int col = (chunk_idx % memcpy_per_row) * elem_per_memcpy;
            int byte_offset = (row * row_stride + col) * sizeof(T);
            __uint128_t raw = llvm_amdgcn_raw_buffer_load_b128(srsrc, byte_offset, col0, 0);
            reg_buffer[c] = *reinterpret_cast<float4*>(&raw);
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
 * @param[in] buffer_size The size of the register buffer, as passed to the load.
 */
template<int N_THREADS = WARP_THREADS, bool wait = true, ducks::st::all ST>
__device__ inline void store_register_buffer_to_shared(ST& dst, const float4* reg_buffer,
                                                       const int buffer_size = stage_calls<ST, N_THREADS>) {
    using T = typename ST::dtype;
    constexpr int elem_per_memcpy = sizeof(float4)/sizeof(T);
    constexpr int memcpy_per_row = ST::cols / elem_per_memcpy;

    uint32_t dst_ptr = reinterpret_cast<uintptr_t>(&dst.data[0]);
    const int laneid = threadIdx.x % N_THREADS;

    constexpr int total_chunks = (ST::rows * ST::cols) / elem_per_memcpy;
    constexpr int total_calls = stage_calls<ST, N_THREADS>;

    // Same slot -> chunk mapping as load_global_to_register_buffer. The CDNA
    // version guarded this with a hardcoded `buf_idx < 64` that had no relation
    // to the buffer the caller actually passed; the size is a parameter here.
    #pragma unroll
    for (int c = 0; c < total_calls; ++c) {
        const int chunk_idx = c * N_THREADS + laneid;
        if (c < buffer_size && chunk_idx < total_chunks) {
            int row = chunk_idx / memcpy_per_row;
            int col = (chunk_idx % memcpy_per_row) * elem_per_memcpy;
            store_shared_vec4(dst.idx(dst_ptr, {row, col}), reg_buffer[c]);
        }
    }
    // Draining here costs a full LDS write latency on every call, and a caller
    // that is about to issue its own ds_reads does not need it: lgkmcnt retires
    // in order, so those reads' waits already cover these writes. What does
    // need it is the barrier that publishes the tile -- s_barrier does not
    // order memory -- and the caller is the one that knows where that is.
    if constexpr (wait) {
        #ifdef BUILTINS_ONLY
        __builtin_amdgcn_s_waitcnt(0);
        #else
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        #endif
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
    const int valid_rows = rows_in_bounds<axis, ST::rows>(dst, unit_coord);

    uint32_t src_ptr = reinterpret_cast<uintptr_t>(&src.data[0]);
    int laneid = threadIdx.x % N_THREADS;

    #pragma unroll
    for(int i = 0; i < total_calls; i++) {

        int load_idx = i * N_THREADS + laneid;
        int row = load_idx / memcpy_per_row;
        int col = (load_idx*elem_per_memcpy) % src.cols;

        if (row < valid_rows) {
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