#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

__device__ inline float4 load_global_vec_new(const float4* gptr) {
    float4 v;
    asm volatile(
        "global_load_dwordx4 %0, %1, off\n"
        : "=v"(v) 
        : "v"(gptr)
        : "memory"
    );
    return v;   
}

__device__ inline void store_shared_vec_new(uint32_t lds_off, float2 val) {
    asm volatile(
        "ds_write_b64 %0, %1\n"
        :
        : "v"(lds_off), "v"(val)
        : "memory"
    );
}

__device__ inline float4 buffer_load_vec4(i32x4 srsrc, uint32_t offset_bytes) {
    __uint128_t raw = llvm_amdgcn_raw_buffer_load_b128(srsrc, offset_bytes, 0, 0);
    return *reinterpret_cast<float4*>(&raw);
}


__device__ inline i32x4 make_srsrc(const void* ptr, uint32_t range_bytes) {
    buffer_resource rsrc = make_buffer_resource(ptr, range_bytes, 0x110000);  // default config
    return *reinterpret_cast<i32x4*>(&rsrc);
}

// Load from global memory to registers with proper batching for cache locality
template<int axis, bool assume_aligned,
         ducks::st::all ST, ducks::gl::all GL,
         ducks::coord::tile COORD = coord<ST>,
         int N_THREADS = WARP_THREADS>
__device__ inline void load_global_to_registers(
    float4* reg_buffer, int buffer_size,
    const GL& src, const COORD& idx, const ST& dst_template)
{
    using T = typename ST::dtype;
    constexpr int elem_per_memcpy = sizeof(float4)/sizeof(T);
    constexpr int memcpy_per_row = ST::cols / elem_per_memcpy;
    constexpr int total_chunks = (ST::rows * ST::cols) / elem_per_memcpy;
    constexpr int total_calls = (total_chunks + N_THREADS - 1) / N_THREADS;
    constexpr int small_calls = 16;
    const int big_calls = (total_calls + small_calls - 1) / small_calls;

    const int row_stride = src.template stride<axis>();
    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    T* base_ptr = (T*)&src[unit_coord];  // global memory pointer
    const int laneid = threadIdx.x % N_THREADS;

    // buffer resource
    const int total_bytes = row_stride * ST::rows * sizeof(T);
    i32x4 srsrc = make_srsrc(base_ptr, total_bytes);

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

                reg_buffer[buf_idx] = buffer_load_vec4(srsrc, byte_offset);
                buf_idx++;
            }
        }
    }
}



// Store from registers to shared memory (preserving the batched pattern)
template<ducks::st::all ST, int N_THREADS = WARP_THREADS>
__device__ inline void store_registers_to_shared(
    const float4* reg_buffer, ST& dst)
{
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
                store_shared_vec_new(dst.idx(dst_ptr, {row, col}), {buf_val.x, buf_val.y});
                store_shared_vec_new(dst.idx(dst_ptr, {row, col + elem_per_half_memcpy}), {buf_val.z, buf_val.w});
                buf_idx++;
            }
        } // Wait for this batch of stores to complete
    }
}

__device__ inline float2 load_shared_vec_async(uint32_t lds_off) {
    float2 result;
    asm volatile(
        "ds_read_b64 %0, %1\n"
        // "s_waitcnt lgkmcnt(0)\n"
        : "=v"(result)              // Output: store result in float2
        : "v"(lds_off)              // Input: LDS offset to read from
        : "memory"
    );
    return result;
}
/**
 * @brief Load data from a shared tile into a register tile.
 *
 * @tparam RT The register tile type
 * @tparam ST The shared tile type
 * @param dst[out] The destination register tile.
 * @param src[in]  The source shared tile.
 */
template<ducks::rt::all RT, ducks::st::all ST>
__device__ inline static void load_async_shared_to_register(RT &dst, const ST &src) {

    static_assert(RT::height == ST::height, "register tile and shared tile must match height");
    static_assert(RT::width  == ST::width,  "register tile and shared tile must match width");

    using T2 = RT::dtype;
    using T  = base_types::packing<T2>::unpacked_type;
    using U  = ST::dtype;
    using U2 = base_types::packing<U >::packed_type;

    const int laneid = kittens::laneid();
    const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&src.data[0]);

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
    for(int i = 0; i < dst.height; i++) {
        const int row = i*dst.tile_size_row + row_offset;
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            const int col = j*dst.tile_size_col + col_offset;
            if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) { // handle the row-major layout

                if constexpr (sizeof(typename ST::dtype) == 4) {
                    // handle float32
                    float2 loaded0 = load_shared_vec_async(src.idx(src_ptr, {row, col}));
                    float2 loaded1 = load_shared_vec_async(src.idx(src_ptr, {row, col+2}));
                    dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(loaded0);
                    dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(loaded1);
                } else {
                    // handle fp16 and bf16
                    float2 loaded = load_shared_vec_async(src.idx(src_ptr, {row, col}));
                    U2* tmp = reinterpret_cast<U2*>(&loaded);
                    dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(tmp[0]);
                    dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(tmp[1]);
                }
            }
            else { // handle the column-major layout
                dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(U2{src[{row, col}], src[{row+1, col}]});
                dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(U2{src[{row+2, col}], src[{row+3, col}]});
            }
        }
    }
}




__device__ inline void store_global_float(float* ptr, float val) {
    asm volatile(
        "global_store_dword %0, %1, off\n"
        :
        : "v"(ptr), "v"(val)
        : "memory"
    );
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
__device__ inline static void store_optimized(const GL &dst, const RT &src, const COORD &idx) {
    using T = base_types::packing<typename RT::dtype>::unpacked_type;
    using U = typename GL::dtype;

    U *dst_ptr = (U*)&dst[(idx.template unit_coord<axis, 3>())];
    const int row_stride = dst.template stride<axis>();
    const int laneid = kittens::laneid();
    const int row_offset = 4*(laneid/16), col_offset = laneid%16;

    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        const int row = i*src.tile_size_row + row_offset;
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            const int col = j*src.tile_size_col + col_offset;
            
            // Use optimized stores for floats, keeping exact same logic
            if constexpr (std::is_same_v<U, float>) {
                store_global_float((float*)&dst_ptr[(row+0)*row_stride + col], 
                                 base_types::convertor<U, T>::convert(src.tiles[i][j].data[0].x));
                store_global_float((float*)&dst_ptr[(row+1)*row_stride + col], 
                                 base_types::convertor<U, T>::convert(src.tiles[i][j].data[0].y));
                store_global_float((float*)&dst_ptr[(row+2)*row_stride + col], 
                                 base_types::convertor<U, T>::convert(src.tiles[i][j].data[1].x));
                store_global_float((float*)&dst_ptr[(row+3)*row_stride + col], 
                                 base_types::convertor<U, T>::convert(src.tiles[i][j].data[1].y));
            } else {
                // Original code for non-float types
                dst_ptr[(row+0)*row_stride + col] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[0].x);
                dst_ptr[(row+1)*row_stride + col] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[0].y);
                dst_ptr[(row+2)*row_stride + col] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[1].x);
                dst_ptr[(row+3)*row_stride + col] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[1].y);
            }
        }
    }
}

template<ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void store_optimized(const GL &dst, const RT &src, const COORD &idx) {
    store<2, RT, GL, COORD>(dst, src, idx);
}

