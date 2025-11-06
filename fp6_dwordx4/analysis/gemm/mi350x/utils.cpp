#include "kittens.cuh"
using namespace kittens;



/*

Assembly and intrinsic functions.

*/


__device__ inline void store_shared_vec_new(uint32_t lds_off, float2 val) {
    asm volatile(
        "ds_write_b64 %0, %1\n"
        :
        : "v"(lds_off), "v"(val)
        : "memory"
    );
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


using as3_uint32_ptr = uint32_t __attribute__((address_space(3)))*;
using index_t = int;
using int32x4_t = int32_t __attribute__((ext_vector_type(4)));



enum class coherency {
	cache_all = 0,
	cache_global = 1,
	cache_stream = 2,
	non_temporal = 3
};

__device__ inline float4 buffer_load_vec4(i32x4 srsrc, uint32_t offset_bytes) {
    const int cc = static_cast<int>(coherency::cache_all);
    __uint128_t raw = llvm_amdgcn_raw_buffer_load_b128(srsrc, offset_bytes, 0, cc);
    return *reinterpret_cast<float4*>(&raw);
}



/*

Load store functions.

*/



// Load from global memory to registers with proper batching for cache locality
template<int axis, bool assume_aligned,
         ducks::st::all ST, ducks::gl::all GL,
         ducks::coord::tile COORD = coord<ST>,
         int N_THREADS = WARP_THREADS>
__device__ inline void load_global_to_registers(
    float4* reg_buffer, int buffer_size,
    const GL& src, const COORD& idx, const ST& dst_template, int offset, int split)
{
    using T = typename ST::dtype;
    constexpr int elem_per_memcpy = sizeof(float4)/sizeof(T);
    constexpr int memcpy_per_row = ST::cols / elem_per_memcpy;
    constexpr int total_chunks = (ST::rows * ST::cols) / elem_per_memcpy;
    constexpr int total_calls = (total_chunks + N_THREADS - 1) / N_THREADS;
    constexpr int small_calls = 16;
    const int big_calls = (total_calls + small_calls - 1) / small_calls;
    const int big_calls_start = (big_calls / split) * offset;
    const int big_calls_end = big_calls_start + (big_calls / split);

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
    for(int j = 0; j < dst.width; j++) {
        const int col = j*dst.tile_size_col + col_offset;
        uint32_t addr = src.idx(src_ptr, {row_offset, col});
        
        #pragma unroll
        for(int i = 0; i < dst.height; i++) {
            const int row = i*dst.tile_size_row + row_offset;

            if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) { // handle the row-major layout
                if constexpr (sizeof(typename ST::dtype) == 4) {
                    // handle float32
                    float2 loaded0 = load_shared_vec_async(src.idx(src_ptr, {row, col}));
                    float2 loaded1 = load_shared_vec_async(src.idx(src_ptr, {row, col+2}));
                    dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(loaded0);
                    dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(loaded1);


                } else {
                    // handle fp16 and bf16
                    // float2 loaded = load_shared_vec_async_offset(addr, i * ST::underlying_cols * kittens::TILE_ROW_DIM<U> * sizeof(U));
                    // U2* tmp = reinterpret_cast<U2*>(&loaded);
                    // dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(tmp[0]);
                    // dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(tmp[1]);
                    
                    // avoid v_bfi_b32
                    asm volatile(
                        "ds_read_b64 %0, %1 offset:%2\n"
                        : "=v"(*reinterpret_cast<uint64_t*>(&dst.tiles[i][j].data[0]))
                        : "v"(addr), "i"(i * ST::underlying_cols * kittens::TILE_ROW_DIM<U> * sizeof(U))
                        : "memory"
                    );
                }
            }
            else { // handle the column-major layout
                dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(U2{src[{row, col}], src[{row+1, col}]});
                dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(U2{src[{row+2, col}], src[{row+3, col}]});
            }
        }
    }
}


extern "C" __device__ void 
llvm_amdgcn_raw_buffer_load_lds(int32x4_t rsrc, // does not change (buffer resource; scalar array?)
                                as3_uint32_ptr lds_ptr, // does not change
                                index_t size, // does not change (16 bytes)
                                index_t voffset, 
                                index_t soffset, 
                                index_t offset,  // does not change (0); instruction offset
                                index_t aux) __asm("llvm.amdgcn.raw.buffer.load.lds"); // cache coherency

// Direct global-to-shared load using buffer load to LDS
template<int axis, bool assume_aligned,
         ducks::st::all ST, ducks::gl::all GL,
         ducks::coord::tile COORD = coord<ST>,
         int N_THREADS = WARP_THREADS, int NUM_WARPS = 1>
__device__ inline void load_global_to_shared_direct(
    const GL& src, const COORD& idx, ST& dst)
{
    using T = typename ST::dtype;
    constexpr int memcpy_per_tile = ST::rows * ST::cols * sizeof(T) / (16 * N_THREADS); // 2
    
    constexpr int elem_per_thread = 16 / sizeof(T);  // 8
    constexpr int elem_per_warp = elem_per_thread * kittens::WARP_THREADS;
    constexpr int threads_per_row = ST::cols / elem_per_thread; 

    const int row_stride = src.template stride<axis>();
    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    T* global_ptr = (T*)&src[unit_coord];
    i32x4 srsrc = make_srsrc(global_ptr, row_stride * ST::rows * sizeof(T));

    uint32_t dst_ptr = reinterpret_cast<uintptr_t>(&dst.data[0]);
    int thread_id = threadIdx.x % N_THREADS;
    int warp_id = thread_id >> 6;
    
    const T* lds_base = &dst.data[0] + (kittens::warpid() * elem_per_warp);

    #pragma unroll
    for (int i = 0; i < memcpy_per_tile; i++) {

        int offset_threads = (i * N_THREADS + threadIdx.x % N_THREADS);
        const int row_in_lds = offset_threads / threads_per_row;
        const int col_in_lds = (offset_threads % threads_per_row) * elem_per_thread;
        const int addr_in_lds = dst.idx(dst_ptr, {row_in_lds, col_in_lds});

        const int offset_in_lds = addr_in_lds - dst_ptr;
        const int swizzled_row_in_lds = (offset_in_lds / sizeof(T)) / ST::cols;
        const int swizzled_col_in_lds = (offset_in_lds / sizeof(T)) % ST::cols;

        const int offset_in_global = (swizzled_row_in_lds * row_stride + swizzled_col_in_lds) * sizeof(T);

        const T* lds_elem_ptr = lds_base + (i * N_THREADS * elem_per_thread);
        uintptr_t lds_addr = reinterpret_cast<uintptr_t>(lds_elem_ptr);
        as3_uint32_ptr lds_ptr = (as3_uint32_ptr)(lds_addr);

        llvm_amdgcn_raw_buffer_load_lds(
            srsrc, // buffer resource
            lds_ptr,
            16, // 16 bytes per thread
            offset_in_global,
            0, 
            0, // instruction offset
            static_cast<index_t>(coherency::cache_all)); // cache coherency
    }
}

// Direct global-to-shared load using buffer load to LDS
template<int axis, bool assume_aligned,
         ducks::st::all ST, ducks::gl::all GL,
         ducks::coord::tile COORD = coord<ST>,
         int N_THREADS = WARP_THREADS, int NUM_WARPS = 1>
__device__ inline void prefill_swizzled_offsets(
    const GL& src, const COORD& idx, ST& dst, uint32_t* swizzled_offsets)
{
    using T = typename ST::dtype;
    constexpr int bytes_per_thread = 16;
    constexpr int bytes_per_memcpy = 16 * N_THREADS;
    constexpr int memcpy_per_tile = ST::rows * ST::cols * sizeof(T) / bytes_per_memcpy;
    
    constexpr int elem_per_thread = bytes_per_thread / sizeof(T);  // e.g., 8 for bf16, 4 for fp32
    constexpr int threads_per_row = ST::cols / elem_per_thread; 

    const int row_stride = src.template stride<axis>();

    uint32_t dst_ptr = reinterpret_cast<uintptr_t>(&dst.data[0]);
    int thread_id = threadIdx.x % N_THREADS;
    int warp_id = thread_id >> 6;

    #pragma unroll
    for (int i = 0; i < memcpy_per_tile; i++) {
        const int row_in_lds = (i * N_THREADS + thread_id) / threads_per_row;
        const int col_in_lds = ((i * N_THREADS + thread_id) % threads_per_row) * elem_per_thread;
        const int addr_in_lds = dst.idx(dst_ptr, {row_in_lds, col_in_lds});

        const int offset_in_lds = addr_in_lds - dst_ptr;
        const int swizzled_row_in_lds = (offset_in_lds / sizeof(T)) / ST::cols;
        const int swizzled_col_in_lds = (offset_in_lds / sizeof(T)) % ST::cols;

        const int offset_in_global = (swizzled_row_in_lds * row_stride + swizzled_col_in_lds) * sizeof(T);

        swizzled_offsets[i] = offset_in_global;
    }
}

template<int axis, bool assume_aligned,
         ducks::st::all ST, ducks::gl::all GL,
         ducks::coord::tile COORD = coord<ST>,
         int N_THREADS = WARP_THREADS, int NUM_WARPS = 1>
__device__ inline void load_global_to_shared_direct_with_swizzled_offsets(
    const GL& src, const COORD& idx, ST& dst, uint32_t* swizzled_offsets)
{
    using T = typename ST::dtype;
    constexpr int bytes_per_thread = 16;
    constexpr int bytes_per_memcpy = bytes_per_thread * N_THREADS;
    constexpr int memcpy_per_tile = ST::rows * ST::cols * sizeof(T) / bytes_per_memcpy;
    
    constexpr int elem_per_thread = bytes_per_thread / sizeof(T);  // e.g., 8 for bf16, 4 for fp32
    constexpr int elem_per_warp = elem_per_thread * kittens::WARP_THREADS;

    const int row_stride = src.template stride<axis>();
    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    T* global_ptr = (T*)&src[unit_coord];
    i32x4 srsrc = make_srsrc(global_ptr, row_stride * ST::rows * sizeof(T));

    const int thread_id = threadIdx.x % N_THREADS;
    const int warp_id = thread_id >> 6;

    #pragma unroll
    for (int i = 0; i < memcpy_per_tile; i++) {
        const T* lds_base = &dst.data[0];
        const T* lds_elem_ptr = lds_base + (i * N_THREADS * elem_per_thread) + (warp_id * elem_per_warp);
        uintptr_t lds_addr = reinterpret_cast<uintptr_t>(lds_elem_ptr);
        as3_uint32_ptr lds_ptr = (as3_uint32_ptr)(lds_addr);

        llvm_amdgcn_raw_buffer_load_lds(
            srsrc, // buffer resource
            lds_ptr,
            bytes_per_thread, // 16 bytes
            swizzled_offsets[i],
            0, 
            0, // instruction offset
            static_cast<index_t>(coherency::cache_all)); // cache coherency
    }
}


__device__ inline float4 load_shared_vec_b128(uint32_t lds_off, uint32_t offset = 0) {
    float4 result;
    asm volatile(
        "ds_read_b128 %0, %1 offset:%2\n"
        // "s_waitcnt lgkmcnt(0)\n"
        : "=v"(result)              // Output: store result in float4
        : "v"(lds_off), "i"(offset)              // Input: LDS offset to read from
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
__device__ inline static void load_lds_reg(RT &dst, const ST &src) {

    static_assert(RT::height == ST::height, "register tile and shared tile must match height");
    static_assert(RT::width  == ST::width,  "register tile and shared tile must match width");

    using T2 = RT::dtype;
    using T  = base_types::packing<T2>::unpacked_type;
    using U  = ST::dtype;
    using U2 = base_types::packing<U >::packed_type;
    static_assert(sizeof(U) == 2, "only supporting 16-bit dtypes");

    const int laneid = kittens::laneid();
    const uint32_t src_ptr = reinterpret_cast<uintptr_t>(&src.data[0]);

    int row_offset, col_offset;
    if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
        row_offset = laneid%32;
        col_offset = 8*(laneid/32);
    }
    else {
        row_offset = 8*(laneid/32);
        col_offset = laneid%32;
    }

    #pragma unroll 
    for (int k = 0; k < 2; k++) {

        #pragma unroll
        for(int j = 0; j < dst.width; j++) {

            int col = 0;
            if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
                col = j*dst.tile_size_col + col_offset + k*16;
            } else {
                col = j*dst.tile_size_col + col_offset;
            }

            uint32_t addr = src.idx(src_ptr, {row_offset, col});

            #pragma unroll
            for(int i = 0; i < dst.height; i++) {

                int row = 0;
                if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) { 
                    row = i*dst.tile_size_row + row_offset;
                } else {
                    row = i*dst.tile_size_row + row_offset + k*16;
                }

                if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) { // handle the row-major layout

                    asm volatile(
                        "ds_read_b128 %0, %1 offset:%2\n"
                        : "=v"(*reinterpret_cast<float4*>(&dst.tiles[i][j].data[k*4]))
                        : "v"(addr), "i"(i * ST::underlying_cols * kittens::TILE_ROW_DIM<U> * sizeof(U))
                        : "memory"
                    );
                }
                else { // handle the column-major layout
                    dst.tiles[i][j].data[0+k*4] = base_types::convertor<T2, U2>::convert(U2{src[{row, col}], src[{row+1, col}]});
                    dst.tiles[i][j].data[1+k*4] = base_types::convertor<T2, U2>::convert(U2{src[{row+2, col}], src[{row+3, col}]});
                    dst.tiles[i][j].data[2+k*4] = base_types::convertor<T2, U2>::convert(U2{src[{row+4, col}], src[{row+5, col}]});
                    dst.tiles[i][j].data[3+k*4] = base_types::convertor<T2, U2>::convert(U2{src[{row+6, col}], src[{row+7, col}]});
                }
            }
        }
    }
}

template<int axis, bool assume_aligned, ducks::st::all ST, ducks::gl::all GL,
         ducks::coord::tile COORD = coord<ST>, int N_THREADS = WARP_THREADS>
__device__ inline void store_linear(const GL &dst, const ST &src, const COORD &idx) {
    
    // determine the number of contiguous vectors we need to load overall
    using T = typename ST::dtype;
    constexpr int vec_len = 16 / sizeof(T);
    constexpr int vecs_per_row = ST::cols / vec_len;
    constexpr int total_vecs = ST::rows * vecs_per_row;

    // take the input idx and extract information from the global layout to get the correct pointer. 
    const int row_stride = dst.template stride<axis>();
    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    T* dst_ptr = (T*)&dst[unit_coord];
    int thread_id = threadIdx.x % N_THREADS;

    for (int i = thread_id; i < total_vecs; i += N_THREADS) {
        const int row = i / vecs_per_row;
        const int col = (i % vecs_per_row) * vec_len;
        const int flat_idx = row * ST::cols + col;
        const T* lds_ptr = &src.data[flat_idx];
        const int gmem_offset = row * row_stride + col;

        float4 val;
        memcpy(&val, lds_ptr, 16);
        memcpy(&dst_ptr[gmem_offset], &val, 16);
    }
}

