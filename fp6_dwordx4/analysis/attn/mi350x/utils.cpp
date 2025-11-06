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
    constexpr int memcpy_per_tile =  ST::rows * ST::cols * sizeof(T) / (16 * N_THREADS); // 16 --> 32
    static_assert(memcpy_per_tile > 0, "memcpy_per_tile must be greater than 0. Please decrease the number of threads.");
    
    constexpr int elem_per_thread = 16 / sizeof(T);  // 8
    constexpr int elem_per_warp = elem_per_thread * kittens::WARP_THREADS; // 512
    constexpr int threads_per_row = ST::cols / elem_per_thread;

    const int row_stride = src.template stride<axis>();
    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    T* global_ptr = (T*)&src[unit_coord];
    i32x4 srsrc = make_srsrc(global_ptr, row_stride * ST::rows * sizeof(T));

    uint32_t dst_ptr = reinterpret_cast<uintptr_t>(&dst.data[0]);
    
    int thread_id = threadIdx.x % N_THREADS;
    int warp_id = thread_id >> 6;
    
    const T* lds_base = &dst.data[0] + (warp_id * elem_per_warp);

    #pragma unroll
    for (int i = 0; i < memcpy_per_tile; i++) {

        int offset_threads = (i * N_THREADS + thread_id);
        const int row_in_lds = offset_threads / threads_per_row;
        const int col_in_lds = (offset_threads % threads_per_row) * elem_per_thread;
        const int addr_in_lds = dst.idx(dst_ptr, {row_in_lds, col_in_lds});

        const int offset_in_lds = addr_in_lds - dst_ptr;
        const int swizzled_col_in_lds = (offset_in_lds / sizeof(T)) % ST::cols;

        const int offset_in_global = (row_in_lds * row_stride + swizzled_col_in_lds) * sizeof(T);

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

// // Direct global-to-shared load using buffer load to LDS
// template<int axis, bool assume_aligned,
//          ducks::rt::all RT,ducks::st::all ST, ducks::gl::all GL,
//          ducks::coord::tile COORD = coord<ST>,
//          int N_THREADS = WARP_THREADS, int NUM_WARPS = 1>
// __device__ inline void load_global_to_shared_direct(
//     const GL& src, const COORD& idx, ST& dst)
// {

//     using T = typename ST::dtype;
//     constexpr int memcpy_per_tile =  ST::rows * ST::cols * sizeof(T) / (16 * N_THREADS); // 16 --> 32
//     static_assert(memcpy_per_tile > 0, "memcpy_per_tile must be greater than 0. Please decrease the number of threads.");
    
//     constexpr int elem_per_thread = 16 / sizeof(T);  // 8
//     constexpr int elem_per_warp = elem_per_thread * kittens::WARP_THREADS; // 512
//     const int thread_id = laneid();
//     const int warp_id = warpid();
//     const T* lds_base = &dst.data[0] + (warp_id * elem_per_warp);
//     uint32_t dst_ptr = reinterpret_cast<uintptr_t>(&dst.data[0]);

//     const int row_stride = src.template stride<axis>();
//     coord<> unit_coord = idx.template unit_coord<axis, 3>();
//     T* global_ptr = (T*)&src[unit_coord];
//     i32x4 srsrc = make_srsrc(global_ptr, row_stride * ST::rows * sizeof(T));

//     constexpr int num_warps = N_THREADS / 64;
//     constexpr int num_register_subtiles = kittens::TILE_ROW_DIM<T> * kittens::TILE_COL_DIM<T> / elem_per_warp;
//     constexpr int num_register_tiles_per_row = ST::cols / kittens::TILE_ROW_DIM<T>;

//     #pragma unroll
//     for (int i = 0; i < memcpy_per_tile; i++) {

//         const int register_tile_id = (warp_id + i * num_warps) / num_register_subtiles;
//         const int register_subtile_id = (warp_id + i * num_warps) % num_register_subtiles;

//         int row_offset, col_offset;

//         if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
//             const int register_subtile_cols = kittens::TILE_COL_DIM<T> / num_register_subtiles;
//             const int num_register_subtiles_per_row = num_register_tiles_per_row * num_register_subtiles;
//             const int warp_col_offset = ((register_tile_id % num_register_tiles_per_row) * num_register_subtiles + register_subtile_id) * register_subtile_cols;
//             const int warp_row_offset = (register_tile_id / num_register_tiles_per_row) * kittens::TILE_ROW_DIM<T>;

//             col_offset = warp_col_offset + (laneid() / 32) * elem_per_thread;
//             row_offset = warp_row_offset + (laneid() % 32);
//         } else if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::col>) {
//             const int register_subtile_rows = kittens::TILE_ROW_DIM<T> / num_register_subtiles;
//             const int num_register_subtiles_per_row = num_register_tiles_per_row;
//             const int warp_col_offset = (register_tile_id % num_register_subtiles_per_row) * kittens::TILE_COL_DIM<T>;
//             const int warp_row_offset = ((register_tile_id / num_register_subtiles_per_row) * num_register_subtiles + register_subtile_id) * register_subtile_rows;
    
//             col_offset = warp_col_offset + (laneid() % 4) * elem_per_thread;
//             row_offset = warp_row_offset + (laneid() / 4);
//         } else {
//             static_assert(false, "Unsupported layout");
//         }

//         const int offset_in_global = (row_offset * row_stride + col_offset) * sizeof(T);

//         const T* lds_elem_ptr = lds_base + (i * N_THREADS * elem_per_thread);
//         uintptr_t lds_addr = reinterpret_cast<uintptr_t>(lds_elem_ptr);
//         as3_uint32_ptr lds_ptr = (as3_uint32_ptr)(lds_addr);

//         llvm_amdgcn_raw_buffer_load_lds(
//             srsrc, // buffer resource
//             lds_ptr,
//             16, // 16 bytes per thread
//             offset_in_global,
//             0, 
//             0, // instruction offset
//             static_cast<index_t>(coherency::cache_all)); // cache coherency
//     }
// }

__device__ inline int get_accum_thread_row_offset(int row) {
    // x in [0..15]
    return ((row & 4) << 1)   // bit-2 → bit-3
            | ((row & 8) >> 1)   // bit-3 → bit-2
            | ( row & 3 );       // bits-0..1 unchanged
}

// Direct global-to-shared load using buffer load to LDS
template<int axis, bool assume_aligned,
         ducks::rt::all RT, ducks::st::all ST, ducks::gl::all GL,
         ducks::coord::tile COORD = coord<ST>,
         int N_THREADS = WARP_THREADS>
__device__ inline void prefill_swizzled_offsets(
    const GL& src, const COORD& idx, ST& dst, uint32_t* swizzled_offsets)
{

    using T = typename ST::dtype;
    constexpr int memcpy_per_tile =  ST::rows * ST::cols * sizeof(T) / (16 * N_THREADS); // 16 --> 32
    static_assert(memcpy_per_tile > 0, "memcpy_per_tile must be greater than 0. Please decrease the number of threads.");
    
    constexpr int elem_per_thread = 16 / sizeof(T);  // 8
    constexpr int elem_per_warp = elem_per_thread * kittens::WARP_THREADS; // 512
    const int warp_id = warpid();
    const int row_stride = src.template stride<axis>();

    constexpr int num_warps = N_THREADS / 64;
    constexpr int num_register_subtiles = kittens::TILE_ROW_DIM<T> * kittens::TILE_COL_DIM<T> / elem_per_warp;
    constexpr int num_register_tiles_per_row = ST::cols / kittens::TILE_ROW_DIM<T>;

    #pragma unroll
    for (int i = 0; i < memcpy_per_tile; i++) {

        const int register_tile_id = (warp_id + i * num_warps) / num_register_subtiles;
        const int register_subtile_id = (warp_id + i * num_warps) % num_register_subtiles;

        int row_offset, col_offset;

        if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
            const int register_subtile_cols = kittens::TILE_COL_DIM<T> / num_register_subtiles;
            const int num_register_subtiles_per_row = num_register_tiles_per_row * num_register_subtiles;
            const int warp_col_offset = ((register_tile_id % num_register_tiles_per_row) * num_register_subtiles + register_subtile_id) * register_subtile_cols;
            const int warp_row_offset = (register_tile_id / num_register_tiles_per_row) * kittens::TILE_ROW_DIM<T>;

            col_offset = warp_col_offset + (laneid() / 32) * elem_per_thread;
            row_offset = warp_row_offset + (laneid() % 32);
        } else if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::col>) {
            const int register_subtile_rows = kittens::TILE_ROW_DIM<T> / num_register_subtiles;
            const int num_register_subtiles_per_row = num_register_tiles_per_row;
            const int warp_col_offset = (register_tile_id % num_register_subtiles_per_row) * kittens::TILE_COL_DIM<T>;
            const int warp_row_offset = ((register_tile_id / num_register_subtiles_per_row) * num_register_subtiles + register_subtile_id) * register_subtile_rows;
    
            col_offset = warp_col_offset + (laneid() % 4) * elem_per_thread;
            row_offset = warp_row_offset + (laneid() / 4);
        } else {
            const int register_subtile_rows = kittens::TILE_ROW_DIM<T> / num_register_subtiles;
            const int num_register_subtiles_per_row = num_register_tiles_per_row;
            const int warp_col_offset = (register_tile_id % num_register_subtiles_per_row) * kittens::TILE_COL_DIM<T>;
            const int warp_row_offset = ((register_tile_id / num_register_subtiles_per_row) * num_register_subtiles + register_subtile_id) * register_subtile_rows;
    
            col_offset = warp_col_offset + (laneid() % 4) * elem_per_thread;
            row_offset = warp_row_offset + get_accum_thread_row_offset(laneid() / 4);
        }

        const int offset_in_global = (row_offset * row_stride + col_offset) * sizeof(T);

        swizzled_offsets[i] = offset_in_global;
    }
}

template<int axis, bool assume_aligned,
         ducks::st::all ST, ducks::gl::all GL,
         ducks::coord::tile COORD = coord<ST>,
         int N_THREADS = WARP_THREADS>
__device__ inline void load_global_to_shared_direct_with_swizzled_offsets(
    const GL& src, const COORD& idx, ST& dst, uint32_t* swizzled_offsets)
{

    using T = typename ST::dtype;
    constexpr int bytes_per_memcpy = 16 * N_THREADS;
    constexpr int memcpy_per_tile = ST::rows * ST::cols * sizeof(T) / bytes_per_memcpy;
    static_assert(memcpy_per_tile > 0, "memcpy_per_tile must be greater than 0. Please decrease the number of threads.");
    
    constexpr int elem_per_thread = 16 / sizeof(T);  // e.g., 8 for bf16, 4 for fp32
    constexpr int elem_per_warp = elem_per_thread * kittens::WARP_THREADS;

    const int row_stride = src.template stride<axis>();
    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    T* global_ptr = (T*)&src[unit_coord];
    i32x4 srsrc = make_srsrc(global_ptr, row_stride * ST::rows * sizeof(T));

    const int warp_id = warpid();
    const T* lds_base = &dst.data[0] + (warp_id * elem_per_warp);

    #pragma unroll
    for (int i = 0; i < memcpy_per_tile; i++) {
        const T* lds_elem_ptr = lds_base + (i * N_THREADS * elem_per_thread);
        uintptr_t lds_addr = reinterpret_cast<uintptr_t>(lds_elem_ptr);
        as3_uint32_ptr lds_ptr = (as3_uint32_ptr)(lds_addr);

        llvm_amdgcn_raw_buffer_load_lds(
            srsrc, // buffer resource
            lds_ptr,
            16, // 16 bytes
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

        if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) {
            col_offset = col_offset + k*16;
        }
        else {
            row_offset = row_offset + k*16;
        }

        // int condition = (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0);

        // if (condition) {
        //     printf("dst.tile_size_row: %d\n", dst.tile_size_row);
        //     printf("dst.tile_size_col: %d\n", dst.tile_size_col);
        //     printf("row_offset: %d\n", row_offset);
        //     printf("col_offset: %d\n", col_offset);
        // }

        #pragma unroll
        for(int j = 0; j < dst.width; j++) {

            int col = j*dst.tile_size_col + col_offset;

            uint32_t addr = src.idx(src_ptr, {row_offset, col});

            #pragma unroll
            for(int i = 0; i < dst.height; i++) {

                if constexpr (std::is_same_v<typename RT::layout, ducks::rt_layout::row>) { // handle the row-major layout

                    asm volatile(
                        "ds_read_b128 %0, %1 offset:%2\n"
                        : "=v"(*reinterpret_cast<float4*>(&dst.tiles[i][j].data[k*4]))
                        : "v"(addr), "i"(i * ST::underlying_cols * kittens::TILE_ROW_DIM<U> * sizeof(U))
                        : "memory"
                    );
                }
                else { // handle the column-major layout
                    int row = i*dst.tile_size_row + row_offset;
                    dst.tiles[i][j].data[0+k*4] = base_types::convertor<T2, U2>::convert(U2{src[{row, col}], src[{row+1, col}]});
                    dst.tiles[i][j].data[1+k*4] = base_types::convertor<T2, U2>::convert(U2{src[{row+2, col}], src[{row+3, col}]});
                    dst.tiles[i][j].data[2+k*4] = base_types::convertor<T2, U2>::convert(U2{src[{row+4, col}], src[{row+5, col}]});
                    dst.tiles[i][j].data[3+k*4] = base_types::convertor<T2, U2>::convert(U2{src[{row+6, col}], src[{row+7, col}]});
                }
            }
        }
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
 template<ducks::rt::row_layout RT, ducks::st::all ST>
 __device__ inline static void load_lds_reg_row(RT &dst, const ST &src) {
 
     static_assert(RT::height == ST::height, "register tile and shared tile must match height");
     static_assert(RT::width  == ST::width,  "register tile and shared tile must match width");
 
     using T2 = RT::dtype;
     using T  = base_types::packing<T2>::unpacked_type;
     using U  = ST::dtype;
     using U2 = base_types::packing<U >::packed_type;
     static_assert(sizeof(U) == 2, "only supporting 16-bit dtypes");
 
     const int laneid = kittens::laneid();
     const int elem_per_thread = 16 / sizeof(U); // 8 
     const uint32_t addr = reinterpret_cast<uintptr_t>(&src.data[laneid * elem_per_thread]);

     const int subtile_stride = kittens::TILE_ROW_DIM<U> * kittens::TILE_COL_DIM<U> * sizeof(U) / 2;
     const int tile_stride = subtile_stride * 2;
     const int row_stride = tile_stride * dst.width;
 
     #pragma unroll
     for(int i = 0; i < dst.height; i++) {

        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
 
            #pragma unroll 
            for (int k = 0; k < 2; k++) {
                asm volatile(
                    "ds_read_b128 %0, %1 offset:%2\n"
                    : "=v"(*reinterpret_cast<float4*>(&dst.tiles[i][j].data[k*4]))
                    : "v"(addr), "i"(i * row_stride + j * tile_stride + k * subtile_stride)
                    : "memory"
                );
             }
         }
     }
 }

 template<ducks::rt::col_layout RT, ducks::st::all ST>
__device__ inline static void load_lds_reg_col(RT &dst, const ST &src) {

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
    row_offset = (laneid % 16) / 4 + (laneid / 32) * 8;
    col_offset = ((laneid % 4) * 4) + 16*((laneid % 32)/16);

    #pragma unroll
    for(int j = 0; j < dst.width; j++) {
        int col = j*dst.tile_size_col + col_offset;

        uint32_t addr0 = src.idx(src_ptr, {row_offset, col});
        uint32_t addr1 = src.idx(src_ptr, {row_offset + 4, col});
        #pragma unroll
        for(int i = 0; i < dst.height; i++) {
            #pragma unroll 
            for (int k = 0; k < 2; k++) {

                asm volatile(
                    "ds_read_b64_tr_b16 %0, %2 offset:%4\n"
                    "ds_read_b64_tr_b16 %1, %3 offset:%4\n"
                    : "=v"(*reinterpret_cast<float2*>(&dst.tiles[i][j].data[k*4])), 
                    "=v"(*reinterpret_cast<float2*>(&dst.tiles[i][j].data[k*4 + 2]))
                    : "v"(addr0), "v"(addr1),
                    "i"((i * 2 + k) * ST::underlying_cols * (kittens::TILE_ROW_DIM<U> >> 1) * sizeof(U))
                    : "memory"
                );  
            }
        }
    }
}

template<ducks::rt::col_layout RT, ducks::st::all ST>
__device__ inline static void load_lds_reg_col_new(RT &dst, const ST &src) {

    static_assert(RT::height == ST::height, "register tile and shared tile must match height");
    static_assert(RT::width  == ST::width,  "register tile and shared tile must match width");

    using T2 = RT::dtype;
    using T  = base_types::packing<T2>::unpacked_type;
    using U  = ST::dtype;
    using U2 = base_types::packing<U >::packed_type;
    static_assert(sizeof(U) == 2, "only supporting 16-bit dtypes");

    const int laneid = kittens::laneid();
    const int row_offset = (laneid % 16) / 4 + (laneid / 32) * 8;
    const int col_offset = ((laneid % 4) * 4) + 16*((laneid % 32)/16);
    const int subtile_offset = row_offset * kittens::TILE_ROW_DIM<U> + col_offset;
    const uint32_t addr = reinterpret_cast<uintptr_t>(&src.data[subtile_offset]);

    const int subtile_stride = kittens::TILE_ROW_DIM<U> * kittens::TILE_COL_DIM<U> * sizeof(U) / 2;
    const int tile_stride = subtile_stride * 2;
    const int row_stride = tile_stride * dst.width;

    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll 
            for (int k = 0; k < 2; k++) {
                asm volatile(
                    "ds_read_b64_tr_b16 %0, %2 offset:%3\n"
                    "ds_read_b64_tr_b16 %1, %2 offset:%4\n"
                    : "=v"(*reinterpret_cast<float2*>(&dst.tiles[i][j].data[k*4])), 
                    "=v"(*reinterpret_cast<float2*>(&dst.tiles[i][j].data[k*4 + 2]))
                    : "v"(addr),
                    "i"(i * row_stride + j * tile_stride + k * subtile_stride),
                    "i"(i * row_stride + j * tile_stride + k * subtile_stride + (4 * kittens::TILE_ROW_DIM<U> * sizeof(U)))
                    : "memory"
                );  
            }
        }
    }
}


template<typename T2, int _rows, int _cols, ducks::rt_layout::all layout>
__device__ static inline void swap_layout_and_transpose(rt<T2, _cols, _rows, typename ducks::rt_layout::transpose<layout>::type> &result, const rt<T2, _rows, _cols, layout> &tile) {

    #pragma unroll
    for (int i = 0; i < tile.height; i++) {
        #pragma unroll
        for (int j = 0; j < tile.width; j++) {
            #pragma unroll
            for (int k = 0; k < tile.packed_per_tile; k++) {
                result.tiles[j][i].data[k] = tile.tiles[i][j].data[k];
            }
        }
    }
}

template<int axis, ducks::rt::accumulator_layout RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void store_transposed(const GL &dst, const RT &src, const COORD &idx) {
    using T = base_types::packing<typename RT::dtype>::unpacked_type;
    using U = typename GL::dtype;

    U *dst_ptr = (U*)&dst[(idx.template unit_coord<axis, 3>())];
    const int row_stride = dst.template stride<axis>();
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
                dst_ptr[row*row_stride + col + (jj * 8) + 0] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[jj * 2].x);
                dst_ptr[row*row_stride + col + (jj * 8) + 1] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[jj * 2].y);
                dst_ptr[row*row_stride + col + (jj * 8) + 2] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[jj * 2 + 1].x);
                dst_ptr[row*row_stride + col + (jj * 8) + 3] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[jj * 2 + 1].y);
            }
        }
    }
}

template<ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void store_transposed(const GL &dst, const RT &src, const COORD &idx) {
    store_transposed<1, RT, GL, COORD>(dst, src, idx);
}



