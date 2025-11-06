#include "kittens.cuh"
using namespace kittens;


/*
Assembly and intrinsic functions.
*/
using as3_uint32_ptr = uint32_t __attribute__((address_space(3)))*;
using index_t = int;
using int32x4_t = int32_t __attribute__((ext_vector_type(4)));


enum class coherency {
    cache_all = 0,
    cache_global = 1,
    cache_stream = 2,
    non_temporal = 3
};

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
         ducks::rt::all RT, ducks::st::all ST, ducks::gl::all GL,
         ducks::coord::tile COORD = coord<ST>,
         int N_THREADS = WARP_THREADS>
__device__ inline void prefill_swizzled_offsets(
    const GL& src, const COORD& idx, ST& dst, uint32_t* swizzled_offsets)
{

    using T = typename ST::dtype;
    constexpr int memcpy_per_tile =  ST::rows * ST::cols * sizeof(T) / (16 * N_THREADS); // 16 --> 32
    static_assert(memcpy_per_tile > 0, "memcpy_per_tile must be greater than 0. Please decrease the number of threads.");

    using U = typename ST::dtype;
    using U2 = base_types::packing<U>::packed_type;
    const int packed_size = sizeof(U2) / sizeof(U);  // 4 for FP6

    constexpr int elem_per_thread = 16 / sizeof(T);  // 8
    constexpr int elem_per_warp = elem_per_thread * kittens::WARP_THREADS; // 512
    const int warp_id = warpid();
    const int row_stride = src.template stride<axis>();

    constexpr int num_warps = N_THREADS / kittens::WARP_THREADS;
    constexpr int num_register_subtiles = kittens::TILE_ROW_DIM<T> * kittens::TILE_COL_DIM<T> / elem_per_warp;
    constexpr int num_register_tiles_per_row = ST::cols / kittens::TILE_COL_DIM<T>;

    #pragma unroll
    for (int i = 0; i < memcpy_per_tile; i++) {

        const int register_tile_id = (warp_id + i * num_warps) / num_register_subtiles;
        const int register_subtile_id = (warp_id + i * num_warps) % num_register_subtiles;

        const int register_subtile_cols = kittens::TILE_COL_DIM<T> / num_register_subtiles;
        const int num_register_subtiles_per_row = num_register_tiles_per_row * num_register_subtiles;
        const int warp_col_offset = ((register_tile_id % num_register_tiles_per_row) * num_register_subtiles + register_subtile_id) * register_subtile_cols;
        const int warp_row_offset = (register_tile_id / num_register_tiles_per_row) * kittens::TILE_ROW_DIM<T>;

        int col_offset = warp_col_offset + (laneid() / 32) * elem_per_thread ;
        int row_offset = warp_row_offset + (laneid() % 32);

        const int offset_in_global = (row_offset * row_stride + col_offset) * sizeof(T);

        swizzled_offsets[i] = offset_in_global;
    }
}


// Direct global-to-shared load using buffer load to LDS
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
    //  static_assert(sizeof(U) == 2, "only supporting 16-bit dtypes");
 
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

// ------------------------------32-packed fp6--------------------------------

// Direct global-to-shared load using buffer load to LDS
template<int axis, bool assume_aligned,
         ducks::st::all ST, ducks::gl::all GL,
         ducks::coord::tile COORD = coord<ST>,
         int N_THREADS = WARP_THREADS>
__device__ inline void prefill_swizzled_offsets_fp6(
    const GL& src, const COORD& idx, ST& dst, uint32_t* swizzled_offsets)
{

    using T = typename ST::dtype;
    constexpr int bytes_per_thread = 16;
    constexpr int memcpy_per_tile =  (ST::rows * ST::cols * 6 / 8) / (bytes_per_thread * N_THREADS);
    static_assert(memcpy_per_tile * bytes_per_thread * N_THREADS == ST::rows * ST::cols * 6 / 8, "memcpy_per_tile * bytes_per_thread * N_THREADS != ST::rows * ST::cols * 6 / 8");

    constexpr int bytes_per_warp = bytes_per_thread * kittens::WARP_THREADS; // 16 * 64 = 1024
    constexpr int bytes_per_block = bytes_per_thread * N_THREADS;
    constexpr int bytes_per_row = ST::cols * 6 / 8;

    const int warp_id = warpid();
    const int laneid = kittens::laneid() % kittens::WARP_THREADS;

    const int row_stride_bytes = src.template stride<axis>() * 6 / 8;

    #pragma unroll
    for (int i = 0; i < memcpy_per_tile; i++) {

        const int warp_byte_offset = (i * bytes_per_block) + (warp_id * bytes_per_warp);
        int lane_byte_offset = laneid * bytes_per_thread + warp_byte_offset;
        const int bottom = (lane_byte_offset % (16 * 96)) / (8 * 96);
        const int right = (lane_byte_offset % (96)) / 48;
        lane_byte_offset += bottom * 48 * (1 - (2 * right));

        const int row_offset = lane_byte_offset / bytes_per_row;
        const int col_byte_offset = lane_byte_offset % bytes_per_row;

        swizzled_offsets[i] = row_offset * row_stride_bytes + col_byte_offset;
    }
}


// Direct global-to-shared load using buffer load to LDS
template<int axis, bool assume_aligned,
         ducks::st::all ST, ducks::gl::all GL,
         ducks::coord::tile COORD = coord<ST>,
         int N_THREADS = WARP_THREADS>
__device__ inline void load_global_to_shared_direct_with_swizzled_offsets_fp6(
    const GL& src, const COORD& idx, ST& dst, uint32_t* swizzled_offsets)
{

    using U = typename ST::dtype;
    constexpr int bytes_per_thread = 16;
    constexpr int memcpy_per_tile =  (ST::rows * ST::cols * 6 / 8) / (bytes_per_thread * N_THREADS);
    static_assert(memcpy_per_tile * bytes_per_thread * N_THREADS == ST::rows * ST::cols * 6 / 8, "memcpy_per_tile * bytes_per_thread * N_THREADS != ST::rows * ST::cols * 6 / 8");     
    
    constexpr int bytes_per_warp = bytes_per_thread * kittens::WARP_THREADS;

    // byte stride
    const int row_stride_bytes = src.template stride<axis>() * 6 / 8;
    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    auto* global_ptr = reinterpret_cast<const uint8_t*>(&src[unit_coord]);
    i32x4 srsrc = make_srsrc(global_ptr, row_stride_bytes * ST::rows); // size in BYTES

    const int warp_id = warpid();
    auto* lds_bytes = reinterpret_cast<uint8_t*>(&dst.data[0]);
    const uint8_t* lds_base = lds_bytes + warp_id * bytes_per_warp;

    #pragma unroll
    for (int i = 0; i < memcpy_per_tile; i++) {
        const uint8_t* lds_elem_ptr = lds_base + i * N_THREADS * bytes_per_thread;
        as3_uint32_ptr lds_ptr = (as3_uint32_ptr)reinterpret_cast<uintptr_t>(lds_elem_ptr);

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

template<ducks::rt::row_layout RT, ducks::st::all ST>
__device__ inline static uint32_t prefill_swizzled_offset_fp6(RT &dst, const ST &src) {
    static_assert(RT::height == ST::height, "register tile and shared tile must match height");
    static_assert(RT::width  == ST::width,  "register tile and shared tile must match width");

    using U  = ST::dtype;
    const int laneid = kittens::laneid() % kittens::WARP_THREADS;
    // auto* lds_bytes = reinterpret_cast<const uint8_t*>(&src.data[0]);

    const int row_offset = laneid % 16;
    const int col_offset = 32 * (laneid / 16);

    int elem_offset = row_offset * kittens::TILE_COL_DIM<U> + col_offset;
    elem_offset ^= ((elem_offset % (16 * 128)) >> 10) << 6;

    int byte_offset = (elem_offset) * 6 / 8;

    // Alternate implementation:
    // const int bottom = (byte_offset % (16 * 96)) / (8 * 96);
    // const int right = (byte_offset % (96)) / 48;
    // byte_offset += bottom * 48 * (1 - (2 * right));

    // const uint32_t addr = reinterpret_cast<uintptr_t>(lds_bytes + byte_offset);
    // return addr;
    return byte_offset;
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
__device__ inline static void load_lds_reg_row_fp6(RT &dst, const ST &src, uint32_t addr) {

    using U  = ST::dtype;

    const int tile_stride = (kittens::TILE_ROW_DIM<U> * kittens::TILE_COL_DIM<U> * 6 / 8);
    const int row_stride = tile_stride * src.underlying_width;

    uint32_t lds_bytes = reinterpret_cast<uintptr_t>(&src.data[0]);

    addr = lds_bytes + addr;

    #pragma unroll
    for(int i = 0; i < dst.height; i++) {

        #pragma unroll
        for(int j = 0; j < dst.width; j++) {

            asm volatile(
                "ds_read_b64 %0, %3 offset:%4\n"
                "ds_read_b64 %1, %3 offset:%5\n"
                "ds_read_b64 %2, %3 offset:%6\n"
                : "=v"(*reinterpret_cast<uint64_t*>(&dst.tiles[i][j].data[0])),
                    "=v"(*reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(&dst.tiles[i][j].data[0]) + 8)),
                    "=v"(*reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(&dst.tiles[i][j].data[0]) + 16))
                : "v"(addr),
                "i"(i * row_stride + j * tile_stride),
                "i"(i * row_stride + j * tile_stride + 8),
                "i"(i * row_stride + j * tile_stride + 16)
                : "memory"
            );
        }
    }
}

 template<int axis, ducks::rt::row_layout RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void store_fp6(const GL &dst, const RT &src, const COORD &idx) {
    using T2 = RT::dtype;
    using U = typename GL::dtype;
    using U2 = base_types::packing<U>::packed_type;

    U *dst_ptr = (U*)&dst[(idx.template unit_coord<axis, 3>())];
    const int row_stride = dst.template stride<axis>();
    int laneid = kittens::laneid() % kittens::WARP_THREADS;

    int row_offset = laneid%16, col_offset = 32*(laneid/16);

    i32x4 srsrc = make_srsrc(dst_ptr, row_stride * RT::rows * 6 / 8);

    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        int row = src.tile_size_row*i + row_offset;
        
        #pragma unroll
        for(int j = 0; j < src.width; j++) {

            #pragma unroll
            for (int k = 0; k < 2; k++) {
                int col = src.tile_size_col*j + col_offset + k * 16;
                
                const __uint96_t val_b96 = *reinterpret_cast<const __uint96_t*>((reinterpret_cast<const uint8_t*>(&src.tiles[i][j].data[0]) + k * 12));
                // const __uint96_t val_b96 = {0x49249249, 0x92492492, 0x24924924};
                llvm_amdgcn_raw_buffer_store_b96(val_b96, srsrc, (row*row_stride + col) * 6 / 8, 0, 0);
            }
        }
    }
}
template<ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void store_fp6(const GL &dst, const RT &src, const COORD &idx) {
    store_fp6<2, RT, GL, COORD>(dst, src, idx);
}

__device__ inline static uint8_t float_to_fp6_bits(float f) {
    if (f == 0.0f) return 0x00;
    
    uint32_t float_bits = __float_as_uint(f);
    uint32_t sign = (float_bits >> 31) & 0x1;
    int32_t exp = ((float_bits >> 23) & 0xFF) - 127 + 1;  // Unbias and add E2M3 bias
    uint32_t mantissa = (float_bits >> 20) & 0x7;
    
    if (exp < 0) return (sign << 5);
    if (exp > 3) return (sign << 5) | 0x1F;
    
    return (sign << 5) | ((exp & 0x3) << 3) | mantissa;
}

template<int axis, ducks::rt::accumulator_layout RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void store_fp6_convert(const GL &dst, const RT &src, const COORD &idx) {
    using T = base_types::packing<typename RT::dtype>::unpacked_type; // float
    using U = typename GL::dtype;  // fp6_e2m3
    
    // Get the base pointer in global memory
    uint8_t *dst_bytes = (uint8_t*)&dst[(idx.template unit_coord<axis, 3>())];
    const int row_stride = dst.template stride<axis>();
    int laneid = kittens::laneid();
    
    int col_offset = laneid % 32;
    int row_offset = laneid / 32;
    
    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = src.tile_size_col * j + col_offset;
            
            #pragma unroll
            for (int ii = 0; ii < 4; ii++) {
                int row = src.tile_size_row * i + ii * 8 + row_offset * 4;
                
                // Convert 4 floats to 4 FP6 values (24 bits total)
                uint8_t fp6_0 = float_to_fp6_bits(src.tiles[i][j].data[ii * 2].x);
                uint8_t fp6_1 = float_to_fp6_bits(src.tiles[i][j].data[ii * 2].y);
                uint8_t fp6_2 = float_to_fp6_bits(src.tiles[i][j].data[ii * 2 + 1].x);
                uint8_t fp6_3 = float_to_fp6_bits(src.tiles[i][j].data[ii * 2 + 1].y);
                
                // Pack and store these 4 FP6 values (24 bits)
                // Calculate bit positions for each value
                int elem0_bit = ((row + 0) * row_stride + col) * 6;
                int elem1_bit = ((row + 1) * row_stride + col) * 6;
                int elem2_bit = ((row + 2) * row_stride + col) * 6;
                int elem3_bit = ((row + 3) * row_stride + col) * 6;
                
                // Use 32-bit atomic operations to update the packed data
                uint32_t *dst_words = (uint32_t*)dst_bytes;
                
                // Helper lambda to pack a single FP6 value
                auto pack_fp6 = [&](int bit_pos, uint8_t fp6_val) {
                    int word_idx = bit_pos / 32;
                    int bit_off = bit_pos % 32;
                    
                    atomicOr(&dst_words[word_idx], uint32_t(fp6_val & 0x3F) << bit_off);
                    
                    if (bit_off + 6 > 32) {
                        atomicOr(&dst_words[word_idx + 1], 
                                uint32_t(fp6_val & 0x3F) >> (32 - bit_off));
                    }
                };
                
                pack_fp6(elem0_bit, fp6_0);
                pack_fp6(elem1_bit, fp6_1);
                pack_fp6(elem2_bit, fp6_2);
                pack_fp6(elem3_bit, fp6_3);
            }
        }
    }
}
template<ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void store_fp6_convert(const GL &dst, const RT &src, const COORD &idx) {
    store_fp6_convert<2, RT, GL, COORD>(dst, src, idx);
}
