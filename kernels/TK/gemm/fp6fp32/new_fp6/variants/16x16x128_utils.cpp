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


// ------------------------------32-packed fp6--------------------------------


// Direct global-to-shared load using buffer load to LDS
template<int axis, bool assume_aligned,
         ducks::st::all ST, ducks::gl::all GL,
         ducks::coord::tile COORD = coord<ST>,
         int N_THREADS = WARP_THREADS>
__device__ inline void load_global_to_shared_fp6(
    const GL& src, const COORD& idx, ST& dst)
{

    using T = typename ST::dtype;
    constexpr int bytes_per_thread = 16;
    constexpr int memcpy_per_tile =  (ST::rows * ST::cols) / (bytes_per_thread * N_THREADS);
    static_assert(memcpy_per_tile > 0, "memcpy_per_tile must be greater than 0. Please decrease the number of threads.");

    constexpr int bytes_per_warp = bytes_per_thread * kittens::WARP_THREADS;
    const int warp_id = warpid();
    const int laneid = kittens::laneid();
    const int row_stride = src.template stride<axis>();
    constexpr int num_warps = N_THREADS / kittens::WARP_THREADS;

    const int shared_base_tile_cols = 128;
    const int shared_base_tile_rows = 16;

    const int bytes_per_shared_base_tile = shared_base_tile_cols * shared_base_tile_rows;
    const int shared_base_tiles_per_row = ST::cols / shared_base_tile_cols;

    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    auto* global_ptr = reinterpret_cast<const uint8_t*>(&src[unit_coord]);
    i32x4 srsrc = make_srsrc(global_ptr, row_stride * ST::rows * 6 / 8); 
    auto* lds_bytes = reinterpret_cast<uint8_t*>(&dst.data[0]);
    const uint8_t* lds_base = lds_bytes + warp_id * bytes_per_warp;

    #pragma unroll
    for (int i = 0; i < memcpy_per_tile; i++) {

        const int byte_offset = ((i * num_warps + warp_id) * bytes_per_warp) + (laneid * bytes_per_thread);
        const int tile_id = byte_offset / bytes_per_shared_base_tile;
        const int tile_row_offset = tile_id / shared_base_tiles_per_row;
        const int tile_col_offset = tile_id % shared_base_tiles_per_row;

        const int base_tile_byte_offset = byte_offset % bytes_per_shared_base_tile;
        const int swizzled_base_tile_byte_offset = base_tile_byte_offset ^ (((base_tile_byte_offset % (8*128)) >> 7) << 4);
        
        const int swizzled_col_offset = (swizzled_base_tile_byte_offset % (shared_base_tile_cols * 1)) / 1;
        const int swizzled_row_offset = swizzled_base_tile_byte_offset / (shared_base_tile_cols * 1);

        const int col_offset_in_global = tile_col_offset * shared_base_tile_cols + swizzled_col_offset;
        const int row_offset_in_global = tile_row_offset * shared_base_tile_rows + swizzled_row_offset;

        const int offset_in_global = (row_offset_in_global * row_stride + col_offset_in_global) * 6 / 8;

        const uint8_t* lds_elem_ptr = lds_base + i * N_THREADS * bytes_per_thread;
        as3_uint32_ptr lds_ptr = (as3_uint32_ptr)reinterpret_cast<uintptr_t>(lds_elem_ptr);

        llvm_amdgcn_raw_buffer_load_lds(
            srsrc, // buffer resource
            lds_ptr,
            12, // 12 bytes is specified but the thread actually writes 16 bytes, last 4 garbage
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
         int N_THREADS = WARP_THREADS>
__device__ inline void prefill_swizzled_offsets_fp6(
    const GL& src, const COORD& idx, ST& dst, uint32_t* swizzled_offsets)
{

    using T = typename ST::dtype;
    constexpr int bytes_per_thread = 16;
    constexpr int memcpy_per_tile =  (ST::rows * ST::cols) / (bytes_per_thread * N_THREADS);
    static_assert(memcpy_per_tile > 0, "memcpy_per_tile must be greater than 0. Please decrease the number of threads.");

    constexpr int bytes_per_warp = bytes_per_thread * kittens::WARP_THREADS;
    const int warp_id = warpid();
    const int laneid = kittens::laneid();
    const int row_stride = src.template stride<axis>();
    constexpr int num_warps = N_THREADS / kittens::WARP_THREADS;

    const int shared_base_tile_cols = 128;
    const int shared_base_tile_rows = 16;

    const int bytes_per_shared_base_tile = shared_base_tile_cols * shared_base_tile_rows;
    const int shared_base_tiles_per_row = ST::cols / shared_base_tile_cols;

    #pragma unroll
    for (int i = 0; i < memcpy_per_tile; i++) {

        const int byte_offset = ((i * num_warps + warp_id) * bytes_per_warp) + (laneid * bytes_per_thread);
        const int tile_id = byte_offset / bytes_per_shared_base_tile;
        const int tile_row_offset = tile_id / shared_base_tiles_per_row;
        const int tile_col_offset = tile_id % shared_base_tiles_per_row;

        const int base_tile_byte_offset = byte_offset % bytes_per_shared_base_tile;
        const int swizzled_base_tile_byte_offset = base_tile_byte_offset ^ (((base_tile_byte_offset % (8*128)) >> 7) << 4);
        
        const int swizzled_col_offset = (swizzled_base_tile_byte_offset % (shared_base_tile_cols * 1)) / 1;
        const int swizzled_row_offset = swizzled_base_tile_byte_offset / (shared_base_tile_cols * 1);

        const int col_offset_in_global = tile_col_offset * shared_base_tile_cols + swizzled_col_offset;
        const int row_offset_in_global = tile_row_offset * shared_base_tile_rows + swizzled_row_offset;

        const int offset_in_global = (row_offset_in_global * row_stride + col_offset_in_global) * 6 / 8;

        swizzled_offsets[i] = offset_in_global;

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
    constexpr int memcpy_per_tile =  (ST::rows * ST::cols * 1) / (bytes_per_thread * N_THREADS);
    static_assert(memcpy_per_tile > 0, "memcpy_per_tile must be greater than 0. Please decrease the number of threads.");
    
    constexpr int bytes_per_warp = bytes_per_thread * kittens::WARP_THREADS;

    // byte stride
    const int row_stride = src.template stride<axis>();
    coord<> unit_coord = idx.template unit_coord<axis, 3>();

    auto* global_ptr = reinterpret_cast<const uint8_t*>(&src[unit_coord]);
    i32x4 srsrc = make_srsrc(global_ptr, row_stride * ST::rows * 6 / 8); // size in BYTES

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
            12, // 12 bytes is specified but the thread actually writes 16 bytes, last 4 garbage
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
 __device__ inline static void load_lds_reg_row_fp6(RT &dst, const ST &src) {
 
     static_assert(RT::height == ST::height, "register tile and shared tile must match height");
     static_assert(RT::width  == ST::width,  "register tile and shared tile must match width");
 
     using U  = ST::dtype;
     const int laneid = kittens::laneid();
     auto* lds_bytes = reinterpret_cast<const uint8_t*>(&src.data[0]);

     const int row_offset = laneid % 16;
     const int col_byte_offset = 32 * (laneid / 16);  // NOTE: This col_byte_offset is in bytes, not elements.

     uint32_t byte_offset = (row_offset * 128 + col_byte_offset);
     uint32_t swizzled_byte_offset = byte_offset ^ (((byte_offset % (8*128)) >> 7) << 4);
     uint32_t byte_offset_second = byte_offset + 16;
     uint32_t swizzled_byte_offset_second = byte_offset_second ^ (((byte_offset_second % (8*128)) >> 7) << 4);
     uint32_t addr = reinterpret_cast<uintptr_t>(lds_bytes + swizzled_byte_offset);
     uint32_t addr_second = reinterpret_cast<uintptr_t>(lds_bytes + swizzled_byte_offset_second);

     const int tile_stride = 16 * 128;
     const int row_stride = tile_stride * src.underlying_width;
 
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {

       #pragma unroll
       for(int j = 0; j < dst.width; j++) {
            asm volatile(
                "ds_read_b96 %0, %1 offset:%2\n"
                : "=v"(*reinterpret_cast<__uint96_t*>((reinterpret_cast<uint8_t*>(&dst.tiles[i][j].data[0]))))
                : "v"(addr),
                "i"(i * row_stride + j * tile_stride)
                : "memory"
            );
            asm volatile(
                "ds_read_b96 %0, %1 offset:%2\n"
                : "=v"(*reinterpret_cast<__uint96_t*>((reinterpret_cast<uint8_t*>(&dst.tiles[i][j].data[0]) + 12)))
                : "v"(addr_second),
                "i"(i * row_stride + j * tile_stride)
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
