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
 int N_THREADS = WARP_THREADS>
 __device__ inline void prefill_swizzled_offsets_fp6(
 const GL& src, const COORD& idx, ST& dst, uint32_t* swizzled_offsets)
 {
 
    using T = typename ST::dtype;
    constexpr int bytes_per_thread = 12;
    constexpr int memcpy_per_tile =  (ST::rows * ST::cols * 6 / 8) / (bytes_per_thread * N_THREADS);
    static_assert(memcpy_per_tile * bytes_per_thread * N_THREADS == ST::rows * ST::cols * 6 / 8, "memcpy_per_tile * bytes_per_thread * N_THREADS != ST::rows * ST::cols * 6 / 8");

    constexpr int bytes_per_warp = bytes_per_thread * kittens::WARP_THREADS; // 16 * 64 = 1024
    constexpr int bytes_per_block = bytes_per_thread * N_THREADS;
    constexpr int bytes_per_base_tile = (kittens::TILE_COL_DIM<T> * kittens::TILE_ROW_DIM<T> * 6 / 8);
    const int warp_id = warpid();
    const int laneid = kittens::laneid() % kittens::WARP_THREADS;
    // row stride
    const int bytes_per_base_tile_row = kittens::TILE_COL_DIM<T> * 6 / 8;
    const int tiles_per_row =  ST::cols / kittens::TILE_COL_DIM<T>;
    const int row_stride_bytes = src.template stride<axis>() * 6 / 8;

 
     #pragma unroll
     for (int i = 0; i < memcpy_per_tile; i++) {
 
        const int warp_byte_offset = (i * bytes_per_block) + (warp_id * bytes_per_warp);
        const int lane_byte_offset = laneid * bytes_per_thread + warp_byte_offset;

        const int tile_id = lane_byte_offset / bytes_per_base_tile;
        const int tile_row_offset = tile_id / tiles_per_row;
        const int tile_col_offset = tile_id % tiles_per_row;

        const int base_tile_byte_offset = lane_byte_offset % bytes_per_base_tile;
        const int base_tile_row_offset = base_tile_byte_offset / bytes_per_base_tile_row;
        const int base_tile_col_byte_offset = base_tile_byte_offset % bytes_per_base_tile_row;

        const int row_offset = tile_row_offset * kittens::TILE_ROW_DIM<T> + base_tile_row_offset;
        const int col_byte_offset = tile_col_offset * bytes_per_base_tile_row + base_tile_col_byte_offset;

        swizzled_offsets[i] = row_offset * row_stride_bytes + col_byte_offset;
     }
 }

//  Direct global-to-shared load using buffer load to LDS
 template<int axis, bool assume_aligned,
 ducks::st::all ST, ducks::gl::all GL,
 ducks::coord::tile COORD = coord<ST>,
 int N_THREADS = WARP_THREADS>
 __device__ inline void load_global_to_shared_direct_with_swizzled_offsets_fp6(
 const GL& src, const COORD& idx, ST& dst, uint32_t* swizzled_offsets)
 {
 
    using U = typename ST::dtype;
    constexpr int bytes_per_thread = 12;
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
        
        // Split the 12-byte load into three 4-byte loads
        as3_uint32_ptr lds_ptr0 = (as3_uint32_ptr)reinterpret_cast<uintptr_t>(lds_elem_ptr);
        as3_uint32_ptr lds_ptr1 = (as3_uint32_ptr)reinterpret_cast<uintptr_t>(lds_elem_ptr + 4);
        as3_uint32_ptr lds_ptr2 = (as3_uint32_ptr)reinterpret_cast<uintptr_t>(lds_elem_ptr + 8);
        
        llvm_amdgcn_raw_buffer_load_lds(srsrc, lds_ptr0, 4, swizzled_offsets[i] + 0, 0, 0, static_cast<index_t>(coherency::cache_all));
        __builtin_amdgcn_s_waitcnt(0);
        llvm_amdgcn_raw_buffer_load_lds(srsrc, lds_ptr1, 4, swizzled_offsets[i] + 4, 0, 0, static_cast<index_t>(coherency::cache_all));
        __builtin_amdgcn_s_waitcnt(0);
        llvm_amdgcn_raw_buffer_load_lds(srsrc, lds_ptr2, 4, swizzled_offsets[i] + 8, 0, 0, static_cast<index_t>(coherency::cache_all));
        __builtin_amdgcn_s_waitcnt(0);

        // llvm_amdgcn_raw_buffer_load_lds(
        //     srsrc, // buffer resource
        //     lds_ptr0,
        //     12, // 12 bytes
        //     swizzled_offsets[i],
        //     0, 
        //     0, // instruction offset
        //     static_cast<index_t>(coherency::cache_all)); // cache coherency
    }
}

/**
* @brief Load data from a shared tile into a register tile with 4-byte load layout.
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

    // Adjust addressing for 4-byte load layout
    // When using 4-byte loads, the hardware places data with (TIDinWave * 4) stride
    // We need to calculate where our 12-byte chunk actually starts
    const int row_offset = laneid % 32;
    const int col_offset = 32 * (laneid / 32);
    const int byte_offset = (row_offset * kittens::TILE_COL_DIM<U> + col_offset) * 6 / 8;
    const uint32_t addr = reinterpret_cast<uintptr_t>(lds_bytes + byte_offset);

    const int subtile_stride = kittens::TILE_ROW_DIM<U> * kittens::TILE_COL_DIM<U> * 6 / 8 / 2;
    const int tile_stride = subtile_stride * 2;
    const int row_stride = tile_stride * src.underlying_width;

    #pragma unroll
    for(int i = 0; i < dst.height; i++) {

        #pragma unroll
        for(int j = 0; j < dst.width; j++) {

            #pragma unroll
            for (int k = 0; k < 2; k++) {
                asm volatile(
                    "ds_read_b96 %0, %1 offset:%2\n"
                    "s_waitcnt lgkmcnt(0)\n"
                    : "=v"(*reinterpret_cast<__uint96_t*>((reinterpret_cast<uint8_t*>(&dst.tiles[i][j].data[0]) + k * 12)))
                    : "v"(addr),
                    "i"(i * row_stride + j * tile_stride + k * subtile_stride)
                    : "memory"
                );
            }
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

int row_offset = laneid%32, col_offset = 16*(laneid/32);

i32x4 srsrc = make_srsrc(dst_ptr, row_stride * RT::rows * 6 / 8);

#pragma unroll
for(int i = 0; i < src.height; i++) {
int row = src.tile_size_row*i + row_offset;

#pragma unroll
for(int j = 0; j < src.width; j++) {

   #pragma unroll
   for (int k = 0; k < 2; k++) {
       int col = src.tile_size_col*j + col_offset + k * 32;
       
       const __uint96_t val_b96 = *reinterpret_cast<const __uint96_t*>((reinterpret_cast<const uint8_t*>(&src.tiles[i][j].data[0]) + k * 12));
       // const __uint96_t val_b96 = {0x11111111, 0x11111111, 0x11111111};
       llvm_amdgcn_raw_buffer_store_b96(val_b96, srsrc, (row*row_stride + col) * 6 / 8, 0, 0);
   }
}
}
}
template<ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void store_fp6(const GL &dst, const RT &src, const COORD &idx) {
store_fp6<2, RT, GL, COORD>(dst, src, idx);
}

