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

