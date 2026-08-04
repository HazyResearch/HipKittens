/**
Load utils for fine-grained interleaving in FP8_4wave.
These are wrappers around the individual assembly instructions.
*/

#include <type_traits>
#include "kittens.cuh"

using namespace kittens;

struct precomputed_addresses {
    i32x4 srsrc;
    uintptr_t lds_base;
};

/**
 * @brief Precompute the buffer resource for the given shared tile and global memory.
 * @param dst The shared tile.
 * @param src The global memory.
 * @param idx The index of the shared tile.
 * @return The buffer resource.
 */
template<typename ST, typename GL>
__device__ __forceinline__ static precomputed_addresses precompute_addresses(ST& dst, const GL& src, const coord<ST>& idx) {
    constexpr int axis = 2;
    using T = typename ST::dtype;

    const int row_stride = src.template stride<axis>();

    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    T* global_ptr = (T*)&src[unit_coord];
    i32x4 srsrc = make_srsrc(global_ptr, row_stride * ST::rows * sizeof(T));

    constexpr int bytes_per_thread = ST::underlying_subtile_bytes_per_thread;
    constexpr int bytes_per_warp = bytes_per_thread * kittens::WARP_THREADS;
    constexpr int elem_per_warp = bytes_per_warp / sizeof(T);
    const int warp_id = warpid();

    uintptr_t lds_base = reinterpret_cast<uintptr_t>(&dst.data[0]) + (warp_id * bytes_per_warp);

    return {srsrc, lds_base};
}

/**
 * @brief Perform one buffer_load_dwordx4 from the global memory to the shared tile.
 * @param dst The shared tile.
 * @param src The global memory.
 * @param srsrc The buffer resource.
 */
template<int i, typename ST, typename GL>
__device__ inline static void load_one(ST& dst, const GL& src, precomputed_addresses addresses)
{
    constexpr int axis = 2;
    const int N_THREADS = kittens::num_warps()*kittens::WARP_THREADS;

    using T = typename ST::dtype;

    constexpr int bytes_per_thread = ST::underlying_subtile_bytes_per_thread;
    constexpr int bytes_per_warp = bytes_per_thread * kittens::WARP_THREADS;
    static_assert(ST::rows * ST::cols * sizeof(T) >= bytes_per_warp, "shared tile must be at least 1024 bytes");

    const int num_warps = N_THREADS / kittens::WARP_THREADS;
    const int laneid = kittens::laneid();
    const int warpid = kittens::warpid() % num_warps;

    const int row_stride = src.template stride<axis>();

    // const uintptr_t lds_base = reinterpret_cast<uintptr_t>(&dst.data[0]) + (warpid * bytes_per_warp);

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

    uintptr_t lds_addr = addresses.lds_base + (i * num_warps * bytes_per_warp);
    as3_uint32_ptr lds_ptr = (as3_uint32_ptr)(lds_addr);

    llvm_amdgcn_raw_buffer_load_lds(
        addresses.srsrc, // buffer resource
        lds_ptr,
        bytes_per_thread,
        swizzled_global_byte_offset,
        0, 
        0, // instruction offset
        static_cast<int>(coherency::cache_all)); // cache coherency
}

/**
 * @brief Prefill the swizzled offsets for the given register tile and shared tile.
 * This function makes a number of assumptions which are true for FP8_4wave gemm, but
 * likely will not be true for other kernels.
 * @param dst The register tile.
 * @param src The shared tile.
 * @param swizzled_offsets The swizzled offsets.
 */
template<int num_offsets, typename RT, typename ST>
__device__ inline static void prefill_swizzled_offsets(RT& dst, ST& src, uint32_t* swizzled_offsets) {
    static_assert(RT::rows == ST::rows, "register tile and shared tile must match rows");
    static_assert(RT::cols == ST::cols,  "register tile and shared tile must match cols");
    static_assert(num_offsets == RT::base_tile_num_strides, "number of offsets must match number of strides");

    using T2 = RT::dtype;
    using T  = base_types::packing<T2>::unpacked_type;
    using U  = ST::dtype;
    using U2 = base_types::packing<U >::packed_type;
    static_assert(sizeof(U) == 2 || sizeof(U) == 1, "only supporting 16 and 8-bit dtypes");
    static_assert((!std::is_same_v<T, fp8e4m3>) || std::is_same_v<U, T>, "global and shared tile must have the same dtype if fp8");

    constexpr int subtile_stride = RT::base_tile_cols * sizeof(U) / 2;
    const int tile_stride = subtile_stride * 2;

    const int elem_per_thread = 16 / sizeof(U); // 8 if bf16, 16 if fp8e4m3
    uint32_t st_offset = (kittens::laneid() % RT::base_tile_rows) * ST::underlying_cols + (kittens::laneid() / RT::base_tile_rows * 16 / sizeof(U));
    uint32_t base_addr = reinterpret_cast<uintptr_t>(&src.data[st_offset]);
    swizzled_offsets[0] = base_addr;
    swizzled_offsets[0] ^= (((swizzled_offsets[0] % (256*8)) >> 8) << 4);
    swizzled_offsets[1] = base_addr + subtile_stride;
    swizzled_offsets[1] ^= (((swizzled_offsets[1] % (256*8)) >> 8) << 4);
}

/**
 * @brief Load data from a shared tile into a register tile.
 * @param dst The register tile.
 * @param src The shared tile.
 */
template<int register_row, int register_col, int k, typename RT, typename ST>
__device__ inline static void load_one(RT& dst, ST& src, uint32_t* swizzled_offsets) {
    using U  = ST::dtype;
    constexpr int packing = base_types::packing<typename RT::dtype>::num();
    const int idx = k * RT::base_tile_stride / packing;
    constexpr int row_stride = RT::base_tile_rows * ST::underlying_cols * sizeof(U);
    asm volatile(
        "ds_read_b128 %0, %1 offset:%2\n"
        : "=v"(*reinterpret_cast<float4*>(&dst.tiles[register_row][register_col].data[idx]))
        : "v"(swizzled_offsets[k]), "i"(register_row * row_stride)
        : "memory"
    );
}

/**
 * @brief Perform one MFMA instruction.
 * @param d_mma The output register tile.
 * @param a_mma The first input register tile.
 * @param b_mma The second input register tile.
 * @param c_mma The input register tile.
 * @param n The row index of the output register tile.
 * @param m The column index of the output register tile.
 * @param k The column index of the first input register tile.
 */
template<typename D, typename A, typename B, typename C>
__device__ inline void mma_ABt_one(D& d_mma, const A& a_mma, const B& b_mma, const C& c_mma, int n, int m, int k) {
    static_assert(D::rows == A::rows && D::cols == B::rows); // Check D matches A, B
    static_assert(A::cols == B::cols); // Check reduction dim is same
    static_assert(D::rows == C::rows && D::cols == C::cols); // Check D matches C
    static_assert(std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, fp8e4m3> &&
                  std::is_same_v<typename B::T, fp8e4m3> && std::is_same_v<typename C::T, float>);
    
    mma_ABt_base(
        d_mma.tiles[n][m],
        a_mma.tiles[n][k],
        b_mma.tiles[m][k],
        c_mma.tiles[n][m]
    );
}
