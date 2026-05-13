/**
 * Load utils for fine-grained interleaving in INT8_4wave.
 * Mirrors fp8fp32/FP8_4wave/utils.cpp, adapted for INT8:
 *   - rt_int8 uses an rt_16x64 base (K=64 MFMA) with stride 16 in sim/HipKittens
 *     (one ds_read_b128 covers all 16 K elements per lane in a single base tile).
 *   - mma_ABt_one accepts (int8/int8 -> int32) inputs.
 */

#include <type_traits>
#include "kittens.cuh"

using namespace kittens;

struct precomputed_addresses {
    i32x4 srsrc;
    uintptr_t lds_base;
};

/**
 * @brief Precompute the buffer resource and per-warp LDS base for a shared-tile fill.
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
    const int warp_id = warpid();

    uintptr_t lds_base = reinterpret_cast<uintptr_t>(&dst.data[0]) + (warp_id * bytes_per_warp);

    return {srsrc, lds_base};
}

/**
 * @brief Issue one buffer_load_dwordx4 from global into the shared tile.
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
        addresses.srsrc,
        lds_ptr,
        bytes_per_thread,
        swizzled_global_byte_offset,
        0,
        0,
        static_cast<int>(coherency::cache_all));
}

/**
 * @brief Prefill the swizzled LDS byte offsets for one (warp, base-row) of a
 * shared->register load. One offset per stride.
 *
 * For INT8 + rt_16x64 stride 16: a single ds_read_b128 covers the lane's full
 * 16-element K segment of one base tile, so num_offsets == RT::num_strides == 1.
 */
template<int num_offsets, typename RT, typename ST>
__device__ inline static void prefill_swizzled_offsets(RT& dst, ST& src, uint32_t* swizzled_offsets) {
    static_assert(RT::rows == ST::rows, "register tile and shared tile must match rows");
    static_assert(RT::cols == ST::cols,  "register tile and shared tile must match cols");
    static_assert(num_offsets == RT::base_tile_num_strides, "number of offsets must match number of strides");

    using T2 = typename RT::dtype;
    using T  = typename base_types::packing<T2>::unpacked_type;
    using U  = typename ST::dtype;
    static_assert(std::is_same_v<T, U>, "register and shared dtypes must match");
    static_assert(sizeof(U) == 1, "INT8 utils.cpp expects a 1-byte dtype");
    static_assert(RT::base_tile_stride == 16, "INT8 utils.cpp expects base_tile_stride == 16 (rt_16x64)");

    // Per-lane bytes covered by one ds_read_b128 = 16 bytes.
    constexpr int bytes_per_read = RT::base_tile_stride * sizeof(U); // 16

    const int lane_row   = kittens::laneid() % RT::base_tile_rows;       // 0..15
    const int lane_group = kittens::laneid() / RT::base_tile_rows;       // 0..3

    uint32_t base_addr = reinterpret_cast<uintptr_t>(&src.data[0])
                        + lane_row * ST::underlying_cols * sizeof(U)
                        + lane_group * bytes_per_read;

    #pragma unroll
    for (int k = 0; k < num_offsets; k++) {
        // num_strides == 1 here, so this is a single iteration.
        uint32_t off = base_addr + k * RT::base_tile_elements_per_stride_group * sizeof(U);
        // st_16x128_s swizzle: offset ^= ((offset % (16*128)) >> 8) << 4
        off ^= (((off % (16 * 128)) >> 8) << 4);
        swizzled_offsets[k] = off;
    }
}

/**
 * @brief Issue one ds_read_b128 from shared into one base tile (register_row,
 * register_col, k-stride) of the register tile.
 */
template<int register_row, int register_col, int k, typename RT, typename ST>
__device__ inline static void load_one(RT& dst, ST& src, uint32_t* swizzled_offsets) {
    using U = typename ST::dtype;
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
 * @brief Single base-tile MFMA: D.tiles[n][m] = A.tiles[n][k] * B.tiles[m][k]^T + C.tiles[n][m].
 */
template<typename D, typename A, typename B, typename C>
__device__ inline void mma_ABt_one(D& d_mma, const A& a_mma, const B& b_mma, const C& c_mma, int n, int m, int k) {
    static_assert(D::rows == A::rows && D::cols == B::rows);
    static_assert(A::cols == B::cols);
    static_assert(D::rows == C::rows && D::cols == C::cols);
    static_assert(std::is_same_v<typename D::T, int>  && std::is_same_v<typename A::T, int8> &&
                  std::is_same_v<typename B::T, int8> && std::is_same_v<typename C::T, int>);

    mma_ABt_base(
        d_mma.tiles[n][m],
        a_mma.tiles[n][k],
        b_mma.tiles[m][k],
        c_mma.tiles[n][m]
    );
}
