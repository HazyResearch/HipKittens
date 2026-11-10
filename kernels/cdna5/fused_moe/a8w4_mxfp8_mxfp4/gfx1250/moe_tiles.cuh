#pragma once
// MoE-specific FLAT shared-tile shape + the preshuffled-weight load. The activation (A) tile uses the
// library shape (st_16x32_padded); only the weight (B) container is kept here, because B is consumed
// pre-shuffled into contiguous WMMA-order 1024 B blocks -- a flat/identity layout that a padded or
// swizzled library shape would not match (see PRESHUFFLE docs). It plugs into the library via
// ducks::st_shape::is_ext_st_shape_v so kittens::st<> and ducks::st_shape::all accept it.
//
// Requires kittens.cuh to be included first (uses ducks::*, fp4e2m1_4, laneid()).

namespace kittens {
namespace ducks {
namespace st_shape {

// FLAT (identity) 1-byte container for the preshuffled B weight. The offline shuffle already lays the
// bytes out in WMMA-consumption order as contiguous 1024 B blocks, so the LDS tile is a plain contiguous
// buffer (no padding/swizzle) that load_b_preshuffled reads address-generation-free.
template<int ROWS, int COLS>
struct st_flat {
    static constexpr int rows = ROWS, cols = COLS;
    template<typename _T> static constexpr int bytes_per_thread() {
        if constexpr (sizeof(_T) == 1) { return 16; } else { static_assert(false, "st_flat is 1-byte"); } }
    template<typename _T> __device__ __host__ __forceinline__ static constexpr uint32_t swizzle(int2 coord) {
        return sizeof(_T) * (coord.x * cols + coord.y); }
};

// Register st_flat with the library shape concept. This extension hook is what keeps the preshuffle
// container local to the fused-MoE kernel instead of naming it in the core library.
template<int R, int C> inline constexpr bool is_ext_st_shape_v<st_flat<R, C>> = true;

} // namespace st_shape
} // namespace ducks

// Load a preshuffled MXFP4 B block from a FLAT 1-byte LDS container into a fp4 register tile.
// Lane L's 32 packed bytes sit contiguously at src.data + block_byte_off + L*32 (no row/half
// address-gen and no padding). block_byte_off selects the K-step block within a contiguous multi-block
// (KSUB x 1024B) tile. Bespoke MoE weight-path helper, deliberately NOT a general `load` overload.
template<ducks::rt::row_layout RT, ducks::st::all ST>
    requires (std::is_same_v<typename RT::shape, ducks::rt_shape::rt_16x128>
              && std::is_same_v<typename RT::dtype, fp4e2m1_4>)
__device__ inline void load_b_preshuffled(RT& dst, const ST& src, int block_byte_off = 0) {
    static_assert(sizeof(typename ST::dtype) == 1, "preshuffle block is a 1-byte contiguous container");
    const int L = kittens::laneid();
    typedef float f4v __attribute__((ext_vector_type(4)));
    using lds_f4 = f4v __attribute__((address_space(3)));
    const lds_f4* sptr = reinterpret_cast<const lds_f4*>(
        reinterpret_cast<uintptr_t>(src.data + block_byte_off + L * 32));

    // dst.tiles[0][0].data is fp4e2m1_4 (2-byte). Write each 16B b128 with memcpy so the store does not
    // assume 4-byte alignment (a uint32_t* store would be misaligned/strict-aliasing UB for fp4).
    uint8_t* d = reinterpret_cast<uint8_t*>(&dst.tiles[0][0].data[0]);
    #pragma unroll
    for (int k = 0; k < 2; k++) {
        f4v v = sptr[k];
        __builtin_memcpy(d + k * 16, &v, 16);
    }
}

} // namespace kittens
