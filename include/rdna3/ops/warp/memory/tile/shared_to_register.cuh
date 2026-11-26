/**
 * @file
 * @brief Functions for transferring data directly between shared memory and registers and back.
 */

#pragma once

#include <type_traits>

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"
#include "../util/util.cuh"

namespace kittens {

/*
 * gfx11 lane -> address mapping.  See rt_base_coord() in
 * types/register/rt_base.cuh for the layouts these derive from.  In short, for
 * base tile (i,j) and lane l:
 *
 *   row layout, bf16/half operand : row l%16, cols 0..15          (32 B contiguous)
 *   col layout, bf16/half operand : col l%16, rows 0..15          (16 strided elements)
 *   col layout, f32 accumulator   : col l%16, rows {2e + l/16}    (8 strided elements)
 *   row layout, f32 accumulator   : row l%16, cols {2e + l/16}    (8 strided elements)
 *
 * Only the first is a vectorizable access, and only when the shared tile holds
 * the same 2-byte type.  That case is the A operand of a GEMM and is worth a
 * fast path; the rest fall back to elementwise accesses, which is what the CDNA
 * tree already does for its column-major path.
 *
 * The fast path issues two 16-byte reads rather than one 32-byte read.  The XOR
 * swizzle in st::idx() permutes 16-byte granules, so 16 contiguous bytes survive
 * it but 32 do not: the two halves of a row may land in either order.  Two
 * separate idx() calls handle that, and two ds_read_b128 is the minimum
 * instruction count a 32-byte-per-lane read can have.  See the swizzle note in
 * types/shared/st.cuh for why 16 bytes is also the granule that keeps the access
 * bank-conflict-free.
 *
 * Note that lanes l and l+16 of an operand tile issue *identical* addresses.
 * That is the 2x mirroring WMMA requires, and LDS broadcasts the matching
 * addresses rather than serializing them, so it costs no extra cycles.
 */

/**
 * @brief Whether load()/store() for this pair take the vectorized ds_read_b128 path.
 */
template<ducks::rt::all RT, ducks::st::all ST>
static constexpr bool lds_vectorizable =
    std::is_same_v<typename RT::layout, ducks::rt_layout::row>
    && rt_base<typename base_types::packing<typename RT::dtype>::unpacked_type,
               typename RT::layout>::element_stride == 1
    && std::is_same_v<typename base_types::packing<typename RT::dtype>::unpacked_type,
                      typename ST::dtype>
    && sizeof(typename ST::dtype) == 2;

/**
 * @brief How many LDS instructions load()/load_async() issues for this pair.
 *
 * Callers pipelining loads against math need this to size their s_waitcnt: see
 * lds_wait() in the memory utilities. Zero means the pair does not take the
 * vectorized path, in which case load_async() is not available.
 */
template<ducks::rt::all RT, ducks::st::all ST>
static constexpr int lds_loads = lds_vectorizable<RT, ST> ? RT::height * RT::width * 2 : 0;

/**
 * @brief Load data from a shared tile into a register tile.
 *
 * @tparam wait Whether to retire the reads before returning. Pass false to keep
 *              them in flight and wait later with lds_wait<>(), which is how a
 *              caller overlaps the reads for one K-slice with the math of the
 *              previous one. Only the vectorized path can do this -- the
 *              elementwise fallback goes through the compiler, which inserts
 *              its own waits, so it is a static_assert rather than a silent
 *              downgrade.
 * @tparam RT The register tile type
 * @tparam ST The shared tile type
 * @param dst[out] The destination register tile.
 * @param src[in]  The source shared tile.
 */
template<bool wait=true, ducks::rt::all RT, ducks::st::all ST>
__device__ inline static void load(RT &dst, const ST &src) {
    static_assert(wait || lds_vectorizable<RT, ST>,
                  "load<false> requires the vectorized path; check lds_loads<RT,ST> != 0");

    static_assert(RT::height == ST::height, "register tile and shared tile must match height");
    static_assert(RT::width  == ST::width,  "register tile and shared tile must match width");

    using T2 = typename RT::dtype;
    using T  = typename base_types::packing<T2>::unpacked_type;
    using U  = typename ST::dtype;

    using L    = typename RT::layout;
    using base = rt_base<T, L>;

    constexpr bool is_row = std::is_same_v<L, ducks::rt_layout::row>;
    constexpr int  E      = base::elements_per_thread;

    // A whole 16-element row, same width in LDS as in registers: 2x ds_read_b128.
    constexpr bool vectorizable = is_row && base::element_stride == 1
                                         && std::is_same_v<T, U> && sizeof(U) == 2;

    const int lane = laneid();
    const int l16  = lane & 15;
    const uint32_t src_ptr = (uint32_t)(uintptr_t)&src.data[0];

    #pragma unroll
    for(int i = 0; i < RT::height; i++) {
        #pragma unroll
        for(int j = 0; j < RT::width; j++) {
            const int row_base = i*base::tile_size_row;
            const int col_base = j*base::tile_size_col;
            if constexpr (vectorizable) {
                // The 16-byte alignment these require holds: tiles are
                // KITTENS_DEFAULT_ALIGN'd, the byte offset of an 8-element
                // column boundary is a multiple of 16, and the swizzle only
                // moves bits 4-6.  The register side goes through memcpy
                // because `data` is only 4-byte aligned as an array.
                #pragma unroll
                for(int b = 0; b < 2; b++) {
                    const uint32_t p = src.idx(src_ptr, {row_base + l16, col_base + 8*b});
                    const float4 v = load_shared_vec4_async(p);
                    __builtin_memcpy((void*)&dst.tiles[i][j].data[4*b], &v, sizeof(v));
                }
            }
            else {
                #pragma unroll
                for(int e = 0; e < E; e++) {
                    const int2 c = rt_base_coord<T, L>(e, lane);
                    const T val = base_types::convertor<T, U>::convert(
                        src[{row_base + c.x, col_base + c.y}]);
                    if (e & 1) dst.tiles[i][j].data[e>>1].y = val;
                    else       dst.tiles[i][j].data[e>>1].x = val;
                }
            }
        }
    }
    // One wait for the whole tile: the reads above were issued back to back so
    // that their latencies overlap, and the compiler cannot insert this itself
    // for an inline-asm ds_read.
    if constexpr (vectorizable && wait) asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
}

/**
 * @brief Issue a tile's LDS reads without waiting for them. See load<false>.
 *
 * The caller owes this tile a `lds_wait_for<N>(dst)` -- not a bare `lds_wait`.
 * The reads land through an inline-asm output operand, so nothing in the IR
 * ties them to the wait, and a consuming WMMA will be scheduled above it. See
 * `lds_bind` in memory/util/util.cuh.
 */
template<ducks::rt::all RT, ducks::st::all ST>
__device__ inline static void load_async(RT &dst, const ST &src) { load<false>(dst, src); }


/**
 * @brief Store data into a shared tile from a register tile.
 *
 * @tparam RT The register tile type
 * @tparam ST The shared tile type
 * @param dst[out] The destination shared tile.
 * @param src[in]  The source register tile.
 */
template<ducks::rt::all RT, ducks::st::all ST>
__device__ inline static void store(ST &dst, const RT &src) {

    static_assert(RT::height == ST::height, "register tile and shared tile must match height");
    static_assert(RT::width  == ST::width,  "register tile and shared tile must match width");

    using T2 = typename RT::dtype;
    using T  = typename base_types::packing<T2>::unpacked_type;
    using U  = typename ST::dtype;

    using L    = typename RT::layout;
    using base = rt_base<T, L>;

    constexpr bool is_row = std::is_same_v<L, ducks::rt_layout::row>;
    constexpr int  E      = base::elements_per_thread;

    constexpr bool vectorizable = is_row && base::element_stride == 1
                                         && std::is_same_v<T, U> && sizeof(U) == 2;

    const int lane = laneid();
    const int l16  = lane & 15;
    const uint32_t dst_ptr = (uint32_t)(uintptr_t)&dst.data[0];

    #pragma unroll
    for(int i = 0; i < RT::height; i++) {
        #pragma unroll
        for(int j = 0; j < RT::width; j++) {
            const int row_base = i*base::tile_size_row;
            const int col_base = j*base::tile_size_col;
            if constexpr (vectorizable) {
                // Lanes l and l+16 write the same bytes with the same values --
                // the mirroring invariant makes this benign, and skipping the
                // upper half would cost a branch for no bandwidth (LDS writes to
                // a matching address collapse the same way reads broadcast).
                #pragma unroll
                for(int b = 0; b < 2; b++) {
                    const uint32_t p = dst.idx(dst_ptr, {row_base + l16, col_base + 8*b});
                    float4 v;
                    __builtin_memcpy(&v, (const void*)&src.tiles[i][j].data[4*b], sizeof(v));
                    store_shared_vec4(p, v);
                }
            }
            else {
                #pragma unroll
                for(int e = 0; e < E; e++) {
                    const int2 c = rt_base_coord<T, L>(e, lane);
                    const T val = (e & 1) ? src.tiles[i][j].data[e>>1].y
                                          : src.tiles[i][j].data[e>>1].x;
                    dst[{row_base + c.x, col_base + c.y}] =
                        base_types::convertor<U, T>::convert(val);
                }
            }
        }
    }
    // Retire the writes before the caller's barrier, and before it can reuse
    // the VGPRs the asm reads from -- neither is something the compiler tracks
    // across an inline-asm ds_write.
    if constexpr (vectorizable) asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
}
}
