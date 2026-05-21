/**
 * @file
 * @brief gfx1250 per-lane async global -> LDS transfers.
 *
 * The async path moves bytes directly from global memory to LDS via
 * `global_load_async_to_lds_b128` (and the multicast variant
 * `cluster_load_async_to_lds_b128`), bypassing the WGP$. Each participating
 * lane issues one 16 B request, so the warp covers `8 * N_THREADS`
 * elements per iteration. Completion lands in `ASYNCcnt`; drain via
 * `async::load_wait<N>` (or `sync::wait_async<N>`).
 *
 * This is a different hardware path from the descriptor-driven Tensor Data
 * Mover (see `tdm.cuh`): per-lane vs per-warp issue, a separate completion
 * counter, and stricter alignment rules for the direct-copy fast path
 * (128 B/256 B aligned base + at least 16 consecutive lanes per chunk;
 * see arch-direct-indirect-copies for the indirect fallback).
 */

#pragma once

#ifdef KITTENS_UDNA1

#include "global_to_shared.cuh"
#include "../../sync/barrier.cuh"

namespace kittens {
namespace async {

namespace detail {
/// @brief 16 B (`int4`) vector types tagged with the address spaces the
///        gfx1250 `*_load_async_to_lds_b128` builtins require.
using i32x4_gvec = int __attribute__((__vector_size__(16))) __attribute__((address_space(1)));
using i32x4_lvec = int __attribute__((__vector_size__(16))) __attribute__((address_space(3)));
} // namespace detail

/**
 * @brief Cooperative async global -> LDS tile copy.
 *
 * Lowers to `global_load_async_to_lds_b128`. Each lane issues one 16 B
 * transfer. Drain with `async::load_wait<N>()` before consuming the LDS
 * slab.
 *
 * @tparam Pad         LDS padding descriptor.
 * @tparam ROWS, COLS  Tile shape (elements).
 * @tparam N_THREADS   Number of threads participating in the load.
 * @param  lds_dst     16 B-aligned LDS pointer (typically `bf16*`).
 * @param  src         Global tile descriptor.
 * @param  idx         Tile coordinate inside `src` (tile-index units).
 * @param  row_stride  Element stride between rows in `src`.
 */
template<typename Pad = lds_nopad, int ROWS = 0, int COLS = 0,
         int N_THREADS = WARP_THREADS,
         typename T, ducks::gl::all GL, ducks::coord::tile COORD = coord<>>
__device__ inline void load(T* __restrict__ lds_dst, const GL& src, const COORD& idx,
                            int row_stride)
{
    static_assert(ROWS > 0 && COLS > 0, "ROWS and COLS must be specified");
    static_assert(sizeof(T) * 8 == 16, "async::load issues one b128 (16 B) per lane");
    constexpr int elems_per_load = 16 / sizeof(T);
    constexpr int total_elems    = ROWS * COLS;
    const int tid = threadIdx.x;
    const int gr_base = idx.r * ROWS;
    const int gc_base = idx.c * COLS;
    const T* base = src.raw_ptr
                  + (((int64_t(idx.b) * src.depth() + idx.d) * src.rows() + gr_base)
                     * src.cols() + gc_base);

    #pragma unroll
    for (int i = tid * elems_per_load; i < total_elems;
         i += N_THREADS * elems_per_load)
    {
        const int row = i / COLS;
        const int col = i % COLS;
        const int lds_flat = ::kittens::detail::subtile_flat<ROWS, COLS, 16, 32>(i);

        // The gfx1250 async-to-LDS builtins want address-space-qualified
        // pointers (AS(1) global, AS(3) LDS). `reinterpret_cast` cannot add
        // an address space, so route through `uintptr_t` + a C-style cast.
        uintptr_t g_uint = reinterpret_cast<uintptr_t>(base + row * row_stride + col);
        uintptr_t l_uint = reinterpret_cast<uintptr_t>(lds_dst + Pad::padded(lds_flat));
        auto* g_ptr = (detail::i32x4_gvec*)(g_uint);
        auto* l_ptr = (detail::i32x4_lvec*)(l_uint);

        __builtin_amdgcn_global_load_async_to_lds_b128(g_ptr, l_ptr, 0, 0);
    }
}

/// @brief Drain pending async transfers, leaving at most `N` in flight.
template<int N = 0>
__device__ __forceinline__ void load_wait() {
    sync::wait_async<N>();
}

/* ----------  CLUSTER-SCOPE (multicast)  ---------- */
//
// Multicast async loads broadcast the line to every workgroup in the
// cluster whose bit is set in the M0 mask. Unlike the TDM cluster path,
// the bit-16 `early_timeout` flag *is* honored here.

namespace cluster {

/**
 * @brief Multicast async global -> LDS tile copy.
 *
 * Lowers to `cluster_load_async_to_lds_b128` with the supplied M0 mask.
 *
 * @param mask The mask returned by `kittens::cluster::mask(wgs, early_timeout)`.
 *             The bit-16 `early_timeout` flag reaches the hardware on this
 *             path (it is dropped by the TDM equivalent).
 */
template<typename Pad = lds_nopad, int ROWS = 0, int COLS = 0,
         int N_THREADS = WARP_THREADS,
         typename T, ducks::gl::all GL, ducks::coord::tile COORD = coord<>>
__device__ inline void load(T* __restrict__ lds_dst, const GL& src, const COORD& idx,
                            uint32_t mask, int row_stride)
{
    static_assert(ROWS > 0 && COLS > 0, "ROWS and COLS must be specified");
    static_assert(sizeof(T) * 8 == 16, "async::cluster::load issues one b128 (16 B) per lane");
    constexpr int elems_per_load = 16 / sizeof(T);
    constexpr int total_elems    = ROWS * COLS;
    const int tid = threadIdx.x;
    const int gr_base = idx.r * ROWS;
    const int gc_base = idx.c * COLS;
    const T* base = src.raw_ptr
                  + (((int64_t(idx.b) * src.depth() + idx.d) * src.rows() + gr_base)
                     * src.cols() + gc_base);

    #pragma unroll
    for (int i = tid * elems_per_load; i < total_elems;
         i += N_THREADS * elems_per_load)
    {
        const int row = i / COLS;
        const int col = i % COLS;
        const int lds_flat = ::kittens::detail::subtile_flat<ROWS, COLS, 16, 32>(i);

        uintptr_t g_uint = reinterpret_cast<uintptr_t>(base + row * row_stride + col);
        uintptr_t l_uint = reinterpret_cast<uintptr_t>(lds_dst + Pad::padded(lds_flat));
        auto* g_ptr = (::kittens::async::detail::i32x4_gvec*)(g_uint);
        auto* l_ptr = (::kittens::async::detail::i32x4_lvec*)(l_uint);

        __builtin_amdgcn_cluster_load_async_to_lds_b128(
            g_ptr, l_ptr, 0, 0, static_cast<int>(mask));
    }
}

} // namespace cluster

} // namespace async
} // namespace kittens

#endif // KITTENS_UDNA1
