/**
 * @file
 * @brief gfx1250 Tensor Data Mover (TDM) tile transfers.
 *
 * The TDM is a per-SIMD-pair DMA engine that moves rectangular tiles
 * between global memory and LDS under descriptor control. A single
 * `tensor_load_to_lds` instruction issued by one wave delivers the entire
 * tile; the descriptor (D#) is built on-device in SGPRs at the call site
 * and encodes the tile shape, source-tensor extents (for HW out-of-bounds
 * checking), row stride, optional LDS padding, optional multicast mask,
 * and -- on the auto-arrive overload below -- the address of an LDS
 * semaphore the TDM unit will arrive at when the transfer retires.
 *
 * Completion is tracked by `TENSORcnt`. Drain with `tdm::load_async_wait<N>`
 * (or, equivalently, `sync::wait_tensor<N>`).
 *
 * The TDM unit ignores `EXEC` and uses no VGPR operands. Issuance is
 * arbitrated per SIMD-pair, so kernels typically nominate one wave (or one
 * lane via `if (laneid() == 0)`) to issue each transfer.
 */

#pragma once

#ifdef KITTENS_UDNA1

#include "global_to_shared.cuh"
#include "../../sync/barrier.cuh"

namespace kittens {
namespace tdm {

namespace detail {

using v4u32 = unsigned int __attribute__((ext_vector_type(4)));
using v8u32 = unsigned int __attribute__((ext_vector_type(8)));

/**
 * @brief Build the 12-DWord TDM descriptor (groups 0 + 1) for a 2D tile.
 *
 * gfx1250 D# layout for 2D transfers (groups 2 and 3 are zero):
 *   group 0 [128 b]: count, lds_addr, global_addr, type
 *   group 1 [256 b]: multicast_mask[15:0], data_size[17:16],
 *                    atomic_barrier_enable[18], iterate_enable[19],
 *                    pad_enable[20], pad_interval[24:22], pad_amount[31:25],
 *                    atomic_barrier_address[w1.15:0],
 *                    tensor_dim0 (32 b), tensor_dim1 (32 b),
 *                    tile_dim0 (16 b), tile_dim1 (16 b),
 *                    tensor_dim0_stride (48 b).
 *
 * `bar_lds_addr` is the LDS byte address of a `kittens::semaphore` cell.
 * When non-zero the helper sets `atomic_barrier_enable` and routes the
 * address into `w1[15:0]`; the TDM unit then emits a
 * `DS_ATOMIC_ASYNC_BARRIER_ARRIVE_B64` on the cell when the transfer
 * retires. Pass 0 for the no-barrier path.
 *
 * Field positions cross-checked against SP3 (the authoritative AMD spec)
 * and the Triton AMD lowering in `TDMUtility.cpp` lines 224-264.
 */
template<typename Pad, int ROWS, int COLS, typename T>
__device__ __forceinline__ void build_d_2d(
    v4u32& g0, v8u32& g1,
    const T* base, T* lds_dst,
    int tensor_rows, int tensor_cols, int row_stride,
    uint32_t cluster_mask, uint32_t bar_lds_addr)
{
    // ---- Group 0: count, lds_addr, global_addr, type ----
    const uint32_t lds_addr = static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(lds_dst));
    const uint64_t gaddr    = reinterpret_cast<uint64_t>(base);

    g0[0] = 1u;
    g0[1] = lds_addr;
    g0[2] = static_cast<uint32_t>(gaddr);
    g0[3] = (static_cast<uint32_t>(gaddr >> 32) & 0x01FFFFFFu) | (2u << 30);

    // ---- Group 1: data_size, padding, dims, stride, optional barrier ----
    // data_size is encoded as log2(bytes_per_element).
    constexpr uint32_t data_size_enc = (sizeof(T) == 1) ? 0
                                     : (sizeof(T) == 2) ? 1
                                     : (sizeof(T) == 4) ? 2
                                     : 3;
    constexpr uint32_t pad_enable   = (Pad::interval > 0) ? 1u : 0u;
    constexpr uint32_t pad_int_enc  = (Pad::interval > 0)
        ? ( __builtin_ctz(Pad::interval * sizeof(T) / 4) ) : 0;
    constexpr uint32_t pad_amt_enc  = (Pad::amount > 0)
        ? ( (Pad::amount * sizeof(T) / 4) - 1 ) : 0;
    const uint32_t atomic_bar_enable = (bar_lds_addr != 0) ? (1u << 18) : 0u;

    uint32_t w0 = (data_size_enc << 16)
                | (pad_enable    << 20)
                |  atomic_bar_enable
                | (pad_int_enc   << 22)
                | (pad_amt_enc   << 25)
                | (cluster_mask  & 0xFFFFu);

    const uint32_t tdim0    = static_cast<uint32_t>(tensor_cols);
    const uint32_t tdim1    = static_cast<uint32_t>(tensor_rows);
    const uint32_t tiledim0 = static_cast<uint32_t>(COLS);
    const uint32_t tiledim1 = static_cast<uint32_t>(ROWS);

    uint32_t w1 = (bar_lds_addr & 0xFFFFu) | (tdim0 << 16);
    uint32_t w2 = (tdim0 >> 16) | (tdim1 << 16);
    uint32_t w3 = (tdim1 >> 16) | (tiledim0 << 16);
    uint32_t w4 = tiledim1;

    const uint64_t stride0 = static_cast<uint64_t>(
        static_cast<uint32_t>(row_stride * sizeof(T)));
    uint32_t w5 = static_cast<uint32_t>(stride0);
    uint32_t w6 = static_cast<uint32_t>(stride0 >> 32);
    uint32_t w7 = 0;

    g1[0] = w0; g1[1] = w1; g1[2] = w2; g1[3] = w3;
    g1[4] = w4; g1[5] = w5; g1[6] = w6; g1[7] = w7;
}

template<typename T, ducks::gl::all GL, ducks::coord::tile COORD>
__device__ __forceinline__ const T* tile_base(const GL& src, const COORD& idx,
                                              int rows, int cols)
{
    const int gr_base = idx.r * rows;
    const int gc_base = idx.c * cols;
    return src.raw_ptr
         + (((int64_t(idx.b) * src.depth() + idx.d) * src.rows() + gr_base)
            * src.cols() + gc_base);
}

} // namespace detail

/* ----------  TDM TILE LOAD (G -> LDS)  ---------- */

/**
 * @brief Issue a TDM tile load; drain via `tdm::load_async_wait<N>`.
 *
 * Builds the D# in-place, issues `tensor_load_to_lds`, and returns. The
 * transfer counts against `TENSORcnt` and is drained by `wait_tensor<N>`
 * (or the alias `load_async_wait<N>` / `store_async_wait<N>`).
 *
 * @tparam Pad         LDS padding descriptor (`lds_nopad` or `lds_padded<I, A>`).
 * @tparam ROWS, COLS  Tile shape in elements.
 * @param  lds_dst     16 B-aligned LDS pointer.
 * @param  src         Global tile descriptor.
 * @param  idx         Tile coordinate inside `src` (tile-index units).
 * @param  tensor_rows Source tensor row extent. The TDM unit clamps reads
 *                    against this bound (OOB reads return zero); pass
 *                    `min(actual_rows, ROWS)` for edge tiles.
 * @param  tensor_cols Source tensor column extent. Same OOB semantics.
 * @param  row_stride  Element stride between rows in the source tensor.
 */
template<typename Pad = lds_nopad, int ROWS = 0, int COLS = 0,
         typename T, ducks::gl::all GL, ducks::coord::tile COORD = coord<>>
__device__ inline void load_async(
    T* __restrict__ lds_dst, const GL& src, const COORD& idx,
    int tensor_rows, int tensor_cols, int row_stride)
{
    static_assert(ROWS > 0 && COLS > 0, "ROWS and COLS must be specified");
    const T* base = detail::tile_base<T, GL, COORD>(src, idx, ROWS, COLS);

    detail::v4u32 g0;
    detail::v8u32 g1;
    detail::build_d_2d<Pad, ROWS, COLS, T>(
        g0, g1, base, lds_dst, tensor_rows, tensor_cols, row_stride,
        /*cluster_mask=*/ 0, /*bar_lds_addr=*/ 0);

    detail::v4u32 g2 = {0, 0, 0, 0};
    detail::v4u32 g3 = {0, 0, 0, 0};
    __builtin_amdgcn_tensor_load_to_lds(g0, g1, g2, g3, 0);
}

/**
 * @brief Issue a TDM tile load that auto-arrives at `sem` on completion.
 *
 * Sets `atomic_barrier_enable` in the D# so the TDM unit emits a
 * `DS_ATOMIC_ASYNC_BARRIER_ARRIVE_B64` on `sem` after the transfer
 * retires. The consumer waits via `kittens::wait(sem, phase)` instead of
 * draining the global `TENSORcnt`, leaving unrelated TDM transfers in
 * flight.
 *
 * The semaphore must be primed via `init_semaphore(sem, arrivals)` before
 * the first call referencing it. `arrivals` is the number of
 * `load_async` invocations that target this semaphore per phase.
 *
 * @note Runtime support for the D# auto-arrive path varies. Triton's AMD
 *       lowering does not rely on it -- it follows the no-bar `load_async`
 *       with an explicit `wait_tensor<0>` + `arrive(sem)`. The
 *       explicit-arrive pattern is portable; this overload is the
 *       lower-overhead alternative on runtimes that honor the auto-arrive.
 *
 * @param sem  LDS-resident semaphore. The cell address must fit in 16 bits
 *             (place it inside an LDS segment).
 */
template<typename Pad = lds_nopad, int ROWS = 0, int COLS = 0,
         typename T, ducks::gl::all GL, ducks::coord::tile COORD = coord<>>
__device__ inline void load_async(
    T* __restrict__ lds_dst, const GL& src, const COORD& idx,
    semaphore& sem,
    int tensor_rows, int tensor_cols, int row_stride)
{
    static_assert(ROWS > 0 && COLS > 0, "ROWS and COLS must be specified");
    const T* base = detail::tile_base<T, GL, COORD>(src, idx, ROWS, COLS);
    const uint32_t bar_lds_addr = static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(&sem.state));

    detail::v4u32 g0;
    detail::v8u32 g1;
    detail::build_d_2d<Pad, ROWS, COLS, T>(
        g0, g1, base, lds_dst, tensor_rows, tensor_cols, row_stride,
        /*cluster_mask=*/ 0, bar_lds_addr);

    detail::v4u32 g2 = {0, 0, 0, 0};
    detail::v4u32 g3 = {0, 0, 0, 0};
    __builtin_amdgcn_tensor_load_to_lds(g0, g1, g2, g3, 0);
}

/* ----------  COMPLETION HELPERS  ---------- */
//
// Both load and store ops increment the same `TENSORcnt` counter on issue,
// so a single drain primitive handles either direction. The load- and
// store-named helpers are convenience aliases.

/// @brief Drain pending TDM transfers, leaving at most `N` in flight.
template<int N = 0>
__device__ __forceinline__ void load_async_wait() {
    sync::wait_tensor<N>();
}

/// @brief Drain pending TDM transfers, leaving at most `N` in flight.
template<int N = 0>
__device__ __forceinline__ void store_async_wait() {
    sync::wait_tensor<N>();
}

/* ----------  L2 PREFETCH  ---------- */

/**
 * @brief Cooperative L2 prefetch for an upcoming tile.
 *
 * Lowers to per-lane `__builtin_amdgcn_global_prefetch` instructions
 * covering the tile. The hint = 0 selects the default cache policy.
 */
template<int ROWS = 0, int COLS = 0, int N_THREADS = WARP_THREADS,
         typename T, ducks::gl::all GL, ducks::coord::tile COORD = coord<>>
__device__ inline void prefetch(const GL& src, const COORD& idx, int row_stride)
{
    static_assert(ROWS > 0 && COLS > 0, "ROWS and COLS must be specified");
    constexpr int elems_per_pf = 16 / sizeof(T);
    constexpr int total_elems  = ROWS * COLS;
    const int tid = threadIdx.x;
    const T* base = detail::tile_base<T, GL, COORD>(src, idx, ROWS, COLS);

    #pragma unroll
    for (int i = tid * elems_per_pf; i < total_elems;
         i += N_THREADS * elems_per_pf)
    {
        const int row = i / COLS;
        const int col = i % COLS;
        const T* addr = base + row * row_stride + col;
        __builtin_amdgcn_global_prefetch(
            (const void __attribute__((address_space(1)))*)addr, 0);
    }
}

/* ----------  CLUSTER-SCOPE TDM (multicast)  ---------- */
//
// When the cluster mask is non-zero, the TDM unit decomposes the transfer
// into `CLUSTER_LOAD_ASYNC_*` micro-ops, broadcasting the line into every
// workgroup whose bit is set in the mask. Only the low 16 bits of the
// mask reach the TDM unit; the `early_timeout` bit (bit 16) is honored by
// the per-lane async path but not by TDM.

namespace cluster {

/**
 * @brief Multicast TDM tile load that auto-arrives at `sem`.
 *
 * @param sem    LDS-resident semaphore to receive the auto-arrive.
 * @param mask   16-bit workgroup-in-cluster mask (`kittens::cluster::mask`).
 */
template<typename Pad = lds_nopad, int ROWS = 0, int COLS = 0,
         typename T, ducks::gl::all GL, ducks::coord::tile COORD = coord<>>
__device__ inline void load_async(
    T* __restrict__ lds_dst, const GL& src, const COORD& idx,
    semaphore& sem, uint32_t mask,
    int tensor_rows, int tensor_cols, int row_stride)
{
    static_assert(ROWS > 0 && COLS > 0, "ROWS and COLS must be specified");
    const T* base = detail::tile_base<T, GL, COORD>(src, idx, ROWS, COLS);
    const uint32_t bar_lds_addr = static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(&sem.state));

    detail::v4u32 g0;
    detail::v8u32 g1;
    detail::build_d_2d<Pad, ROWS, COLS, T>(
        g0, g1, base, lds_dst, tensor_rows, tensor_cols, row_stride,
        mask, bar_lds_addr);

    detail::v4u32 g2 = {0, 0, 0, 0};
    detail::v4u32 g3 = {0, 0, 0, 0};
    __builtin_amdgcn_tensor_load_to_lds(g0, g1, g2, g3, 0);
}

/**
 * @brief Multicast TDM tile load; drain via `tdm::load_async_wait<N>`.
 */
template<typename Pad = lds_nopad, int ROWS = 0, int COLS = 0,
         typename T, ducks::gl::all GL, ducks::coord::tile COORD = coord<>>
__device__ inline void load_async(
    T* __restrict__ lds_dst, const GL& src, const COORD& idx,
    uint32_t mask,
    int tensor_rows, int tensor_cols, int row_stride)
{
    static_assert(ROWS > 0 && COLS > 0, "ROWS and COLS must be specified");
    const T* base = detail::tile_base<T, GL, COORD>(src, idx, ROWS, COLS);

    detail::v4u32 g0;
    detail::v8u32 g1;
    detail::build_d_2d<Pad, ROWS, COLS, T>(
        g0, g1, base, lds_dst, tensor_rows, tensor_cols, row_stride,
        mask, /*bar_lds_addr=*/ 0);

    detail::v4u32 g2 = {0, 0, 0, 0};
    detail::v4u32 g3 = {0, 0, 0, 0};
    __builtin_amdgcn_tensor_load_to_lds(g0, g1, g2, g3, 0);
}

} // namespace cluster

} // namespace tdm
} // namespace kittens

#endif // KITTENS_UDNA1
