/**
 * @file
 * @brief gfx1250 Tensor Data Mover (TDM) tile transfers.
 *
 * The TDM is a per-SIMD-pair DMA engine that moves rectangular tiles
 * between global memory and LDS under descriptor control. A single
 * `tensor_load_to_lds` / `tensor_store_from_lds` instruction issued by
 * one wave delivers the entire tile. The descriptor (D#) is built
 * on-device in SGPRs at the call site; this header exposes:
 *
 *   - `tdm::tile<int... DIMS>`            -- compile-time tile shape (1D-5D)
 *   - `tdm::gather_tile<int COLS>`        -- compile-time tile shape for
 *                                            row-indexed gather (2D only)
 *   - `tdm::load_async` / `store_async`   -- N-dimensional bulk copy
 *   - `tdm::gather_async`/`scatter_async` -- row-indexed 2D bulk copy
 *   - `tdm::cluster::load_async`          -- multicast load variant
 *   - `tdm::prefetch`                     -- L2 prefetch covering a tile
 *   - `tdm::load_async_wait<N>` / `tdm::store_async_wait<N>` -- drains
 *
 * Each `load_async` / `store_async` / `gather_async` / `scatter_async`
 * has two overloads: a no-semaphore variant (caller drains via
 * `load_async_wait<N>` / `store_async_wait<N>`) and a `semaphore&`
 * variant that sets `atomic_barrier_enable` in the D# so the TDM unit
 * arrives at the cell when the transfer retires. The semaphore overload
 * is the fine-grained ordering path; the no-bar overload is the
 * fire-and-drain path.
 *
 * The D# layout matches SP3 §4.10 and the AMD compiler-team annotation
 * in Triton's `TDMUtility.cpp`:
 *
 *   group0 (4 DWord)   pred, gather-enable, gather-index-size, lds_addr,
 *                      global_addr, type
 *   group1 (8 DWord)   multicast_mask, data_size, atomic_barrier_*,
 *                      pad_*, tensor_dim0/1, tile_dim0/1/2,
 *                      tensor_stride0/1
 *   group2 (4 DWord)   tensor_dim2/3, tensor_stride2, tile_dim3
 *                      -- or row_indices_0..7 (u16) / 0..3 (u32) in
 *                         gather mode
 *   group3 (4 DWord)   tensor_stride3, tensor_dim4, tile_dim4
 *                      -- or row_indices_8..15 (u16) / 4..7 (u32) in
 *                         gather mode
 *
 * For ≤2D non-gather, only groups 0+1 (12 DWord / 48 B) are filled.
 * 3D adds group 2 (16 DWord / 64 B); 4D-5D adds group 3 (20 DWord / 80 B).
 */

#pragma once

#ifdef KITTENS_UDNA1

#include "global_to_shared.cuh"
#include "../../sync/barrier.cuh"
#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace kittens {
namespace tdm {

/* ----------  COMPILE-TIME TILE SHAPES  ---------- */

/**
 * @brief N-dimensional tile shape (1D through 5D).
 *
 * The pack `DIMS...` is innermost-last (matches the kernel-side reading
 * order `{batch, depth, row, col}`). At descriptor-build time these are
 * reversed into the TDM's innermost-first numbering.
 */
template<int... DIMS>
struct tile {
    static constexpr int rank = sizeof...(DIMS);
    static_assert(rank >= 1 && rank <= 5, "TDM supports 1D-5D tiles");
    static constexpr std::array<int, rank> dims{DIMS...};

    using extents = std::array<int32_t, rank>;
    using strides = std::array<int32_t, rank == 0 ? 0 : rank - 1>;
};

/**
 * @brief Tile shape for row-indexed gather / scatter.
 *
 * The row count is HW-fixed at 16 (when indices are `uint16_t`) or 8
 * (when indices are `uint32_t`); only the column count is a compile-time
 * choice.
 */
template<int COLS>
struct gather_tile {
    static_assert(COLS > 0, "gather_tile requires COLS > 0");
    static constexpr int cols = COLS;
};

/* ----------  INTERNAL HELPERS  ---------- */

namespace detail {

using v4u32 = unsigned int __attribute__((ext_vector_type(4)));
using v8u32 = unsigned int __attribute__((ext_vector_type(8)));

/**
 * @brief Compute the base global pointer for a tile coordinate.
 *
 * Maps `{b, d, r, c}` tile-index coordinates to an element pointer at the
 * tile's top-left corner. The two trailing dims are scaled by the tile's
 * trailing two extents; the leading dims are passed straight through.
 */
template<int TILE_R, int TILE_C, typename T, ducks::gl::all GL,
         ducks::coord::tile COORD>
__device__ __forceinline__ const T* tile_base_2d(
    const GL& src, const COORD& idx)
{
    const int gr_base = idx.r * TILE_R;
    const int gc_base = idx.c * TILE_C;
    return src.raw_ptr
         + (((int64_t(idx.b) * src.depth() + idx.d) * src.rows() + gr_base)
            * src.cols() + gc_base);
}

/* --- group0 builder (shared by N-D and gather modes) --- */

template<typename T>
__device__ __forceinline__ void fill_group0(
    v4u32& g0, const T* base, void* lds_dst,
    bool gather_enable, bool gather_idx_32bit)
{
    const uint32_t lds_addr = static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(lds_dst));
    const uint64_t gaddr = reinterpret_cast<uint64_t>(base);

    // DWord 0: bits 1..0 = pred (must be non-zero for the op to execute),
    //          bit 30 = gather index size (0=u16, 1=u32),
    //          bit 31 = gather enable (0=normal, 1=gather/scatter mode).
    g0[0] = 1u   // pred = 1
          | (gather_enable    ? (1u << 31) : 0u)
          | (gather_idx_32bit ? (1u << 30) : 0u);
    g0[1] = lds_addr;
    g0[2] = static_cast<uint32_t>(gaddr);
    // DWord 3: bits 24..0 = global_addr[56:32]; bits 127..126 (i.e. dword3 bits 31..30) = type = 0x2.
    g0[3] = (static_cast<uint32_t>(gaddr >> 32) & 0x01FFFFFFu) | (2u << 30);
}

/* --- group1 builder (N-D, non-gather) ---
 *
 * Always fills the data_size / pad / barrier / multicast control word,
 * the two innermost tensor dims, the two innermost tile dims, and the
 * innermost stride. tile_dim2 in DWord 4[31:16] is filled when rank >= 3.
 * stride1 in DWords 6-7 is filled when rank >= 3.
 */
template<typename Tile, typename Pad, typename T>
__device__ __forceinline__ void fill_group1_ctrl(
    v8u32& g1, uint32_t cluster_mask, uint32_t bar_lds_addr)
{
    constexpr uint32_t data_size_enc = (sizeof(T) == 1) ? 0
                                     : (sizeof(T) == 2) ? 1
                                     : (sizeof(T) == 4) ? 2
                                     : 3;
    constexpr uint32_t pad_enable = (Pad::interval > 0) ? 1u : 0u;
    // pad_interval is encoded as log2(interval_in_dwords) - 1 per SP3.
    // For lds_padded<128, 8> with bf16: interval_in_dwords = 64,
    //   encoded = log2(64) - 1 = 5.
    constexpr uint32_t pad_int_enc = (Pad::interval > 0)
        ? __builtin_ctz(Pad::interval * sizeof(T) / 4) - 1 : 0;
    // pad_amount is encoded as amount_in_dwords - 1.
    constexpr uint32_t pad_amt_enc = (Pad::amount > 0)
        ? (Pad::amount * sizeof(T) / 4) - 1 : 0;
    const uint32_t bar_enable = (bar_lds_addr != 0) ? (1u << 18) : 0u;

    g1[0] = (cluster_mask  & 0xFFFFu)
          | (data_size_enc << 16)
          |  bar_enable
          | (pad_enable    << 20)
          | (pad_int_enc   << 22)
          | (pad_amt_enc   << 25);
}

template<typename Tile, typename T>
__device__ __forceinline__ void fill_group1_dims(
    v8u32& g1,
    typename Tile::extents const& tensor_dims,
    typename Tile::strides const& strides,
    uint32_t bar_lds_addr)
{
    constexpr int rank = Tile::rank;

    // TDM numbering is innermost-first; HK passes outermost-first arrays.
    // Use `if constexpr` so out-of-range array indices never instantiate --
    // a runtime ternary would let the compiler constant-evaluate the
    // unchosen branch and leak garbage into the descriptor.
    uint32_t tdim0 = 0, tdim1 = 0;
    uint32_t tildim0 = 0, tildim1 = 0, tildim2 = 0;

    if constexpr (rank >= 1) {
        tdim0   = uint32_t(tensor_dims[rank - 1]);
        tildim0 = uint32_t(Tile::dims[rank - 1]);
    }
    if constexpr (rank >= 2) {
        tdim1   = uint32_t(tensor_dims[rank - 2]);
        tildim1 = uint32_t(Tile::dims[rank - 2]);
    }
    if constexpr (rank >= 3) {
        tildim2 = uint32_t(Tile::dims[rank - 3]);
    }

    g1[1] = (bar_lds_addr & 0xFFFFu) | (tdim0 << 16);
    g1[2] = (tdim0 >> 16) | (tdim1 << 16);
    g1[3] = (tdim1 >> 16) | (tildim0 << 16);
    g1[4] = tildim1 | (tildim2 << 16);

    // stride0 = elements between innermost rows of the tensor.
    // The TDM HW interprets the stride field in element units (cross-checked
    // against Triton's driver.c line 189 which writes
    // `desc->group1_5 = strides[rank - 2]` raw, no sizeof multiplication).
    g1[5] = 0;
    g1[6] = 0;
    g1[7] = 0;
    if constexpr (rank >= 2) {
        const uint64_t s0 = uint32_t(strides[rank - 2]);
        g1[5] = uint32_t(s0);
        g1[6] = uint32_t(s0 >> 32);
    }

    // stride1 (48 bits) = elements between rank-2 rows.
    if constexpr (rank >= 3) {
        const uint64_t s1 = uint32_t(strides[rank - 3]);
        g1[6] |= (uint32_t(s1) & 0xFFFFu) << 16;
        g1[7]  = uint32_t(s1 >> 16);
    }
}

template<typename Tile, typename T>
__device__ __forceinline__ void fill_group2_nd(
    v4u32& g2,
    typename Tile::extents const& tensor_dims,
    typename Tile::strides const& strides)
{
    constexpr int rank = Tile::rank;
    g2 = v4u32{0, 0, 0, 0};

    if constexpr (rank >= 3) {
        // tensor_dim2 = full 32-bit.
        g2[0] = uint32_t(tensor_dims[rank - 3]);
    }
    if constexpr (rank >= 4) {
        // tensor_dim3 = full 32-bit, in DWord 1.
        g2[1] = uint32_t(tensor_dims[rank - 4]);
        // tensor_dim2_stride: 48-bit (in elements), low 32 in DWord 2,
        // high 16 in DWord 3[15:0].
        const uint64_t s2 = uint32_t(strides[rank - 4]);
        g2[2] = uint32_t(s2);
        g2[3] = uint32_t(s2 >> 32) & 0xFFFFu;
        // tile_dim3 in DWord 3[31:16].
        g2[3] |= uint32_t(Tile::dims[rank - 4]) << 16;
    }
}

template<typename Tile, typename T>
__device__ __forceinline__ void fill_group3_nd(
    v4u32& g3,
    typename Tile::extents const& tensor_dims,
    typename Tile::strides const& strides)
{
    constexpr int rank = Tile::rank;
    g3 = v4u32{0, 0, 0, 0};

    if constexpr (rank == 5) {
        // tensor_dim3_stride: 48-bit (in elements), low 32 in DWord 0,
        // high 16 in DWord 1[15:0].
        const uint64_t s3 = uint32_t(strides[rank - 5]);
        g3[0] = uint32_t(s3);
        g3[1] = uint32_t(s3 >> 32) & 0xFFFFu;
        // tensor_dim4: 32-bit split — low 16 in DWord 1[31:16], high 16 in DWord 2[15:0].
        const uint32_t td4 = uint32_t(tensor_dims[0]);
        g3[1] |= (td4 & 0xFFFFu) << 16;
        g3[2]  = (td4 >> 16) & 0xFFFFu;
        // tile_dim4 in DWord 2[31:16].
        g3[2] |= uint32_t(Tile::dims[0]) << 16;
    }
}

/**
 * @brief Build the full N-D TDM descriptor for non-gather mode.
 *
 * `bar_lds_addr` is the LDS byte address of a `kittens::semaphore` cell
 * (`&sem.state`). When non-zero, sets `atomic_barrier_enable` and the
 * barrier address in group 1; the TDM unit issues a
 * `DS_ATOMIC_ASYNC_BARRIER_ARRIVE_B64` on the cell when the transfer
 * retires. Pass 0 for the no-barrier (drain-via-counter) path.
 *
 * `cluster_mask` is the workgroup-in-cluster mask (only the low 16 bits
 * reach the TDM unit). Zero for single-WG loads. Stores ignore the mask.
 */
template<typename Tile, typename Pad, typename T>
__device__ __forceinline__ void build_d_nd(
    v4u32& g0, v8u32& g1, v4u32& g2, v4u32& g3,
    const T* base, void* lds_dst,
    typename Tile::extents const& tensor_dims,
    typename Tile::strides const& strides,
    uint32_t cluster_mask, uint32_t bar_lds_addr)
{
    fill_group0<T>(g0, base, lds_dst,
                   /*gather_enable=*/ false,
                   /*gather_idx_32bit=*/ false);
    fill_group1_ctrl<Tile, Pad, T>(g1, cluster_mask, bar_lds_addr);
    fill_group1_dims<Tile, T>(g1, tensor_dims, strides, bar_lds_addr);
    fill_group2_nd<Tile, T>(g2, tensor_dims, strides);
    fill_group3_nd<Tile, T>(g3, tensor_dims, strides);
}

/* --- group2/3 builders (gather mode) ---
 *
 * Gather is 2D-only. The HW mode is selected by index dtype:
 *   - uint16_t → 16 indices, packed 2-per-DWord, 8 across group2 and 8
 *                across group3.
 *   - uint32_t → 8 indices, 1-per-DWord, 4 across group2 and 4 across
 *                group3.
 */
template<size_t NROWS>
__device__ __forceinline__ void pack_gather_indices(
    v4u32& g2, v4u32& g3, const std::array<uint16_t, NROWS>& idx)
{
    static_assert(NROWS == 16, "u16 gather mode requires 16 indices");
    g2[0] = uint32_t(idx[0])  | (uint32_t(idx[1])  << 16);
    g2[1] = uint32_t(idx[2])  | (uint32_t(idx[3])  << 16);
    g2[2] = uint32_t(idx[4])  | (uint32_t(idx[5])  << 16);
    g2[3] = uint32_t(idx[6])  | (uint32_t(idx[7])  << 16);
    g3[0] = uint32_t(idx[8])  | (uint32_t(idx[9])  << 16);
    g3[1] = uint32_t(idx[10]) | (uint32_t(idx[11]) << 16);
    g3[2] = uint32_t(idx[12]) | (uint32_t(idx[13]) << 16);
    g3[3] = uint32_t(idx[14]) | (uint32_t(idx[15]) << 16);
}

template<size_t NROWS>
__device__ __forceinline__ void pack_gather_indices(
    v4u32& g2, v4u32& g3, const std::array<uint32_t, NROWS>& idx)
{
    static_assert(NROWS == 8, "u32 gather mode requires 8 indices");
    g2[0] = idx[0]; g2[1] = idx[1]; g2[2] = idx[2]; g2[3] = idx[3];
    g3[0] = idx[4]; g3[1] = idx[5]; g3[2] = idx[6]; g3[3] = idx[7];
}

/**
 * @brief Build the 2D gather/scatter TDM descriptor.
 *
 * Layout: groups 0+1 follow the regular 2D form (with the gather-enable
 * bit set in group0[0] and the index-size bit selecting u16 vs u32).
 * Groups 2-3 carry the row index list per the SP3 field map.
 *
 * `NROWS` is the runtime row count; the tile shape is therefore
 * `gather_tile<COLS>` × `NROWS` rows (HW-fixed by the index dtype).
 */
template<typename GTile, typename Pad, typename T, typename IdxT, size_t NROWS>
__device__ __forceinline__ void build_d_gather(
    v4u32& g0, v8u32& g1, v4u32& g2, v4u32& g3,
    const T* base, void* lds_dst,
    int tensor_rows, int tensor_cols, int row_stride,
    const std::array<IdxT, NROWS>& row_indices,
    uint32_t bar_lds_addr)
{
    static_assert(
        (std::is_same_v<IdxT, uint16_t> && NROWS == 16) ||
        (std::is_same_v<IdxT, uint32_t> && NROWS == 8),
        "TDM gather supports 16 rows of uint16_t indices or 8 rows of uint32_t indices");
    constexpr bool idx_32bit = std::is_same_v<IdxT, uint32_t>;

    fill_group0<T>(g0, base, lds_dst,
                   /*gather_enable=*/ true,
                   /*gather_idx_32bit=*/ idx_32bit);

    // Reuse the regular 2D group1 builder. Treat NROWS as tile_dim1 so the
    // HW knows how many rows to gather into LDS.
    using GTile2D = tile<int(NROWS), GTile::cols>;
    typename GTile2D::extents tensor_dims{int32_t(tensor_rows),
                                          int32_t(tensor_cols)};
    typename GTile2D::strides strides{int32_t(row_stride)};

    fill_group1_ctrl<GTile2D, Pad, T>(g1, /*cluster_mask=*/ 0, bar_lds_addr);
    fill_group1_dims<GTile2D, T>(g1, tensor_dims, strides, bar_lds_addr);

    pack_gather_indices<NROWS>(g2, g3, row_indices);
}

} // namespace detail

/* ----------  TDM N-D TILE LOAD (G -> LDS)  ---------- */

/**
 * @brief Issue an N-D TDM tile load; drain via `tdm::load_async_wait<N>`.
 *
 * Builds the D# in-place and issues `tensor_load_to_lds`. The transfer
 * counts against `TENSORcnt` and is drained by `wait_tensor<N>` (or the
 * `load_async_wait<N>` / `store_async_wait<N>` aliases).
 *
 * @tparam Tile  `tdm::tile<DIMS...>` — compile-time tile shape (1D-5D).
 * @param  lds_dst     16 B-aligned LDS pointer.
 * @param  src         Global tile descriptor.
 * @param  idx         Tile coordinate inside `src` (tile-index units).
 * @param  tensor_dims Outermost-first per-dim extents of the source tensor.
 *                     The TDM unit clamps reads against these bounds (OOB
 *                     reads return zero); pass `min(actual, tile)` for the
 *                     trailing dim on edge tiles.
 * @param  strides     Outermost-first per-dim element strides (one entry
 *                     per dim **except the innermost**, which is unit).
 */
template<typename Tile, typename Pad = lds_nopad,
         typename T, ducks::gl::all GL, ducks::coord::tile COORD = coord<>>
__device__ inline void load_async(
    T* __restrict__ lds_dst, const GL& src, const COORD& idx,
    typename Tile::extents const& tensor_dims,
    typename Tile::strides const& strides)
{
    static_assert(Tile::rank >= 1 && Tile::rank <= 5, "TDM supports 1D-5D");
    constexpr int R = Tile::rank;
    const T* base = detail::tile_base_2d<
        R >= 2 ? Tile::dims[R - 2] : 1,
        Tile::dims[R - 1], T, GL, COORD>(src, idx);

    detail::v4u32 g0;
    detail::v8u32 g1;
    detail::v4u32 g2, g3;
    detail::build_d_nd<Tile, Pad, T>(g0, g1, g2, g3, base, lds_dst,
                                     tensor_dims, strides,
                                     /*cluster_mask=*/ 0,
                                     /*bar_lds_addr=*/ 0);
    __builtin_amdgcn_tensor_load_to_lds(g0, g1, g2, g3, 0);
}

/**
 * @brief Issue an N-D TDM tile load that auto-arrives at `sem`.
 *
 * Sets `atomic_barrier_enable` in the D# so the TDM unit emits a
 * `DS_ATOMIC_ASYNC_BARRIER_ARRIVE_B64` on `sem` after the transfer
 * retires. The consumer waits via `kittens::wait(sem, phase)`.
 *
 * @note Runtime support for the D# auto-arrive path varies. Triton's AMD
 *       lowering follows `load_async` (no-bar) with explicit
 *       `wait_tensor<0>` + `arrive(sem)` instead. The explicit pattern is
 *       portable; this overload is the lower-overhead alternative on
 *       runtimes that honor the auto-arrive.
 */
template<typename Tile, typename Pad = lds_nopad,
         typename T, ducks::gl::all GL, ducks::coord::tile COORD = coord<>>
__device__ inline void load_async(
    T* __restrict__ lds_dst, const GL& src, const COORD& idx,
    semaphore& sem,
    typename Tile::extents const& tensor_dims,
    typename Tile::strides const& strides)
{
    static_assert(Tile::rank >= 1 && Tile::rank <= 5, "TDM supports 1D-5D");
    constexpr int R = Tile::rank;
    const T* base = detail::tile_base_2d<
        R >= 2 ? Tile::dims[R - 2] : 1,
        Tile::dims[R - 1], T, GL, COORD>(src, idx);
    const uint32_t bar_lds_addr = static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(&sem.state));

    detail::v4u32 g0;
    detail::v8u32 g1;
    detail::v4u32 g2, g3;
    detail::build_d_nd<Tile, Pad, T>(g0, g1, g2, g3, base, lds_dst,
                                     tensor_dims, strides,
                                     /*cluster_mask=*/ 0, bar_lds_addr);
    __builtin_amdgcn_tensor_load_to_lds(g0, g1, g2, g3, 0);
}

/* ----------  TDM N-D TILE STORE (LDS -> G)  ---------- */

/**
 * @brief Issue an N-D TDM tile store; drain via `tdm::store_async_wait<N>`.
 *
 * Mirror of `load_async`. Drains against the same `TENSORcnt` counter as
 * loads -- `tdm::store_async_wait<N>` and `tdm::load_async_wait<N>` are
 * the same drain primitive under different names.
 *
 * @note Per SP3, `workgroup_mask` is load-only; there is no cluster
 *       (multicast) store variant. Stores omit the mask argument.
 */
template<typename Tile, typename Pad = lds_nopad,
         typename T, ducks::gl::all GL, ducks::coord::tile COORD = coord<>>
__device__ inline void store_async(
    const GL& dst, const T* __restrict__ lds_src, const COORD& idx,
    typename Tile::extents const& tensor_dims,
    typename Tile::strides const& strides)
{
    static_assert(Tile::rank >= 1 && Tile::rank <= 5, "TDM supports 1D-5D");
    constexpr int R = Tile::rank;
    const T* base = detail::tile_base_2d<
        R >= 2 ? Tile::dims[R - 2] : 1,
        Tile::dims[R - 1], T, GL, COORD>(dst, idx);

    detail::v4u32 g0;
    detail::v8u32 g1;
    detail::v4u32 g2, g3;
    detail::build_d_nd<Tile, Pad, T>(g0, g1, g2, g3,
                                     base, const_cast<T*>(lds_src),
                                     tensor_dims, strides,
                                     /*cluster_mask=*/ 0,
                                     /*bar_lds_addr=*/ 0);
    __builtin_amdgcn_tensor_store_from_lds(g0, g1, g2, g3, 0);
}

/// @brief Store with HW auto-arrive at `sem`.
template<typename Tile, typename Pad = lds_nopad,
         typename T, ducks::gl::all GL, ducks::coord::tile COORD = coord<>>
__device__ inline void store_async(
    const GL& dst, const T* __restrict__ lds_src, const COORD& idx,
    semaphore& sem,
    typename Tile::extents const& tensor_dims,
    typename Tile::strides const& strides)
{
    static_assert(Tile::rank >= 1 && Tile::rank <= 5, "TDM supports 1D-5D");
    constexpr int R = Tile::rank;
    const T* base = detail::tile_base_2d<
        R >= 2 ? Tile::dims[R - 2] : 1,
        Tile::dims[R - 1], T, GL, COORD>(dst, idx);
    const uint32_t bar_lds_addr = static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(&sem.state));

    detail::v4u32 g0;
    detail::v8u32 g1;
    detail::v4u32 g2, g3;
    detail::build_d_nd<Tile, Pad, T>(g0, g1, g2, g3,
                                     base, const_cast<T*>(lds_src),
                                     tensor_dims, strides,
                                     /*cluster_mask=*/ 0, bar_lds_addr);
    __builtin_amdgcn_tensor_store_from_lds(g0, g1, g2, g3, 0);
}

/* ----------  GATHER / SCATTER (HW-native, 2D, row-indexed)  ---------- */

/**
 * @brief Row-indexed gather load from a 2D global tensor.
 *
 * Reads `NROWS` rows of `COLS` contiguous columns each (starting at
 * `col_origin`) into LDS, with rows selected by `row_indices`. The HW
 * mode (16-row u16 vs 8-row u32) is selected by the index dtype.
 *
 * `row_indices` must be **strictly increasing**, no duplicates (HW
 * requirement; not statically checkable). OOB row indices are clamped
 * by the same `tensor_rows` mechanism as regular loads -- rows beyond
 * the tensor extent return zero.
 *
 * @tparam GTile  `tdm::gather_tile<COLS>` -- column count per row.
 * @param  lds_dst     16 B-aligned LDS pointer. Gathered rows land
 *                     contiguously: row `i` of the gather at
 *                     `lds_dst + i * COLS`.
 * @param  col_origin  Column index of the first column to gather.
 * @param  tensor_cols Column extent of the source tensor (for OOB).
 * @param  row_stride  Element stride between rows in the source tensor.
 * @param  row_indices Per-row global row indices, strictly increasing.
 */
template<typename GTile, typename Pad = lds_nopad,
         typename IdxT, size_t NROWS,
         typename T, ducks::gl::all GL>
__device__ inline void gather_async(
    T* __restrict__ lds_dst, const GL& src,
    int col_origin, int tensor_rows, int tensor_cols, int row_stride,
    const std::array<IdxT, NROWS>& row_indices)
{
    static_assert(
        (std::is_same_v<IdxT, uint16_t> && NROWS == 16) ||
        (std::is_same_v<IdxT, uint32_t> && NROWS == 8),
        "TDM gather supports 16 rows of uint16_t indices or 8 rows of uint32_t indices");

    // Base = first element of the column window in row 0.
    const T* base = src.raw_ptr + col_origin;

    detail::v4u32 g0;
    detail::v8u32 g1;
    detail::v4u32 g2, g3;
    detail::build_d_gather<GTile, Pad, T, IdxT, NROWS>(
        g0, g1, g2, g3, base, lds_dst,
        tensor_rows, tensor_cols, row_stride,
        row_indices, /*bar_lds_addr=*/ 0);
    __builtin_amdgcn_tensor_load_to_lds(g0, g1, g2, g3, 0);
}

/// @brief Gather load with HW auto-arrive at `sem`.
template<typename GTile, typename Pad = lds_nopad,
         typename IdxT, size_t NROWS,
         typename T, ducks::gl::all GL>
__device__ inline void gather_async(
    T* __restrict__ lds_dst, const GL& src, semaphore& sem,
    int col_origin, int tensor_rows, int tensor_cols, int row_stride,
    const std::array<IdxT, NROWS>& row_indices)
{
    static_assert(
        (std::is_same_v<IdxT, uint16_t> && NROWS == 16) ||
        (std::is_same_v<IdxT, uint32_t> && NROWS == 8),
        "TDM gather supports 16 rows of uint16_t indices or 8 rows of uint32_t indices");

    const T* base = src.raw_ptr + col_origin;
    const uint32_t bar_lds_addr = static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(&sem.state));

    detail::v4u32 g0;
    detail::v8u32 g1;
    detail::v4u32 g2, g3;
    detail::build_d_gather<GTile, Pad, T, IdxT, NROWS>(
        g0, g1, g2, g3, base, lds_dst,
        tensor_rows, tensor_cols, row_stride,
        row_indices, bar_lds_addr);
    __builtin_amdgcn_tensor_load_to_lds(g0, g1, g2, g3, 0);
}

/**
 * @brief Row-indexed scatter store to a 2D global tensor.
 *
 * Mirror of `gather_async`: writes `NROWS` LDS rows back into a 2D
 * tensor at the row indices given by `row_indices`. Columns are
 * contiguous from `col_origin`. Same dtype-driven mode selection
 * (u16 = 16 rows, u32 = 8 rows).
 */
template<typename GTile, typename Pad = lds_nopad,
         typename IdxT, size_t NROWS,
         typename T, ducks::gl::all GL>
__device__ inline void scatter_async(
    const GL& dst, const T* __restrict__ lds_src,
    int col_origin, int tensor_rows, int tensor_cols, int row_stride,
    const std::array<IdxT, NROWS>& row_indices)
{
    static_assert(
        (std::is_same_v<IdxT, uint16_t> && NROWS == 16) ||
        (std::is_same_v<IdxT, uint32_t> && NROWS == 8),
        "TDM scatter supports 16 rows of uint16_t indices or 8 rows of uint32_t indices");

    const T* base = dst.raw_ptr + col_origin;

    detail::v4u32 g0;
    detail::v8u32 g1;
    detail::v4u32 g2, g3;
    detail::build_d_gather<GTile, Pad, T, IdxT, NROWS>(
        g0, g1, g2, g3, base, const_cast<T*>(lds_src),
        tensor_rows, tensor_cols, row_stride,
        row_indices, /*bar_lds_addr=*/ 0);
    __builtin_amdgcn_tensor_store_from_lds(g0, g1, g2, g3, 0);
}

/// @brief Scatter store with HW auto-arrive at `sem`.
template<typename GTile, typename Pad = lds_nopad,
         typename IdxT, size_t NROWS,
         typename T, ducks::gl::all GL>
__device__ inline void scatter_async(
    const GL& dst, const T* __restrict__ lds_src, semaphore& sem,
    int col_origin, int tensor_rows, int tensor_cols, int row_stride,
    const std::array<IdxT, NROWS>& row_indices)
{
    static_assert(
        (std::is_same_v<IdxT, uint16_t> && NROWS == 16) ||
        (std::is_same_v<IdxT, uint32_t> && NROWS == 8),
        "TDM scatter supports 16 rows of uint16_t indices or 8 rows of uint32_t indices");

    const T* base = dst.raw_ptr + col_origin;
    const uint32_t bar_lds_addr = static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(&sem.state));

    detail::v4u32 g0;
    detail::v8u32 g1;
    detail::v4u32 g2, g3;
    detail::build_d_gather<GTile, Pad, T, IdxT, NROWS>(
        g0, g1, g2, g3, base, const_cast<T*>(lds_src),
        tensor_rows, tensor_cols, row_stride,
        row_indices, bar_lds_addr);
    __builtin_amdgcn_tensor_store_from_lds(g0, g1, g2, g3, 0);
}

/* ----------  COMPLETION HELPERS  ---------- */
//
// Load and store ops both increment the same `TENSORcnt` on issue, so a
// single drain primitive handles either direction. The load- and
// store-named helpers are convenience aliases for clearer call sites.

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
 * @brief Cooperative L2 prefetch for an upcoming 2D tile.
 *
 * Lowers to per-lane `__builtin_amdgcn_global_prefetch` instructions
 * covering the tile. Hint = 0 selects the default cache policy.
 */
template<int ROWS = 0, int COLS = 0, int N_THREADS = WARP_THREADS,
         typename T, ducks::gl::all GL, ducks::coord::tile COORD = coord<>>
__device__ inline void prefetch(const GL& src, const COORD& idx, int row_stride)
{
    static_assert(ROWS > 0 && COLS > 0, "ROWS and COLS must be specified");
    constexpr int elems_per_pf = 16 / sizeof(T);
    constexpr int total_elems  = ROWS * COLS;
    const int tid = threadIdx.x;
    const T* base = detail::tile_base_2d<ROWS, COLS, T, GL, COORD>(src, idx);

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

/* ----------  CLUSTER-SCOPE TDM (multicast load only)  ---------- */
//
// When the cluster mask is non-zero, the TDM unit decomposes the load
// into `CLUSTER_LOAD_ASYNC_*` micro-ops, broadcasting the line into every
// workgroup whose bit is set in the mask. Only the low 16 bits of the
// mask reach the TDM unit; the `early_timeout` bit (bit 16) is honored
// by the per-lane async path but not by TDM. `workgroup_mask` is
// load-only per SP3 -- no cluster store/scatter variants exist.

namespace cluster {

/**
 * @brief Multicast N-D TDM tile load that auto-arrives at `sem`.
 *
 * @param sem    LDS-resident semaphore to receive the auto-arrive.
 * @param mask   16-bit workgroup-in-cluster mask
 *               (`kittens::cluster::mask_for_tdm(...)`).
 */
template<typename Tile, typename Pad = lds_nopad,
         typename T, ducks::gl::all GL, ducks::coord::tile COORD = coord<>>
__device__ inline void load_async(
    T* __restrict__ lds_dst, const GL& src, const COORD& idx,
    semaphore& sem, uint32_t mask,
    typename Tile::extents const& tensor_dims,
    typename Tile::strides const& strides)
{
    static_assert(Tile::rank >= 1 && Tile::rank <= 5, "TDM supports 1D-5D");
    constexpr int R = Tile::rank;
    const T* base = detail::tile_base_2d<
        R >= 2 ? Tile::dims[R - 2] : 1,
        Tile::dims[R - 1], T, GL, COORD>(src, idx);
    const uint32_t bar_lds_addr = static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(&sem.state));

    detail::v4u32 g0;
    detail::v8u32 g1;
    detail::v4u32 g2, g3;
    detail::build_d_nd<Tile, Pad, T>(g0, g1, g2, g3, base, lds_dst,
                                     tensor_dims, strides,
                                     mask, bar_lds_addr);
    __builtin_amdgcn_tensor_load_to_lds(g0, g1, g2, g3, 0);
}

/// @brief Multicast N-D TDM tile load; drain via `tdm::load_async_wait<N>`.
template<typename Tile, typename Pad = lds_nopad,
         typename T, ducks::gl::all GL, ducks::coord::tile COORD = coord<>>
__device__ inline void load_async(
    T* __restrict__ lds_dst, const GL& src, const COORD& idx,
    uint32_t mask,
    typename Tile::extents const& tensor_dims,
    typename Tile::strides const& strides)
{
    static_assert(Tile::rank >= 1 && Tile::rank <= 5, "TDM supports 1D-5D");
    constexpr int R = Tile::rank;
    const T* base = detail::tile_base_2d<
        R >= 2 ? Tile::dims[R - 2] : 1,
        Tile::dims[R - 1], T, GL, COORD>(src, idx);

    detail::v4u32 g0;
    detail::v8u32 g1;
    detail::v4u32 g2, g3;
    detail::build_d_nd<Tile, Pad, T>(g0, g1, g2, g3, base, lds_dst,
                                     tensor_dims, strides,
                                     mask, /*bar_lds_addr=*/ 0);
    __builtin_amdgcn_tensor_load_to_lds(g0, g1, g2, g3, 0);
}

} // namespace cluster

} // namespace tdm
} // namespace kittens

#endif // KITTENS_UDNA1
