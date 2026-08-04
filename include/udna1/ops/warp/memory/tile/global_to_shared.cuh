/**
 * @file
 * @brief Functions for transferring data directly between global and shared memory and back.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

namespace detail {

/**
 * @brief Re-attach `addrspace(3)` to a pointer into an `st` tile's LDS storage.
 *
 * `shared_allocator` hands out a plain `int*` from `reinterpret_cast`ing `&__shm[0]`, which
 * strips the address space beyond what `InferAddressSpaces` can recover, so without this an LDS
 * access lowers to `flat_*` rather than `ds_*` -- a 64-bit address through the generic datapath
 * instead of a 32-bit LDS offset.
 *
 * Only valid for a pointer that really is LDS: the narrowing keeps the low 32 bits, which is the
 * LDS byte offset for a shared pointer and garbage otherwise. The round trip through `uintptr_t`
 * is required, as `reinterpret_cast` alone cannot add an address space. The device-pass guard is
 * also required: `T` is usually a class type, and on the host pass address space 3 is unrelated,
 * so `lds[i] = value` fails to resolve there.
 */
template<typename T>
__device__ __forceinline__ static auto lds_ptr(T *p) {
#if defined(__HIP_DEVICE_COMPILE__)
    return reinterpret_cast<T __attribute__((address_space(3)))*>(
        reinterpret_cast<uintptr_t>(p));
#else
    return p;
#endif
}

/**
 * @brief Row-major flat index of a shared tile -> physical element offset in `ST::data`.
 *
 * Which map applies is a property of the shape: padded shapes are row-major with the shape's
 * periodic padding, which is what the TDM deposits; size-preserving shapes are subtile-major
 * with the shape's within-subtile swizzle. Note `st::lds_offset()` is a third spelling,
 * `padded(subtile_flat(.))`, agreeing with the padded map only when `cols == Shape::cols`.
 */
template<ducks::st::all ST>
__device__ __forceinline__ static int st_element_offset(int flat) {
    using T = typename ST::dtype;
    if constexpr (requires { ST::shape::padded(0); }) {
        /* Taken on the FLAT index, not a recomposed (row, col): chunks step by a constant and
         * `padded` is affine on each interval, so the walk needs one address register instead of
         * one per iteration. Through the shape rather than `ST::padded`, which `st_subtile` lacks. */
        return ST::shape::padded(flat);
    } else {
        constexpr int SR = ST::underlying_subtile_rows;
        constexpr int SC = ST::underlying_subtile_cols;
        const int row = flat / ST::cols, col = flat % ST::cols;
        const int sub_id = (row / SR) * ST::subtiles_per_row + (col / SC);
        return sub_id * ST::underlying_subtile_elements
             + int(ST::swizzle({row % SR, col % SC})) / int(sizeof(T));
    }
}

/**
 * @brief Checks whether a run of `E` logical elements starting at a multiple of `E` is a single
 *        aligned run of `E * sizeof(T)` bytes in LDS.
 *
 * For a padded shape `padded(f) = f + (f/PI)*PA`, so the run stays contiguous iff `PI % E == 0`
 * and aligned iff `PA % E == 0`; `st_16x32_padded<64,4>` fails the second at bf16, which is why
 * this is computed rather than assumed. Every size-preserving swizzle moves an offset by a
 * multiple of 8 bytes using only bits 7 and above, so it is constant across a run of at most
 * 16 bytes -- a shape with finer swizzle granularity would break that and must be added here.
 */
template<ducks::st::all ST, int E>
__device__ __host__ static constexpr bool st_run_is_contiguous() {
    if constexpr (requires { ST::shape::pad_interval; })
        return (ST::shape::pad_interval % E == 0) && (ST::shape::pad_amount % E == 0);
    else
        return true;
}

/**
 * @brief Move one thread's run of `E` logical elements from the shared tile to global memory.
 *
 * @tparam WIDE Take the single sized global store. Decided once by the caller, outside the loop
 *              nest: a per-iteration alignment branch stops the store widening at all.
 *
 * `dst_ptr` stays a typed `U*` derived from `&dst[coord]` rather than round-tripping through
 * `uintptr_t`, which loses the provenance the backend needs to prove the address is global and
 * lowers to `flat_store`.
 */
template<bool WIDE, int E, ducks::st::all ST, typename U>
__device__ __forceinline__ static void st_run_to_global(
    U *dst_ptr, int row_stride, const ST &src, int flat)
{
    using T = typename ST::dtype;
    constexpr int BYTES = E * (int)sizeof(T);

    const T *in  = &src.data[st_element_offset<ST>(flat)];
    /* 32-bit index arithmetic, matching `read_back`. `row * row_stride + col` cannot overflow
     * -- `row` is bounded by the tile and `row_stride` by the tensor -- and the whole tensor
     * offset is already carried in `dst_ptr`, which `gl::idx()` computed in 64 bits. Widening
     * this to int64 costs the SGPR base: the backend gives up on proving the base wave-uniform
     * and materialises it into a VGPR pair. */
    U       *out = dst_ptr + (flat / ST::cols) * row_stride + (flat % ST::cols);

    T buf[E];
    if constexpr (st_run_is_contiguous<ST, E>() &&
                  (BYTES == 2 || BYTES == 4 || BYTES == 8 || BYTES == 16)) {
        if      constexpr (BYTES == 16) *reinterpret_cast<__uint128_t*>(buf) = *reinterpret_cast<const __uint128_t*>(in);
        else if constexpr (BYTES == 8)  *reinterpret_cast<uint64_t*>(buf)    = *reinterpret_cast<const uint64_t*>(in);
        else if constexpr (BYTES == 4)  *reinterpret_cast<uint32_t*>(buf)    = *reinterpret_cast<const uint32_t*>(in);
        else                            *reinterpret_cast<uint16_t*>(buf)    = *reinterpret_cast<const uint16_t*>(in);
    } else {
        #pragma unroll
        for (int j = 0; j < E; j++) buf[j] = in[j];
    }

    if constexpr (WIDE) {
        // `T` and `U` are the same type (asserted by the caller), so the gathered run already
        // holds the destination bytes and the conversion below is the identity.
        if      constexpr (BYTES == 16) *reinterpret_cast<__uint128_t*>(out) = *reinterpret_cast<const __uint128_t*>(buf);
        else if constexpr (BYTES == 8)  *reinterpret_cast<uint64_t*>(out)    = *reinterpret_cast<const uint64_t*>(buf);
        else if constexpr (BYTES == 4)  *reinterpret_cast<uint32_t*>(out)    = *reinterpret_cast<const uint32_t*>(buf);
        else                            *reinterpret_cast<uint16_t*>(out)    = *reinterpret_cast<const uint16_t*>(buf);
    } else {
        #pragma unroll
        for (int j = 0; j < E; j++)
            out[j] = kittens::base_types::convertor<U, T>::convert(buf[j]);
    }
}

} // namespace detail

/// Refused: the body's `row * row_stride + col` assumes a unit-stride column axis, which
/// inverts under col_major and would silently address the wrong elements.
template<int axis, bool assume_aligned,
        ducks::st::all ST, ducks::gl::col_layout GL,
        ducks::coord::tile COORD=coord<ST>, int N_THREADS=WARP_THREADS>
__device__ static inline void store(const GL &dst, const ST &src, const COORD &idx) {
    static_assert(ducks::gl_layout::unhandled<typename GL::layout>,
        "store(dst, shared_tile, coord) is implemented for ducks::gl_layout::row_major only: "
        "its addressing gives the column axis an implicit unit stride. A column-major "
        "destination also wants the tile streamed out in COLUMN-major linear order, which is a "
        "different traversal and not merely a different stride -- implement that path rather "
        "than passing this descriptor. For a WMMA accumulator, use the staging overload "
        "store<N_THREADS>(dst, acc, scratch, block_row, block_col, warp_rbase, warp_cbase), "
        "which is correct for both layouts.");
}

/**
 * @brief Stores data from a shared memory tile into global memory.
 *
 * @tparam ST The type of the shared tile.
 * @param[out] dst The destination global memory array.
 * @param[in] src The source shared memory tile.
 * @param idx[in] The tile coordinate in the destination array.
 *
 * Walks the tile in global order: each thread owns `E` logical elements consecutive in one row,
 * so its run is contiguous in global memory and is gathered from whatever (swizzled or padded)
 * LDS slots hold it. Running the other way, contiguous in LDS, scatters the side coalescing is
 * paid for and lowers to one narrow store per element.
 *
 * The run is gathered into a register buffer and written as one sized store, since contiguous
 * addresses alone do not get a wide store out of the compiler. That store is taken only when
 * alignment is provable: the row stride and tile origin are runtime values the type cannot see,
 * so it is checked rather than silently required. The test is wave-uniform and hoisted out of
 * the loop nest, because deciding per iteration stops the widening entirely.
 * `assume_aligned` skips the check for a caller that can certify it.
 */
template<int axis, bool assume_aligned, 
        ducks::st::all ST, ducks::gl::row_layout GL, 
        ducks::coord::tile COORD=coord<ST>, int N_THREADS=WARP_THREADS>
__device__ static inline void store(const GL &dst, const ST &src, const COORD &idx) {

    using T = typename ST::dtype;
    using U = typename GL::dtype;

    static_assert(std::is_same_v<T, U>, "T and U must be the same type");
    static_assert(!std::is_same_v<T, fp8e4m3>, "Unsupported type for store");

    constexpr int bytes_per_thread = ST::underlying_subtile_bytes_per_thread;
    constexpr int elems_per_thread = bytes_per_thread / sizeof(T);
    constexpr int memcpy_per_tile =  ST::rows * ST::cols * sizeof(T) / (bytes_per_thread * N_THREADS);

    // A run must stay inside one row, or it is neither one global store nor one LDS read.
    static_assert(ST::cols % elems_per_thread == 0,
        "a thread's run must not straddle a tile row: ST::cols must be a multiple of "
        "bytes_per_thread/sizeof(T)");

    constexpr int num_warps = N_THREADS / kittens::WARP_THREADS;
    const int laneid = kittens::laneid();
    const int warpid = kittens::warpid() % num_warps;
    const int tid    = warpid * kittens::WARP_THREADS + laneid;

    const int row_stride = dst.template stride<axis>();

    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    U *dst_ptr = &dst[unit_coord];

    /* One wave-uniform decision for the whole tile. `sized` is the compile-time half (a run
     * has to be a width the ISA has); the runtime half is the destination's alignment, which
     * depends on the caller's leading dimension and tile origin. */
    constexpr bool sized = (bytes_per_thread == 2 || bytes_per_thread == 4 ||
                            bytes_per_thread == 8 || bytes_per_thread == 16);
    const bool wide = sized &&
        (assume_aligned || (((row_stride % elems_per_thread) == 0) &&
                            ((reinterpret_cast<uintptr_t>(dst_ptr) % bytes_per_thread) == 0)));

    /* The two nests are identical but for the `WIDE` template argument. Written out rather
     * than folded into one nest with a runtime flag, because the flag inside the unrolled body
     * is exactly what collapses the wide arm. */
#define KITTENS_ST_TO_GLOBAL_NEST(WIDE_ARM)                                                   \
    if constexpr (memcpy_per_tile > 0) {                                                      \
        _Pragma("unroll")                                                                     \
        for (int i = 0; i < memcpy_per_tile; i++) {                                           \
            detail::st_run_to_global<WIDE_ARM, elems_per_thread>(                             \
                dst_ptr, row_stride, src, (tid + i * N_THREADS) * elems_per_thread);          \
        }                                                                                     \
    }                                                                                         \
    if constexpr (memcpy_per_tile * (bytes_per_thread * N_THREADS)                            \
                  != ST::rows * ST::cols * sizeof(T)) {                                       \
        constexpr int leftover_bytes   = ST::rows * ST::cols * sizeof(T)                      \
                                       - memcpy_per_tile * (bytes_per_thread * N_THREADS);    \
        constexpr int leftover_threads = leftover_bytes / bytes_per_thread;                   \
        constexpr int leftover_warps   = leftover_threads / kittens::WARP_THREADS;            \
        if (warpid < leftover_warps) {                                                        \
            detail::st_run_to_global<WIDE_ARM, elems_per_thread>(                             \
                dst_ptr, row_stride, src,                                                     \
                (tid + memcpy_per_tile * N_THREADS) * elems_per_thread);                      \
        }                                                                                     \
    }

    if (wide) { KITTENS_ST_TO_GLOBAL_NEST(true) }
    else      { KITTENS_ST_TO_GLOBAL_NEST(false) }
#undef KITTENS_ST_TO_GLOBAL_NEST
}
/* A pure delegator: it contains no layout-dependent code of its own, so it stays on
 * `ducks::gl::all` and takes its diagnostics from the overload it resolves to. */
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store(const GL &dst, const ST &src, const COORD &idx) {
    store<2, false, ST, GL, COORD, WARP_THREADS>(dst, src, idx);
}

/**
 * gfx1250 raw-pointer global <-> LDS transfers
 *
 * Three hardware paths move a global tile into LDS, all landing straight in
 * LDS with no VGPR staging:
 *
 *   - `global_load_async_to_lds_*`: each active thread copies B bytes
 *     (B8/B32/B64/B128 = 1/4/8/16 B) from global to LDS, so a b128 load moves
 *     16 B x 32 threads = 512 B per wave per instruction, into this
 *     workgroup's LDS. Drained with `wait_async`.
 *   - `cluster_load_async_to_lds_*`: the same per-wave payload, except the one
 *     L2 return is broadcast into the LDS of several workgroups in a cluster at
 *     once (up to ~5x amplification; bypasses L1) -- for workgroup-cluster
 *     kernels where multiple workgroups want the same tile. Also drained with
 *     `wait_async`.
 *   - `tensor_load_to_lds` (TDM): a dedicated DMA-style engine, 
 *     moves a WHOLE tile per instruction from an SGPR descriptor 
 *     and does its own address generation. Drained with `wait_tdm`.
 *
 * These ops dispatch through the gfx1250 shared tile `st`, which owns its LDS
 * storage and address map, mirroring the canonical `load(tile, gl, idx)`
 * surface -- no separate padding descriptor. Kernels allocate an `st_bf` tile
 * (optionally via `shared_allocator::allocate_in<segment<I>>`) and pass it
 * straight in.
 *
 */

/// Refused: two independent row-major assumptions, a base composed as
/// `((b*depth + d)*rows + gr)*cols + gc` rather than through `gl::idx()`, and a
/// `row*row_stride + col` walk. Both invert under col_major, and `row_stride` cannot rescue
/// either -- the base offset is wrong whatever is passed.
template<int N_THREADS = WARP_THREADS, typename T, int ROWS, int COLS,
         ducks::st_shape::all Shape, ducks::gl::col_layout GL, ducks::coord::tile COORD = coord<>>
__device__ inline void load(st<T, ROWS, COLS, Shape>& dst, const GL& src,
                            const COORD& idx, int row_stride)
{
    static_assert(ducks::gl_layout::unhandled<typename GL::layout>,
        "load(shared_tile, src, coord, row_stride) is implemented for ducks::gl_layout::row_major "
        "only: it composes its base address row-major and then indexes row*row_stride + col, "
        "giving the column axis an implicit unit stride. A column-major source needs the walk "
        "transposed as well as the base recomputed -- implement that path rather than passing "
        "this descriptor.");
}

/**
 * @brief Cooperative register-mediated global -> LDS tile copy (gfx1250 baseline).
 *
 * Plain `global_load` -> VGPR -> `ds_store` path. Use this when no async
 * intrinsic is available or for correctness baselines. The destination
 * `st` tile owns the subtile-major + padding LDS address map.
 *
 * The LDS write goes through `detail::lds_ptr`, without which it lowers to `flat_store` rather
 * than `ds_store`.
 */
template<int N_THREADS = WARP_THREADS, typename T, int ROWS, int COLS,
         ducks::st_shape::all Shape, ducks::gl::row_layout GL, ducks::coord::tile COORD = coord<>>
__device__ inline void load(st<T, ROWS, COLS, Shape>& dst, const GL& src,
                            const COORD& idx, int row_stride)
{
    constexpr int total_elems = ROWS * COLS;
    const int tid = threadIdx.x;
    // The COORD is interpreted as tile-index coordinates `{b, d, tile_row, tile_col}`
    // -- convert to element coordinates by multiplying the trailing two by ROWS/COLS.
    const int gr_base = idx.r * ROWS;
    const int gc_base = idx.c * COLS;
    const T* base = src.raw_ptr
                  + (((int64_t(idx.b) * src.depth() + idx.d) * src.rows() + gr_base)
                     * src.cols() + gc_base);

    auto* lds = detail::lds_ptr(dst.data);

    #pragma unroll
    for (int i = tid; i < total_elems; i += N_THREADS) {
        const int row = i / COLS;
        const int col = i % COLS;
        // st maps the logical (row-major) index to its subtile-major,
        // padded LDS slot.
        lds[dst.lds_offset(i)] = base[row * row_stride + col];
    }
}

/// Refused: the inverse of the load above, with the same two row-major assumptions, and it
/// inverts under col_major the same way.
template<int N_THREADS = WARP_THREADS, typename T, int ROWS, int COLS,
         ducks::st_shape::all Shape, ducks::gl::col_layout GL, ducks::coord::tile COORD = coord<>>
__device__ inline void store(const GL& dst, const st<T, ROWS, COLS, Shape>& src,
                             const COORD& idx, int row_stride)
{
    static_assert(ducks::gl_layout::unhandled<typename GL::layout>,
        "store(dst, shared_tile, coord, row_stride) is implemented for ducks::gl_layout::row_major "
        "only: it composes its base address row-major and then indexes row*row_stride + col, "
        "giving the column axis an implicit unit stride. For a WMMA accumulator use the staging "
        "overload store<N_THREADS>(dst, acc, scratch, block_row, block_col, warp_rbase, "
        "warp_cbase), which is correct for both layouts; otherwise implement the transposed "
        "traversal rather than passing this descriptor.");
}

/**
 * @brief Cooperative register-mediated LDS -> global tile copy (gfx1250).
 *
 * Inverse of the register-mediated `load(st, gl, idx, row_stride)`: reads
 * each element from the tile's subtile-major/padded slot `lds_offset(flat)`
 * and scatters it back to global memory. Pairs with `load` / `load_async` /
 * `load_tdm`, which all land data in the same LDS address map.
 *
 * The LDS read goes through `detail::lds_ptr` for the same reason the `load` above writes
 * through it, kept symmetric so the pair cannot drift.
 */
template<int N_THREADS = WARP_THREADS, typename T, int ROWS, int COLS,
         ducks::st_shape::all Shape, ducks::gl::row_layout GL, ducks::coord::tile COORD = coord<>>
__device__ inline void store(const GL& dst, const st<T, ROWS, COLS, Shape>& src,
                             const COORD& idx, int row_stride)
{
    constexpr int total_elems = ROWS * COLS;
    const int tid = threadIdx.x;
    const int gr_base = idx.r * ROWS;
    const int gc_base = idx.c * COLS;
    T* base = dst.raw_ptr
            + (((int64_t(idx.b) * dst.depth() + idx.d) * dst.rows() + gr_base)
               * dst.cols() + gc_base);

    const auto* lds = detail::lds_ptr(src.data);

    #pragma unroll
    for (int i = tid; i < total_elems; i += N_THREADS) {
        const int row = i / COLS;
        const int col = i % COLS;
        base[row * row_stride + col] = lds[src.lds_offset(i)];
    }
}

/**
* @brief Refused: a column-major source does not merely move the addresses, it dissolves the
*        16-byte transfer this path is built on.
*
* The base is composed row-major and the walk is `row*row_stride + col`, as in the two routines
* above. But there is a second, sharper reason here: each lane issues ONE `b128`, so the
* `elems_per_load` consecutive values of `col` it covers have to be one contiguous 16-byte run.
* That is true only when the column axis is unit-stride. Under `col_major` those elements are
* `rows()` apart and there is no single transfer that moves them -- the operation is not
* re-addressable, it needs a different instruction shape.
*
* Constrained rather than left on `ducks::gl::all`, where a column-major descriptor is accepted
* and silently deposits `ROWS*COLS` elements gathered from the wrong places.
*/
template<int N_THREADS = WARP_THREADS, bool RATE_ONLY = false, typename T, int ROWS, int COLS,
         ducks::st_shape::all Shape, ducks::gl::col_layout GL, ducks::coord::tile COORD = coord<>>
__device__ inline void load_async(st<T, ROWS, COLS, Shape>& dst, const GL& src,
                                  const COORD& idx, int row_stride, uint32_t cluster_mask = 0)
{
    static_assert(ducks::gl_layout::unhandled<typename GL::layout>,
        "load_async is implemented for ducks::gl_layout::row_major only. It composes its base "
        "address row-major and indexes row*row_stride + col, and -- decisively -- each lane's "
        "single b128 assumes its run of columns is 16 contiguous bytes, which holds only when "
        "the column axis is unit-stride. A column-major source needs a different transfer "
        "shape, not a different address; use load_tdm with a descriptor built for that layout, "
        "or implement the path.");
}

/**
 * @brief Cooperative async global -> LDS tile copy on gfx1250.
 *
 * Lowers to `global_load_async_to_lds_b128` (single-WG) when `cluster_mask == 0`,
 * and to `cluster_load_async_to_lds_b128` (multicast) when non-zero. Each lane
 * issues one 16-byte transfer; the warp covers `8 * N_THREADS` elements per
 * iteration. Drain with `kittens::sync::wait_async()` before consuming.
 *
 * @tparam N_THREADS    Number of threads participating in the load.
 * @param  dst          Destination `st` tile (owns the padded LDS map).
 * @param  src          Global tile descriptor.
 * @param  idx          Tile coordinate inside `src`.
 * @param  row_stride   Element stride between rows in `src`.
 * @param  cluster_mask `M0` cluster multicast mask (0 for single-WG, non-zero for a workgroup cluster).
 *
 * ⚠ This writes `padded(subtile_flat(.))` where `load_tdm` writes plain `padded(row-major)`, so
 * it is layout-compatible with the TDM and with the shared->register `load` only when
 * `subtile_flat` is the identity, i.e. `COLS == Shape::cols`. The static_assert below enforces
 * it; violating it corrupts silently and deterministically.
 *
 * @tparam RATE_ONLY Skip that check, for harnesses that measure transfer rate and never read the
 *                   data back. Never set it in a kernel whose output is verified.
 */
template<int N_THREADS = WARP_THREADS, bool RATE_ONLY = false, typename T, int ROWS, int COLS,
         ducks::st_shape::all Shape, ducks::gl::row_layout GL, ducks::coord::tile COORD = coord<>>
__device__ inline void load_async(st<T, ROWS, COLS, Shape>& dst, const GL& src,
                                  const COORD& idx, int row_stride, uint32_t cluster_mask = 0)
{
    static_assert(sizeof(T) * 8 == 16, "load_async issues one b128 (16B) per lane");
    static_assert(RATE_ONLY || COLS == Shape::cols,
        "load_async layout mismatch: this tile is wider than one subtile column, so "
        "`subtile_flat` is not the identity and load_async writes a different LDS layout "
        "from the one load_tdm writes and kittens::load reads. Ignoring this corrupts roughly "
        "half the output, silently and deterministically, while looking correct at small "
        "sizes. Either keep the tile at cols == Shape::cols, or fill it with load_tdm, or -- "
        "if you are measuring transfer rate and never read the data back -- pass "
        "RATE_ONLY=true explicitly.");
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

        // The gfx1250 async-to-LDS builtins want address-space-qualified
        // pointers (AS(1) global, AS(3) LDS). `reinterpret_cast` cannot add
        // an address space, so route through `uintptr_t` + a C-style cast,
        // matching the pattern used elsewhere in this file for AS(3).
        uintptr_t g_uint = reinterpret_cast<uintptr_t>(base + row * row_stride + col);
        uintptr_t l_uint = reinterpret_cast<uintptr_t>(dst.data + dst.lds_offset(i));
        auto* g_ptr = (detail::i32x4_gvec*)(g_uint);
        auto* l_ptr = (detail::i32x4_lvec*)(l_uint);

        if (cluster_mask) {
            __builtin_amdgcn_cluster_load_async_to_lds_b128(
                g_ptr, l_ptr, 0, 0, static_cast<int>(cluster_mask));
        } else {
            __builtin_amdgcn_global_load_async_to_lds_b128(g_ptr, l_ptr, 0, 0);
        }
    }
}

/**
 * @brief Hardware tile DMA (TDM) global -> LDS load on gfx1250.
 *
 * Issues a single `tensor_load_to_lds` instruction whose D# descriptor
 * encodes the 2D tile shape, source tensor extents, row stride, and optional
 * LDS padding.
 *
 * The transfer is issued once by the whole wave, not per thread: it uses no
 * vector registers (VGPRs) and ignores the active-thread mask, so
 * which threads are active makes no difference. The entire tile is described
 * by a small block of scalar registers.
 *
 * A CU has one TDM per SIMD-pair (a gfx1250 CU is four SIMDx32s grouped into two pairs). 
 * That single engine handles one request stream and is shared by the waves on its pair, so
 * extra issuers don't make the copy faster, they just contend for it and use
 * up its in-flight slots (at most 3 transfers per wave, 6 per SIMD).
 *
 * Two issuers land on different engines iff their warp ids differ in parity: warps with
 * `warpid % 4` in {0, 2} share one, {1, 3} share the other. Neither `warpid % 4` nor
 * `SIMD >> 1` gives the pairing.
 *
 * Drain with `kittens::sync::wait_tdm()`.
 *
 * @param  dst         Destination `st` tile (its shape's pad fields drive the D#).
 * @param  src         Global tile descriptor.
 * @param  idx         Tile coordinate.
 * @param  tensor_rows,tensor_cols  Source tensor extents (elements).
 * @param  row_stride  Source row stride (elements).
 * @param  cluster_mask Optional `workgroup_mask` (0 for single-WG, non-zero
 *                     to switch the load into `CLUSTER_LOAD_ASYNC` micro-ops).
 */
namespace detail {

using v4u32 = unsigned int __attribute__((ext_vector_type(4)));
using v8u32 = unsigned int __attribute__((ext_vector_type(8)));

/**
 * @brief Build the 12-DWord TDM D# (groups 0 + 1) for a 2D tile transfer.
 *
 * Encapsulates the bit-packing shared by `load_tdm` and `load_tdm_arrive`.
 * The LDS padding fields are read from the tile shape (`Shape::pad_interval`
 * / `Shape::pad_amount`).
 *
 * @tparam WITH_BARRIER Set `atomic_barrier_enable` and pack `bar_lds_addr`, so the TDM unit
 *                      auto-arrives at that cell when the transfer retires. A template
 *                      parameter and not `bar_lds_addr != 0`, because 0 is a legal cell -- it
 *                      is the first address `shared_allocator` hands out.
 */
template<typename Shape, int ROWS, int COLS, typename T, bool WITH_BARRIER = false>
__device__ __forceinline__ void build_tdm_descriptor_2d(
    v4u32& g0, v8u32& g1,
    const T* base, T* lds_dst,
    int tensor_rows, int tensor_cols, int row_stride,
    uint32_t cluster_mask, uint32_t bar_lds_addr, uint32_t count = 1)
{
    // ---- Group 0: count, lds_addr, global_addr, type ----
    // count == 0 is the NULL descriptor: no memory copied, no atomic_barrier sent. It is the
    // branch-free skip for a tail transfer, since a TDM cannot be EXEC-masked.
    const uint32_t lds_addr = static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(lds_dst));
    const uint64_t gaddr    = reinterpret_cast<uint64_t>(base);

    g0[0] = count;                                              // count (0 => NULL)
    g0[1] = lds_addr;
    g0[2] = static_cast<uint32_t>(gaddr);
    g0[3] = (static_cast<uint32_t>(gaddr >> 32) & 0x01FFFFFFu) | (2u << 30);

    // ---- Group 1: data_size, padding, dims, stride, optional barrier ----
    // data_size encoded as log2(bytes_per_element).
    constexpr uint32_t data_size_enc = (sizeof(T) == 1) ? 0
                                     : (sizeof(T) == 2) ? 1
                                     : (sizeof(T) == 4) ? 2
                                     : 3;
    constexpr int shape_pad_interval = [] {
        if constexpr (requires { Shape::pad_interval; }) return Shape::pad_interval;
        else return 0;
    }();
    constexpr int shape_pad_amount = [] {
        if constexpr (requires { Shape::pad_amount; }) return Shape::pad_amount;
        else return 0;
    }();
    // pad_interval and pad_amount must be both zero or both non-zero, so all three pad fields
    // are gated on both shape values.
    constexpr bool do_pad = (shape_pad_interval > 0) && (shape_pad_amount > 0);
    constexpr uint32_t pad_enable   = do_pad ? 1u : 0u;
    // pad_interval field (group1 w0 bits 24:22) encodes 2^(enc+1) DWORDs, so the
    // encoder is log2(interval_dwords) - 1. For bf16 / 256 B (=128 elems = 64
    // DWORDs) this is ctz(64)-1 = 5, NOT ctz(64)=6. (Equivalently ctz(...*
    // sizeof(T)/8).)
    constexpr uint32_t pad_int_enc  = do_pad
        ? ( __builtin_ctz(shape_pad_interval * sizeof(T) / 4) - 1 ) : 0;
    // pad_amount field (bits 31:25) encodes (amount_dwords - 1).
    constexpr uint32_t pad_amt_enc  = do_pad
        ? ( (shape_pad_amount * sizeof(T) / 4) - 1 ) : 0;

    // atomic_barrier_enable lives at bit 18 of group 1 word 0
    // (per the MI400 TDM D# layout: w0 = multicast_mask[15:0],
    // data_size[17:16], atomic_barrier_enable[18], iterate_enable[19],
    // pad_enable[20], pad_interval[24:22], pad_amount[31:25]).
    constexpr uint32_t atomic_bar_enable = WITH_BARRIER ? (1u << 18) : 0u;

    // Only the two fields the caller owns are let through: Workgroup_mask[15:0] and
    // early_timeout[21]. Every other w0 bit belongs to a field computed here, so a stray bit in
    // `cluster_mask` must not reach the descriptor -- bit 16 in particular is data_size's low
    // bit, and passing it would declare bf16 operands 4-byte.
    // A mask must name only workgroups that do issue a matching request, and at most 5;
    // otherwise the multicast is silently demoted to a regular request, which costs throughput
    // rather than correctness.
    // Spelled out locally rather than taken from `cluster::`, which this header precedes.
    constexpr uint32_t ET = 1u << 21;
    const uint32_t w0_mcast = (cluster_mask & 0xFFFFu) | (cluster_mask & ET);
    uint32_t w0 = (data_size_enc << 16)
                | (pad_enable    << 20)
                |  atomic_bar_enable
                | (pad_int_enc   << 22)
                | (pad_amt_enc   << 25)
                |  w0_mcast;

    const uint32_t tdim0    = static_cast<uint32_t>(tensor_cols);
    const uint32_t tdim1    = static_cast<uint32_t>(tensor_rows);
    const uint32_t tiledim0 = static_cast<uint32_t>(COLS);
    const uint32_t tiledim1 = static_cast<uint32_t>(ROWS);

    // barrier_addr occupies w1[15:0]; tensor_dim0 lo16 occupies w1[31:16].
    // atomic_barrier_address holds LDS addr bits [18:3] (8-byte-aligned cell),
    // so the address is shifted right by 3 before packing into the 16-bit field.
    // Gated on WITH_BARRIER too, so the field is zero whenever the enable is clear rather
    // than carrying whatever the caller happened to pass.
    uint32_t w1 = (WITH_BARRIER ? ((bar_lds_addr >> 3) & 0xFFFFu) : 0u) | (tdim0 << 16);
    uint32_t w2 = (tdim0 >> 16) | (tdim1 << 16);
    uint32_t w3 = (tdim1 >> 16) | (tiledim0 << 16);
    uint32_t w4 = tiledim1;

    // tensor_dim0_stride (group1 words 5-6, 48-bit) is "in elements of
    // data_size" -- the engine multiplies by data_size internally -- so the row
    // stride is passed in ELEMENTS, not bytes (no * sizeof(T)).
    const uint64_t stride0 = static_cast<uint64_t>(
        static_cast<uint32_t>(row_stride));
    uint32_t w5 = static_cast<uint32_t>(stride0);
    uint32_t w6 = static_cast<uint32_t>(stride0 >> 32);
    uint32_t w7 = 0;

    g1[0] = w0; g1[1] = w1; g1[2] = w2; g1[3] = w3;
    g1[4] = w4; g1[5] = w5; g1[6] = w6; g1[7] = w7;
}

// TDM builtin ABI shim. clang 23 (LLVM 23) added a 5th D# operand -- "group 4"
// (v8i32) -- to `__builtin_amdgcn_tensor_{load_to,store_from}_lds`, documented
// in llvm/IR/IntrinsicsAMDGPU.td (class AMDGPUTensorLoadStore) as: "reserved
// for future targets, use <8 x i32> zeroinitializer for now. This argument
// will be silently ignored." clang 22 (ROCm 7.2, r7.03 baseline) still has the
// 5-operand form. The version guard keeps the known-good clang-22 path emitting
// the exact original 5-operand call; clang 23+ passes an extra zero group.
__device__ inline void emit_tensor_load_to_lds(v4u32 g0, v8u32 g1,
                                               v4u32 g2, v4u32 g3) {
#if defined(__clang_major__) && __clang_major__ >= 23
    v8u32 g4 = {0, 0, 0, 0, 0, 0, 0, 0};
    __builtin_amdgcn_tensor_load_to_lds(g0, g1, g2, g3, g4, 0);
#else
    __builtin_amdgcn_tensor_load_to_lds(g0, g1, g2, g3, 0);
#endif
}

} // namespace detail

/// Refused: the D# bakes the layout in, naming the column axis as contiguous, and the base is
/// composed row-major on top of it. Swapping the caller's `tensor_rows`/`tensor_cols` does not
/// fix the base, so a column-major source needs the axes exchanged AND the base recomposed.
template<typename T, int ROWS, int COLS, ducks::st_shape::all Shape,
         ducks::gl::col_layout GL, ducks::coord::tile COORD = coord<>>
__device__ inline void load_tdm(st<T, ROWS, COLS, Shape>& dst, const GL& src,
                                const COORD& idx,
                                int tensor_rows, int tensor_cols, int row_stride,
                                uint32_t cluster_mask = 0, uint32_t count = 1)
{
    static_assert(ducks::gl_layout::unhandled<typename GL::layout>,
        "load_tdm is implemented for ducks::gl_layout::row_major only. The D# it builds puts the "
        "COLUMN extent in tensor_dim0/tile_dim0 and the row stride in tensor_dim0_stride, which "
        "describes a tensor whose columns are contiguous; the base address is composed row-major "
        "too. Passing tensor_rows/tensor_cols the other way round does not correct the base. "
        "Build a column-major descriptor path rather than passing this descriptor.");
}

template<typename T, int ROWS, int COLS, ducks::st_shape::all Shape,
         ducks::gl::row_layout GL, ducks::coord::tile COORD = coord<>>
__device__ inline void load_tdm(st<T, ROWS, COLS, Shape>& dst, const GL& src,
                                const COORD& idx,
                                int tensor_rows, int tensor_cols, int row_stride,
                                uint32_t cluster_mask = 0, uint32_t count = 1)
{
    const int gr_base = idx.r * ROWS;
    const int gc_base = idx.c * COLS;
    const T* base = src.raw_ptr
                  + (((int64_t(idx.b) * src.depth() + idx.d) * src.rows() + gr_base)
                     * src.cols() + gc_base);

    detail::v4u32 g0;
    detail::v8u32 g1;
    // count==0 => ISA NULL descriptor: moves no memory, sends no barrier. This is the
    // branch-free tail skip; a clamped-index real fill instead races the epilogue's
    // C staging for the same LDS.
    detail::build_tdm_descriptor_2d<Shape, ROWS, COLS, T,
                                    /*WITH_BARRIER=*/false>(
        g0, g1, base, dst.data, tensor_rows, tensor_cols, row_stride,
        cluster_mask, /*bar_lds_addr=*/ 0, count);

    detail::v4u32 g2 = {0, 0, 0, 0};
    detail::v4u32 g3 = {0, 0, 0, 0};
    detail::emit_tensor_load_to_lds(g0, g1, g2, g3);
}

/**
 * @brief TDM load that auto-arrives at an LDS barrier on completion.
 * @experimental
 *
 * Sets `atomic_barrier_enable` in the D# so the TDM unit emits a
 * `DS_ATOMIC_ASYNC_BARRIER_ARRIVE_B64` on `bar` after the transfer retires.
 * The consumer waits on `bar`'s phase flip via
 * `kittens::sync::wait_barrier(bar, phase)` instead of draining the global
 * `tensorcnt`, leaving unrelated TDM transfers in flight.
 *
 * The barrier must be primed via `kittens::sync::init_barrier(bar, count)`
 * before the first call referencing it. `count` is the number of
 * `load_tdm_arrive` invocations that target this barrier per phase.
 *
 * ⚠ The arrive is a signal, not a fence: only an explicit `wait_tdm` drains TENSORCNT, so
 * publishing through a barrier without one reads a partly-filled tile. A partial `wait_tdm<N>()`
 * suffices, but `N` must be no larger than the number of transfers guaranteed outstanding on
 * every path, not in the steady state -- where fewer have been issued it retires nothing while
 * reading like a fence.
 *
 * @note The D# bit positions for `atomic_barrier_enable` (`w0` bit 18) and
 * `atomic_barrier_address` (`w1[15:0]`) match the field table documented
 * in the Triton AMD backend (third_party/amd/lib/TritonAMDGPUToLLVM/
 * TDMUtility.cpp lines 224-264). The Triton lowering itself does not use
 * the D# auto-arrive path -- it follows `load_tdm` with an explicit
 * `wait_tdm()` + `async_barrier_arrive()` sequence. This overload is provided for
 * runtimes that model TDM auto-arrive natively; on simulators that don't,
 * use the explicit-arrive pattern instead.
 *
 * @param bar  Pointer to a 64-bit LDS barrier counter (a `sync::barrier_lds`
 *             cell). Must point at LDS storage; must be 8-byte aligned.
 *             LDS byte address 0 is a valid cell: the enable is a template
 *             argument on the descriptor builder, not `bar != 0`.
 */

/// Refused: `load_tdm` plus an auto-arrive bit, so it carries `load_tdm`'s assumption too.
/// Identical base composition and D#; the arrive bit changes nothing about the addressing.
template<typename T, int ROWS, int COLS, ducks::st_shape::all Shape,
         ducks::gl::col_layout GL, ducks::coord::tile COORD = coord<>>
__device__ inline void load_tdm_arrive(
    st<T, ROWS, COLS, Shape>& dst, const GL& src, const COORD& idx,
    int tensor_rows, int tensor_cols, int row_stride,
    uint64_t* bar, uint32_t cluster_mask = 0)
{
    static_assert(ducks::gl_layout::unhandled<typename GL::layout>,
        "load_tdm_arrive is implemented for ducks::gl_layout::row_major only, for exactly the "
        "reasons load_tdm is: the D# names the column axis as the contiguous one and the base "
        "address is composed row-major. The auto-arrive bit changes nothing about the "
        "addressing.");
}

template<typename T, int ROWS, int COLS, ducks::st_shape::all Shape,
         ducks::gl::row_layout GL, ducks::coord::tile COORD = coord<>>
__device__ inline void load_tdm_arrive(
    st<T, ROWS, COLS, Shape>& dst, const GL& src, const COORD& idx,
    int tensor_rows, int tensor_cols, int row_stride,
    uint64_t* bar, uint32_t cluster_mask = 0)
{
    const int gr_base = idx.r * ROWS;
    const int gc_base = idx.c * COLS;
    const T* base = src.raw_ptr
                  + (((int64_t(idx.b) * src.depth() + idx.d) * src.rows() + gr_base)
                     * src.cols() + gc_base);

    const uint32_t bar_lds_addr = static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(bar));

    detail::v4u32 g0;
    detail::v8u32 g1;
    detail::build_tdm_descriptor_2d<Shape, ROWS, COLS, T,
                                    /*WITH_BARRIER=*/true>(
        g0, g1, base, dst.data, tensor_rows, tensor_cols, row_stride,
        cluster_mask, bar_lds_addr);

    detail::v4u32 g2 = {0, 0, 0, 0};
    detail::v4u32 g3 = {0, 0, 0, 0};
    detail::emit_tensor_load_to_lds(g0, g1, g2, g3);
}

/**
 * @brief Cooperative L2 prefetch for an upcoming tile.
 *
 * Lowers to `global_prefetch_b8` per participating lane. Fire-and-forget: no VGPR result and
 * nothing to wait on. The 2nd builtin argument is the temporal hint, not a cache scope --
 * 0 is `TH_LOAD_RT`, 1 `NT`, 2 `HT`, 3 `LU` -- and staging into GL2 for a later fill wants the
 * line kept, hence the 2 below.
 *
 * Walks at 16 B per prefetch while the instruction fills a whole line, so it issues about 8x
 * the instructions needed for the same coverage.
 */

template<int ROWS = 0, int COLS = 0, int N_THREADS = WARP_THREADS,
         ducks::gl::col_layout GL, ducks::coord::tile COORD = coord<>>
__device__ inline void prefetch_l2(const GL& src, const COORD& idx, int row_stride)
{
    static_assert(ducks::gl_layout::unhandled<typename GL::layout>,
        "prefetch_l2 is implemented for ducks::gl_layout::row_major only: it composes its base "
        "address row-major and walks row*row_stride + col, so against a column-major tensor it "
        "warms the wrong cache lines. That cannot corrupt an output -- it just silently stops "
        "prefetching what the kernel is about to read, which no correctness gate can see.");
}

template<int ROWS = 0, int COLS = 0, int N_THREADS = WARP_THREADS,
         ducks::gl::row_layout GL, ducks::coord::tile COORD = coord<>>
__device__ inline void prefetch_l2(const GL& src, const COORD& idx, int row_stride)
{
    static_assert(ROWS > 0 && COLS > 0, "ROWS and COLS must be specified");
    using T = typename GL::dtype;
    constexpr int elems_per_pf = 16 / sizeof(T);
    constexpr int total_elems  = ROWS * COLS;
    const int tid = threadIdx.x;
    const int gr_base = idx.r * ROWS;
    const int gc_base = idx.c * COLS;
    const T* base = src.raw_ptr
                  + (((int64_t(idx.b) * src.depth() + idx.d) * src.rows() + gr_base)
                     * src.cols() + gc_base);

    #pragma unroll
    for (int i = tid * elems_per_pf; i < total_elems;
         i += N_THREADS * elems_per_pf)
    {
        const int row = i / COLS;
        const int col = i % COLS;
        const T* addr = base + row * row_stride + col;
        // 2 = th:TH_LOAD_HT -- retain in the higher-level (L2/MALL) caches, which
        // is what "use L2 as another buffering level" needs. Verified from asm.
        __builtin_amdgcn_global_prefetch(
            (const void __attribute__((address_space(1)))*)addr, 2);
    }
}

}
