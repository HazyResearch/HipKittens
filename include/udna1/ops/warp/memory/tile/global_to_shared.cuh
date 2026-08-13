/**
 * @file
 * @brief Functions for transferring data directly between global and shared memory and back.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"
#include "../util/util.cuh"

namespace kittens {

/// Refused: the body's `row * row_stride + col` assumes a unit-stride column axis, which
/// inverts under col_major and would silently address the wrong elements. A col-LAYOUT tile has its
/// own overload below; this one catches the row-layout tile, for which the traversal is still wrong.
template<int N_THREADS = WARP_THREADS, int axis = 2, bool assume_aligned = false,
        ducks::st::all ST, ducks::gl::col_layout GL,
        ducks::coord::tile COORD = coord<ST>>
__device__ static inline void store(const GL &dst, const ST &src, const COORD &idx) {
    static_assert(ducks::gl_layout::unhandled<typename GL::layout>,
        "store(dst, shared_tile, coord) is implemented for ducks::gl_layout::row_major only, unless "
        "the shared tile is col-layout: its addressing gives the column axis an implicit unit "
        "stride. A column-major destination wants the tile streamed out in COLUMN-major linear "
        "order, which is a different traversal and not merely a different stride -- stage the "
        "accumulator into a col-layout tile (st_bf<R, C, Shape, ducks::st_layout::col>) and store "
        "that, which walks the right way round.");
}

/**
 * @brief Move one thread's run of `E` logical elements from a shared tile to global memory.
 *
 * @tparam WIDE Take the single sized global store. A template parameter and not a runtime flag,
 *              because the width has to be selected by `if constexpr`: a per-iteration alignment
 *              branch inside the unrolled body stops the store widening at all. The caller decides
 *              once, wave-uniformly.
 *
 * The run is gathered into a register buffer and written as one sized store, since contiguous
 * addresses alone do not get a wide store out of the compiler.
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

    const int row = flat / ST::cols, col = flat % ST::cols;
    const T *in  = &src.data[ST::idx(row, col)];
    /* 32-bit index arithmetic. `row * row_stride + col` cannot overflow -- `row` is bounded by the
     * tile and `row_stride` by the tensor -- and the whole tensor offset is already carried in
     * `dst_ptr`, which `gl::idx()` computed in 64 bits. Widening this to int64 costs the SGPR base:
     * the backend gives up on proving the base wave-uniform and materialises it into a VGPR pair. */
    U       *out = dst_ptr + row * row_stride + col;

    T buf[E];
    // Whether a run is one contiguous aligned span is a property of the SHAPE, so it is asked of the
    // shape directly; the register->LDS stage in shared_to_register.cuh asks the same question.
    if constexpr (ducks::st_shape::run_is_contiguous<typename ST::shape, E>() && sized_word_ok<BYTES>) {
        using W = sized_word_t<BYTES>;
        *reinterpret_cast<W*>(buf) = *reinterpret_cast<const W*>(in);
    } else {
        #pragma unroll
        for (int j = 0; j < E; j++) buf[j] = in[j];
    }

    if constexpr (WIDE) {
        // `T` and `U` are the same type (asserted by the caller), so the gathered run already
        // holds the destination bytes and the conversion below is the identity.
        using W = sized_word_t<BYTES>;
        *reinterpret_cast<W*>(out) = *reinterpret_cast<const W*>(buf);
    } else {
        #pragma unroll
        for (int j = 0; j < E; j++)
            out[j] = kittens::base_types::convertor<U, T>::convert(buf[j]);
    }
}

/// @brief Stream a whole row-layout tile out in row-major order, for one store width. Counterpart of
/// `st_col_to_global`; `WIDE` is a template parameter for the same reason.
template<bool WIDE, int N_THREADS, ducks::st::all ST, typename U>
__device__ __forceinline__ static void st_row_to_global(
    U *dst_ptr, int row_stride, const ST &src, int tid, int warpid)
{
    using T = typename ST::dtype;
    constexpr int bytes_per_thread = ST::underlying_subtile_bytes_per_thread;
    constexpr int elems_per_thread = bytes_per_thread / sizeof(T);
    constexpr int memcpy_per_tile  = ST::rows * ST::cols * sizeof(T) / (bytes_per_thread * N_THREADS);

    if constexpr (memcpy_per_tile > 0) {
        #pragma unroll
        for (int i = 0; i < memcpy_per_tile; i++) {
            st_run_to_global<WIDE, elems_per_thread>(
                dst_ptr, row_stride, src, (tid + i * N_THREADS) * elems_per_thread);
        }
    }
    if constexpr (memcpy_per_tile * (bytes_per_thread * N_THREADS)
                  != ST::rows * ST::cols * sizeof(T)) {
        constexpr int leftover_bytes   = ST::rows * ST::cols * sizeof(T)
                                       - memcpy_per_tile * (bytes_per_thread * N_THREADS);
        constexpr int leftover_threads = leftover_bytes / bytes_per_thread;
        constexpr int leftover_warps   = leftover_threads / kittens::WARP_THREADS;
        if (warpid < leftover_warps) {
            st_run_to_global<WIDE, elems_per_thread>(
                dst_ptr, row_stride, src,
                (tid + memcpy_per_tile * N_THREADS) * elems_per_thread);
        }
    }
}

/**
 * @brief Stream a whole col-layout tile out to a column-major destination, for one store width.
 *
 * `WIDE` is a template parameter and not a runtime flag because the sized store has to be selected by
 * `if constexpr`: a runtime test inside the unrolled body is exactly what stops the store widening.
 * The caller decides once, wave-uniformly, and calls this for the arm it wants.
 */
template<bool WIDE, int E, int N_THREADS, ducks::st::col_layout ST, typename U>
__device__ __forceinline__ static void st_col_to_global(
    U *dst_ptr, int ldm, const ST &src, int tid)
{
    using T = typename ST::dtype;
    constexpr int BYTES = E * (int)sizeof(T);
    constexpr int total = ST::rows * ST::cols;

    #pragma unroll
    for (int base = tid * E; base < total; base += N_THREADS * E) {
        const int c = base / ST::rows, r = base % ST::rows;
        const T *in  = &src.data[ST::idx(r, c)];
        U       *out = dst_ptr + (int64_t)c * ldm + r;
        if constexpr (WIDE) {
            using W = sized_word_t<BYTES>;
            *reinterpret_cast<W*>(out) = *reinterpret_cast<const W*>(in);
        } else {
            #pragma unroll
            for (int j = 0; j < E; j++) out[j] = src.data[ST::idx(r + j, c)];
        }
    }
}

/**
 * @brief Whether the destination admits the sized store for a run of `E` elements.
 *
 * Two things have to hold: the run is a width the ISA has, and the address it lands on carries
 * that width's alignment. The second depends on the caller's leading dimension and tile origin,
 * runtime values no type can see, so it is tested rather than silently required. `assume_aligned`
 * drops the test for a caller that can certify it.
 *
 * The answer is the same for every lane and every iteration, so it is taken once, here, and handed
 * to the streamer as a template argument. Asking it inside the unrolled body instead leaves the
 * width a runtime choice and the compiler falls back on one narrow store per element: the branch
 * costs precisely the widening it guards.
 *
 * Only the destination is asked about. Whether the run is also contiguous in LDS is a property of
 * the shape, and the two streamers need that answer in different places -- `st_col_to_global`
 * reads the run with one sized load, so its caller conjoins it onto this one, while
 * `st_run_to_global` stages through registers and asks the shape itself, per run.
 */
template<ducks::st::all ST, int E, bool assume_aligned, typename U>
__device__ __forceinline__ static bool store_is_wide(const U *dst_ptr, int ldm) {
    using T = typename ST::dtype;
    constexpr int BYTES = E * (int)sizeof(T);
    return sized_word_ok<BYTES> &&
        (assume_aligned || (((ldm % E) == 0) &&
                            ((reinterpret_cast<uintptr_t>(dst_ptr) % BYTES) == 0)));
}

/**
 * @brief Stream a col-layout shared tile out to a column-major global destination.
 *
 * The column-major counterpart of the row-major overload below, and the second half of the
 * decomposed epilogue: it walks the tile in COLUMN-major linear order, so each thread's run of `E`
 * elements is `E` consecutive ROWS of one column -- contiguous in a column-major tensor, and
 * contiguous in a col-layout tile -- and the wave covers one contiguous span at each end.
 *
 * The tile is read through `ST::idx`, the same map `store(st, rt)` staged it with. That pairing is
 * the point: the two halves of an epilogue have to agree on where an element lives.
 */
template<int N_THREADS = WARP_THREADS, int axis = 2, bool assume_aligned = false,
        ducks::st::col_layout ST, ducks::gl::col_layout GL,
        ducks::coord::tile COORD = coord<ST>>
__device__ static inline void store(const GL &dst, const ST &src, const COORD &idx) {

    using T = typename ST::dtype;
    using U = typename GL::dtype;
    static_assert(std::is_same_v<T, U>, "T and U must be the same type");
    static_assert(!std::is_same_v<T, fp8e4m3>, "Unsupported type for store");

    constexpr int E = ST::underlying_subtile_bytes_per_thread / sizeof(T);
    static_assert(ST::rows % E == 0,
        "a thread's run must not straddle a tile column: ST::rows must be a multiple of "
        "bytes_per_thread/sizeof(T)");

    constexpr int num_warps = N_THREADS / kittens::WARP_THREADS;
    const int tid = (kittens::warpid() % num_warps) * kittens::WARP_THREADS + kittens::laneid();

    /* Leading dimension is the distance between neighbouring COLUMNS, which under col_major is
     * `stride<3>()`; `stride<2>()` is 1 here, so taking the row-major overload's `stride<axis>()`
     * would silently index as if the tensor were one column wide. The tile origin goes through the
     * descriptor's own `idx()`, which is layout-aware, rather than being composed here. */
    const int ldm = (int)dst.template stride<3>();
    U *dst_ptr = &dst[coord<>{idx.b, idx.d, idx.r * ST::rows, idx.c * ST::cols}];

    /* One wave-uniform decision for the whole tile. The shape's answer is conjoined onto the
     * destination's because `st_col_to_global` takes the run straight out of LDS with one sized
     * load: padding falling inside the run rules the wide arm out on the source side too. */
    const bool wide = ducks::st_shape::run_is_contiguous<typename ST::shape, E>()
                      && store_is_wide<ST, E, assume_aligned>(dst_ptr, ldm);

    if (wide) st_col_to_global<true,  E, N_THREADS>(dst_ptr, ldm, src, tid);
    else      st_col_to_global<false, E, N_THREADS>(dst_ptr, ldm, src, tid);
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
template<int N_THREADS = WARP_THREADS, int axis = 2, bool assume_aligned = false,
        ducks::st::all ST, ducks::gl::row_layout GL,
        ducks::coord::tile COORD = coord<ST>>
__device__ static inline void store(const GL &dst, const ST &src, const COORD &idx) {

    using T = typename ST::dtype;
    using U = typename GL::dtype;

    static_assert(std::is_same_v<T, U>, "T and U must be the same type");
    static_assert(!std::is_same_v<T, fp8e4m3>, "Unsupported type for store");

    constexpr int bytes_per_thread = ST::underlying_subtile_bytes_per_thread;
    constexpr int elems_per_thread = bytes_per_thread / sizeof(T);

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

    /* One wave-uniform decision for the whole tile, on the destination alone. The shape's
     * contiguity does not gate it here: the run is staged through registers, so a padded run is
     * gathered element-wise by `st_run_to_global` and the wide store still stands. */
    const bool wide = store_is_wide<ST, elems_per_thread, assume_aligned>(dst_ptr, row_stride);

    if (wide) st_row_to_global<true,  N_THREADS>(dst_ptr, row_stride, src, tid, warpid);
    else      st_row_to_global<false, N_THREADS>(dst_ptr, row_stride, src, tid, warpid);
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

/// Refused: the inverse of the store below, with the same two row-major assumptions, and it
/// inverts under col_major the same way.
template<int N_THREADS = WARP_THREADS, typename T, int ROWS, int COLS,
         ducks::st_shape::all Shape, ducks::gl::col_layout GL, ducks::coord::tile COORD = coord<>>
__device__ inline void store(const GL& dst, const st<T, ROWS, COLS, Shape>& src,
                             const COORD& idx, int row_stride)
{
    static_assert(ducks::gl_layout::unhandled<typename GL::layout>,
        "store(dst, shared_tile, coord, row_stride) is implemented for ducks::gl_layout::row_major "
        "only: it composes its base address row-major and then indexes row*row_stride + col, "
        "giving the column axis an implicit unit stride. For a WMMA accumulator, store it straight "
        "out with store(dst, acc, coord), or stage it through a col-layout shared tile and use "
        "store<N_THREADS>(dst, shared_tile, coord), which walks column-major.");
}

/**
 * @brief Cooperative LDS -> global tile copy (gfx1250).
 *
 * Reads each element from its slot via `ST::idx` and scatters it back to global memory. Pairs
 * with `load` and `tdm::load_async`, which land data in the same LDS address map. There is no
 * direct LDS -> global engine outside `tdm::store_async`, so this one does stage through
 * registers.
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

    #pragma unroll
    for (int i = tid; i < total_elems; i += N_THREADS) {
        const int row = i / COLS;
        const int col = i % COLS;
        base[row * row_stride + col] = src.data[src.idx(row, col)];
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
__device__ inline void load(st<T, ROWS, COLS, Shape>& dst, const GL& src,
                                  const COORD& idx, int row_stride, uint32_t cluster_mask = 0)
{
    static_assert(ducks::gl_layout::unhandled<typename GL::layout>,
        "load is implemented for ducks::gl_layout::row_major only. It composes its base "
        "address row-major and indexes row*row_stride + col, and -- decisively -- each lane's "
        "single b128 assumes its run of columns is 16 contiguous bytes, which holds only when "
        "the column axis is unit-stride. A column-major source needs a different transfer "
        "shape, not a different address; use tdm::load_async with a descriptor built for that layout, "
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
 * ⚠ Restricted to tiles exactly one subtile column wide (`COLS == Shape::cols`), enforced by the
 * static_assert below. Violating it corrupts silently and deterministically.
 *
 * @tparam RATE_ONLY Skip that check, for harnesses that measure transfer rate and never read the
 *                   data back. Never set it in a kernel whose output is verified.
 */
template<int N_THREADS = WARP_THREADS, bool RATE_ONLY = false, typename T, int ROWS, int COLS,
         ducks::st_shape::all Shape, ducks::gl::row_layout GL, ducks::coord::tile COORD = coord<>>
__device__ inline void load(st<T, ROWS, COLS, Shape>& dst, const GL& src,
                                  const COORD& idx, int row_stride, uint32_t cluster_mask = 0)
{
    static_assert(sizeof(T) * 8 == 16, "load issues one b128 (16B) per lane");
    static_assert(RATE_ONLY || COLS == Shape::cols,
        "load is restricted to tiles one subtile column wide (cols == Shape::cols). "
        "Ignoring this corrupts roughly "
        "half the output, silently and deterministically, while looking correct at small "
        "sizes. Either keep the tile at cols == Shape::cols, or fill it with tdm::load_async, or -- "
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
        uintptr_t l_uint = reinterpret_cast<uintptr_t>(dst.data + dst.idx(row, col));
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
