/**
 * @file
 * @brief Functions for transferring data directly between global and shared memory and back.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"
#include "../util/util.cuh"

namespace kittens {

// Staging vector for the SPLIT fill (see `load`'s SPLIT parameter). Exposed at
// namespace scope so a kernel can hold the staged registers ACROSS a K-step,
// which is what decouples the global load from the ds_store that consumes it.
typedef float g2s_stage_vec __attribute__((ext_vector_type(4)));


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
    if constexpr (ducks::st_shape::run_is_contiguous<typename ST::shape, E, typename ST::dtype>() && sized_word_ok<BYTES>) {
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
    const bool wide = ducks::st_shape::run_is_contiguous<typename ST::shape, E, typename ST::dtype>()
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
 * Global <-> LDS tile transfers on gfx1201
 *
 * gfx1201 has no direct global -> LDS datapath: none of gfx1250's
 * `global_load_async_to_lds_*`, `cluster_load_async_to_lds_*` or
 * `tensor_load_to_lds` (TDM) exist here, so this tree has no `load_async`,
 * `load_tdm` or `prefetch_l2`. A tile copy is always `buffer_load -> VGPR ->
 * ds_store`, through `load(st, gl, idx, row_stride)` below, which dispatches
 * through the shared tile `st` and its LDS address map.
 */

/**
 * @brief Register staging buffer for a split global -> LDS tile copy.
 *
 * gfx1201 has no direct global->LDS datapath, so a tile copy is always
 * `buffer_load -> VGPR -> ds_store`. Holding the intermediate VGPRs in an
 * explicit, caller-owned object rather than inside one function lets the two
 * halves be separated in time: `stage()` issues the global loads, and
 * `commit()` writes them to LDS. Anything placed between the two -- most
 * usefully the WMMA work of the *previous* K step -- executes while those
 * loads are in flight.
 *
 * That is the software-pipelining trick usually spelled "prefetch global read"
 * (Tensile's PrefetchGlobalRead). It costs `4 * per_thread` VGPRs held live
 * across the compute section, so check occupancy with
 * `-Rpass-analysis=kernel-resource-usage` after changing tile shapes.
 *
 * `valid` records whether the fast path applied at stage() time; if it did not,
 * commit() falls back to the scalar element-wise copy, which re-reads global
 * and therefore is not prefetched. Only edge tiles take that path.
 */
template<int N_THREADS, typename T, int ROWS, int COLS>
struct gts_staged {
    static constexpr int VEC        = 16 / sizeof(T);
    static constexpr int nvec       = (ROWS * COLS) / VEC;
    static constexpr int per_thread = (nvec + N_THREADS - 1) / N_THREADS;
    typedef float f32x4 __attribute__((ext_vector_type(4)));
    f32x4 v[per_thread];
    bool  valid = false;
};

/**
 * @brief Issue the global loads for a tile into `s`, without touching LDS.
 *
 * Mirrors `load()`'s fast-path conditions exactly; when they do not hold,
 * `s.valid` is left false and no loads are issued.
 */
template<int N_THREADS = WARP_THREADS, typename T, int ROWS, int COLS,
         ducks::st_shape::all Shape, ducks::gl::row_layout GL, ducks::coord::tile COORD = coord<>>
__device__ inline void stage(gts_staged<N_THREADS, T, ROWS, COLS>& s,
                             const st<T, ROWS, COLS, Shape>& dst, const GL& src,
                             const COORD& idx, int row_stride,
                             int col_limit = 0x7fffffff,
                             int row_limit = 0x7fffffff)
{
    using S = gts_staged<N_THREADS, T, ROWS, COLS>;
    constexpr int VEC = S::VEC;
    constexpr int total_elems = ROWS * COLS;
    const int tid = threadIdx.x;
    const int gr_base = idx.r * ROWS;
    const int gc_base = idx.c * COLS;
    const T* base = src.raw_ptr
                  + (((int64_t(idx.b) * src.depth() + idx.d) * src.rows() + gr_base)
                     * src.cols() + gc_base);

    s.valid = false;
    if constexpr (COLS % VEC == 0 && total_elems % VEC == 0
                  && ducks::st_shape::run_is_contiguous<Shape, VEC, T>()) {
        // NO ALIGNMENT CONDITION.
        //
        // This used to also require `row_stride % VEC == 0` and a 16-byte aligned
        // base. Both are unnecessary, and requiring them was expensive: with
        // row_stride = K, any K not a multiple of 8 made this path invalid on
        // EVERY iteration, so the caller fell back to the scalar element-by-
        // element loader -- ~30x per element, and a 70% throughput loss overall
        // whenever K % 8 != 0.
        //
        // Measured on gfx1201:
        //   raw_buffer_load_b128 returns CORRECT data at every 2-byte-aligned
        //   offset -- verified at element offsets 0..9 -- and costs
        //   8.77 TB/s aligned, 8.36 at 8B, 8.37 at 4B, 5.81 at 2B.
        // So misalignment is correct and costs at most 34%, against 30x for
        // scalarising.
        //
        // The per-DWORD straddle hazard does not apply here: the two bounds
        // tests below already restrict this path to tiles that are ENTIRELY in
        // range, so no load can reach num_records at all.
        if (gc_base + COLS <= col_limit &&
            gr_base + ROWS <= row_limit) {
            constexpr uint32_t kGfx12RawBufferCfg = BUFFER_RESOURCE_CONFIG;
            // The whole tensor, not one matrix: elem_base below includes the
            // (batch, depth) offset from raw_ptr, so a one-matrix range put
            // every b > 0 or d > 0 tile out of range, which reads as zeros.
            const int64_t mat_elems = (int64_t)src.batch() * src.depth() * src.rows() * src.cols();
            auto rsrc = __builtin_amdgcn_make_buffer_rsrc(
                (void*)src.raw_ptr, /*stride*/0,
                (int)(mat_elems * sizeof(T)), kGfx12RawBufferCfg);
            const int elem_base =
                // 32-BIT. This is a buffer-descriptor element offset and the
                // hardware's offset operand is 32-bit, so computing it as
                // int64_t emitted a v_lshlrev_b64 plus a
                // v_add_co_u32 / v_add_co_ci_u32_e64 carry pair whose high half
                // was then discarded by this very cast -- provably dead work, in
                // the K loop, once per tile load. The same defect was found and
                // fixed in store_acc16 long ago; the loaders still had it.
                //
                // Range: a buffer descriptor cannot address beyond 4 GB, which
                // at bf16 is 2^31 elements, so int is exactly as capable as the
                // hardware it feeds.
                ((idx.b * (int)src.depth() + idx.d) * (int)src.rows()
                       + gr_base) * (int)src.cols() + gc_base;
            #pragma unroll
            for (int u = 0; u < S::per_thread; ++u) {
                const int vi = tid + u * N_THREADS;
                if (S::nvec % N_THREADS != 0 && vi >= S::nvec) break;
                const int i   = vi * VEC;
                const int row = i / COLS;
                const int col = i % COLS;
                const int off = (elem_base + row * row_stride + col) * (int)sizeof(T);
                s.v[u] = __builtin_amdgcn_raw_buffer_load_b128(rsrc, off, 0, 0);   // see the voffset/soffset note in load()
            }
            s.valid = true;
        }
    }
}

/**
 * @brief Write a previously staged tile into LDS.
 *
 * If `stage()` took the fast path this is pure `ds_store_b128`; the loads have
 * already been issued and only the wait for them remains. Otherwise it performs
 * the scalar bounds-checked copy, identical to `load()`'s fallback.
 */
template<int N_THREADS = WARP_THREADS, typename T, int ROWS, int COLS,
         ducks::st_shape::all Shape, ducks::gl::all GL, ducks::coord::tile COORD = coord<>>
__device__ inline void commit(st<T, ROWS, COLS, Shape>& dst,
                              const gts_staged<N_THREADS, T, ROWS, COLS>& s,
                              const GL& src, const COORD& idx, int row_stride,
                              int col_limit = 0x7fffffff,
                              int row_limit = 0x7fffffff)
{
    static_assert(ducks::st_shape::run_is_contiguous<Shape, 16 / sizeof(T), T>(),
        "this commit writes 16-byte runs, and this shared-tile shape does not keep a 16-byte run "
        "contiguous (its swizzle is finer); use a padded shape or st_16x16");
    using S = gts_staged<N_THREADS, T, ROWS, COLS>;
    constexpr int VEC = S::VEC;
    constexpr int total_elems = ROWS * COLS;
    const int tid = threadIdx.x;

    if (s.valid) {
        typedef typename S::f32x4 f32x4;
        typedef __attribute__((address_space(3))) f32x4 lds_f32x4_w;
        #pragma unroll
        for (int u = 0; u < S::per_thread; ++u) {
            const int vi = tid + u * N_THREADS;
            if (S::nvec % N_THREADS != 0 && vi >= S::nvec) break;
            *(lds_f32x4_w*)(&dst.data[dst.lds_offset(vi * VEC)]) = s.v[u];
        }
        return;
    }

    const int gr_base = idx.r * ROWS;
    const int gc_base = idx.c * COLS;
    const T* base = src.raw_ptr
                  + (((int64_t(idx.b) * src.depth() + idx.d) * src.rows() + gr_base)
                     * src.cols() + gc_base);
    #pragma unroll
    for (int i = tid; i < total_elems; i += N_THREADS) {
        const int row = i / COLS;
        const int col = i % COLS;
        const bool in_bounds = (gc_base + col < col_limit) &&
                               (gr_base + row < row_limit);
        dst.data[dst.lds_offset(i)] =
            in_bounds ? base[row * row_stride + col]
                      : base_types::convertor<T, float>::convert(0.0f);
    }
}

/**
 * @brief `commit()` without the scalar fallback.
 *
 * The general `commit()` keeps the source global tile, its coordinate and the
 * two limits live so it can fall back to an element-wise copy when `stage()`
 * declined the fast path. When the caller has already established that the tile
 * is interior, that liveness is pure cost -- and on a kernel sitting one
 * register above an occupancy boundary it is the difference between 8 and 7
 * waves per SIMD.
 *
 * Caller must guarantee `s.valid`.
 */
template<int N_THREADS = WARP_THREADS, typename T, int ROWS, int COLS,
         ducks::st_shape::all Shape>
__device__ inline void commit_fast(st<T, ROWS, COLS, Shape>& dst,
                                   const gts_staged<N_THREADS, T, ROWS, COLS>& s)
{
    static_assert(ducks::st_shape::run_is_contiguous<Shape, 16 / sizeof(T), T>(),
        "this commit writes 16-byte runs, and this shared-tile shape does not keep a 16-byte run "
        "contiguous (its swizzle is finer); use a padded shape or st_16x16");
    using S = gts_staged<N_THREADS, T, ROWS, COLS>;
    constexpr int VEC = S::VEC;
    typedef typename S::f32x4 f32x4;
    typedef __attribute__((address_space(3))) f32x4 lds_f32x4_w;
    const int tid = threadIdx.x;
    #pragma unroll
    for (int u = 0; u < S::per_thread; ++u) {
        const int vi = tid + u * N_THREADS;
        if (S::nvec % N_THREADS != 0 && vi >= S::nvec) break;
        *(lds_f32x4_w*)(&dst.data[dst.lds_offset(vi * VEC)]) = s.v[u];
    }
}

/// Refused: `load` composes its base address row-major and indexes row*row_stride + col, and each
/// lane's wide load assumes its run of columns is contiguous -- true only for a unit-stride column
/// axis. A column-major source needs a different transfer shape, not a different address.
template<int N_THREADS = WARP_THREADS, typename T, int ROWS, int COLS,
         ducks::st_shape::all Shape, ducks::gl::col_layout GL, ducks::coord::tile COORD = coord<>>
__device__ inline void load(st<T, ROWS, COLS, Shape>& dst, const GL& src,
                            const COORD& idx, int row_stride)
{
    static_assert(ducks::gl_layout::unhandled<typename GL::layout>,
        "load(shared_tile, src, coord, row_stride) is implemented for ducks::gl_layout::row_major only: "
        "it indexes row*row_stride + col and loads each lane's run of columns as one contiguous "
        "transfer, which holds only when the column axis is unit-stride.");
}

/**
 * @brief Cooperative register-mediated global -> LDS tile copy (gfx1250 baseline).
 *
 * Plain `global_load` -> VGPR -> `ds_store` path. Use this when no async
 * intrinsic is available or for correctness baselines. The destination
 * `st` tile owns the subtile-major + padding LDS address map.
 */
// ALWAYS_VEC: the caller guarantees `row_stride % VEC == 0` and a 16-byte
// aligned base, so the vectorised buffer-load path is unconditional and the
// scalar fallback is not emitted at all.
//
// This matters for what the fallback drags in, not for the two compares. Its
// per-element path uses POINTER dereferences, so the compiler emits
// v_lshlrev_b64 plus a v_add_co_u32 / v_add_co_ci_u32_e64 carry pair to form a
// 64-bit address -- and hoists that arithmetic ABOVE the `s_cbranch_execz` that
// skips the loads. The loads never run in the fast case; the address maths and
// its two `s_wait_alu` are paid on every K iteration regardless.
//
// In gemm_fat's FAST instantiation the guard is provably true: FAST encodes
// K % K_STEP == 0, hence K % 8 == 0, and hipMalloc is 256-byte aligned.
// SPLIT: 0 = load and store together (default, unchanged); 1 = issue the global
// loads into `ext_stage` and DO NOT store; 2 = ds_store from `ext_stage` and do
// NOT load. Splitting lets the caller issue the loads for step k+2 and store the
// step-k+1 data that arrived last iteration, so the ds_store never waits on its
// own load. An ATT trace puts that wait -- `WAIT load+ds` -- at 135.9 cyc/hit.
// SPLIT != 0 requires STAGE_DEPTH == 0 so the chunk loop is a single pass and
// `ext_stage` has a fixed size of `per_thread` vectors.
template<int N_THREADS = WARP_THREADS, bool ALWAYS_VEC = false, int SUB_ROWS = 0,
         int STAGE_DEPTH = 0, bool FULL_COLS = false, int SPLIT = 0,
         typename T, int ROWS, int COLS,
         ducks::st_shape::all Shape, ducks::gl::row_layout GL, ducks::coord::tile COORD = coord<>>
__device__ inline void load(st<T, ROWS, COLS, Shape>& dst, const GL& src,
                            const COORD& idx, int row_stride,
                            int col_limit = 0x7fffffff,
                            int row_limit = 0x7fffffff,
                            int tid_in = -1,
                            int band_row0 = 0,
                            g2s_stage_vec* ext_stage = nullptr)
{
    static_assert(SPLIT == 0 || STAGE_DEPTH == 0,
                  "SPLIT requires STAGE_DEPTH == 0 (single chunk pass)");
    // tid_in lets a SUBSET of the block drive this cooperative load: pass
    // `threadIdx.x - first_thread_in_group` and instantiate N_THREADS with the
    // group size. Needed by the ping-pong schedule (gemm_pp.cpp), where half the
    // block fills a tile while the other half computes. Defaulted to -1, which
    // leaves every existing call site emitting exactly what it did before.
    //
    // col_limit bounds the valid global column extent (i.e. K for an A/B tile).
    // Columns at or beyond it are filled with zero rather than read out of
    // bounds, so a partial final K step contributes nothing to the accumulation.
    // Without this, k_iters = K / K_STEP truncates and the dropped K remainder
    // is missing from EVERY output element -- measured 64% of outputs wrong at
    // 4096x4096x2000.
    // ROW-BAND WRITE. SUB_ROWS > 0 fills only rows [band_row0, band_row0+SUB_ROWS)
    // of `dst`, from the matching global rows of the tile the COORD names. Both
    // sides take the same offset, so the band is a horizontal slice.
    //
    // This exists so a SUBSET of the block can fill a slice of a SHARED tile --
    // the ping-pong schedule (gemm_pp.cpp), where one wave group fills while the
    // other computes. Without it each group needs its own tile object, and a
    // runtime-selected LDS object decays to a GENERIC pointer: measured at
    // +73,056 64-bit v_add_co pairs and +32,768 v_cmp_ne_u64 addrspacecast
    // checks, i.e. 2.37 VALU per WMMA against 0.46. One shared object plus a
    // runtime ROW OFFSET is the cheap pattern the FAT kernel already uses.
    //
    // SUB_ROWS == 0 keeps whole-tile behaviour; every existing call site is
    // byte-identical.
    constexpr int RSUB = (SUB_ROWS > 0) ? SUB_ROWS : ROWS;
    constexpr int total_elems = RSUB * COLS;
    const int band_off = band_row0 * COLS;
    const int tid = (tid_in >= 0) ? tid_in : (int)threadIdx.x;
    // The COORD is interpreted as tile-index coordinates `{b, d, tile_row, tile_col}`
    // -- convert to element coordinates by multiplying the trailing two by ROWS/COLS.
    const int gr_base = idx.r * ROWS + band_row0;
    const int gc_base = idx.c * COLS;
    const T* base = src.raw_ptr
                  + (((int64_t(idx.b) * src.depth() + idx.d) * src.rows() + gr_base)
                     * src.cols() + gc_base);

    // Vectorised path: 16 bytes (8 bf16) per thread per step instead of one
    // element. The scalar form below compiles to `global_load_u16` +
    // `flat_store_b16` -- a single 16-bit element per thread per iteration,
    // which is the dominant cost of the K-step prologue.
    //
    // Both sides stay 16-byte aligned:
    //   - global: gc_base is a multiple of COLS and col a multiple of VEC, so
    //     the address is aligned whenever row_stride is (checked at runtime).
    //   - LDS: for an 8-aligned logical index, lds_offset() is a multiple of 8
    //     elements, since COLS and the subtile width are multiples of 8 and the
    //     pad amount is 8. Eight consecutive logical indices lie in the same
    //     subtile row and the same 128-element pad block, so they map to eight
    //     consecutive LDS slots.
    constexpr int VEC = 16 / sizeof(T);
    if constexpr (COLS % VEC == 0 && total_elems % VEC == 0
                  && ducks::st_shape::run_is_contiguous<Shape, VEC, T>()) {
        // NOTE the row limit is deliberately NOT tested here.
        //
        // The buffer descriptor's num_records spans the whole matrix, so an
        // offset belonging to a row at or past `row_limit` is >= num_records and
        // the hardware returns ZERO for it. Zeros in an A/B tile contribute
        // nothing to the dot product, and the epilogue is separately bounded, so
        // a partial row block is already correct on the vectorised path.
        //
        // The COLUMN test must stay: a column past col_limit is still a valid
        // linear offset -- it lands in the next row -- so the hardware cannot
        // catch it.
        //
        // Testing the row limit here sent every workgroup on a partial M block
        // down the scalar fallback, which loads ONE 2-byte element per thread
        // per iteration instead of four b128s. Measured: 1000x1000x1024 ran at
        // 15.80 TFLOP/s against 1024^3's 81.83 -- identical grid, identical
        // k_iters, identical work, 5.2x slower -- with only 1/8 of workgroups on
        // the fallback. That implies the fallback is ~30x slower than the
        // vectorised path, which matches 7/8 + 1/8 x 30 = 4.6x.
        // The column tail no longer forces the scalar path either. A vector that
        // straddles col_limit is still loaded whole -- reading past K within a
        // row lands in the next row, which is valid memory (and past the last
        // row, num_records returns zero) -- and the out-of-range ELEMENTS are
        // masked to zero in registers before the ds_store. That keeps b128 loads
        // for the K tail instead of dropping to one 2-byte element per thread.
        //
        // Measured: with the tail on the scalar path, 1024x1024x1000 ran at
        // 61.46 against 1024^3's 81.83 -- a 1.33x penalty for one K step in
        // sixteen.
        // FULL_COLS: the caller guarantees the tile never straddles col_limit,
        // i.e. K % K_STEP == 0. Then the per-element column mask below is
        // provably dead. It is NOT implied by ALWAYS_VEC -- that promises only
        // K % 8 == 0, and gemm_fat's !FAST instantiation passes ALWAYS_VEC=true
        // on genuinely unaligned K, so folding the two would silently corrupt it.
        //
        // Worth 80+ masking instructions per K-step: census puts kernels using
        // this path at ~86 v_and/v_cndmask/s_or against nbuf2sk16's 4, and
        // nbuf2sk16 differs precisely by not going through it.
        const bool col_full = FULL_COLS || (gc_base + COLS <= col_limit);
        // STRADDLE TEST INSTEAD OF AN ALIGNMENT TEST.
        //
        // b128 is correct at any 2-byte-aligned offset on gfx1201 and costs at
        // most 34% (measured), so alignment per se is not
        // a reason to scalarise. The real hazard is the per-DWORD out-of-range
        // rule: a load whose 16 bytes CROSS num_records has its in-range half
        // read as zero.
        //
        // Unlike stage(), this path deliberately reads out of range -- rows past
        // row_limit are meant to come back zero -- so the hazard is real here.
        // But it can only bite the ONE tile that spans the end of the matrix.
        // A tile entirely inside num_records is correct, and a tile entirely
        // past it reads all-zero, which is the required fill. Only a tile
        // containing the boundary must scalarise.
        //
        // This keeps the vectorised path for every tile but the last, instead of
        // scalarising all of them whenever K is not a multiple of 8.
        // The straddle is handled per load inside the loop below, but the test
        // itself must not be paid when it cannot fire. `need_straddle_check` is
        // uniform and hoisted so the compiler can unswitch the unrolled body:
        // with row_stride a multiple of VEC no b128 can ever cross num_records,
        // and a tile wholly inside or wholly past it cannot either. Leaving the
        // compare in unconditionally cost 12.5% at 4096^3 and 13.7% at 6144^3,
        // which reach here through the FAT path.
        const int64_t n_elems = (int64_t)src.rows() * src.cols();
        const int64_t tile_hi =
            (int64_t)(gr_base + RSUB - 1) * row_stride + gc_base + COLS;
        const int64_t tile_lo = (int64_t)gr_base * row_stride + gc_base;
        // ALWAYS_VEC compiles the whole apparatus out. The caller then promises
        // K is a multiple of VEC, which is exactly what the FAT translation unit
        // guarantees (it is routed only for aligned K, being -DFAT_NO_KTAIL).
        // Leaving the path merely PRESENT cost 10-13% at 4096^3 and 6144^3 --
        // not the compare, but what the fallback drags into register allocation,
        // the same effect this parameter was introduced for.
        const bool need_straddle_check = !ALWAYS_VEC &&
            ((row_stride % VEC) != 0) && (tile_hi > n_elems) && (tile_lo < n_elems);

        if (true) {   // vectorised path
            // Buffer loads instead of flat/global loads -- the mechanism the
            // CDNA3 HipKittens path uses. Two benefits on gfx1201: 32-bit
            // offsets against a resource descriptor rather than the 64-bit
            // address arithmetic a flat load needs, and hardware bounds
            // checking (num_records covers the whole matrix, so out-of-range
            // offsets return zero).
            //
            // The descriptor config word is load-bearing and target-specific.
            // Verified by round-trip against a known array on gfx1201:
            //   flags = 0          -> reads return ZERO (silently wrong)
            //   flags = 0x00020000 -> reads return ZERO   (this is the CDNA value)
            //   flags = 0x31014000 -> correct
            // Passing 0 or the CDNA constant produces a fast, silently wrong
            // kernel; there is no fault, just zeros.
            constexpr uint32_t kGfx12RawBufferCfg = BUFFER_RESOURCE_CONFIG;
            typedef float f32x4 __attribute__((ext_vector_type(4)));

            // The whole tensor, not one matrix: elem_base below includes the
            // (batch, depth) offset from raw_ptr, so a one-matrix range put
            // every b > 0 or d > 0 tile out of range, which reads as zeros.
            const int64_t mat_elems = (int64_t)src.batch() * src.depth() * src.rows() * src.cols();
            auto rsrc = __builtin_amdgcn_make_buffer_rsrc(
                (void*)src.raw_ptr, /*stride*/0,
                (int)(mat_elems * sizeof(T)), kGfx12RawBufferCfg);

            const int elem_base =
                // 32-BIT. This is a buffer-descriptor element offset and the
                // hardware's offset operand is 32-bit, so computing it as
                // int64_t emitted a v_lshlrev_b64 plus a
                // v_add_co_u32 / v_add_co_ci_u32_e64 carry pair whose high half
                // was then discarded by this very cast -- provably dead work, in
                // the K loop, once per tile load. The same defect was found and
                // fixed in store_acc16 long ago; the loaders still had it.
                //
                // Range: a buffer descriptor cannot address beyond 4 GB, which
                // at bf16 is 2^31 elements, so int is exactly as capable as the
                // hardware it feeds.
                ((idx.b * (int)src.depth() + idx.d) * (int)src.rows()
                       + gr_base) * (int)src.cols() + gc_base;

            constexpr int nvec = total_elems / VEC;
            typedef __attribute__((address_space(3))) f32x4 lds_f32x4_w;

            // BATCHED LOAD, THEN BATCHED STORE.
            //
            // The obvious loop -- load one vector, store it to LDS, repeat --
            // compiles to `buffer_load / s_waitcnt / ds_store` repeated, with a
            // maximum of ONE load in flight. Every iteration pays the full
            // cache latency serially. Measured on gfx1201: that structure
            // delivers 1.47 TB/s, while an independent global->LDS copy with
            // many loads in flight reaches 2.35 TB/s on the Infinity-Cache
            // plateau (measured). The gap is pure exposed
            // latency, not bandwidth.
            //
            // The compiler serialises because reusing a single VGPR quad for
            // every load minimises register pressure. Forcing the loads into
            // distinct registers -- an explicit array, filled in one unrolled
            // loop and drained in a second -- gets all `per_thread` loads
            // issued before the first `s_waitcnt`. Costs 4 VGPRs per staged
            // vector, which is why this is a fast path and not the default:
            // check occupancy with -Rpass-analysis=kernel-resource-usage after
            // changing tile shapes.
            constexpr int per_thread = (nvec + N_THREADS - 1) / N_THREADS;
            // STAGE_DEPTH bounds how many vectors are held in flight at once.
            //
            // The default (0 = per_thread) loads ALL of them before storing any,
            // which is what gets the loads issued back to back instead of
            // `buffer_load / s_waitcnt / ds_store` serialised -- worth 1.47 ->
            // 2.35 TB/s when it was introduced. It costs 4 VGPRs per staged
            // vector.
            //
            // A smaller depth trades that memory-level parallelism for
            // registers. Worth it where the latency is covered by something
            // else: in the ping-pong schedule the fill sits in a phase whose
            // latency the partner group's compute is already hiding, and in the
            // !FAST fallback the path is off the hot loop entirely.
            constexpr int DEPTH = (STAGE_DEPTH > 0 && STAGE_DEPTH < per_thread)
                                ? STAGE_DEPTH : per_thread;
            f32x4 staged_local[SPLIT == 0 ? DEPTH : 1];
            // When SPLIT != 0 the registers live in the CALLER, across K-steps.
            //
            // The pointer must be chosen with `if constexpr`, not a ternary. The
            // ternary form -- even with SPLIT constexpr -- left the compiler
            // unable to prove the callee writes the caller's array in place, and
            // it emitted register COPIES: ATT put
            // `v_dual_mov_b32 v64, v140 :: v_dual_mov_b32 v65, v141` at 11.1% of
            // all stall and 302 cyc/hit, which is most of why PP_PGR2 lost.
            f32x4* __restrict__ staged;
            if constexpr (SPLIT == 0) staged = &staged_local[0];
            else                      staged = (f32x4*)ext_stage;

            #pragma unroll
            for (int ub = 0; ub < per_thread; ub += DEPTH) {

            if constexpr (SPLIT != 2) {   // SPLIT==2: store only, skip the loads
            #pragma unroll
            for (int d = 0; d < DEPTH; ++d) {
                const int u = ub + d;
                if (u >= per_thread) break;
                const int v = tid + u * N_THREADS;
                if (nvec % N_THREADS != 0 && v >= nvec) break;
                const int i   = v * VEC;
                const int row = i / COLS;
                const int col = i % COLS;
                // SPLIT: per-thread part in voffset, uniform part in soffset.
                // `row * row_stride + col` derives from tid and the unroll index
                // only, so it is loop-INVARIANT and hoists out of the K loop.
                // `elem_base` carries gc_base = k*K_STEP, which changes every
                // iteration but is UNIFORM across the workgroup, so it belongs
                // in an SGPR.
                //
                // The same change load_direct makes for the B operand.
                // hipBLASLt's inner loop has ZERO VALU precisely because every
                // address is built this way: the per-lane component once,
                // outside; the uniform component incremented in SGPRs.
                const int voff = (row * row_stride + col) * (int)sizeof(T);
                const int soff = elem_base * (int)sizeof(T);

                // PER-LOAD STRADDLE PATCH.
                //
                // Out-of-range checking is per-dword relative to the load
                // address, so a b128 that CROSSES num_records has in-range
                // elements zeroed along with the out-of-range ones. Exactly one
                // load in the whole matrix can cross it. Testing that per TILE
                // (the previous form) bailed out every M-edge tile, because such
                // a tile always reaches past the last row -- which is why
                // unaligned K and edge tiles interacted so badly: 87.9 measured
                // against 105.1 predicted from the two effects separately.
                //
                // Testing per LOAD keeps the vector path for every load but the
                // one that actually crosses.
                // The whole straddle apparatus lives behind one uniform branch.
                // Computing `lin` unconditionally -- even with the compare
                // short-circuited -- cost 13% at 4096^3 and 14% at 6144^3, which
                // reach here via FAT: it is int64 arithmetic per load that the
                // compiler would not sink past the short circuit.
                f32x4 vec;
                if (__builtin_expect(need_straddle_check, 0)) {
                    const int64_t lin = (int64_t)elem_base + row * row_stride + col;
                    if (lin < mat_elems && lin + VEC > mat_elems) {
                        unsigned short se[VEC];
                        #pragma unroll
                        for (int q = 0; q < VEC; ++q)
                            se[q] = (lin + q < mat_elems)
                                  ? __builtin_amdgcn_raw_buffer_load_b16(
                                        rsrc, voff + q * (int)sizeof(T), soff, 0)
                                  : (unsigned short)0;
                        __builtin_memcpy(&vec, se, sizeof(vec));
                    } else {
                        vec = __builtin_amdgcn_raw_buffer_load_b128(rsrc, voff, soff, 0);
                    }
                } else {
                    vec = __builtin_amdgcn_raw_buffer_load_b128(rsrc, voff, soff, 0);
                }
                if (!col_full) {
                    // Zero any element whose global column is at or past the
                    // limit. Only the straddling vector is affected; the rest
                    // are wholly in or wholly out and fold away.
                    //
                    // Masked through a real unsigned short[VEC], NOT through an
                    // (unsigned short*) aliased onto the f32x4. The aliased
                    // form is a strict-aliasing violation; it happens to
                    // survive here but the identical construct in load_direct
                    // had its stores elided, silently corrupting every odd-K
                    // shape. Same shape of bug, one inlining decision apart.
                    unsigned short e[VEC];
                    __builtin_memcpy(e, &vec, sizeof(vec));
                    #pragma unroll
                    for (int q = 0; q < VEC; ++q)
                        if (gc_base + col + q >= col_limit) e[q] = 0;
                    __builtin_memcpy(&staged[d], e, sizeof(e));
                } else {
                    staged[d] = vec;
                }
            }
            }   // if constexpr (SPLIT != 2)
            if constexpr (SPLIT != 1) {   // SPLIT==1: load only, skip the stores
            #pragma unroll
            for (int d = 0; d < DEPTH; ++d) {
                const int u = ub + d;
                if (u >= per_thread) break;
                const int v = tid + u * N_THREADS;
                if (nvec % N_THREADS != 0 && v >= nvec) break;
                // Address-space-qualified LDS store. A plain write through
                // dst.data compiles to `flat_store_b128`: the compiler cannot
                // prove the pointer is LDS and falls back to generic flat
                // addressing. Casting to address_space(3) yields ds_store_b128
                // with 32-bit addressing. This is the write-side counterpart of
                // the same fix already applied to the LDS reads in
                // shared_to_register.cuh.
                *(lds_f32x4_w*)(&dst.data[dst.lds_offset(band_off + v * VEC)]) = staged[d];
                }
            }   // if constexpr (SPLIT != 1)
                // Without this the chunking is cosmetic: nothing orders the
                // chunks, so the scheduler hoists every load above every store
                // and reconstructs the full-depth DAG. The barrier is emitted
                // only when the depth actually bites.
                if constexpr (DEPTH < per_thread) __builtin_amdgcn_sched_barrier(0);
            }   // chunk loop: STAGE_DEPTH vectors in flight at a time
#ifdef KITTENS_FILL_SGB          // opt-in: measured neutral
            // FORCE loads-then-stores. The two-pass structure above INTENDS to
            // issue every global load before any ds_store, but the scheduler
            // re-interleaves them. Measured in the shipped kernel's ISA:
            //
            //     buffer_load v[154:157]
            //     buffer_load v[158:161]
            //     s_wait_loadcnt 0x1      <-- 972.5 cyc/hit, 49.4% of ALL stall
            //     ds_store    v145, v[154:157]
            //     s_wait_loadcnt 0x0
            //     ds_store    v144, v[158:161]
            //
            // Two loads in flight and the store waits on its own load with zero
            // instructions of distance. Source-level batching does not survive
            // (the compiler re-sinks the loads); sched_group_barrier does.
            // 0x020 = VMEM read, 0x200 = DS write.
            if constexpr (SPLIT == 0) {
                __builtin_amdgcn_sched_group_barrier(0x020, per_thread, 0);
                __builtin_amdgcn_sched_group_barrier(0x200, per_thread, 0);
            }
#endif
            return;
        }
    }

    // Scalar fallback: non-16-byte-friendly shapes/strides, or a partial tile
    // at the K edge. Out-of-range columns are zeroed.
    #pragma unroll
    for (int i = tid; i < total_elems; i += N_THREADS) {
        const int row = i / COLS;
        const int col = i % COLS;
        // st maps the logical (row-major) index to its subtile-major,
        // padded LDS slot.
        const bool in_bounds = (gc_base + col < col_limit) &&
                               (gr_base + row < row_limit);
        dst.data[dst.lds_offset(band_off + i)] =
            in_bounds ? base[row * row_stride + col]
                      : base_types::convertor<T, float>::convert(0.0f);
    }
}

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
 * @brief Cooperative LDS -> global tile copy.
 *
 * Reads each element from its slot via `ST::idx` and scatters it back to global memory. Pairs
 * with `load`, which lands data in the same LDS address map.
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
 * @brief DirectToVgpr: load a WMMA operand fragment straight from global.
 *
 * Tensile's `DirectToVgprA/B`, which every shipped gfx1201 bf16 solution
 * enables. The operand never enters LDS: each wave reads its own fragment from
 * global into registers in WMMA layout.
 *
 * The trade. LDS holds a tile so that the waves sharing it read it once; drop
 * the tile and each of the `redundancy` waves that shared it issues its own
 * global read. Global request volume for that operand therefore multiplies by
 * the warp-grid extent along the other axis. What is bought is LDS footprint,
 * and hence occupancy -- for a 256x128x64 tile, removing B takes LDS from
 * 48 KB to 32 KB, i.e. 1 -> 2 workgroups per CU.
 *
 * The bet is that the redundant reads are nearly free: the waves sharing a tile
 * are in one workgroup on one CU issuing within a few microseconds of each
 * other, so all but the first should hit L0/L1 rather than reaching the
 * Infinity Cache. Measured L2-resident bandwidth is 7-10 TB/s against 2.34 TB/s
 * for the MALL (measured), so a 4x request count against a
 * 3-4x faster cache level is roughly break-even on paper.
 *
 * Fragment layout, measured on gfx1201 with a one-hot lane probe:
 *
 *     A[row][k] -> lane 16*(k/8) + row, element k%8
 *
 * so lane L supplies row `L % 16` and K-half `L / 16`, reading 8 contiguous
 * bf16 = one `buffer_load_b128`. Identical for a B operand stored N-major with
 * K contiguous (Tensile's `Bjlk`), which is what this GEMM uses.
 *
 * @param gr_base  global row of this wave's fragment origin
 * @param gc_base  global column (K) of this wave's fragment origin
 */
// ALWAYS_IN: the caller guarantees the whole fragment is in range, so the
// bounds test -- and with it the branch -- is not emitted at all.
//
// This matters for SCHEDULING, not for the two compare instructions it saves.
// With the test present, `if (in_bounds) { loads; return; }` puts the loads in
// their own basic block, and the disassembly of the FAST kernel shows the
// consequence: ZERO buffer/global instructions inside the steady-state K loop,
// because they were sunk into a separate block and cannot be interleaved with
// the WMMA stream. hipBLASLt's loop carries 24 buffer_load_b128 inline, at
// 0.375 per WMMA, where each load issued early is covered by 64 WMMA of work.
//
// In gemm_sb's FAST instantiation the test is provably always true -- FAST
// already encodes K % K_STEP == 0, so gc_base + WARP_K <= col_limit holds for
// every k -- but the compiler cannot see that through the runtime arguments.
template<int WARP_M, int WARP_K, bool KTAIL = false, bool ALWAYS_IN = false,
         ducks::gl::row_layout GL>
__device__ inline void load_direct(
    rt_bf<WARP_M, WARP_K, ducks::rt_layout::row, ducks::rt_shape::rt_16x16>& dst,
    const GL& src, int gr_base, int gc_base, int row_limit, int col_limit)
{
    constexpr int RSUB   = 16;
    constexpr int height = WARP_M / RSUB;
    constexpr int rwidth = WARP_K / RSUB;

    const int L          = kittens::laneid();
    const int row_in_sub = L % RSUB;
    const int khalf      = L / RSUB;

    typedef float f32x4 __attribute__((ext_vector_type(4)));
    constexpr uint32_t kGfx12RawBufferCfg = BUFFER_RESOURCE_CONFIG;

    const int   ld         = src.cols();
    const int64_t mat_elems = (int64_t)src.rows() * ld;
    auto rsrc = __builtin_amdgcn_make_buffer_rsrc(
        (void*)src.raw_ptr, /*stride*/0,
        (int)(mat_elems * sizeof(kittens::bf16)), kGfx12RawBufferCfg);

    // Whole-fragment bounds test, hoisted out of the unrolled body: every lane
    // of every subtile is in range together or not at all.
    // Row limit deliberately not tested -- same reasoning as load() above. The
    // descriptor's num_records spans the matrix, so a fragment row at or past
    // row_limit reads as zero in hardware, which is exactly the fill required:
    // zeros contribute nothing to the dot product and the epilogue is bounded
    // separately. Only the column test is needed, because a column past
    // col_limit is a valid offset landing in the next row.
    //
    // For the B operand `row_limit` is N, so this is what sent every wave on a
    // partial N block to the per-element fallback.
    const bool in_bounds = ALWAYS_IN ? true : (gc_base + WARP_K <= col_limit);

    // Load straight into the destination fragment registers. A separate
    // staging array would be pure overhead here: unlike the global->shared
    // path, where the staging buffer exists because the destination is LDS and
    // cannot hold a load result, `dst.tiles[ti][tj]` is ALREADY a distinct set
    // of registers. Writing into it batches the loads just as well and costs
    // nothing. Staging instead measured +56 VGPRs, which capped occupancy at 7
    // -- below the 8 the LDS saving was supposed to improve on.
    if (ALWAYS_IN || in_bounds) {
        #pragma unroll
        for (int ti = 0; ti < height; ti++) {
            #pragma unroll
            for (int tj = 0; tj < rwidth; tj++) {
                // SPLIT THE OFFSET: per-lane part in voffset, uniform part in
                // soffset. `raw_buffer_load` takes both, and this loader had
                // been folding everything into voffset with soffset = 0.
                //
                // `row * ld + tj*RSUB + khalf*(RSUB/2)` depends only on the
                // lane and the unroll indices, so it is loop-INVARIANT and gets
                // hoisted out of the K loop once split out. `gc_base` is
                // uniform across the wave, so it belongs in an SGPR rather than
                // being added per-lane on the VALU every iteration.
                //
                // That per-lane add is what the ISA showed: v_add_lshl_u32
                // chains feeding `s_wait_alu`, the largest wait category in the
                // loop at 24 per 64 WMMA against hipBLASLt's near-zero VALU.
                // Tensile emits none precisely because it addresses this way.
                const int row  = gr_base + ti * RSUB + row_in_sub;
                const int voff = (row * ld + tj * RSUB + khalf * (RSUB / 2))
                                 * (int)sizeof(kittens::bf16);
                const int soff = gc_base * (int)sizeof(kittens::bf16);
                const f32x4 v = __builtin_amdgcn_raw_buffer_load_b128(rsrc, voff, soff, 0);
                __builtin_memcpy(&dst.tiles[ti][tj].data[0], &v, sizeof(v));
            }
        }
        return;
    }

    // KTAIL selects the fallback at COMPILE TIME.
    //
    // The masked vectorised tail is strictly faster when it runs (+26% at
    // 2048x2048x63) but it is a second copy of the loop nest, and merely having
    // it present cost 3.6% at 4096^3 where it is dead code. The kernel already
    // knows from the host whether K divides the K step, so the path that cannot
    // be reached is simply not compiled.
    //
    // GUARD: the masked vector path is only valid when no b128 load can have a
    // DWORD straddling num_records. Buffer out-of-range checking on gfx12 is
    // per-dword relative to the load address, NOT per element: an in-range
    // 16-bit element that shares a dword with an out-of-range byte reads as
    // ZERO. Measured directly on gfx1201. So
    // raw_buffer_load_b128 is NOT equivalent to 8x raw_buffer_load_b16 at the
    // end of the buffer, and the scalar fallback (which loads each element
    // separately, each individually in range) is not merely slower, it is
    // correct where the vector load is not.
    //
    // ld % 8 == 0 makes every row start a multiple of 16 bytes AND makes
    // num_records a multiple of 16, so no dword can ever straddle. This is the
    // same condition load() already enforces, which is why load() never showed
    // the bug. Without it: 403 wrong elements at 1024x1024x1023, all of them in
    // the last output column, i.e. the last row of B.
    if constexpr (KTAIL) if ((ld % 8) == 0) {
        #pragma unroll
        for (int ti = 0; ti < height; ti++) {
            #pragma unroll
            for (int tj = 0; tj < rwidth; tj++) {
                const int row = gr_base + ti * RSUB + row_in_sub;
                const int k   = gc_base + tj * RSUB + khalf * (RSUB / 2);
                const int off = (row * ld + k) * (int)sizeof(kittens::bf16);
                const f32x4 vec = __builtin_amdgcn_raw_buffer_load_b128(rsrc, off, 0, 0);
                // Mask through a real unsigned short[8], copied in and out.
                // Writing 16-bit zeros through an (unsigned short*) aliased
                // onto the f32x4 is a strict-aliasing violation and the
                // compiler DID elide the stores: it produced 403 wrong
                // elements at 1024x1024x1023 and 108 at 333x777x1111 while
                // looking correct in source. Every shape it broke had odd K.
                unsigned short e[8];
                __builtin_memcpy(e, &vec, sizeof(vec));
                #pragma unroll
                for (int q = 0; q < 8; ++q)
                    if (k + q >= col_limit) e[q] = 0;
                __builtin_memcpy(&dst.tiles[ti][tj].data[0], e, sizeof(e));
            }
        }
        return;
    }

    #pragma unroll
    for (int ti = 0; ti < height; ti++) {
        #pragma unroll
        for (int tj = 0; tj < rwidth; tj++) {
            bf16_2 p[4];
            {
                // Edge fragment: rebuild element-wise and zero-fill anything
                // past the M or K limit. Zeros contribute nothing to the dot
                // product, which is what makes a partial K step correct.
                // Buffer loads, not pointer dereferences. `src.raw_ptr[...]`
                // needs a full 64-bit address per element -- a v_lshlrev_b64
                // plus a v_add_co_u32 / v_add_co_ci_u32_e64 pair -- whereas a
                // buffer load takes a 32-bit offset against the descriptor
                // already built above. With 64 elements per fragment that is
                // the largest single VALU category in the whole kernel.
                //
                // Out-of-range ROWS are discarded by num_records in hardware.
                // The column test must stay: a column past col_limit is still a
                // valid linear offset and would land in the next row.
                const int row = gr_base + ti * RSUB + row_in_sub;
                unsigned short pe[8];
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int k = gc_base + tj * RSUB + khalf * (RSUB / 2) + e;
                    const bool ok = (row < row_limit) && (k < col_limit);
                    pe[e] = ok ? __builtin_amdgcn_raw_buffer_load_b16(
                                     rsrc, (row * ld + k) * (int)sizeof(kittens::bf16), 0, 0)
                               : (unsigned short)0;
                }
                __builtin_memcpy(p, pe, sizeof(pe));
            }
            __builtin_memcpy(&dst.tiles[ti][tj].data[0], p, sizeof(p));
        }
    }
}

}
