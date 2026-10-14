/**
 * @file
 * @brief Functions for transferring data directly between global memory and registers and back.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"
#include "../util/util.cuh"

namespace kittens {

namespace detail {
// __builtin_nontemporal_store takes only native scalar/vector types, and the
// tile element types here are classes (__hip_bfloat16, HIP_vector_type). These
// are the same-width native types to memcpy through; nothing is converted, the
// bits go out unchanged.
template<int Bytes> struct nt_word;
template<> struct nt_word<2> { using type = unsigned short; };
template<> struct nt_word<4> { using type = unsigned int;   };
template<int Bytes> using nt_word_t = typename nt_word<Bytes>::type;
using nt_vec4 = unsigned int __attribute__((ext_vector_type(4)));

/**
 * @brief Assemble the dwords of one contiguous destination run that a lane
 *        shares with its wave-half partner.
 *
 * Why this exists: the f32 accumulator layout is *interleaved*, not mirrored.
 * rt_base_coord puts element e of lane l at column `2e + (l>>4)`, so lane l and
 * lane l+16 hold alternating columns of the same 16-wide run and neither owns a
 * contiguous byte range. Nothing the compiler can do merges that -- a store of
 * an f32 tile into an int8 global came out as 32 `global_store_b8` per tile and
 * ran 3.8x slower than the same kernel writing bf16 (measured, 16384x4096: 2.25
 * ms vs 0.59 ms for twice the bytes). The 2-byte `T == U` path above never hits
 * this because it is the *operand* layout, where element_stride is 1.
 *
 * So do the interleave in registers instead of in the memory system: each lane
 * places its E values at their byte offsets within the run, ORs in the
 * partner's copy of the same dwords, and both lanes then store the completed
 * run. They write identical bytes, which is the same benign duplicate the
 * mirrored operand path already relies on.
 *
 * `__shfl_xor(..., 16, 32)` is the exchange; on gfx11 wave32 that is the two
 * halves of the wave, which is exactly the pairing `halves_interleave`
 * describes.
 *
 * @tparam U Destination element type (narrower than, or equal to, the tile's).
 * @tparam E Elements one lane holds per base tile.
 * @param[in] val This lane's E values, already converted to U.
 * @param[in] half `laneid() >> 4`.
 * @param[out] w The run's dwords, identical in both lanes of the pair.
 */
template<typename U, int E>
__device__ inline void pack_interleaved_run(const U (&val)[E], int half,
                                            unsigned int (&w)[2*E*sizeof(U)/4]) {
    constexpr int NW = 2*E*sizeof(U)/4;
    #pragma unroll
    for(int k = 0; k < NW; k++) w[k] = 0u;
    #pragma unroll
    for(int e = 0; e < E; e++) {
        // Byte offset of this element inside the run. `2*e + half` is the
        // column, straight out of rt_base_coord.
        const int byte = (2*e + half) * int(sizeof(U));
        unsigned int bits = 0u;                       // zero-extend, then place
        __builtin_memcpy(&bits, &val[e], sizeof(U));
        w[byte >> 2] |= bits << ((byte & 3) * 8);
    }
    #pragma unroll
    for(int k = 0; k < NW; k++) w[k] |= __shfl_xor(w[k], 16, 32);
}
}

/*
 * Same gfx11 lane mapping as the shared-memory path (rt_base_coord in
 * types/register/rt_base.cuh), minus the swizzle: global tiles are plain
 * row-major with `row_stride` between rows.
 *
 * Only the row-layout 2-byte operand is contiguous per lane -- 16 elements =
 * 32 B, issued as two 16-byte accesses. Everything else is strided and goes
 * elementwise, which is what the CDNA tree already does for its column path.
 *
 * As there, lanes l and l+16 of an operand tile address identical bytes. That
 * is the mirroring WMMA requires; the coalescer merges them.
 *
 * Unlike LDS, global rows are only as aligned as `row_stride` makes them: the
 * allocation base is 256 B from hipMalloc, but row r starts at r*row_stride
 * elements, so a stride of e.g. 8200 bf16 leaves every odd row 16-byte
 * misaligned. The wide path is therefore selected at runtime (a warp-uniform
 * scalar branch) and falls back to the elementwise loop, which needs no
 * alignment at all, when the tile does not qualify.
 */

/**
 * @brief Load data from a source array into a register tile.
 *
 * @tparam RT The register tile type.
 * @tparam GL The global layout type.
 * @param dst[out] The destination tile to load data into.
 * @param src[in] The source array to load data from.
 * @param idx[in] The index of the tile to load data from.
 */
template<int axis, ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void load(RT &dst, const GL &src, const COORD &idx) {
    using T  = typename base_types::packing<typename RT::dtype>::unpacked_type;
    using U  = typename GL::dtype;

    using L    = typename RT::layout;
    using base = rt_base<T, L>;

    constexpr bool is_row = std::is_same_v<L, ducks::rt_layout::row>;
    constexpr int  E      = base::elements_per_thread;
    constexpr bool vectorizable = is_row && base::element_stride == 1
                                         && std::is_same_v<T, U> && sizeof(U) == 2;

    /*
     * The widening load: a narrow global (bf16, fp16, int8) into a wider tile.
     *
     * This is the read-side twin of the `packable` store below, and it exists
     * for the same reason. In a row-layout tile every element a lane holds is
     * in row `lane & 15` -- rt_base_coord's `.x` does not depend on `e` -- so
     * a lane's E elements all live inside the base tile's 16-element run of
     * that row. For the f32 accumulator those elements are every other column
     * of the run, which the elementwise loop issues as E separate narrow
     * loads: 32 `global_load_u16` per 16x64 tile per lane where two
     * `global_load_dwordx4` would do.
     *
     * Unlike the store, the memory *traffic* is already fine -- lanes 0..15
     * cover 16 rows and the coalescer merges the halves, so each cache line is
     * fetched once -- so what this buys is issue slots, not bytes. It is still
     * worth it: every Phase 2 op reads a 16-bit tensor into an fp32 tile, and
     * at 75% of HBM the instruction stream is what is left to give back.
     *
     * The run is read once and indexed by `c.y`, which is why this needs no
     * shuffle and no knowledge of which layout it is serving: lane l and lane
     * l+16 read the *same* bytes and each keeps its own half of them. Reading
     * the same address from both halves is what the mirrored operand path
     * already does, and the coalescer merges it.
     *
     * Not enabled for `sizeof(U) == sizeof(T)`: the run would be 64 B per lane
     * and the staging registers cost more than the issue slots are worth.
     */
    constexpr int  RUN  = base::tile_size_col;            // elements in the run
    constexpr bool widening = is_row && !vectorizable && sizeof(U) < sizeof(T)
                                     && (RUN * sizeof(U)) % 16 == 0;

    const U *src_ptr = (const U*)&src[(idx.template unit_coord<axis, 3>())];
    const int row_stride = src.template stride<axis>();
    const int lane = laneid();
    const int l16  = lane & 15;

    // 16-byte alignment of every row this warp will touch. Uniform across the
    // warp, so the branches below are scalar branches, not divergence.
    const bool aligned =
        ((reinterpret_cast<uintptr_t>(src_ptr) | (row_stride * sizeof(U))) & 15) == 0;
    const bool wide = vectorizable && aligned;
    const int  half = lane >> 4;

    auto elementwise = [&](int i, int j, int row_base, int col_base) {
        #pragma unroll
        for(int e = 0; e < E; e++) {
            const int2 c = rt_base_coord<T, L>(e, lane);
            const T val = base_types::convertor<T, U>::convert(
                src_ptr[(row_base + c.x)*row_stride + col_base + c.y]);
            if (e & 1) dst.tiles[i][j].data[e>>1].y = val;
            else       dst.tiles[i][j].data[e>>1].x = val;
        }
    };

    // Read this lane's whole 16-element run, then pick the E entries of it the
    // lane actually owns. See the `widening` comment above.
    auto widened = [&](int i, int j, int row_base, int col_base) {
        constexpr int NV = RUN * sizeof(U) / 16;
        constexpr int PER = 16 / sizeof(U);   // U's per 16-byte access
        const auto *p = reinterpret_cast<const detail::nt_vec4*>(
            &src_ptr[(row_base + l16)*row_stride + col_base]);
        U run[RUN];
        #pragma unroll
        for(int k = 0; k < NV; k++) {
            // memcpy rather than a reinterpret_cast of the vector array: U is
            // a class (__hip_bfloat16) and reading uint storage through it is
            // an aliasing violation. The copy does not survive SROA.
            const detail::nt_vec4 v = p[k];
            __builtin_memcpy(&run[k*PER], &v, sizeof(v));
        }
        #pragma unroll
        for(int e = 0; e < E; e++) {
            // c.x is l16 for every e in a row-layout tile -- that is the
            // precondition this path rests on -- so only c.y is needed.
            //
            // Both positions are read and one is selected, rather than
            // indexing by `rt_base_coord(e, lane).y` directly, because that
            // index depends on the wave half and a *dynamic* index into a
            // local array is not a register file: the first version of this
            // compiled `raw` to 48 B of scratch per lane with zero spills
            // reported, which is the same silent-reordering hazard a spill is.
            // Asking for lane 0 and lane 16 makes both indices constant, and
            // the select is one v_cndmask. Where the halves do not interleave
            // the two are the same index and it folds away entirely.
            const int c0 = rt_base_coord<T, L>(e, 0).y;
            const int c1 = rt_base_coord<T, L>(e, 16).y;
            const T v0 = base_types::convertor<T, U>::convert(run[c0]);
            const T v1 = base_types::convertor<T, U>::convert(run[c1]);
            const T val = half ? v1 : v0;
            if (e & 1) dst.tiles[i][j].data[e>>1].y = val;
            else       dst.tiles[i][j].data[e>>1].x = val;
        }
    };

    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            const int row_base = i*dst.tile_size_row;
            const int col_base = j*dst.tile_size_col;
            if constexpr (vectorizable) {
                if (wide) {
                    const U *p = &src_ptr[(row_base + l16)*row_stride + col_base];
                    #pragma unroll
                    for(int h = 0; h < 2; h++) {
                        const float4 v = *reinterpret_cast<const float4*>(p + 8*h);
                        __builtin_memcpy((void*)&dst.tiles[i][j].data[4*h], &v, sizeof(v));
                    }
                }
                else elementwise(i, j, row_base, col_base);
            }
            else if constexpr (widening) {
                if (aligned) widened(i, j, row_base, col_base);
                else elementwise(i, j, row_base, col_base);
            }
            else elementwise(i, j, row_base, col_base);
        }
    }
}

template<ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void load(RT &dst, const GL &src, const COORD &idx) {
    load<2, RT, GL, COORD>(dst, src, idx);
}

/**
 * @brief Store a register tile to a raw global pointer.
 *
 * The addressing half of store() -- turning a gl and a coord into a base
 * pointer and a row stride -- split out so a caller that already has the
 * pointer can skip it. A distributed epilogue is the case that needs this: the
 * destination may be a peer GPU's buffer, which is an ordinary global pointer
 * but is not describable as a gl, because gl has no default constructor and a
 * host-only one, so a kernel argument cannot hold an array of them.
 *
 * Everything below the pointer is identical to the gl version -- which is the
 * point, the two share this body rather than duplicating the vectorization
 * decision.
 *
 * @tparam RT The register tile type.
 * @tparam U The element type of the destination.
 * @tparam NT Bypass the cache hierarchy on the way out (see store_at_nt).
 * @param[out] dst_ptr Where the tile's (0,0) element goes.
 * @param[in] row_stride Elements between consecutive rows at the destination.
 * @param[in] src The source register tile.
 */
template<ducks::rt::all RT, typename U, bool NT = false>
__device__ inline static void store_at(U *dst_ptr, int row_stride, const RT &src) {
    using T  = typename base_types::packing<typename RT::dtype>::unpacked_type;

    using L    = typename RT::layout;
    using base = rt_base<T, L>;

    constexpr bool is_row = std::is_same_v<L, ducks::rt_layout::row>;
    constexpr int  E      = base::elements_per_thread;
    constexpr bool vectorizable = is_row && base::element_stride == 1
                                         && std::is_same_v<T, U> && sizeof(U) == 2;

    /*
     * The interleaved (f32 accumulator) layout, narrowing on the way out.
     *
     * `sizeof(U) < sizeof(T)` is the whole point and also the limit: the win
     * comes from the destination run being *narrower* than the source, so the
     * dwords a lane assembles are few. Same-width would need 16 exchanges per
     * base tile to replace 8 plain dword stores, which is a loss.
     *
     * NT is excluded because a peer-directed store has to carry the bypass bits
     * on every access, and the duplicate write this path relies on is not
     * something to reason about over the fabric. Local stores are what it is
     * for.
     *
     * Note this makes store() a warp-collective op for these types: the
     * exchange requires the whole wave converged. Every caller in the tree
     * diverges at warp granularity or coarser, which is the granularity the
     * rest of the library already assumes.
     */
    constexpr bool packable = is_row && !vectorizable && !NT
                                     && base::element_stride == 2
                                     && base::halves_interleave
                                     && sizeof(U) < sizeof(T)
                                     && (2*E*sizeof(U)) % 16 == 0;

    const int lane = laneid();
    const int l16  = lane & 15;
    const int half = lane >> 4;

    // 16-byte alignment of every row this warp touches, uniform across the
    // warp, so this is a scalar branch rather than divergence.
    const bool aligned =
        ((reinterpret_cast<uintptr_t>(dst_ptr) | (row_stride * sizeof(U))) & 15) == 0;
    const bool wide = vectorizable && aligned;

    // Lanes l and l+16 of an operand tile write the same bytes with the same
    // values. That redundancy is exactly the WMMA mirroring invariant, so the
    // write is benign whichever lane lands last.
    auto elementwise = [&](int i, int j, int row_base, int col_base) {
        #pragma unroll
        for(int e = 0; e < E; e++) {
            const int2 c = rt_base_coord<T, L>(e, lane);
            const T val = (e & 1) ? src.tiles[i][j].data[e>>1].y
                                  : src.tiles[i][j].data[e>>1].x;
            U *p = &dst_ptr[(row_base + c.x)*row_stride + col_base + c.y];
            const U cv = base_types::convertor<U, T>::convert(val);
            // __builtin_nontemporal_store only accepts native scalar/vector
            // types, and U is a class (__hip_bfloat16), so go through an
            // integer of the same width. Pure bit shuffling, no conversion.
            if constexpr (NT) {
                using raw = detail::nt_word_t<sizeof(U)>;
                raw w; __builtin_memcpy(&w, &cv, sizeof(U));
                __builtin_nontemporal_store(w, reinterpret_cast<raw*>(p));
            }
            else *p = cv;
        }
    };

    // Convert this lane's E elements and write the run they share with the
    // partner lane. See detail::pack_interleaved_run.
    auto packed = [&](int i, int j, int row_base, int col_base) {
        U val[E];
        #pragma unroll
        for(int e = 0; e < E; e++) {
            const T v = (e & 1) ? src.tiles[i][j].data[e>>1].y
                                : src.tiles[i][j].data[e>>1].x;
            val[e] = base_types::convertor<U, T>::convert(v);
        }
        constexpr int NW = 2*E*sizeof(U)/4;
        unsigned int w[NW];
        detail::pack_interleaved_run<U, E>(val, half, w);
        // The run starts at this lane's row (rt_base_coord's `.x` is l16 for
        // every e in a row-layout tile) and at the base tile's first column.
        auto *q = reinterpret_cast<detail::nt_vec4 *>(
            &dst_ptr[(row_base + l16)*row_stride + col_base]);
        #pragma unroll
        for(int k = 0; k < NW/4; k++) {
            detail::nt_vec4 v;
            __builtin_memcpy(&v, &w[4*k], sizeof(v));
            q[k] = v;
        }
    };

    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            const int row_base = i*src.tile_size_row;
            const int col_base = j*src.tile_size_col;
            if constexpr (vectorizable) {
                if (wide) {
                    U *p = &dst_ptr[(row_base + l16)*row_stride + col_base];
                    #pragma unroll
                    for(int h = 0; h < 2; h++) {
                        float4 v;
                        __builtin_memcpy(&v, (const void*)&src.tiles[i][j].data[4*h], sizeof(v));
                        float4 *q = reinterpret_cast<float4*>(p + 8*h);
                        if constexpr (NT) {
                            detail::nt_vec4 w;
                            __builtin_memcpy(&w, &v, sizeof(v));
                            __builtin_nontemporal_store(
                                w, reinterpret_cast<detail::nt_vec4*>(q));
                        }
                        else *q = v;
                    }
                }
                else elementwise(i, j, row_base, col_base);
            }
            else if constexpr (packable) {
                if (aligned) packed(i, j, row_base, col_base);
                else elementwise(i, j, row_base, col_base);
            }
            else elementwise(i, j, row_base, col_base);
        }
    }
}

/**
 * @brief store_at(), but the stores bypass the cache hierarchy.
 *
 * For a destination that is another GPU's memory. gfx1100 maps a peer's HBM
 * cacheable in the *local* L2, and nothing available to a kernel writes that L2
 * back: __threadfence_system() compiles to s_waitcnt_vscnt plus L0/L1
 * invalidates and emits no writeback at all, and a system-scope release store
 * compiles to a plain global_store_b32 with no cache bits. So an ordinary store
 * aimed at a peer can sit dirty in my own L2 until something evicts it, which
 * is long after the barrier has told the peer its data is ready.
 *
 * __builtin_nontemporal_store is the one form that carries the bypass bits --
 * it emits global_store_* ... glc slc dlc -- so the write goes out to the
 * fabric instead of parking. Only use this for peer destinations: for local
 * memory it is a pure loss, since the L2 hit is what makes the normal path fast.
 *
 * @tparam RT The register tile type.
 * @tparam U The element type of the destination.
 */
template<ducks::rt::all RT, typename U>
__device__ inline static void store_at_nt(U *dst_ptr, int row_stride, const RT &src) {
    store_at<RT, U, true>(dst_ptr, row_stride, src);
}

/**
 * @brief Store data from a register tile to a destination array in global memory.
 *
 * @tparam RT The register tile type.
 * @tparam GL The global layout type.
 * @param[out] dst The destination array in global memory to store data into.
 * @param[in] src The source register tile to store data from.
 * @param[in] idx The index of the tile to store data to.
 */
template<int axis, ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void store(const GL &dst, const RT &src, const COORD &idx) {
    using U = typename GL::dtype;
    store_at<RT, U>((U*)&dst[(idx.template unit_coord<axis, 3>())],
                    dst.template stride<axis>(), src);
}

template<ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void store(const GL &dst, const RT &src, const COORD &idx) {
    store<2, RT, GL, COORD>(dst, src, idx);
}

/*
 * Row-predicated load and store.
 *
 * Separate functions rather than a flag on load()/store_at(), and on buffer
 * instructions rather than on those functions' pointers. Both choices were
 * measured. A predicate threaded through the pointer path -- even one that is
 * a compile-time constant on the unbounded side -- changed how LLVM addresses
 * every store (32 base+offset stores became 32 separate address pairs, 36 ->
 * 91 VGPRs on an elementwise op) and spilled the GEMM. A buffer resource whose
 * range ends at the tensor's last row needs no predicate in the code at all:
 * the hardware returns zero for a load past the range and drops a store past
 * it, with no branch, no masked lane, and the instruction count of a whole
 * tile. The bytes past the row never move.
 *
 * The voffset is 32-bit, so the tile's rows from its first must span less
 * than 4 GB -- true of any tensor this library is handed.
 */
namespace detail {

template<int axis, int ROWS, typename GL, typename COORD>
__device__ inline i32x4 row_bounded_srsrc(const GL &g, const COORD &unit, const void *base,
                                          int limit = ROWS) {
    // `limit` clips further, for a caller whose rows end before the
    // tensor's do: a query tile that runs into the next request's tokens.
    const int in = rows_in_bounds<axis, ROWS>(g, unit);
    const int valid = limit < 0 ? 0 : (limit < in ? limit : in);
    const uint32_t bytes = uint32_t(valid) * uint32_t(g.template stride<axis>()) *
                           uint32_t(sizeof(typename GL::dtype));
    return make_srsrc(base, bytes);
}

template<typename U>
__device__ inline U buffer_load_elem(i32x4 srsrc, uint32_t off, uint32_t soff) {
    static_assert(sizeof(U) == 2 || sizeof(U) == 4, "16- or 32-bit elements");
    U v;
    if constexpr (sizeof(U) == 2) {
        const uint16_t raw = llvm_amdgcn_raw_buffer_load_b16(srsrc, off, soff, 0);
        __builtin_memcpy(&v, &raw, 2);
    } else {
        const uint32_t raw = llvm_amdgcn_raw_buffer_load_b32(srsrc, off, soff, 0);
        __builtin_memcpy(&v, &raw, 4);
    }
    return v;
}

template<typename U>
__device__ inline void buffer_store_elem(i32x4 srsrc, uint32_t off, U v) {
    static_assert(sizeof(U) == 2 || sizeof(U) == 4, "16- or 32-bit elements");
    if constexpr (sizeof(U) == 2) {
        uint16_t raw; __builtin_memcpy(&raw, &v, 2);
        llvm_amdgcn_raw_buffer_store_b16(raw, srsrc, off, 0, 0);
    } else {
        uint32_t raw; __builtin_memcpy(&raw, &v, 4);
        llvm_amdgcn_raw_buffer_store_b32(raw, srsrc, off, 0, 0);
    }
}

} // namespace detail

/**
 * @brief load(), predicated on the tensor's rows: rows of the tile past the
 * last row of `src` read as zero, and their memory is never touched.
 *
 * Same lane mapping and the same two 16-byte accesses per base tile as load()
 * on its vectorised path; everything else goes element by element, including
 * the widening case load() vectorises.
 */
template<int axis, ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void load_bounded(RT &dst, const GL &src, const COORD &idx,
                                           int limit = RT::rows) {
    using T  = typename base_types::packing<typename RT::dtype>::unpacked_type;
    using U  = typename GL::dtype;
    using L    = typename RT::layout;
    using base = rt_base<T, L>;

    constexpr bool is_row = std::is_same_v<L, ducks::rt_layout::row>;
    constexpr int  E      = base::elements_per_thread;
    constexpr bool vectorizable = is_row && base::element_stride == 1
                                         && std::is_same_v<T, U> && sizeof(U) == 2;

    // The buffer starts at the tile's first row, column 0, and the column
    // offset rides in soffset. In a K loop that makes the descriptor and every
    // lane's voffset loop-invariant and the one moving part a scalar: with the
    // column folded into the base instead, LLVM hoisted eight voffsets out of
    // the decode GEMM's loop and the kernel lost a step of occupancy.
    const auto unit = idx.template unit_coord<axis, 3>();
    auto row0 = unit;
    row0.c = 0;
    const U *row_ptr = (const U*)&src[row0];
    // An SGPR operand, and uniform by the same argument as the descriptor.
    const uint32_t col0 = __builtin_amdgcn_readfirstlane(uint32_t(unit.c) * sizeof(U));
    const int row_stride = src.template stride<axis>();
    const i32x4 srsrc = detail::row_bounded_srsrc<axis, RT::rows>(src, unit, row_ptr, limit);
    const int lane = laneid();
    const int l16  = lane & 15;

    auto elementwise = [&](int i, int j, int row_base, int col_base) {
        #pragma unroll
        for(int e = 0; e < E; e++) {
            const int2 c = rt_base_coord<T, L>(e, lane);
            const uint32_t off = ((row_base + c.x)*row_stride + col_base + c.y) * sizeof(U);
            const T val = base_types::convertor<T, U>::convert(
                detail::buffer_load_elem<U>(srsrc, off, col0));
            if (e & 1) dst.tiles[i][j].data[e>>1].y = val;
            else       dst.tiles[i][j].data[e>>1].x = val;
        }
    };

    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            const int row_base = i*dst.tile_size_row;
            const int col_base = j*dst.tile_size_col;
            if constexpr (vectorizable) {
                // No alignment branch, unlike load(): gfx11 runs with
                // unaligned access enabled -- LLVM merges load()'s own
                // elementwise fallback into the same 16-byte loads -- and a
                // fallback here is 16 separate buffer loads per base tile that
                // LLVM cannot merge, whose registers the allocator reserves
                // whether the branch runs or not (+10 VGPRs on the decode GEMM).
                const uint32_t off = ((row_base + l16)*row_stride + col_base) * sizeof(U);
                #pragma unroll
                for(int h = 0; h < 2; h++) {
                    const __uint128_t raw =
                        llvm_amdgcn_raw_buffer_load_b128(srsrc, off + 16*h, col0, 0);
                    __builtin_memcpy((void*)&dst.tiles[i][j].data[4*h], &raw, sizeof(raw));
                }
            }
            else elementwise(i, j, row_base, col_base);
        }
    }
}
template<ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void load_bounded(RT &dst, const GL &src, const COORD &idx,
                                           int limit = RT::rows) {
    load_bounded<2, RT, GL, COORD>(dst, src, idx, limit);
}

/**
 * @brief store(), predicated on the tensor's rows: rows of the tile past the
 * last row of `dst` are not written.
 *
 * The vectorised and the packed (fp32 accumulator narrowed to 16 bits) paths
 * are store_at()'s, writing through the buffer; the rest go element by element.
 */
template<int axis, ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void store_bounded(const GL &dst, const RT &src, const COORD &idx,
                                            int limit = RT::rows) {
    using T  = typename base_types::packing<typename RT::dtype>::unpacked_type;
    using U  = typename GL::dtype;
    using L    = typename RT::layout;
    using base = rt_base<T, L>;

    constexpr bool is_row = std::is_same_v<L, ducks::rt_layout::row>;
    constexpr int  E      = base::elements_per_thread;
    constexpr bool vectorizable = is_row && base::element_stride == 1
                                         && std::is_same_v<T, U> && sizeof(U) == 2;
    constexpr bool packable = is_row && !vectorizable
                                     && base::element_stride == 2
                                     && base::halves_interleave
                                     && sizeof(U) < sizeof(T)
                                     && (2*E*sizeof(U)) % 16 == 0;

    const auto unit = idx.template unit_coord<axis, 3>();
    U *dst_ptr = (U*)&dst[unit];
    const int row_stride = dst.template stride<axis>();
    const i32x4 srsrc = detail::row_bounded_srsrc<axis, RT::rows>(dst, unit, dst_ptr, limit);
    const int lane = laneid();
    const int l16  = lane & 15;
    const int half = lane >> 4;
    const bool aligned =
        ((reinterpret_cast<uintptr_t>(dst_ptr) | (row_stride * sizeof(U))) & 15) == 0;

    auto elementwise = [&](int i, int j, int row_base, int col_base) {
        #pragma unroll
        for(int e = 0; e < E; e++) {
            const int2 c = rt_base_coord<T, L>(e, lane);
            const T v = (e & 1) ? src.tiles[i][j].data[e>>1].y
                                : src.tiles[i][j].data[e>>1].x;
            detail::buffer_store_elem<U>(
                srsrc, ((row_base + c.x)*row_stride + col_base + c.y) * sizeof(U),
                base_types::convertor<U, T>::convert(v));
        }
    };

    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            const int row_base = i*src.tile_size_row;
            const int col_base = j*src.tile_size_col;
            const uint32_t run = ((row_base + l16)*row_stride + col_base) * sizeof(U);
            if constexpr (vectorizable) {
                if (aligned) {
                    #pragma unroll
                    for(int h = 0; h < 2; h++) {
                        __uint128_t raw;
                        __builtin_memcpy(&raw, (const void*)&src.tiles[i][j].data[4*h], sizeof(raw));
                        llvm_amdgcn_raw_buffer_store_b128(raw, srsrc, run + 16*h, 0, 0);
                    }
                }
                else elementwise(i, j, row_base, col_base);
            }
            else if constexpr (packable) {
                if (aligned) {
                    U val[E];
                    #pragma unroll
                    for(int e = 0; e < E; e++) {
                        const T v = (e & 1) ? src.tiles[i][j].data[e>>1].y
                                            : src.tiles[i][j].data[e>>1].x;
                        val[e] = base_types::convertor<U, T>::convert(v);
                    }
                    constexpr int NW = 2*E*sizeof(U)/4;
                    unsigned int w[NW];
                    detail::pack_interleaved_run<U, E>(val, half, w);
                    #pragma unroll
                    for(int k = 0; k < NW/4; k++) {
                        __uint128_t raw;
                        __builtin_memcpy(&raw, &w[4*k], sizeof(raw));
                        llvm_amdgcn_raw_buffer_store_b128(raw, srsrc, run + 16*k, 0, 0);
                    }
                }
                else elementwise(i, j, row_base, col_base);
            }
            else elementwise(i, j, row_base, col_base);
        }
    }
}
template<ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void store_bounded(const GL &dst, const RT &src, const COORD &idx,
                                            int limit = RT::rows) {
    store_bounded<2, RT, GL, COORD>(dst, src, idx, limit);
}

}
