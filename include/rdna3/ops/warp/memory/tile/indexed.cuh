/**
 * @file
 * @brief Global <-> register tile by row index, and global atomics.
 *
 * Row gather and scatter are what a mixture-of-experts GEMM reads and writes
 * its tokens with, in expert order; an embedding, its vocabulary rows; a
 * KV cache write, its slots. Atomics are the reduction a kernel cannot do in
 * registers because its partials live in different workgroups.
 */

#pragma once

#include "global_to_register.cuh"

namespace kittens {

namespace detail {

/// A buffer resource over every row of `g` (batch, depth and rows flattened),
/// and the voffset a row index past it gets: far enough out that no column
/// offset added to it can wrap back in.
template<ducks::gl::all GL>
__device__ inline i32x4 whole_srsrc(const GL &g, uint32_t &oob, int &nrows) {
    using U = typename GL::dtype;
    nrows = g.batch() * g.depth() * g.rows();
    const uint64_t bytes = uint64_t(nrows) * uint64_t(g.template stride<2>()) * sizeof(U);
    const uint32_t cap = 0xF0000000u;
    oob = bytes < cap ? uint32_t(bytes) : cap;
    return make_srsrc(g.raw_ptr, oob);
}

/// This lane's row index for base-tile row `i`: idx[at + 16*i + lane%16],
/// or -1 at or past `count`.
template<ducks::gl::all IL>
__device__ inline int gather_index(const IL &idx, int at, int count, int i) {
    static_assert(std::is_same_v<typename IL::dtype, int>, "row indices are int32");
    const int pos = at + 16 * i + (laneid() & 15);
    return pos < count ? idx.raw_ptr[pos] : -1;
}

} // namespace detail

/**
 * @brief load() with the tile's rows chosen by index: tile row r is row
 * `idx[at + r]` of `src`, from column `col`.
 *
 * One buffer resource spans the whole of `src` and each lane's row comes in
 * as its voffset, so an index outside [0, rows) -- the padding vLLM's
 * moe_align_block_size fills with numel, a -1 sentinel -- reads zeros with
 * no branch, as does a position at or past `count`. `src` is indexed by its
 * flattened rows (batch, depth and rows). The bytes it spans must stay
 * below 3.75 GB, since the voffset is 32-bit.
 *
 * Row layout only: a lane's elements of a base tile are all in its row,
 * which is what lets the row's address be a per-lane value.
 */
template<ducks::rt::all RT, ducks::gl::all GL, ducks::gl::all IL>
__device__ inline static void load_gather(RT &dst, const GL &src, const IL &idx,
                                          int at, int count, int col) {
    using T  = typename base_types::packing<typename RT::dtype>::unpacked_type;
    using U  = typename GL::dtype;
    using L  = typename RT::layout;
    using base = rt_base<T, L>;
    static_assert(std::is_same_v<L, ducks::rt_layout::row>, "gather fills row-layout tiles");
    constexpr int  E    = base::elements_per_thread;
    constexpr int  RUN  = base::tile_size_col;
    constexpr bool vectorizable = base::element_stride == 1 && std::is_same_v<T, U> &&
                                  sizeof(U) == 2;
    constexpr bool widening = !vectorizable && sizeof(U) < sizeof(T) &&
                              (RUN * sizeof(U)) % 16 == 0;

    uint32_t oob; int nrows;
    const i32x4 srsrc = detail::whole_srsrc(src, oob, nrows);
    const uint32_t row_bytes = uint32_t(src.template stride<2>()) * sizeof(U);
    const int lane = laneid(), half = lane >> 4;

    #pragma unroll
    for (int i = 0; i < dst.height; i++) {
        const int r = detail::gather_index(idx, at, count, i);
        const uint32_t row_off = (r >= 0 && r < nrows)
            ? uint32_t(r) * row_bytes + uint32_t(col) * sizeof(U) : oob;
        #pragma unroll
        for (int j = 0; j < dst.width; j++) {
            const uint32_t off = row_off == oob ? oob
                : row_off + uint32_t(j * dst.tile_size_col) * sizeof(U);
            if constexpr (vectorizable) {
                #pragma unroll
                for (int h = 0; h < 2; h++) {
                    const __uint128_t raw = llvm_amdgcn_raw_buffer_load_b128(srsrc, off + 16 * h, 0, 0);
                    __builtin_memcpy((void *)&dst.tiles[i][j].data[4 * h], &raw, sizeof(raw));
                }
            } else if constexpr (widening) {
                constexpr int NV = RUN * sizeof(U) / 16, PER = 16 / sizeof(U);
                U run[RUN];
                #pragma unroll
                for (int k = 0; k < NV; k++) {
                    const __uint128_t raw = llvm_amdgcn_raw_buffer_load_b128(srsrc, off + 16 * k, 0, 0);
                    __builtin_memcpy(&run[k * PER], &raw, sizeof(raw));
                }
                #pragma unroll
                for (int e = 0; e < E; e++) {
                    const T v0 = base_types::convertor<T, U>::convert(run[rt_base_coord<T, L>(e, 0).y]);
                    const T v1 = base_types::convertor<T, U>::convert(run[rt_base_coord<T, L>(e, 16).y]);
                    const T val = half ? v1 : v0;
                    if (e & 1) dst.tiles[i][j].data[e >> 1].y = val;
                    else       dst.tiles[i][j].data[e >> 1].x = val;
                }
            } else {
                #pragma unroll
                for (int e = 0; e < E; e++) {
                    const int cy = rt_base_coord<T, L>(e, lane).y;
                    const T val = base_types::convertor<T, U>::convert(
                        detail::buffer_load_elem<U>(srsrc, off == oob ? oob : off + cy * sizeof(U), 0));
                    if (e & 1) dst.tiles[i][j].data[e >> 1].y = val;
                    else       dst.tiles[i][j].data[e >> 1].x = val;
                }
            }
        }
    }
}

/**
 * @brief store() with the tile's rows chosen by index: tile row r goes to row
 * `idx[at + r]` of `dst`, from column `col`. An index outside [0, rows), or
 * a position at or past `count`, writes nothing (the buffer drops it).
 * Two tile rows naming the same row race; which one lands is unspecified.
 */
template<ducks::rt::all RT, ducks::gl::all GL, ducks::gl::all IL>
__device__ inline static void store_scatter(const GL &dst, const RT &src, const IL &idx,
                                            int at, int count, int col) {
    using T  = typename base_types::packing<typename RT::dtype>::unpacked_type;
    using U  = typename GL::dtype;
    using L  = typename RT::layout;
    using base = rt_base<T, L>;
    static_assert(std::is_same_v<L, ducks::rt_layout::row>, "scatter stores row-layout tiles");
    constexpr int  E    = base::elements_per_thread;
    constexpr bool vectorizable = base::element_stride == 1 && std::is_same_v<T, U> &&
                                  sizeof(U) == 2;
    constexpr bool packable = !vectorizable && base::element_stride == 2 &&
                              base::halves_interleave && sizeof(U) < sizeof(T) &&
                              (2 * E * sizeof(U)) % 16 == 0;

    uint32_t oob; int nrows;
    const i32x4 srsrc = detail::whole_srsrc(dst, oob, nrows);
    const uint32_t row_bytes = uint32_t(dst.template stride<2>()) * sizeof(U);
    const int lane = laneid(), half = lane >> 4;

    #pragma unroll
    for (int i = 0; i < src.height; i++) {
        const int r = detail::gather_index(idx, at, count, i);
        const uint32_t row_off = (r >= 0 && r < nrows)
            ? uint32_t(r) * row_bytes + uint32_t(col) * sizeof(U) : oob;
        #pragma unroll
        for (int j = 0; j < src.width; j++) {
            const uint32_t off = row_off == oob ? oob
                : row_off + uint32_t(j * src.tile_size_col) * sizeof(U);
            if constexpr (vectorizable) {
                #pragma unroll
                for (int h = 0; h < 2; h++) {
                    __uint128_t raw;
                    __builtin_memcpy(&raw, (const void *)&src.tiles[i][j].data[4 * h], sizeof(raw));
                    llvm_amdgcn_raw_buffer_store_b128(raw, srsrc, off + 16 * h, 0, 0);
                }
            } else if constexpr (packable) {
                U val[E];
                #pragma unroll
                for (int e = 0; e < E; e++) {
                    const T v = (e & 1) ? src.tiles[i][j].data[e >> 1].y
                                        : src.tiles[i][j].data[e >> 1].x;
                    val[e] = base_types::convertor<U, T>::convert(v);
                }
                constexpr int NW = 2 * E * sizeof(U) / 4;
                unsigned int w[NW];
                detail::pack_interleaved_run<U, E>(val, half, w);
                #pragma unroll
                for (int k = 0; k < NW / 4; k++) {
                    __uint128_t raw;
                    __builtin_memcpy(&raw, &w[4 * k], sizeof(raw));
                    llvm_amdgcn_raw_buffer_store_b128(raw, srsrc, off + 16 * k, 0, 0);
                }
            } else {
                #pragma unroll
                for (int e = 0; e < E; e++) {
                    const int cy = rt_base_coord<T, L>(e, lane).y;
                    const T v = (e & 1) ? src.tiles[i][j].data[e >> 1].y
                                        : src.tiles[i][j].data[e >> 1].x;
                    detail::buffer_store_elem<U>(srsrc, off == oob ? oob : off + cy * sizeof(U),
                                                 base_types::convertor<U, T>::convert(v));
                }
            }
        }
    }
}

/**
 * @brief Add an fp32 tile into a float global, element by element, atomically.
 *
 * The reduction for partials that live in different workgroups -- split-K,
 * a scatter-add -- without a second pass. fp32 tiles only: a 16-bit operand
 * tile is replicated across the wave halves, and adding it would count every
 * element twice; the fp32 layouts hold each element once.
 */
template<ducks::rt::all RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void atomic_add(const GL &dst, const RT &src, const COORD &idx) {
    using T = typename base_types::packing<typename RT::dtype>::unpacked_type;
    using L = typename RT::layout;
    static_assert(std::is_same_v<T, float> && std::is_same_v<typename GL::dtype, float>,
                  "atomic_add adds an fp32 tile into a float global");
    constexpr int E = rt_base<T, L>::elements_per_thread;
    float *p = (float *)&dst[(idx.template unit_coord<2, 3>())];
    const int row_stride = dst.template stride<2>();
    const int lane = laneid();
    #pragma unroll
    for (int i = 0; i < src.height; i++) {
        #pragma unroll
        for (int j = 0; j < src.width; j++) {
            #pragma unroll
            for (int e = 0; e < E; e++) {
                const int2 c = rt_base_coord<T, L>(e, lane);
                const float v = (e & 1) ? src.tiles[i][j].data[e >> 1].y
                                        : src.tiles[i][j].data[e >> 1].x;
                atomicAdd(&p[(i * src.tile_size_row + c.x) * row_stride +
                             j * src.tile_size_col + c.y], v);
            }
        }
    }
}

/**
 * @brief One atomic on one int32 element, once per warp: `op` 0 adds `v`, 1
 * takes the max. Returns, in every lane, the element's value before it.
 *
 * Lane 0 issues it and the result is broadcast: issued by every lane it
 * would be 32 atomics, and the counter would count lanes rather than warps.
 * A workgroup of W warps that calls it performs W of them.
 */
template<ducks::gl::all GL, typename COORD=coord<>>
__device__ inline static int atomic_scalar(const GL &dst, const COORD &idx, int v, int op) {
    static_assert(std::is_same_v<typename GL::dtype, int>, "atomic_scalar works on int32");
    int old = 0;
    if (laneid() == 0) {
        int *p = (int *)&dst[(idx.template unit_coord<-1, 3>())];
        old = op == 0 ? atomicAdd(p, v) : atomicMax(p, v);
    }
    return __builtin_amdgcn_readfirstlane(old);
}

} // namespace kittens
