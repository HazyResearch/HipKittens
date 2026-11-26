/**
 * @file
 * @brief Conversions between data layouts and types for register tiles.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

/*
 * Everything in this file is derived from one picture. Index a base tile by
 * (lane_low, pos), where lane_low = laneid & 15 and pos is the coordinate that
 * rt_base_coord() builds from the element index:
 *
 *   row layout: (row, col) = (lane_low, pos)
 *   col layout: (row, col) = (pos, lane_low)
 *
 * How pos is physically stored is the whole difference between the two kinds of
 * tile gfx11 has:
 *
 *   operands (bf16/half, replication 2): pos == the element index, 0..15, and
 *     lanes 0-15 mirror lanes 16-31. The wave-half bit carries no information.
 *
 *   accumulators (float, replication 1): pos = 2*e + (laneid >> 4). Bit 0 of pos
 *     lives in the wave-half bit; bits 3:1 are the element index. Nothing is
 *     replicated.
 *
 * That is why the conversions here are not the CDNA ones with a different lane
 * formula -- on CDNA every tile has the same replication, so copy() between two
 * dtypes is pointwise. Here a float tile and a bf16 tile of the same shape hold
 * their values in genuinely different places, and copy() has to move data.
 */

/* ----------  LAYOUT SWAPS  ---------- */

namespace detail {

/**
 * @brief One stage of the transpose network: swap bit i of lane_low with bit i of pos.
 *
 * Every value whose two bits disagree moves to the lane with LANEMASK flipped and
 * the element with ELEMMASK flipped. That is a symmetric pair swap, so a single
 * butterfly carries both directions at once.
 *
 * The selects are written as ternaries rather than branches on purpose: `a` is
 * divergent, and a lane exchange has to be issued with the whole wave active.
 */
template<int LANEMASK, int ELEMMASK, typename T, int E>
__device__ static inline void xpose_stage(T (&v)[E], const int lane) {
    const bool a = (lane & LANEMASK) != 0;
    #pragma unroll
    for(int e = 0; e < E; e++) {
        if((e & ELEMMASK) != 0) continue;              // visit each pair once
        const T lo = v[e], hi = v[e | ELEMMASK];
        const T recv = lane_xor<LANEMASK>(a ? lo : hi);
        v[e]            = a ? recv : lo;
        v[e | ELEMMASK] = a ? hi   : recv;
    }
}

/**
 * @brief Transpose a base tile's 16x16 (lane_low, pos) grid.
 *
 * swap_layout and transpose are the same movement in that grid -- one relabels
 * the axes, the other relabels the matrix -- so they share this routine and
 * differ only in the type they hand back.
 *
 * Four stages, stage i swapping bit i of lane_low with bit i of pos:
 *
 *   operands: pos is the element index, so every stage pairs lane_xor<1<<i> with
 *     an element-index flip. The wave-half bit is never touched, so the mirroring
 *     survives without any extra work.
 *
 *   accumulators: pos bit 0 is the wave-half bit, so stage 0 has to swap lane bit
 *     0 with lane bit 4. That is not a power-of-two butterfly; it is lane ^ 17,
 *     applied only to the lanes where the two bits disagree (those lanes swap
 *     their entire register file, since all of a lane's values share pos bit 0).
 *     Stages 1..3 are then the ordinary butterfly, shifted down one element bit.
 *
 * Cost per base tile: 20 lane exchanges for an operand, 8 permlanex16 plus 12
 * DPP movs for an accumulator. Compare a round trip through LDS, which is what
 * the CDNA code's dynamically-indexed __shfl lowers to.
 */
template<bool halves_interleave, typename T2, int P>
__device__ static inline void transpose_base_data(T2 (&dst)[P], const T2 (&src)[P], const int lane) {
    using T = typename base_types::packing<T2>::unpacked_type;
    constexpr int E = 2 * P;

    T v[E];
    #pragma unroll
    for(int k = 0; k < P; k++) { v[2*k] = src[k].x; v[2*k+1] = src[k].y; }

    if constexpr (halves_interleave) {
        static_assert(E == 8, "an accumulator base tile holds 8 elements per lane");
        const bool swap_halves = ((lane ^ (lane >> 4)) & 1) != 0;
        #pragma unroll
        for(int e = 0; e < E; e++) {
            const T other = lane_xor17(v[e]);
            v[e] = swap_halves ? other : v[e];
        }
        xpose_stage<2, 1, T, E>(v, lane);
        xpose_stage<4, 2, T, E>(v, lane);
        xpose_stage<8, 4, T, E>(v, lane);
    }
    else {
        static_assert(E == 16, "an operand base tile holds 16 elements per lane");
        xpose_stage<1, 1, T, E>(v, lane);
        xpose_stage<2, 2, T, E>(v, lane);
        xpose_stage<4, 4, T, E>(v, lane);
        xpose_stage<8, 8, T, E>(v, lane);
    }

    #pragma unroll
    for(int k = 0; k < P; k++) { dst[k].x = v[2*k]; dst[k].y = v[2*k+1]; }
}

} // namespace detail

/**
 * @brief Swaps the layout of a register base tile.
 *
 * The matrix is unchanged; only the mapping from lanes and registers to its
 * entries is. See detail::transpose_base_data for why that is a transpose.
 *
 * @tparam T The data type of the register tile elements.
 * @tparam layout The current layout of the register tile.
 * @param dst[out] Reference to the destination register base tile where the result will be stored.
 * @param src[in] Reference to the source register base tile to be swapped.
 */
template<typename T, ducks::rt_layout::all layout>
__device__ inline void swap_layout(rt_base<T, typename ducks::rt_layout::transpose<layout>::type> &dst, const rt_base<T, layout> &src) {
    using base = rt_base<T, layout>;
    detail::transpose_base_data<base::halves_interleave>(dst.data, src.data, laneid());
}

/**
 * @brief Swaps the layout of a register tile.
 *
 * @tparam T2 The data type of the register tile elements.
 * @tparam _height The height of the register tile.
 * @tparam _width The width of the register tile.
 * @tparam layout The current layout of the register tile.
 * @param dst[out] Reference to the destination register tile where the result will be stored.
 * @param src[in] Reference to the source register tile to be swapped.
 */
template<typename T2, int _height, int _width, ducks::rt_layout::all layout>
__device__ static inline void swap_layout(rt<T2, _height, _width, typename ducks::rt_layout::transpose<layout>::type> &dst, const rt<T2, _height, _width, layout> &src) {

    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            swap_layout(dst.tiles[i][j], src.tiles[i][j]);
        }
    }
}

/**
 * @brief Swaps the layout of a register base tile in place.
 *
 * @tparam T2 The data type of the register tile elements.
 * @tparam layout The current layout of the register tile.
 * @param src[in] Reference to the register base tile to be swapped in place.
 * @return A reference to the swapped register base tile.
 */
template<typename T2, ducks::rt_layout::all layout>
__device__ inline rt_base<T2, typename ducks::rt_layout::transpose<layout>::type>& swap_layout_inplace(const rt_base<T2, layout> &src) {
    rt_base<T2, typename ducks::rt_layout::transpose<layout>::type> &dst = *(rt_base<T2, typename ducks::rt_layout::transpose<layout>::type>*)(&src);
    swap_layout(dst, src);
    return dst;
}

/**
 * @brief Swaps the layout of a register tile in place.
 *
 * @tparam T2 The data type of the register tile elements.
 * @tparam _height The height of the register tile.
 * @tparam _width The width of the register tile.
 * @tparam layout The current layout of the register tile.
 * @param tile[in,out] Reference to the register tile to be swapped in place.
 * @return A reference to the swapped register tile.
 */
template<typename T2, int _rows, int _cols, ducks::rt_layout::all layout>
__device__ static inline rt<T2, _rows, _cols, typename ducks::rt_layout::transpose<layout>::type>& swap_layout_inplace(rt<T2, _rows, _cols, layout> &tile) {
    #pragma unroll
    for(int i = 0; i < tile.height; i++) {
        #pragma unroll
        for(int j = 0; j < tile.width; j++) {
            swap_layout_inplace(tile.tiles[i][j]);
        }
    }
    return *(rt<T2, _rows, _cols, typename ducks::rt_layout::transpose<layout>::type>*)(&tile);
}

/* ----------  TRANSPOSE  ---------- */

/**
 * @brief Transposes a register base tile.
 *
 * Same movement as swap_layout, and safe when dst and src alias, because the
 * values are lifted into a local array first.
 *
 * @tparam T The data type of the register tile elements.
 * @tparam layout The current layout of the register tile.
 * @param dst[out] Reference to the register tile in which to store the transposed src.
 * @param src[in] Reference to the register tile to be transposed.
 */
template<typename T, ducks::rt_layout::all layout>
__device__ inline void transpose(rt_base<T, layout> &dst, const rt_base<T, layout> &src) {
    using base = rt_base<T, layout>;
    detail::transpose_base_data<base::halves_interleave>(dst.data, src.data, laneid());
}

/**
 * @brief Transposes a register tile.
 *
 * This function is marked "sep", which means that the registers underlying dst MUST be separate
 * from the registers underlying src.
 *
 * @tparam RT The type of the destination register tile.
 * @param dst[out] Reference to the register tile in which to store the transposed src.
 * @param src[in] Reference to the register tile to be transposed.
 */
template<ducks::rt::all RT>
__device__ static inline void transpose_sep(RT &dst, const rt<typename RT::T, RT::cols, RT::rows, typename RT::layout> &src) {
    #pragma unroll
    for(int i = 0; i < RT::height; i++) {
        #pragma unroll
        for(int j = 0; j < RT::width; j++) {
            transpose(dst.tiles[i][j], src.tiles[j][i]);
        }
    }
}

/**
 * @brief Transposes a register tile by relabeling, without moving any data.
 *
 * The other transpose above moves values; this one does not move anything at all.
 * A base tile's storage depends only on its dtype -- the layout tag is purely how
 * (lane_low, pos) is read, (row, col) for row and (col, row) for col -- so reading
 * the same registers under the opposite tag already yields the transpose. Pair that
 * with sending tiles[i][j] to tiles[j][i] and the whole tile is transposed for free.
 *
 * dst and src must not alias; the grid permutation would overwrite as it goes.
 *
 * @param result[out] The transposed tile, of the mirrored shape and opposite layout.
 * @param tile[in] The tile to transpose.
 */
template<typename T2, int _rows, int _cols, ducks::rt_layout::all layout>
__device__ static inline void transpose(rt<T2, _cols, _rows, typename ducks::rt_layout::transpose<layout>::type> &result, const rt<T2, _rows, _cols, layout> &tile) {
    #pragma unroll
    for (int i = 0; i < tile.height; i++) {
        #pragma unroll
        for (int j = 0; j < tile.width; j++) {
            #pragma unroll
            for (int k = 0; k < tile.packed_per_tile; k++) {
                result.tiles[j][i].data[k] = tile.tiles[i][j].data[k];
            }
        }
    }
}

/**
 * @brief Transposes a register base tile in-place.
 *
 * @tparam T2 The data type of the register base tile elements.
 * @tparam layout The current layout of the register base tile.
 * @param src[in] Reference to the register tile to be transposed.
 * @return A reference to the transposed register base tile.
 */
template<typename T2, ducks::rt_layout::all layout>
__device__ inline rt_base<T2, layout>& transpose_inplace(rt_base<T2, layout> &src) {
    transpose(src, src);
    return src;
}
/**
 * @brief Transposes a square register tile in-place.
 *
 * @tparam T2 The data type of the register tile elements.
 * @tparam _rows The height of the register tile. (Must equal _cols.)
 * @tparam _cols The width of the register tile. (Must equal _rows.)
 * @tparam layout The current layout of the register tile.
 * @param src[in] Reference to the register tile to be transposed.
 * @return A reference to the transposed register tile.
 */
template<typename T2, int _rows, int _cols, ducks::rt_layout::all layout>
__device__ static inline rt<T2, _rows, _cols, layout>& transpose_inplace(rt<T2, _rows, _cols, layout> &tile) {
    static_assert(_cols == _rows, "in-place register tile transpose is only allowed for square tiles.");
    #pragma unroll
    for(int i = 0; i < tile.height; i++) {
        #pragma unroll
        for(int j = 0; j < i; j++) {
            rt_base<T2, layout> tmp;
            copy(tmp, tile.tiles[i][j]);
            transpose(tile.tiles[i][j], tile.tiles[j][i]);
            transpose(tile.tiles[j][i], tmp);
        }
        transpose_inplace(tile.tiles[i][i]);
    }
    return tile;
}

/* ----------  TYPE SWAPS  ---------- */

/**
 * @brief Copies a register base tile, converting the underlying type if necessary.
 *
 * Between two tiles of the same replication this is pointwise, as on CDNA. Across
 * replications it is not, because the two kinds of tile spread `pos` differently:
 *
 *   accumulator -> operand (float -> bf16/half). A lane owns the positions whose
 *     parity matches its wave half; its mirror owns the rest. One permlanex16 per
 *     element fetches the other parity, and interleaving the two gives the full
 *     K vector, identical in both halves. This is the conversion attention needs
 *     between the score matrix and the second matmul's operand, and it is the
 *     reason to pay for a replicated operand layout at all -- the alternative is
 *     a full transpose.
 *
 *   operand -> accumulator. Pure discard: keep the positions this lane owns.
 *
 * @tparam T The data type of the destination register elements.
 * @tparam U The data type of the source register elements.
 * @tparam layout The current layout of the register base tile.
 * @param[out] dst A reference to the destination register base tile.
 * @param[in] src A reference to the source register base tile.
 */
template<typename T, typename U, ducks::rt_layout::all layout>
__device__ static inline void copy(rt_base<T, layout> &dst, const rt_base<U, layout> &src) {
    using D  = rt_base<T, layout>;
    using S  = rt_base<U, layout>;
    using T2 = typename base_types::packing<T>::packed_type;
    using U2 = typename base_types::packing<U>::packed_type;

    if constexpr (D::replication == S::replication) {
        #pragma unroll
        for(int k = 0; k < D::packed_per_thread; k++) {
            dst.data[k] = base_types::convertor<T2, U2>::convert(src.data[k]);
        }
    }
    else if constexpr (S::replication == 1) {
        const int h = laneid() >> 4;
        #pragma unroll
        for(int p = 0; p < S::elements_per_thread; p++) {
            // Element p of the source is position 2p + h. The mirror lane holds
            // position 2p + (1-h) in the same element, so one exchange completes
            // the pair; h decides which way round they go.
            const U own   = (p & 1) ? src.data[p>>1].y : src.data[p>>1].x;
            const U other = lane_xor<HALF_WAVE_SWAP>(own);
            dst.data[p].x = base_types::convertor<T, U>::convert(h ? other : own);
            dst.data[p].y = base_types::convertor<T, U>::convert(h ? own   : other);
        }
    }
    else {
        const int h = laneid() >> 4;
        #pragma unroll
        for(int p = 0; p < D::elements_per_thread; p++) {
            // Destination element p is position 2p + h, which the source holds at
            // element 2p + h, i.e. the h-th half of its packed register p.
            //
            // Convert before selecting, not after. U is bf16 or half here, which
            // are classes wrapping a short; a runtime select between two of them
            // materialises a temporary, and that is enough to make LLVM give up
            // on SROA and park the whole source tile in scratch. T is float in
            // this branch -- selecting on it stays in registers.
            const T lo = base_types::convertor<T, U>::convert(src.data[p].x);
            const T hi = base_types::convertor<T, U>::convert(src.data[p].y);
            const T c  = h ? hi : lo;
            if(p & 1) dst.data[p>>1].y = c;
            else      dst.data[p>>1].x = c;
        }
    }
}

/**
 * @brief Copies a register tile, converting the underlying type if necessary.
 *
 * @tparam T2 The data type of the destination register elements.
 * @tparam U2 The data type of the source register elements.
 * @tparam _height The height (in units of 16) of the register tiles.
 * @tparam _width The width (in units of 16) of the register tiles.
 * @tparam layout The current layout of the register tile.
 * @param[out] dst A reference to the destination register tile.
 * @param[in] src A reference to the source register tile.
 */
template<typename T2, typename U2, int _height, int _width, ducks::rt_layout::all layout>
__device__ static inline void copy(rt<T2, _height, _width, layout> &dst, const rt<U2, _height, _width, layout> &src) {
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            copy(dst.tiles[i][j], src.tiles[i][j]);
        }
    }
}

/* ----------  MASKS AND FILLS  ---------- */

namespace detail {

/**
 * @brief Fill on one side of a shifted diagonal, without materialising the diagonal.
 *
 * The readable lowering asks rt_base_coord() for each element's (row, col) and
 * writes the predicate out in full. That is the right shape for an axis
 * predicate, where only one of the two is needed and it is a single register.
 * A *diagonal* predicate needs both, and `col - row` varies with the element
 * index, so the same lowering holds one register per element of the tile --
 * eight, for every layout on this architecture -- live for as long as the mask
 * is in a loop. That is what it costs: the attention kernel's causal mask
 * spilled 152 B/lane written that way, the whole Q register file along with
 * it, and spilling is a wrong answer in a kernel that hand-places `s_waitcnt`.
 *
 * The element index is not actually on the lane axis, though. Writing out
 * rt_base_coord(),
 *
 *     row layout:  col - row = (element_stride*e + h) - l16
 *     col layout:  col - row = l16 - (element_stride*e + h)
 *
 * with `l16 = lane & 15` and `h` the wave half, so
 *
 *     col - row = dbase +/- element_stride*e,   dbase = +/-(h - l16)
 *
 * -- one lane-dependent register for the whole tile, and a per-element *shift*
 * that is a compile-time constant once the loops unroll. Which side of the
 * compare it lands on matters. `dbase >= -row_idx - off - se` is one VGPR and a
 * scalar threshold on paper, but the right-hand side is loop-invariant per
 * element and LLVM reassociates it straight back into eight hoisted registers;
 * that was measured too, and it spills harder than the naive form. Folding
 * `row_idx` into the lane value instead leaves
 *
 *     keep  <=>  dbase + row_idx >= -(off + se)
 *
 * whose right-hand side is an *integer literal*. There is nothing left to
 * hoist: one VGPR for the whole tile, and `elements_per_thread` compares
 * against immediates.
 *
 * @tparam KEEP_GE true keeps `col - row >= -row_idx` (triu's complement),
 *                 false keeps `col - row <= -row_idx` (tril's).
 */
template<bool KEEP_GE, ducks::rt::all RT>
__device__ static inline void diag_fill(
        RT &dst, const RT &src, const int row_idx,
        const typename base_types::packing<typename RT::dtype>::unpacked_type &val) {
    using T    = typename RT::T;
    using L    = typename RT::layout;
    using base = rt_base<T, L>;
    constexpr bool is_row = std::is_same_v<L, ducks::rt_layout::row>;
    const int lane  = laneid();
    const int h     = base::halves_interleave ? (lane >> 4) : 0;
    const int dbase = is_row ? (h - (lane & 15)) : ((lane & 15) - h);
    // The one register the predicate costs. Everything else is an immediate.
    const int d = dbase + row_idx;
    #pragma unroll
    for(int i = 0; i < RT::height; i++) {
        #pragma unroll
        for(int j = 0; j < RT::width; j++) {
            // The tile's own offset into the wider tile, on the same axis.
            const int off = j * base::tile_size_col - i * base::tile_size_row;
            #pragma unroll
            for(int e = 0; e < base::elements_per_thread; e++) {
                const int se  = is_row ? base::element_stride * e
                                       : -base::element_stride * e;
                const int c = -(off + se);
                const bool keep = KEEP_GE ? (d >= c) : (d <= c);
                const T v = (e & 1) ? src.tiles[i][j].data[e>>1].y
                                    : src.tiles[i][j].data[e>>1].x;
                const T out = keep ? v : T(val);
                if(e & 1) dst.tiles[i][j].data[e>>1].y = out;
                else      dst.tiles[i][j].data[e>>1].x = out;
            }
        }
    }
}

/**
 * @brief Fill on one side of an axis index, without materialising the index.
 *
 * The same economy as `diag_fill`, for the rectangular masks. Half of these
 * are already cheap: a tile's row index is `lane & 15` in row layout and its
 * column index is `lane & 15` in col layout, one register either way. The other
 * half are not -- the *element* axis reads `element_stride*e + h`, which is a
 * value per element of the tile, and a mask inside a loop holds all of them
 * live. Attention's tail mask is on that axis (a row of S^T is a kv index) and
 * cost six hoisted, spilled registers until it was hidden behind a branch.
 *
 * Writing the coordinate as `K + ce + base` -- `K` the tile's offset, `ce` the
 * per-element shift, `base` the one lane-dependent term -- and moving the
 * constants across the compare leaves `base - idx` against an immediate. One
 * register, and no reason left to guard the mask with a branch.
 *
 * @tparam AXIS_IS_ROW true for the row masks (upper/lower), false for the
 *                     column masks (left/right).
 * @tparam FILL_LT     true fills where the coordinate is `< idx`, false where
 *                     it is `>= idx`.
 */
template<bool AXIS_IS_ROW, bool FILL_LT, ducks::rt::all RT>
__device__ static inline void axis_fill(
        RT &dst, const RT &src, const int idx,
        const typename base_types::packing<typename RT::dtype>::unpacked_type &val) {
    using T    = typename RT::T;
    using L    = typename RT::layout;
    using base = rt_base<T, L>;
    constexpr bool is_row = std::is_same_v<L, ducks::rt_layout::row>;
    // row layout puts the elements on the column axis and col layout on the
    // row axis, so exactly one of the two masks pays the per-element shift.
    constexpr bool on_elements = (AXIS_IS_ROW != is_row);
    const int lane = laneid();
    const int h    = base::halves_interleave ? (lane >> 4) : 0;
    const int t    = (on_elements ? h : (lane & 15)) - idx;
    #pragma unroll
    for(int i = 0; i < RT::height; i++) {
        #pragma unroll
        for(int j = 0; j < RT::width; j++) {
            const int K = AXIS_IS_ROW ? i * base::tile_size_row
                                      : j * base::tile_size_col;
            #pragma unroll
            for(int e = 0; e < base::elements_per_thread; e++) {
                const int c = -(K + (on_elements ? base::element_stride * e : 0));
                const bool fill = FILL_LT ? (t < c) : (t >= c);
                const T v = (e & 1) ? src.tiles[i][j].data[e>>1].y
                                    : src.tiles[i][j].data[e>>1].x;
                const T out = fill ? T(val) : v;
                if(e & 1) dst.tiles[i][j].data[e>>1].y = out;
                else      dst.tiles[i][j].data[e>>1].x = out;
            }
        }
    }
}

} // namespace detail

/* ----------  CAUSAL  ---------- */

/**
 * @brief Makes a square register tile causal by zeroing elements above the main diagonal.
 *
 * @tparam RT The type of the register tile.
 * @param dst[out] The destination register tile.
 * @param src[in] The source register tile.
 * @param val[in] The value to fill the upper triangle with.
 */
template<ducks::rt::all RT>
__device__ static inline void make_causal(RT &dst, const RT &src, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    detail::diag_fill<false>(dst, src, 0, val);
}

/**
 * @brief Makes a square register tile anti-causal by zeroing elements below the main diagonal.
 *
 * @tparam RT The type of the register tile.
 * @param dst[out] The destination register tile.
 * @param src[in] The source register tile.
 * @param val[in] The value to fill the lower triangle with.
 */
template<ducks::rt::all RT>
__device__ static inline void make_causal_t(RT &dst, const RT &src, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    detail::diag_fill<true>(dst, src, 0, val);
}

/* ----------  TRIANGULAR FILLS  ---------- */

/**
 * @brief Makes a register tile triangular by filling elements above the shifted diagonal.
 *
 * @tparam RT The type of the register tile.
 * @param dst[out] The destination register tile.
 * @param src[in] The source register tile.
 * @param row_idx[in] The row index to triangularize from.
 * @param val[in] The value to fill with.
 */
template<ducks::rt::all RT>
__device__ static inline void tril(RT &dst, const RT &src, const int row_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    detail::diag_fill<false>(dst, src, row_idx, val);
}

/**
 * @brief Makes a register tile triangular by filling elements below the shifted diagonal.
 *
 * @tparam RT The type of the register tile.
 * @param dst[out] The destination register tile.
 * @param src[in] The source register tile.
 * @param row_idx[in] The row index to triangularize from.
 * @param val[in] The value to fill with.
 */
template<ducks::rt::all RT>
__device__ static inline void triu(RT &dst, const RT &src, const int row_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    detail::diag_fill<true>(dst, src, row_idx, val);
}

/* ----------  RECTANGULAR FILLS  ---------- */

/**
 * @brief Fills a register tile with a value from a column index rightwards.
 *
 * @tparam RT The type of the register tile.
 * @param dst[in,out] The register tile to be filled.
 * @param src[in] The register tile to copy from.
 * @param col_idx[in] The column index to fill from and onwards to the right.
 * @param val[in] The value to fill with.
 */
template<ducks::rt::all RT>
__device__ static inline void right_fill(RT &dst, const RT &src, const int col_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    if(col_idx >= RT::cols) return;
    detail::axis_fill<false, false>(dst, src, col_idx, val);
}

/**
 * @brief Fills a register tile with a value to the left of a column index.
 *
 * @tparam RT The type of the register tile.
 * @param dst[in,out] The register tile to be filled.
 * @param src[in] The register tile to copy from.
 * @param col_idx[in] The column index to fill to the left of (exclusive).
 * @param val[in] The value to fill with.
 */
template<ducks::rt::all RT>
__device__ static inline void left_fill(RT &dst, const RT &src, const int col_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    if(col_idx <= 0) return;
    detail::axis_fill<false, true>(dst, src, col_idx, val);
}

/**
 * @brief Fills a register tile with a value above a row index.
 *
 * @tparam RT The type of the register tile.
 * @param dst[in,out] The register tile to be filled.
 * @param src[in] The register tile to copy from.
 * @param row_idx[in] The row index to fill to, from the top (exclusive).
 * @param val[in] The value to fill with.
 */
template<ducks::rt::all RT>
__device__ static inline void upper_fill(RT &dst, const RT &src, const int row_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    if(row_idx <= 0) return;
    detail::axis_fill<true, true>(dst, src, row_idx, val);
}

/**
 * @brief Fills a register tile with a value from a row index downwards.
 *
 * @tparam RT The type of the register tile.
 * @param dst[in,out] The register tile to be filled.
 * @param src[in] The register tile to copy from.
 * @param row_idx[in] The row index to fill from and onwards to the bottom (inclusive).
 * @param val[in] The value to fill with.
 */
template<ducks::rt::all RT>
__device__ static inline void lower_fill(RT &dst, const RT &src, const int row_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0) {
    if(row_idx >= RT::rows) return;
    detail::axis_fill<true, false>(dst, src, row_idx, val);
}

/* ----------  SUBTILE  ---------- */

/**
* @brief Returns a reference to a subtile of the given tile.
*
* @tparam subtile_height The height of the subtile.
* @tparam RT The type of the input tile, which must satisfy the ducks::rt::all concept.
* @param src The input tile.
* @param idx The coord of the subtile.
* @return A reference to the subtile.
*
* @note The subtile height must evenly divide the tile height.
*/
template<int subtile_rows, ducks::rt::all RT>
__device__ inline rt<typename RT::T, subtile_rows, RT::cols, typename RT::layout> &subtile_inplace(RT & src, int idx) {
    using T = typename RT::T;
    static_assert(RT::height % (subtile_rows / TILE_ROW_DIM<T>) == 0, "subtile height should evenly divide tile height.");
    return reinterpret_cast<rt<typename RT::T, subtile_rows, RT::cols, typename RT::layout>&>(
        src.tiles[idx*(subtile_rows / TILE_ROW_DIM<T>)]
    );
}

}
