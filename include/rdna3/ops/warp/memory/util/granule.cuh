/**
 * @file
 * @brief Granule-decomposed LDS addressing for 16x16 operand fragments.
 *
 * `load`/`store` on a shared tile form one full VGPR address per ds_read_b128.
 * That is the right default when a kernel touches a tile or two.  A kernel that
 * reads dozens of fragments out of the *same* tile inside a loop cannot afford
 * it: every one of those addresses is loop-invariant, LLVM hoists them all, and
 * they spill.  The decomposition below collapses them to `swizzle_bytes/16`
 * registers per tile by moving the whole fragment-coordinate dependence into
 * ds_read_b128's 16-bit immediate offset, where it costs no register at all.
 *
 * Found in kernels/rdna3/attn/fwd/attn.cpp; the derivation and the exhaustive
 * check against `st::idx` are in the comment below.
 */
#pragma once

#include <hip/hip_runtime.h>
#include <type_traits>
#include <utility>

namespace kittens {

// ---- LDS addressing for the two hot matmuls --------------------------------
//
// The library's load() forms one full VGPR address per ds_read_b128.  That is
// the right default for a tile or two.  In the attention forward kernel it is
// fatal: the QK and PV loops between them issue 128 reads whose addresses are every one of them
// loop-invariant, so LLVM hoists 128 addresses out of the KV loop and then
// spills most of them.  Measured, on the version that used load():
//
//     91 scratch_store, all in the loop preheader, none in the body
//     87 scratch_load,  all in the body
//     every spilled value is `v_add_nc_u32 v0, s6|s7, vN`  -- an LDS address
//
// 372 bytes/lane of scratch, and in a kernel that hand-manages s_waitcnt a
// spill is a correctness bug, not a slowdown.
//
// Those 128 addresses collapse to nine, because st::idx is affine in almost all
// of its arguments.  Write SB for the tile's swizzle_bytes and SUB = SB/2 for
// its subtile_cols.  Then for any 2-byte tile with SB >= 64,
//
//     idx(ptr, {R + l, c}) = idx(ptr, {l, 8*((c % SUB) / 8)})    <- lane, SB/16 of them
//                          + (c / SUB) * rows * SB + SB * R      <- compile-time
//
// for l in 0..15 and R a multiple of 16.  Two facts make it true.  The swizzle
// key is bits 7..9 of the tile-relative offset and both dropped terms are
// multiples of 1024, so neither perturbs the key -- which is what lets them
// leave the XOR and become addends.  And the swizzle is a permutation of
// 16-byte granules *within* a row, so the lane-dependent part takes SB/16
// values, one per granule, however wide the tile is and however many fragments
// are read out of it.
//
// SB >= 64 is the real precondition, not an artifact: at SB = 32 the row stride
// is small enough that 16 rows reach bit 9, the key changes with R, and the row
// term stops being constant.  A 32-byte swizzle means a 16-column tile, which
// nothing here stages.
//
// So: SB/16 granule addresses computed once, and the entire (R, c) dependence
// moves into ds_read_b128's 16-bit offset field, where it costs no register at
// all.  Checked exhaustively against st::idx over every (rows, cols, R, l, c)
// with SB >= 64 up to 256x256.
#ifndef HK_SYNC_LDS
#define HK_SYNC_LDS 0
#endif

typedef float hk_v4f __attribute__((ext_vector_type(4)));

template<int OFF>
__device__ inline float4 ds_read_b128_off(uint32_t addr) {
    static_assert(OFF >= 0 && OFF < 65536, "ds_read_b128's offset field is 16 bits");
    hk_v4f v;
    asm volatile("ds_read_b128 %0, %1 offset:%2\n"
#if HK_SYNC_LDS
                 "s_waitcnt lgkmcnt(0)\n"
#endif
                 : "=v"(v) : "v"(addr), "i"(OFF) : "memory");
    float4 r;
    __builtin_memcpy(&r, &v, sizeof(v));
    return r;
}

// The granule addresses of one shared tile, for this lane.
template<typename ST> struct lds_granules {
    static_assert(sizeof(typename ST::dtype) == 2, "2-byte tiles only");
    static_assert(ST::swizzle_bytes >= 64,
                  "a 32-byte swizzle puts the key bits inside 16 rows of stride, "
                  "so the row offset is no longer a compile-time addend");
    static constexpr int NG = ST::swizzle_bytes / 16;
    uint32_t g[NG];
    __device__ inline void init(const ST &t) {
        const uint32_t p = (uint32_t)(uintptr_t)&t.data[0];
        const int l16 = (int)(kittens::laneid() & 15);
        #pragma unroll
        for (int x = 0; x < NG; x++) g[x] = ST::idx(p, int2{l16, 8 * x});
    }
    // Move every address to the other LDS buffer. The two buffers differ in one
    // bit by construction (see config::LDS_XOR), so this is NG v_xor and keeps
    // the addresses in the registers they were already in -- re-deriving them
    // would redo the swizzle. With X == 0 it compiles away.
    template<uint32_t X> __device__ inline void swap() {
        if constexpr (X != 0) {
            #pragma unroll
            for (int x = 0; x < NG; x++) g[x] ^= X;
        }
    }
};

// One 16x16 bf16 operand fragment out of a shared tile, at tile coords (R, C).
// Two ds_read_b128, neither of which waits -- the caller batches the waits.
template<int R, int C, typename ST, typename BT>
__device__ inline void lds_read_frag(BT &dst, const lds_granules<ST> &gr) {
    constexpr int SB = ST::swizzle_bytes, SUB = SB / 2;
    static_assert(R % 16 == 0 && C % 16 == 0, "fragment coords are in 16s");
    static_assert(R + 16 <= ST::rows && C + 16 <= ST::cols, "fragment out of tile");
    constexpr int G0 = ((C    ) % SUB) / 8, I0 = ((C    ) / SUB) * ST::rows * SB + SB * R;
    constexpr int G1 = ((C + 8) % SUB) / 8, I1 = ((C + 8) / SUB) * ST::rows * SB + SB * R;
    const float4 v0 = ds_read_b128_off<I0>(gr.g[G0]);
    const float4 v1 = ds_read_b128_off<I1>(gr.g[G1]);
    __builtin_memcpy((void*)&dst.data[0], &v0, sizeof(v0));
    __builtin_memcpy((void*)&dst.data[4], &v1, sizeof(v1));
}

// The same two, taking a whole 1x1 register tile rather than its base. A
// caller that already has the fragment decomposed (the handwritten attention
// kernel indexes `dst.tiles[r][c]`) wants the base overloads above; a caller
// whose unit of value *is* one 16x16 tile (anything hk-dsl generates,
// where every fragment is its own IR value) wants these. The static_assert is
// the whole implementation: a wider tile has more than one base and the
// fragment coordinate would be ambiguous.
template<int R, int C, typename ST, typename T, int RR, int CC, typename L>
__device__ inline void lds_read_frag(rt<T, RR, CC, L> &dst,
                                     const lds_granules<ST> &gr) {
    static_assert(rt<T, RR, CC, L>::height == 1 && rt<T, RR, CC, L>::width == 1,
                  "the tile overload reads one 16x16 fragment; index a wider "
                  "tile's .tiles[r][c] and use the base overload");
    lds_read_frag<R, C>(dst.tiles[0][0], gr);
}

template<int R, int C, typename ST, typename T, int RR, int CC, typename L>
__device__ inline void lds_write_frag(const lds_granules<ST> &gr,
                                      uint32_t row_off,
                                      const rt<T, RR, CC, L> &src) {
    static_assert(rt<T, RR, CC, L>::height == 1 && rt<T, RR, CC, L>::width == 1,
                  "the tile overload writes one 16x16 fragment; index a wider "
                  "tile's .tiles[r][c] and use the base overload");
    lds_write_frag<R, C>(gr, row_off, src.tiles[0][0]);
}

// Every `lds_wait<0>()` below is followed by `lds_bind` on the fragments it
// retires.  This is not decoration: s_waitcnt carries no register dependence,
// so without the bind the consuming WMMA is scheduled *above* the wait and
// reads stale LDS, nondeterministically and with ScratchSize still 0.  The
// mechanism and the fix are written out at `lds_bind` in
// include/rdna3/ops/warp/memory/util/util.cuh; the attention kernel is where
// it was found.

template<int OFF>
__device__ inline void ds_write_b128_off(uint32_t addr, float4 val) {
    static_assert(OFF >= 0 && OFF < 65536, "ds_write_b128's offset field is 16 bits");
    hk_v4f v;
    __builtin_memcpy(&v, &val, sizeof(v));
    asm volatile("ds_write_b128 %0, %1 offset:%2\n"
                 :: "v"(addr), "v"(v), "i"(OFF) : "memory");
}

// The write side of the same decomposition, for the V^T staging.  Here the row
// is only known at runtime (it is the warp's slice of the head dimension), so
// the caller passes it pre-scaled as a byte offset; R is the fragment's row
// *within* that slice.  Any multiple of 16 works, compile-time or not -- the
// decomposition only needs R % 16 == 0.
template<int R, int C, typename ST, typename BT>
__device__ inline void lds_write_frag(const lds_granules<ST> &gr, uint32_t row_off,
                                      const BT &src) {
    constexpr int SB = ST::swizzle_bytes, SUB = SB / 2;
    static_assert(R % 16 == 0 && C % 16 == 0, "fragment coords are in 16s");
    static_assert(C + 16 <= ST::cols, "fragment out of tile");
    constexpr int G0 = ((C    ) % SUB) / 8, I0 = ((C    ) / SUB) * ST::rows * SB + SB * R;
    constexpr int G1 = ((C + 8) % SUB) / 8, I1 = ((C + 8) / SUB) * ST::rows * SB + SB * R;
    float4 v0, v1;
    __builtin_memcpy(&v0, (const void*)&src.data[0], sizeof(v0));
    __builtin_memcpy(&v1, (const void*)&src.data[4], sizeof(v1));
    ds_write_b128_off<I0>(gr.g[G0] + row_off, v0);
    ds_write_b128_off<I1>(gr.g[G1] + row_off, v1);
}

} // namespace kittens
