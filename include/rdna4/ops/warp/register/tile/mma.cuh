/**
 * @file
 * @brief Matrix multiply-accumulate for RDNA4 (gfx1200/gfx1201).
 *
 * Derived from the cdna5 (gfx1250) tree, which is the closest relative: both are
 * wave32 and both use WMMA rather than MFMA. Two differences drive everything.
 *
 * 1. K DEPTH. gfx1250 has `v_wmma_f32_16x16x32_{f16,bf16}`; RDNA4 has only
 *    `v_wmma_f32_16x16x16_*`. So the A/B operand is 8 elements per lane instead
 *    of 16, and a K=32 step becomes two instructions rather than one. The
 *    accumulator is unchanged -- 8 floats per lane either way -- so the
 *    accumulator tile type and every downstream store path carry over as-is.
 *
 * 2. NO MFMA. RDNA4 has no matrix-fused-multiply-add at all, so the mfma*
 *    helpers cdna5 carries alongside its WMMA path are absent here rather than
 *    guarded.
 *
 * There are also no operand-reuse hints: the gfx12 builtin takes (a, b, c) and
 * nothing else, so cdna5's A_reuse/B_reuse template parameters have no analogue.
 * They are accepted and ignored below so kernels written against the cdna5 API
 * port across unchanged.
 *
 * REGISTER LAYOUT (measured, not assumed)
 * ---------------------------------------
 * Derived empirically on gfx1201 by a probe kernel that constructs the accumulator mapping from one-hot products and then validates
 * the whole triple against a CPU reference on random matrices (worst relative
 * error ~3e-7, f16 and bf16):
 *
 *   A[row][k]   -> lane 16*(k/8)   + row, element k%8
 *   B[k][col]   -> lane 16*(k/8)   + col, element k%8
 *   D[row][col] -> lane 16*(row/8) + col, element row%8
 *
 * This is the same structural form cdna5 uses -- lane carries the non-reduction
 * index, element carries k -- with the K split at 8 instead of 16, which is why
 * the shared->register loaders port with only a change of subtile width.
 *
 * A and B share a register form and differ only in which logical axis the lane
 * selects, so mma_AB (B as K x N, col layout) and mma_ABt (B as N x K, row
 * layout) both reduce to one instruction with no data movement.
 *
 * The accumulator is the mirror of A -- row in the element position, column in
 * the lane -- so a WMMA result cannot be fed back as a left operand without a
 * transpose. CDNA has the same property; it is why mma_AtB/mma_AtBt are absent.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

/* ----------  RAW INSTRUCTION WRAPPERS  ---------- */

/**
 * @brief D = A*B + C for one 16x16x16 tile, bf16 in, fp32 accumulate.
 *
 * At wave32 a 16x16 bf16 tile is 8 elements per lane and a 16x16 fp32 tile is
 * 8 floats per lane -- exactly the instruction's operand widths. The
 * rt_base::data arrays are therefore already correctly shaped and only need
 * reinterpreting: no packing, no shuffles.
 *
 * The template parameters exist for source compatibility with the gfx1250
 * operand-reuse hints and are ignored; RDNA4's builtin has no such control.
 */
// The operands are moved through __builtin_memcpy rather than a reinterpret
// cast. The cast form -- `*(bf16x8_t*)A` -- is what the CDNA and cdna5 trees
// use, and it is doubly ill-formed here: it type-puns through incompatible
// pointer types (strict aliasing), and it asserts an alignment the source does
// not have. `bf16_2 data[4]` in rt_base is 4-byte aligned and `float2 data[4]`
// is 8-byte aligned, while bf16x8_t wants 16 and floatx8_t wants 32. Nothing
// in rt_base declares that alignment.
//
// memcpy of a trivially-copyable object is well defined, carries no alignment
// requirement, and lowers to plain register moves at -O3 -- the emitted ISA is
// the same 8 v_wmma instructions with no extra traffic. This is not a
// pessimisation; it is the same code without the undefined behaviour.
template<bool A_reuse = false, bool B_reuse = false>
__device__ static inline void wmma161616(      float2 (&D)[4],
                                         const bf16_2 (&A)[4],
                                         const bf16_2 (&B)[4],
                                         const float2 (&C)[4]) {
    typedef __attribute__((__vector_size__(8 * sizeof(__bf16)))) __bf16 bf16x8_t;
    typedef __attribute__((__vector_size__(8 * sizeof(float)))) float floatx8_t;

    bf16x8_t av, bv;
    floatx8_t cv;
    __builtin_memcpy(&av, &A[0], sizeof(av));
    __builtin_memcpy(&bv, &B[0], sizeof(bv));
    __builtin_memcpy(&cv, &C[0], sizeof(cv));

    const floatx8_t dv =
        __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(av, bv, cv);
    __builtin_memcpy(&D[0], &dv, sizeof(dv));
}

/// @brief fp16 counterpart, identical shape.
template<bool A_reuse = false, bool B_reuse = false>
__device__ static inline void wmma161616(      float2 (&D)[4],
                                         const half_2 (&A)[4],
                                         const half_2 (&B)[4],
                                         const float2 (&C)[4]) {
    typedef __attribute__((__vector_size__(8 * sizeof(_Float16)))) _Float16 fp16x8_t;
    typedef __attribute__((__vector_size__(8 * sizeof(float)))) float floatx8_t;

    fp16x8_t av, bv;
    floatx8_t cv;
    __builtin_memcpy(&av, &A[0], sizeof(av));
    __builtin_memcpy(&bv, &B[0], sizeof(bv));
    __builtin_memcpy(&cv, &C[0], sizeof(cv));

    const floatx8_t dv =
        __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(av, bv, cv);
    __builtin_memcpy(&D[0], &dv, sizeof(dv));
}

/**
 * @brief Accumulate-in-place form: DC = A*B + DC.
 *
 * THIS EXISTS FOR A CORRECTNESS REASON, not for convenience.
 *
 * The four-argument form above takes the accumulator twice -- once as
 * `float2 (&D)[4]` and once as `const float2 (&C)[4]`. Every accumulating GEMM
 * calls it that way, binding both references to the same storage. That is a
 * const/non-const aliasing lie: the compiler is told C is immutable while D,
 * the same object, is written. It is entitled to reorder the read of C against
 * the write of D, and at -O3 it does.
 *
 * Measured on gfx1201, single WMMA, operands independently proven sane and the
 * accumulator explicitly zeroed:
 *
 *     D and C aliased    -> max_abs_err 4.3e16   (garbage)
 *     D and C distinct   -> max_abs_err 9.96     (bounded, correct magnitude)
 *
 * Routing the memcpy through locals does *not* fix it -- the reordering happens
 * at the memory operations on C and D themselves, not inside the marshalling.
 *
 * So express the in-place case honestly: one reference, mutated. No const lie,
 * nothing for the optimizer to exploit. The same latent defect exists in the
 * CDNA and cdna5 trees, where `mma_AB`/`mma_ABt` pass `d.tiles[n][m]` as both
 * destination and accumulator inside the K loop.
 */
template<bool A_reuse = false, bool B_reuse = false>
__device__ static inline void wmma161616_acc(      float2 (&DC)[4],
                                             const bf16_2 (&A)[4],
                                             const bf16_2 (&B)[4]) {
    typedef __attribute__((__vector_size__(8 * sizeof(__bf16)))) __bf16 bf16x8_t;
    typedef __attribute__((__vector_size__(8 * sizeof(float)))) float floatx8_t;

    bf16x8_t av, bv;
    floatx8_t cv;
    __builtin_memcpy(&av, &A[0],  sizeof(av));
#if defined(WMMA_BANK_PAD) && WMMA_BANK_PAD > 0
    // DIAGNOSTIC. v_wmma_f32_16x16x16_bf16 wants (srcB - srcA) == 2 (mod 4);
    // anything else costs 1.1-1.7 cyc_real (~9%, measured). Disassembly of the ws winner shows 87.5% of our
    // WMMAs at distance 0 (congruent bases, the worst case). This tries to nudge
    // the allocator by holding WMMA_BANK_PAD VGPRs live between the two operand
    // materialisations. Verify by counting the ISA distribution, NOT by timing.
    float bank_pad[WMMA_BANK_PAD];
    #pragma unroll
    for (int i = 0; i < WMMA_BANK_PAD; ++i) asm volatile("" : "=v"(bank_pad[i]));
#endif
    __builtin_memcpy(&bv, &B[0],  sizeof(bv));
    __builtin_memcpy(&cv, &DC[0], sizeof(cv));
#if defined(WMMA_BANK_PAD) && WMMA_BANK_PAD > 0
    #pragma unroll
    for (int i = 0; i < WMMA_BANK_PAD; ++i) asm volatile("" :: "v"(bank_pad[i]));
#endif

    const floatx8_t dv =
        __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(av, bv, cv);
    __builtin_memcpy(&DC[0], &dv, sizeof(dv));
}

/**
 * @brief int8 form: DC = A*B + DC, int8 in, int32 accumulate.
 *
 * `v_wmma_i32_16x16x16_iu8` is the SAME 16x16x16 shape as the bf16 op but
 * measures 1.99x its rate at burst (409.0 vs 205.3 TOPS on gfx1201), because it
 * issues in half the cycles while reading half the operand bytes. srcA/srcB are
 * 2 VGPRs, not 4 -- the assembler rejects the 4-VGPR form outright.
 *
 * THE SIGNEDNESS ARGUMENTS ARE NOT OPTIONAL. The builtin's first and third
 * parameters select signed operands, lowering to `neg_lo:[1,1,0]`. Passing
 * false reads the operands as u8, and a signed input set then comes out around
 * 127.5^2 * K -- a plausible-looking large positive number, not a NaN or a
 * crash. **If an int8 result is large, positive, and scales as 127.5^2*K, the
 * operands are being read unsigned.** That cost an hour in the assembly kernel.
 *
 * The last argument is `clamp`; false gives wrapping i32 accumulation, which is
 * what a GEMM wants -- with |a|,|b| <= 127 the sum over K stays inside i32 for
 * any K a GPU will see (127^2 * 8 * K/16 ... K = 2^31/8064 ~ 266k).
 */
template<bool A_reuse = false, bool B_reuse = false>
__device__ static inline void wmma161616_acc(      int2    (&DC)[4],
                                             const int8_4 (&A)[2],
                                             const int8_4 (&B)[2]) {
    typedef __attribute__((__vector_size__(2 * sizeof(int)))) int intx2_t;
    typedef __attribute__((__vector_size__(8 * sizeof(int)))) int intx8_t;

    intx2_t av, bv;
    intx8_t cv;
    __builtin_memcpy(&av, &A[0],  sizeof(av));
    __builtin_memcpy(&bv, &B[0],  sizeof(bv));
    __builtin_memcpy(&cv, &DC[0], sizeof(cv));

    const intx8_t dv = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
        /*signA=*/true, av, /*signB=*/true, bv, cv, /*clamp=*/false);
    __builtin_memcpy(&DC[0], &dv, sizeof(dv));
}

/// @brief Four-argument int8 form, for symmetry with the float paths.
template<bool A_reuse = false, bool B_reuse = false>
__device__ static inline void wmma161616(      int2    (&D)[4],
                                         const int8_4 (&A)[2],
                                         const int8_4 (&B)[2],
                                         const int2   (&C)[4]) {
    typedef __attribute__((__vector_size__(2 * sizeof(int)))) int intx2_t;
    typedef __attribute__((__vector_size__(8 * sizeof(int)))) int intx8_t;

    intx2_t av, bv;
    intx8_t cv;
    __builtin_memcpy(&av, &A[0], sizeof(av));
    __builtin_memcpy(&bv, &B[0], sizeof(bv));
    __builtin_memcpy(&cv, &C[0], sizeof(cv));

    const intx8_t dv = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
        true, av, true, bv, cv, false);
    __builtin_memcpy(&D[0], &dv, sizeof(dv));
}

/// @brief fp16 counterpart of the accumulate-in-place form.
template<bool A_reuse = false, bool B_reuse = false>
__device__ static inline void wmma161616_acc(      float2 (&DC)[4],
                                             const half_2 (&A)[4],
                                             const half_2 (&B)[4]) {
    typedef __attribute__((__vector_size__(8 * sizeof(_Float16)))) _Float16 fp16x8_t;
    typedef __attribute__((__vector_size__(8 * sizeof(float)))) float floatx8_t;

    fp16x8_t av, bv;
    floatx8_t cv;
    __builtin_memcpy(&av, &A[0],  sizeof(av));
    __builtin_memcpy(&bv, &B[0],  sizeof(bv));
    __builtin_memcpy(&cv, &DC[0], sizeof(cv));

    const floatx8_t dv =
        __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(av, bv, cv);
    __builtin_memcpy(&DC[0], &dv, sizeof(dv));
}

/* ----------  BASE TILE OPERATIONS  ---------- */

/**
 * @brief One 16x16x16 tile: D(col) = A(row) * B(col) + C(col).
 */
template<bool A_reuse = false, bool B_reuse = false, typename T,
         ducks::rt_shape::all D_shape, ducks::rt_shape::all A_shape,
         ducks::rt_shape::all B_shape, ducks::rt_shape::all C_shape>
__device__ static inline void mma_AB_base(rt_base<float, ducks::rt_layout::col, D_shape> &d,
                                          const rt_base<T, ducks::rt_layout::row, A_shape> &a,
                                          const rt_base<T, ducks::rt_layout::col, B_shape> &b,
                                          const rt_base<float, ducks::rt_layout::col, C_shape> &c) {
    static_assert(std::is_same_v<D_shape, C_shape>, "D and C must have the same shape");
    static_assert(std::is_same_v<D_shape, typename ducks::rt_shape::rt_16x16> &&
                  std::is_same_v<A_shape, typename ducks::rt_shape::rt_16x16> &&
                  std::is_same_v<B_shape, typename ducks::rt_shape::rt_16x16>,
                  "RDNA4 WMMA is 16x16x16 only -- no 32x32 as on CDNA, no 16x16x32 as on "
                  "gfx1250. Compose larger tiles from 16x16.");
    wmma161616<A_reuse, B_reuse>(d.data, a.data, b.data, c.data);
}

/**
 * @brief One 16x16x16 tile with B transposed: D(col) = A(row) * B(row)^T + C(col).
 *
 * B arrives as N x K in row layout: its lane index already selects the output
 * column and its element index already carries k, bit-identical to the
 * col-layout K x N case. Same instruction, no transpose.
 */
template<bool A_reuse = false, bool B_reuse = false, typename T,
         ducks::rt_shape::all D_shape, ducks::rt_shape::all A_shape,
         ducks::rt_shape::all B_shape, ducks::rt_shape::all C_shape>
__device__ static inline void mma_ABt_base(rt_base<float, ducks::rt_layout::col, D_shape> &d,
                                           const rt_base<T, ducks::rt_layout::row, A_shape> &a,
                                           const rt_base<T, ducks::rt_layout::row, B_shape> &b,
                                           const rt_base<float, ducks::rt_layout::col, C_shape> &c) {
    static_assert(std::is_same_v<D_shape, C_shape>, "D and C must have the same shape");
    static_assert(std::is_same_v<D_shape, typename ducks::rt_shape::rt_16x16> &&
                  std::is_same_v<A_shape, typename ducks::rt_shape::rt_16x16> &&
                  std::is_same_v<B_shape, typename ducks::rt_shape::rt_16x16>,
                  "RDNA4 WMMA is 16x16x16 only.");
    wmma161616<A_reuse, B_reuse>(d.data, a.data, b.data, c.data);
}

/* ----------  TILE-ARRAY OPERATIONS  ---------- */

/**
 * @brief D = A * B + C over an array of 16x16 tiles.
 *
 * The K reduction accumulates in place, so the accumulator stays register
 * resident across the whole reduction.
 */
template<ducks::rt::col_layout D, ducks::rt::row_layout A, ducks::rt::col_layout B, ducks::rt::col_layout C>
__device__ static inline void mma_AB(D &d, const A &a, const B &b, const C &c) {
    static_assert(D::rows == A::rows && D::cols == B::cols);
    static_assert(A::cols == B::rows);
    static_assert(D::rows == C::rows && D::cols == C::cols);
    static_assert(
        std::is_same_v<typename D::T, float> && std::is_same_v<typename C::T, float> &&
        std::is_same_v<typename A::T, typename B::T> &&
        (std::is_same_v<typename A::T, bf16> || std::is_same_v<typename A::T, half>),
        "RDNA4 WMMA: A and B must both be bf16 or both half; accumulator must be float.");
    static_assert(
        std::is_same_v<typename D::shape, ducks::rt_shape::rt_16x16> &&
        std::is_same_v<typename A::shape, ducks::rt_shape::rt_16x16> &&
        std::is_same_v<typename B::shape, ducks::rt_shape::rt_16x16> &&
        std::is_same_v<typename C::shape, ducks::rt_shape::rt_16x16>,
        "RDNA4 WMMA is 16x16x16 only: D, A, B and C must all use rt_16x16 base tiles.");

    #pragma unroll
    for(int n = 0; n < D::height; n++) {
        #pragma unroll
        for(int m = 0; m < D::width; m++) {
            // Seed the destination from C, then accumulate in place. Never
            // pass the same tile as both destination and const accumulator --
            // see the note on wmma161616_acc. Self-assignment when d and c are
            // the same object (the usual case) is a harmless POD copy.
            // Accumulate in a LOCAL, then write out. d and c are the same
            // object in every accumulating call (mma_ABt(C,A,B,C)), so touching
            // d before c has been fully read is a read/write hazard on aliased
            // storage. Staging through a local reads c completely first and
            // writes d exactly once. Measured: direct in-place accumulation
            // into d.tiles[n][m] gives inf; this form gives 0.06 at K=64,
            // which is bf16 rounding.
            float2 acc[4];
            #pragma unroll
            for(int i = 0; i < 4; i++) acc[i] = c.tiles[n][m].data[i];
            #pragma unroll
            for(int k = 0; k < A::width; k++) {
                wmma161616_acc(acc, a.tiles[n][k].data, b.tiles[k][m].data);
            }
            #pragma unroll
            for(int i = 0; i < 4; i++) d.tiles[n][m].data[i] = acc[i];
        }
    }
}

/**
 * @brief D = A^T * B + C over an array of 16x16 tiles, A given as K x M (col layout).
 *
 * The transposed-A variant. Attention needs it: the flash-attention formulation
 * this tree ports from computes S^T and O^T throughout (which is what makes its
 * softmax reductions column-wise, hence `col_max`/`col_sum`), so both of its
 * GEMMs are A^T * B. CDNA4 has it; RDNA4 did not until this.
 *
 * Identical to `mma_AB` except for how `a` is indexed. With A in COLUMN layout
 * the (n, k) fragment lives at `a.tiles[k][n]`, not `a.tiles[n][k]` -- that
 * transposition IS the operation. Everything else, including the accumulate-in-
 * a-local hazard workaround, carries over unchanged.
 *
 * > A wrong index here compiles, runs, and returns plausible but wrong numbers.
 * > Verified against a CPU reference before use.
 */
template<ducks::rt::col_layout D, ducks::rt::col_layout A, ducks::rt::col_layout B, ducks::rt::col_layout C>
__device__ static inline void mma_AtB(D &d, const A &a, const B &b, const C &c) {
    static_assert(D::rows == A::cols && D::cols == B::cols);
    static_assert(A::rows == B::rows);
    static_assert(D::rows == C::rows && D::cols == C::cols);
    static_assert(
        std::is_same_v<typename D::T, float> && std::is_same_v<typename C::T, float> &&
        std::is_same_v<typename A::T, typename B::T> &&
        (std::is_same_v<typename A::T, bf16> || std::is_same_v<typename A::T, half>),
        "RDNA4 WMMA: A and B must both be bf16 or both half; accumulator must be float.");
    static_assert(
        std::is_same_v<typename D::shape, ducks::rt_shape::rt_16x16> &&
        std::is_same_v<typename A::shape, ducks::rt_shape::rt_16x16> &&
        std::is_same_v<typename B::shape, ducks::rt_shape::rt_16x16> &&
        std::is_same_v<typename C::shape, ducks::rt_shape::rt_16x16>,
        "RDNA4 WMMA is 16x16x16 only: D, A, B and C must all use rt_16x16 base tiles.");

    #pragma unroll
    for(int n = 0; n < D::height; n++) {
        #pragma unroll
        for(int m = 0; m < D::width; m++) {
            float2 acc[4];
            #pragma unroll
            for(int i = 0; i < 4; i++) acc[i] = c.tiles[n][m].data[i];
            #pragma unroll
            for(int k = 0; k < A::height; k++) {
                wmma161616_acc(acc, a.tiles[k][n].data, b.tiles[k][m].data);
            }
            #pragma unroll
            for(int i = 0; i < 4; i++) d.tiles[n][m].data[i] = acc[i];
        }
    }
}

/**
 * @brief D = A * B^T + C over an array of 16x16 tiles, B given as N x K.
 *
 * Traversal is n-outer / m-inner so A's subtile is invariant across the inner
 * loop. On gfx1250 that is what the operand-reuse hints exploit; RDNA4 has no
 * such hint, but the ordering still helps the scheduler keep A's registers live
 * and keeps the two trees' loop structure identical.
 */
template<ducks::rt::col_layout D, ducks::rt::row_layout A, ducks::rt::row_layout B, ducks::rt::col_layout C>
__device__ static inline void mma_ABt(D &d, const A &a, const B &b, const C &c) {
    static_assert(D::rows == A::rows && D::cols == B::rows);
    static_assert(A::cols == B::cols);
    static_assert(D::rows == C::rows && D::cols == C::cols);
    static_assert(
        std::is_same_v<typename D::T, float> && std::is_same_v<typename C::T, float> &&
        std::is_same_v<typename A::T, typename B::T> &&
        (std::is_same_v<typename A::T, bf16> || std::is_same_v<typename A::T, half>),
        "RDNA4 WMMA: A and B must both be bf16 or both half; accumulator must be float.");
    static_assert(
        std::is_same_v<typename D::shape, ducks::rt_shape::rt_16x16> &&
        std::is_same_v<typename A::shape, ducks::rt_shape::rt_16x16> &&
        std::is_same_v<typename B::shape, ducks::rt_shape::rt_16x16> &&
        std::is_same_v<typename C::shape, ducks::rt_shape::rt_16x16>,
        "RDNA4 WMMA is 16x16x16 only: D, A, B and C must all use rt_16x16 base tiles.");

    #pragma unroll
    for(int n = 0; n < D::height; n++) {
        #pragma unroll
        for(int m = 0; m < D::width; m++) {
            // See mma_AB: seed from C, then accumulate in place.
            // Accumulate in a LOCAL, then write out. d and c are the same
            // object in every accumulating call (mma_ABt(C,A,B,C)), so touching
            // d before c has been fully read is a read/write hazard on aliased
            // storage. Staging through a local reads c completely first and
            // writes d exactly once. Measured: direct in-place accumulation
            // into d.tiles[n][m] gives inf; this form gives 0.06 at K=64,
            // which is bf16 rounding.
            float2 acc[4];
            #pragma unroll
            for(int i = 0; i < 4; i++) acc[i] = c.tiles[n][m].data[i];
            #pragma unroll
            for(int k = 0; k < A::width; k++) {
                wmma161616_acc(acc, a.tiles[n][k].data, b.tiles[m][k].data);
            }
            #pragma unroll
            for(int i = 0; i < 4; i++) d.tiles[n][m].data[i] = acc[i];
        }
    }
}

/*
 * mma_AtB / mma_AtBt are intentionally absent on RDNA4.
 *
 * They need the left operand with the lane index selecting k rather than the
 * row, which on this layout is a real cross-lane transpose. Providing them as
 * silent shuffle sequences would hand callers a working kernel with no signal
 * that it had left the fast path. Transpose explicitly at the tile level.
 */

} // namespace kittens
