/**
 * @file
 * @brief Matrix multiply-accumulate operations for tiles stored in registers.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

/*
 * gfx11 WMMA.  One shape only: 16x16x16, wave32, f32 accumulate.
 *
 * Operands are v16f16 / v16i16 -- a full K=16 vector per lane -- and lanes 0-15
 * must mirror lanes 16-31.  rt_base<bf16|half, *> is sized for exactly that
 * (replication 2, 8 VGPRs), so the fragments are a straight reinterpret of
 * `.data`, same as the MFMA path on CDNA.  The accumulator is v8f32, which
 * rt_base<float, col> holds as float2[4].
 *
 * The mirroring is not enforced here.  It is an invariant of every RDNA3 tile
 * of operand type: shared_to_register loads it that way (lane l and l^16 read
 * the same address), elementwise maps preserve it because they are pointwise,
 * and the layout conversions in conversions.cuh restore it explicitly.  Feeding
 * WMMA halves that disagree is undefined on the hardware, not merely wrong.
 */

typedef __attribute__((__vector_size__(16 * sizeof(__fp16)))) __fp16 wmma_half16_t;
typedef __attribute__((__vector_size__(16 * sizeof(short))))  short  wmma_short16_t;
typedef __attribute__((__vector_size__(8  * sizeof(float))))  float  wmma_float8_t;

__device__ static inline void wmma161616(float2 (&D)[4],
                                         const half_2 (&A)[8],
                                         const half_2 (&B)[8],
                                         const float2 (&C)[4]) {
    *(wmma_float8_t*)D = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
        *(wmma_half16_t*)A,
        *(wmma_half16_t*)B,
        *(wmma_float8_t*)C
    );
}

__device__ static inline void wmma161616(float2 (&D)[4],
                                         const bf16_2 (&A)[8],
                                         const bf16_2 (&B)[8],
                                         const float2 (&C)[4]) {
    *(wmma_float8_t*)D = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
        *(wmma_short16_t*)A,
        *(wmma_short16_t*)B,
        *(wmma_float8_t*)C
    );
}

/**
 * @brief Base matrix multiply-accumulate operation for row layout.
 *
 * @param[out] d The output rt_base<float, col_layout> accumulator.
 * @param[in] a The first input rt_base<bf16, row_layout> matrix.
 * @param[in] b The second input rt_base<bf16, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_base<float, col_layout> accumulator matrix.
 */
__device__ static inline void mma_AB_base(rt_base<float, ducks::rt_layout::col> &d,
                                    const rt_base<half, ducks::rt_layout::row> &a,
                                    const rt_base<half, ducks::rt_layout::col> &b, // in col-major mode
                                    const rt_base<float, ducks::rt_layout::col> &c) {
    wmma161616(d.data, a.data, b.data, c.data);
}
__device__ static inline void mma_AB_base(rt_base<float, ducks::rt_layout::col> &d,
                                    const rt_base<bf16, ducks::rt_layout::row> &a,
                                    const rt_base<bf16, ducks::rt_layout::col> &b, // in col-major mode
                                    const rt_base<float, ducks::rt_layout::col> &c) {
    wmma161616(d.data, a.data, b.data, c.data);
}

/**
 * @brief Base dot product operation for row layout.
 *
 * A row-layout operand and a col-layout operand have bit-identical register
 * contents on gfx11 -- lane l holds index l%16 of whichever axis the layout
 * names, and elements run along k either way -- so transposing an operand is
 * purely a relabelling and every variant below lowers to the same instruction.
 */
__device__ static inline void mma_ABt_base(rt_base<float, ducks::rt_layout::col> &d,
                                     const rt_base<half, ducks::rt_layout::row> &a,
                                     const rt_base<half, ducks::rt_layout::row> &b, // in row-major mode
                                     const rt_base<float, ducks::rt_layout::col> &c) {
    wmma161616(d.data, a.data, b.data, c.data);
}
__device__ static inline void mma_ABt_base(rt_base<float, ducks::rt_layout::col> &d,
                                     const rt_base<bf16, ducks::rt_layout::row> &a,
                                     const rt_base<bf16, ducks::rt_layout::row> &b, // in row-major mode
                                     const rt_base<float, ducks::rt_layout::col> &c) {
    wmma161616(d.data, a.data, b.data, c.data);
}
/**
 * @brief Base matrix multiply-accumulate operation for row layout with transposed A.
 */
__device__ static inline void mma_AtB_base(rt_base<float, ducks::rt_layout::col> &d,
                                     const rt_base<half, ducks::rt_layout::col> &a,
                                     const rt_base<half, ducks::rt_layout::col> &b, // in col-major mode
                                     const rt_base<float, ducks::rt_layout::col> &c) {
    wmma161616(d.data, a.data, b.data, c.data);
}
__device__ static inline void mma_AtB_base(rt_base<float, ducks::rt_layout::col> &d,
                                     const rt_base<bf16, ducks::rt_layout::col> &a,
                                     const rt_base<bf16, ducks::rt_layout::col> &b, // in col-major mode
                                     const rt_base<float, ducks::rt_layout::col> &c) {
    wmma161616(d.data, a.data, b.data, c.data);
}
/**
 * @brief Base matrix multiply-accumulate operation for row layout with transposed A and B.
 */
__device__ static inline void mma_AtBt_base(rt_base<float, ducks::rt_layout::col> &d,
                                      const rt_base<half, ducks::rt_layout::col> &a,
                                      const rt_base<half, ducks::rt_layout::row> &b, // in col-major mode
                                      const rt_base<float, ducks::rt_layout::col> &c) {
    wmma161616(d.data, a.data, b.data, c.data);
}
__device__ static inline void mma_AtBt_base(rt_base<float, ducks::rt_layout::col> &d,
                                      const rt_base<bf16, ducks::rt_layout::col> &a,
                                      const rt_base<bf16, ducks::rt_layout::row> &b, // in col-major mode
                                      const rt_base<float, ducks::rt_layout::col> &c) {
    wmma161616(d.data, a.data, b.data, c.data);
}

/// The type combinations gfx11 WMMA can actually take. Unlike CDNA there is no
/// fp8 variant, and the f16-accumulate form is not wired up here.
template<typename D, typename A, typename B, typename C>
constexpr bool mma_types_ok =
    std::is_same_v<typename D::T, float> && std::is_same_v<typename C::T, float> &&
    std::is_same_v<typename A::T, typename B::T> &&
    (std::is_same_v<typename A::T, bf16> || std::is_same_v<typename A::T, half>);

/**
 * @brief Matrix multiply-accumulate operation.
 *
 * @tparam N The number of row tiles.
 * @tparam K The number of column tiles for the A matrix and row tiles for the B matrix.
 * @tparam M The number of column tiles for the B matrix.
 * @param[out] d The output rt_fl<N, M, col_layout> accumulator.
 * @param[in] a The first input rt_bf<N, K, row_layout> matrix.
 * @param[in] b The second input rt_bf<K, M, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_fl<N, M, col_layout> accumulator matrix.
 */
template<ducks::rt::col_layout D, ducks::rt::row_layout A, ducks::rt::col_layout B, ducks::rt::col_layout C>
__device__ static inline void mma_AB(D &d,
                               const A &a,
                               const B &b,
                               const C &c) {
    static_assert(D::rows == A::rows && D::cols == B::cols); // Check D matches A, B
    static_assert(A::cols == B::rows); // Check reduction dim is same
    static_assert(D::rows == C::rows && D::cols == C::cols); // Check D matches C
    static_assert(mma_types_ok<D, A, B, C>, "unsupported type combination for gfx11 WMMA");

    #pragma unroll
    for(int n = 0; n < D::height; n++) {
        #pragma unroll
        for(int m = 0; m < D::width; m++) {
            mma_AB_base(
                d.tiles[n][m],
                a.tiles[n][0],
                b.tiles[0][m],
                c.tiles[n][m]
            );
            #pragma unroll
            for(int k = 1; k < A::width; k++) {
                mma_AB_base(
                    d.tiles[n][m],
                    a.tiles[n][k],
                    b.tiles[k][m],
                    d.tiles[n][m]
                );
            }
        }
    }
}

/**
 * @brief Dot product operation for row layout.
 *
 * @param[out] d The output rt_fl<N, M, col_layout> accumulator.
 * @param[in] a The first input rt_bf<N, K, row_layout> matrix.
 * @param[in] b The second input rt_bf<M, K, row_layout> matrix in row-major mode.
 * @param[in] c The input rt_fl<N, M, col_layout> accumulator matrix.
 */
template<ducks::rt::col_layout D, ducks::rt::row_layout A, ducks::rt::row_layout B, ducks::rt::col_layout C>
__device__ static inline void mma_ABt(D &d,
                                const A &a,
                                const B &b, // notice row and (M, K) instead of col and (K, M)
                                const C &c) {
    static_assert(D::rows == A::rows && D::cols == B::rows); // Check D matches A, B
    static_assert(A::cols == B::cols); // Check reduction dim is same
    static_assert(D::rows == C::rows && D::cols == C::cols); // Check D matches C
    static_assert(mma_types_ok<D, A, B, C>, "unsupported type combination for gfx11 WMMA");

    #pragma unroll
    for(int n = 0; n < D::height; n++) {
        #pragma unroll
        for(int m = 0; m < D::width; m++) {
            mma_ABt_base(
                d.tiles[n][m],
                a.tiles[n][0],
                b.tiles[m][0],
                c.tiles[n][m]
            );
            #pragma unroll
            for(int k = 1; k < A::width; k++) {
                mma_ABt_base(
                    d.tiles[n][m],
                    a.tiles[n][k],
                    b.tiles[m][k],
                    d.tiles[n][m]
                );
            }
        }
    }
}
/**
 * @brief Matrix multiply-accumulate operation with transposed A.
 *
 * @param[out] d The output rt_fl<N, M, col_layout> accumulator.
 * @param[in] a The first input rt_bf<K, N, col_layout> matrix.
 * @param[in] b The second input rt_bf<K, M, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_fl<N, M, col_layout> accumulator matrix.
 */
template<ducks::rt::col_layout D, ducks::rt::col_layout A, ducks::rt::col_layout B, ducks::rt::col_layout C>
__device__ static inline void mma_AtB(D &d,
                                const A &a,
                                const B &b,
                                const C &c) {
    static_assert(D::rows == A::cols && D::cols == B::cols); // Check D matches A, B
    static_assert(A::rows == B::rows); // Check reduction dim is same
    static_assert(D::rows == C::rows && D::cols == C::cols); // Check D matches C
    static_assert(mma_types_ok<D, A, B, C>, "unsupported type combination for gfx11 WMMA");

    #pragma unroll
    for(int n = 0; n < D::height; n++) {
        #pragma unroll
        for(int m = 0; m < D::width; m++) {
            mma_AtB_base(
                d.tiles[n][m],
                a.tiles[0][n],
                b.tiles[0][m],
                c.tiles[n][m]
            );
            #pragma unroll
            for(int k = 1; k < A::height; k++) {
                mma_AtB_base(
                    d.tiles[n][m],
                    a.tiles[k][n],
                    b.tiles[k][m],
                    d.tiles[n][m]
                );
            }
        }
    }
}

/**
 * @brief Matrix multiply-accumulate operation with transposed A and B.
 *
 * @param[out] d The output rt_fl<N, M, col_layout> accumulator.
 * @param[in] a The first input rt_bf<K, N, col_layout> matrix.
 * @param[in] b The second input rt_bf<M, K, row_layout> matrix in column-major mode.
 * @param[in] c The input rt_fl<N, M, col_layout> accumulator matrix.
 */
template<ducks::rt::col_layout D, ducks::rt::col_layout A, ducks::rt::row_layout B, ducks::rt::col_layout C>
__device__ static inline void mma_AtBt(D &d,
                                 const A &a,
                                 const B &b,
                                 const C &c) {
    static_assert(D::rows == A::cols && D::cols == B::rows); // Check D matches A, B
    static_assert(A::rows == B::cols); // Check reduction dim is same
    static_assert(D::rows == C::rows && D::cols == C::cols); // Check D matches C
    static_assert(mma_types_ok<D, A, B, C>, "unsupported type combination for gfx11 WMMA");

    #pragma unroll
    for(int n = 0; n < D::height; n++) {
        #pragma unroll
        for(int m = 0; m < D::width; m++) {
            mma_AtBt_base(
                d.tiles[n][m],
                a.tiles[0][n],
                b.tiles[m][0],
                c.tiles[n][m]
            );
            #pragma unroll
            for(int k = 1; k < A::height; k++) {
                mma_AtBt_base(
                    d.tiles[n][m],
                    a.tiles[k][n],
                    b.tiles[m][k],
                    d.tiles[n][m]
                );
            }
        }
    }
}
}
