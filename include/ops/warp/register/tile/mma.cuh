/**
 * @file
 * @brief Matrix multiply-accumulate operations for tiles stored in registers.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {


#ifdef KITTENS_CDNA4
__device__ static inline void mfma323216(      float2 (&D)[8],
                                         const bf16_2 (&A)[4],
                                         const bf16_2 (&B)[4],
                                         const float2 (&C)[8]) {
    // Cast to the correct vector types that the intrinsic expects
    typedef __attribute__((__vector_size__(8 * sizeof(__bf16)))) __bf16 bf16x8_t;
    typedef __attribute__((__vector_size__(16 * sizeof(float)))) float floatx16_t;
    
    *(floatx16_t*)D = __builtin_amdgcn_mfma_f32_32x32x16_bf16(
        *(bf16x8_t*)A,
        *(bf16x8_t*)B,
        *(floatx16_t*)C,
        0, 0, 0
    );
}

__device__ static inline void mfma323216(      float2 (&D)[8],
                                         const half_2 (&A)[4],
                                         const half_2 (&B)[4],
                                         const float2 (&C)[8]) {
    // Cast to the correct vector types that the intrinsic expects
    typedef __attribute__((__vector_size__(8 * sizeof(__fp16)))) __fp16 fp16x8_t;
    typedef __attribute__((__vector_size__(16 * sizeof(float)))) float floatx16_t;
    
    *(floatx16_t*)D = __builtin_amdgcn_mfma_f32_32x32x16_f16(
        *(fp16x8_t*)A,
        *(fp16x8_t*)B,
        *(floatx16_t*)C,
        0, 0, 0
    );
}
#else
__device__ static inline void mfma161616(      float2 (&D)[2],
                                         const half_2 (&A)[2],
                                         const half_2 (&B)[2],
                                         const float2 (&C)[2]) {
    (*(float4*)D).data = {__builtin_amdgcn_mfma_f32_16x16x16f16(
        (*(short4*)A).data,
        (*(short4*)B).data,
        (*(float4*)C).data,
        0, 0, 0
    )};
}

__device__ static inline void mfma161616(      float2 (&D)[2],
                                         const bf16_2 (&A)[2],
                                         const bf16_2 (&B)[2],
                                         const float2 (&C)[2]) {
    (*(float4*)D).data = {__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
        (*(short4*)A).data,
        (*(short4*)B).data,
        (*(float4*)C).data,
        0, 0, 0
    )};
}
#endif


/**
 * @brief Base matrix multiply-accumulate operation for row layout.
 *
 * This function performs the base matrix multiply-accumulate operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<bf16_2, row_layout> matrix.
 * @param[in] b The second input rt_base<bf16_2, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
#ifdef KITTENS_CDNA4

#else
__device__ static inline void mma_AB_base(rt_base<float, ducks::rt_layout::col> &d,
                                    const rt_base<half, ducks::rt_layout::row> &a,
                                    const rt_base<half, ducks::rt_layout::col> &b, // in col-major mode
                                    const rt_base<float, ducks::rt_layout::col> &c) {
    mfma161616(d.data, a.data, b.data, c.data);
}
__device__ static inline void mma_AB_base(rt_base<float, ducks::rt_layout::col> &d,
                                    const rt_base<bf16, ducks::rt_layout::row> &a,
                                    const rt_base<bf16, ducks::rt_layout::col> &b, // in col-major mode
                                    const rt_base<float, ducks::rt_layout::col> &c) {
    mfma161616(d.data, a.data, b.data, c.data);
}
#endif
/**
 * @brief Base dot product operation for row layout.
 *
 * This function performs the base dot product operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<bf16_2, row_layout> matrix.
 * @param[in] b The second input rt_base<bf16_2, row_layout> matrix in row-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
#ifdef KITTENS_CDNA4
__device__ static inline void mma_ABt_base(rt_base<float, ducks::rt_layout::accumulator> &d,
                                     const rt_base<half, ducks::rt_layout::row> &a,
                                     const rt_base<half, ducks::rt_layout::row> &b, // in row-major mode
                                     const rt_base<float, ducks::rt_layout::accumulator> &c) {
    mfma323216(d.data, a.data, b.data, c.data);
}
__device__ static inline void mma_ABt_base(rt_base<float, ducks::rt_layout::accumulator> &d,
                                     const rt_base<bf16, ducks::rt_layout::row> &a,
                                     const rt_base<bf16, ducks::rt_layout::row> &b, // in row-major mode
                                     const rt_base<float, ducks::rt_layout::accumulator> &c) {
    mfma323216(d.data, a.data, b.data, c.data);
}
#else
__device__ static inline void mma_ABt_base(rt_base<float, ducks::rt_layout::col> &d,
                                     const rt_base<half, ducks::rt_layout::row> &a,
                                     const rt_base<half, ducks::rt_layout::row> &b, // in row-major mode
                                     const rt_base<float, ducks::rt_layout::col> &c) {
    mfma161616(d.data, a.data, b.data, c.data);
}
__device__ static inline void mma_ABt_base(rt_base<float, ducks::rt_layout::col> &d,
                                     const rt_base<bf16, ducks::rt_layout::row> &a,
                                     const rt_base<bf16, ducks::rt_layout::row> &b, // in row-major mode
                                     const rt_base<float, ducks::rt_layout::col> &c) {
    mfma161616(d.data, a.data, b.data, c.data);
}
#endif
/**
 * @brief Base matrix multiply-accumulate operation for row layout with transposed A.
 *
 * This function performs the base matrix multiply-accumulate operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<bf16_2, col_layout> matrix.
 * @param[in] b The second input rt_base<bf16_2, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
#ifdef KITTENS_CDNA4

#else
__device__ static inline void mma_AtB_base(rt_base<float, ducks::rt_layout::col> &d,
                                     const rt_base<half, ducks::rt_layout::col> &a,
                                     const rt_base<half, ducks::rt_layout::col> &b, // in col-major mode
                                     const rt_base<float, ducks::rt_layout::col> &c) {
    mfma161616(d.data, a.data, b.data, c.data);
}
__device__ static inline void mma_AtB_base(rt_base<float, ducks::rt_layout::col> &d,
                                     const rt_base<bf16, ducks::rt_layout::col> &a,
                                     const rt_base<bf16, ducks::rt_layout::col> &b, // in col-major mode
                                     const rt_base<float, ducks::rt_layout::col> &c) {
    mfma161616(d.data, a.data, b.data, c.data);
}
#endif
/**
 * @brief Base matrix multiply-accumulate operation for row layout with transposed A and B.
 *
 * This function performs the base matrix multiply-accumulate operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<bf16_2, col_layout> matrix.
 * @param[in] b The second input rt_base<bf16_2, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
#ifdef KITTENS_CDNA4

#else
__device__ static inline void mma_AtBt_base(rt_base<float, ducks::rt_layout::col> &d,
                                      const rt_base<half, ducks::rt_layout::col> &a,
                                      const rt_base<half, ducks::rt_layout::row> &b, // in col-major mode
                                      const rt_base<float, ducks::rt_layout::col> &c) {
    mfma161616(d.data, a.data, b.data, c.data);
}
__device__ static inline void mma_AtBt_base(rt_base<float, ducks::rt_layout::col> &d,
                                      const rt_base<bf16, ducks::rt_layout::col> &a,
                                      const rt_base<bf16, ducks::rt_layout::row> &b, // in col-major mode
                                      const rt_base<float, ducks::rt_layout::col> &c) {
    mfma161616(d.data, a.data, b.data, c.data);
}
#endif
/**
 * @brief Matrix multiply-accumulate operation.
 *
 * This function performs the matrix multiply-accumulate operation
 * using the `hmma16816` function.
 *
 * @tparam N The number of row tiles.
 * @tparam K The number of column tiles for the A matrix and row tiles for the B matrix.
 * @tparam M The number of column tiles for the B matrix.
 * @param[out] d The output rt_hf<N, M, row_layout> accumulator.
 * @param[in] a The first input rt_hf<N, K, row_layout> matrix.
 * @param[in] b The second input rt_hf<K, M, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_hf<N, M, row_layout> accumulator matrix.
 */
#ifdef KITTENS_CDNA4

#else
template<ducks::rt::col_layout D, ducks::rt::row_layout A, ducks::rt::col_layout B, ducks::rt::col_layout C>
__device__ static inline void mma_AB(D &d,
                               const A &a,
                               const B &b,
                               const C &c) {
    static_assert(D::rows == A::rows && D::cols == B::cols); // Check D matches A, B
    static_assert(A::cols == B::rows); // Check reduction dim is same
    static_assert(D::rows == C::rows && D::cols == C::cols); // Check D matches C

    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)
    );

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
#endif
/**
 * @brief Dot product operation for row layout.
 *
 * This function performs the dot product operation
 * using the `hmma16816` function.
 *
 * @tparam N The number of row tiles.
 * @tparam K The number of column tiles for the A matrix and row tiles for the B matrix.
 * @tparam M The number of column tiles for the B matrix.
 * @param[out] d The output rt_fl<N, M, row_layout> accumulator.
 * @param[in] a The first input rt_bf<N, K, row_layout> matrix.
 * @param[in] b The second input rt_bf<M, K, row_layout> matrix in row-major mode.
 * @param[in] c The input rt_fl<N, M, row_layout> accumulator matrix.
 */
#ifdef KITTENS_CDNA4
template<ducks::rt::accumulator_layout D, ducks::rt::row_layout A, ducks::rt::row_layout B, ducks::rt::accumulator_layout C>
__device__ static inline void mma_ABt(D &d,
                                const A &a,
                                const B &b, // notice row and (M, K) instead of col and (K, M)
                                const C &c) {
    static_assert(D::rows == A::rows && D::cols == B::rows); // Check D matches A, B
    static_assert(A::cols == B::cols); // Check reduction dim is same
    static_assert(D::rows == C::rows && D::cols == C::cols); // Check D matches C

    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)
    );

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
#else
template<ducks::rt::col_layout D, ducks::rt::row_layout A, ducks::rt::row_layout B, ducks::rt::col_layout C>
__device__ static inline void mma_ABt(D &d,
                                const A &a,
                                const B &b, // notice row and (M, K) instead of col and (K, M)
                                const C &c) {
    static_assert(D::rows == A::rows && D::cols == B::rows); // Check D matches A, B
    static_assert(A::cols == B::cols); // Check reduction dim is same
    static_assert(D::rows == C::rows && D::cols == C::cols); // Check D matches C

    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)
    );

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
#endif
/**
 * @brief Matrix multiply-accumulate operation with transposed A.
 *
 * This function performs the matrix multiply-accumulate operation
 * using the `hmma16816` instruction.
 *
 * @tparam N The number of row tiles.
 * @tparam K The number of column tiles for the A matrix and row tiles for the B matrix.
 * @tparam M The number of column tiles for the B matrix.
 * @param[out] d The output rt_fl<N, M, row_layout> accumulator.
 * @param[in] a The first input rt_bf<K, N, row_layout> matrix.
 * @param[in] b The second input rt_bf<K, M, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_fl<N, M, row_layout> accumulator matrix.
 */
#ifdef KITTENS_CDNA4

#else
template<ducks::rt::col_layout D, ducks::rt::col_layout A, ducks::rt::col_layout B, ducks::rt::col_layout C>
__device__ static inline void mma_AtB(D &d,
                                const A &a,
                                const B &b,
                                const C &c) {
    static_assert(D::rows == A::cols && D::cols == B::cols); // Check D matches A, B
    static_assert(A::rows == B::rows); // Check reduction dim is same
    static_assert(D::rows == C::rows && D::cols == C::cols); // Check D matches C

    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)
    );

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
#endif
/**
 * @brief Matrix multiply-accumulate operation with transposed A and B.
 *
 * This function performs the matrix multiply-accumulate operation
 * using the `hmma16816` instruction.
 *
 * @tparam N The number of row tiles.
 * @tparam K The number of column tiles for the A matrix and row tiles for the B matrix.
 * @tparam M The number of column tiles for the B matrix.
 * @param[out] d The output rt_fl<N, M, row_layout> accumulator.
 * @param[in] a The first input rt_bf<K, N, col_layout> matrix.
 * @param[in] b The second input rt_bf<M, K, row_layout> matrix in column-major mode.
 * @param[in] c The input rt_fl<N, M, row_layout> accumulator matrix.
 */
#ifdef KITTENS_CDNA4

#else
template<ducks::rt::col_layout D, ducks::rt::col_layout A, ducks::rt::row_layout B, ducks::rt::col_layout C>
__device__ static inline void mma_AtBt(D &d,
                                 const A &a,
                                 const B &b,
                                 const C &c) {
    static_assert(D::rows == A::cols && D::cols == B::rows); // Check D matches A, B
    static_assert(A::rows == B::cols); // Check reduction dim is same
    static_assert(D::rows == C::rows && D::cols == C::cols); // Check D matches C

    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)
    );

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
#endif
}