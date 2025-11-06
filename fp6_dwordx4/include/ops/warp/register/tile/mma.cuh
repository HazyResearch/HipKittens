/**
 * @file
 * @brief Matrix multiply-accumulate operations for tiles stored in registers.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {
__device__ static inline void mfma1616128_fp6(float2 (&D)[2],
                                             const fp6_e2m3_32 (&A)[1], 
                                             const fp6_e2m3_32 (&B)[1],
                                             const float2 (&C)[2]) {

    typedef __attribute__((__vector_size__(8 * sizeof(int)))) int int8x_t;
    typedef __attribute__((__vector_size__(4 * sizeof(float)))) float floatx4_t;

    *(floatx4_t*)D = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
        *(const int8x_t*)&A[0],
        *(const int8x_t*)&B[0],
        *(floatx4_t*)C,
        2, // cbsz
        2, // blgp
        0, 0, 0, 0
    );
}

__device__ static inline void mma_ABt_base(rt_base<float, ducks::rt_layout::accumulator> &d,
                                     const rt_base<fp6_e2m3, ducks::rt_layout::row> &a,
                                     const rt_base<fp6_e2m3, ducks::rt_layout::row> &b,
                                     const rt_base<float, ducks::rt_layout::accumulator> &c) {
       mfma1616128_fp6(d.data, a.data, b.data, c.data);
}


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
#else
template<ducks::rt::col_layout D, ducks::rt::row_layout A, ducks::rt::row_layout B, ducks::rt::col_layout C>
#endif
__device__ static inline void mma_ABt(D &d,
                                const A &a,
                                const B &b, // notice row and (M, K) instead of col and (K, M)
                                const C &c) {
    static_assert(D::rows == A::rows && D::cols == B::rows); // Check D matches A, B
    static_assert(A::cols == B::cols); // Check reduction dim is same
    static_assert(D::rows == C::rows && D::cols == C::cols); // Check D matches C

    #ifdef KITTENS_CDNA4
    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>) ||
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, fp6_e2m3> &&
            std::is_same_v<typename B::T, fp6_e2m3> && std::is_same_v<typename C::T, float>)
    );
    #else
    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)
    );
    #endif

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
}