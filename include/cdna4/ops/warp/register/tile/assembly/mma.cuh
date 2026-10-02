/**
 * @file
 * @brief Matrix multiply-accumulate operations for tiles stored in registers.
 */

#pragma once

#include "../../../../../common/common.cuh"
#include "../../../../../types/types.cuh"

namespace kittens {
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
template<typename AccumulatorShape, typename InputType, typename RegisterRangeA, typename RegisterRangeB, typename RegisterRangeC, typename RegisterRangeD>
__device__ static inline void mma_ABt_base() {
    if constexpr (std::is_same_v<AccumulatorShape, ducks::rt_shape::rt_16x16>)
    {
        if constexpr (std::is_same_v<InputType, fp8e4m3>)
        {
            macros::mfma_f32_16x16x32_fp8_fp8<RegisterRangeA::lo, RegisterRangeB::lo, RegisterRangeC::lo, RegisterRangeD::lo>();
        }
        else
        {
            macros::mfma_f32_16x16x32_bf16<RegisterRangeA::lo, RegisterRangeB::lo, RegisterRangeC::lo, RegisterRangeD::lo>();
        }
    }
    else
    {
        macros::mfma_f32_16x16x32_bf16<RegisterRangeA::lo, RegisterRangeB::lo, RegisterRangeC::lo, RegisterRangeD::lo>();
    }
}

template<typename AccumulatorShape, typename InputType, typename RegisterRangeA, typename RegisterRangeB, typename RegisterRangeD>
__device__ static inline void mma_ABt_base_zero_accum() {
    if constexpr (std::is_same_v<AccumulatorShape, ducks::rt_shape::rt_16x16>)
    {
        if constexpr (std::is_same_v<InputType, fp8e4m3>)
        {
            macros::mfma_f32_16x16x32_fp8_fp8_zero_accum<RegisterRangeA::lo, RegisterRangeB::lo, RegisterRangeD::lo>();
        }
        else
        {
            macros::mfma_f32_16x16x32_bf16_zero_accum<RegisterRangeA::lo, RegisterRangeB::lo, RegisterRangeD::lo>();
        }
    }
    else
    {
        macros::mfma_f32_16x16x32_bf16_zero_accum<RegisterRangeA::lo, RegisterRangeB::lo, RegisterRangeD::lo>();
    }
}

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
template<typename AccumulatorShape, typename RegisterRangeA, typename RegisterRangeB, typename RegisterRangeC, typename RegisterRangeD>
__device__ static inline void mma_AtB_base() {

    if constexpr (std::is_same_v<AccumulatorShape, ducks::rt_shape::rt_32x32>) {
        macros::mfma_f32_32x32x16_bf16<RegisterRangeA::lo, RegisterRangeB::lo, RegisterRangeC::lo, RegisterRangeD::lo>();
    } else {
        macros::mfma_f32_16x16x32_bf16<RegisterRangeA::lo, RegisterRangeB::lo, RegisterRangeC::lo, RegisterRangeD::lo>();
    }
}
template<typename AccumulatorShape, typename RegisterRangeA, typename RegisterRangeB, typename RegisterRangeD>
__device__ static inline void mma_AtB_base_zero_accum() {
    if constexpr (std::is_same_v<AccumulatorShape, ducks::rt_shape::rt_32x32>) {
        macros::mfma_f32_32x32x16_bf16_zero_accum<RegisterRangeA::lo, RegisterRangeB::lo, RegisterRangeD::lo>();
    } else {
        macros::mfma_f32_16x16x32_bf16_zero_accum<RegisterRangeA::lo, RegisterRangeB::lo, RegisterRangeD::lo>();
    }
}

/**
 * @brief Matrix multiply-accumulate operation for rt types.
 *
 * This function performs the matrix multiply-accumulate operation D = A * B^T + C
 * specifically optimized for rt types with explicit register management.
 *
 * @tparam D The output rt matrix type
 * @tparam A The input rt matrix type A
 * @tparam B The input rt matrix type B (will be transposed)
 * @tparam C The input rt accumulator matrix type
 */
template<int N, int M, int K, ducks::art::all D, ducks::art::all A, ducks::art::all B, ducks::art::all C>
__device__ static inline void mma_ABt(D &d,
                                const A &a,
                                const B &b,
                                const C &c) {

    static_assert(std::is_same_v<typename D::layout, ducks::rt_layout::col>, "D must be a col layout");
    static_assert(std::is_same_v<typename A::layout, ducks::rt_layout::row>, "A must be a row layout");
    static_assert(std::is_same_v<typename B::layout, ducks::rt_layout::row>, "B must be a row layout");
    static_assert(std::is_same_v<typename C::layout, ducks::rt_layout::col>, "C must be a col layout");

    static_assert(D::rows == A::rows && D::cols == B::rows); // Check D matches A, B
    static_assert(A::cols == B::cols); // Check reduction dim is same
    static_assert(D::rows == C::rows && D::cols == C::cols); // Check D matches C

    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half>) ||
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, fp8e4m3> &&
            std::is_same_v<typename B::T, fp8e4m3>)
    );

    // Helper function template for compile-time MMA operations
    using range_type_A = ducks::art::get_nth_range_t<typename A::register_ranges, N * A::width + K>;
    using range_type_B = ducks::art::get_nth_range_t<typename B::register_ranges, M * B::width + K>;
    using range_type_C = ducks::art::get_nth_range_t<typename C::register_ranges, N * C::width + M>;
    using range_type_D = ducks::art::get_nth_range_t<typename D::register_ranges, N * D::width + M>;
    mma_ABt_base<typename D::shape, typename A::T, range_type_A, range_type_B, range_type_C, range_type_D>();
}

template<ducks::art::all D, ducks::art::all A, ducks::art::all B, ducks::art::all C>
__device__ static inline void mma_ABt(D &d,
                                const A &a,
                                const B &b,
                                const C &c) {

    static_assert(std::is_same_v<typename D::layout, ducks::rt_layout::col>, "D must be a col layout");
    static_assert(std::is_same_v<typename A::layout, ducks::rt_layout::row>, "A must be a row layout");
    static_assert(std::is_same_v<typename B::layout, ducks::rt_layout::row>, "B must be a row layout");
    static_assert(std::is_same_v<typename C::layout, ducks::rt_layout::col>, "C must be a col layout");

    static_assert(D::rows == A::rows && D::cols == B::rows); // Check D matches A, B
    static_assert(A::cols == B::cols); // Check reduction dim is same
    static_assert(D::rows == C::rows && D::cols == C::cols); // Check D matches C

    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half>) ||
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, fp8e4m3> &&
            std::is_same_v<typename B::T, fp8e4m3>)
    );

    // Helper function template for compile-time MMA operations
    auto perform_mma_at = []<int N, int M>() {
        // First MMA operation with k=0
        using range_type_A = ducks::art::get_nth_range_t<typename A::register_ranges, N * A::width>;
        using range_type_B = ducks::art::get_nth_range_t<typename B::register_ranges, M * B::width>;
        using range_type_C = ducks::art::get_nth_range_t<typename C::register_ranges, N * C::width + M>;
        using range_type_D = ducks::art::get_nth_range_t<typename D::register_ranges, N * D::width + M>;
        mma_ABt_base<typename D::shape, typename A::T, range_type_A, range_type_B, range_type_C, range_type_D>();

        // Subsequent MMA operations for k=1 to A::width-1
        [&]<std::size_t... Ks>(std::index_sequence<Ks...>) {
            ([&] {
                constexpr int k = Ks + 1;
                if constexpr (k < A::width) {
                    using range_type_A = ducks::art::get_nth_range_t<typename A::register_ranges, k + N * A::width>;
                    using range_type_B = ducks::art::get_nth_range_t<typename B::register_ranges, k + M * B::width>;
                    using range_type_C = ducks::art::get_nth_range_t<typename C::register_ranges, N * C::width + M>;
                    using range_type_D = ducks::art::get_nth_range_t<typename D::register_ranges, N * D::width + M>;
                    mma_ABt_base<typename D::shape, typename A::T, range_type_A, range_type_B, range_type_C, range_type_D>();
                }
            }(), ...);
        }(std::make_index_sequence<A::width>{});
    };

    // Compile-time nested loops over N and M
    [&]<std::size_t... Ns>(std::index_sequence<Ns...>) {
        ([&]<std::size_t N>() {
            [&]<std::size_t... Ms>(std::index_sequence<Ms...>) {
                ([&]<std::size_t M>() {
                    perform_mma_at.template operator()<N, M>();
                }.template operator()<Ms>(), ...);
            }(std::make_index_sequence<D::width>{});
        }.template operator()<Ns>(), ...);
    }(std::make_index_sequence<D::height>{});
}

template<int N, int M, int K, ducks::art::all D, ducks::art::all A, ducks::art::all B>
__device__ static inline void mma_ABt(D &d,
                                const A &a,
                                const B &b) {

    static_assert(std::is_same_v<typename D::layout, ducks::rt_layout::col>, "D must be a col layout");
    static_assert(std::is_same_v<typename A::layout, ducks::rt_layout::row>, "A must be a row layout");
    static_assert(std::is_same_v<typename B::layout, ducks::rt_layout::row>, "B must be a row layout");

    static_assert(D::rows == A::rows && D::cols == B::rows); // Check D matches A, B
    static_assert(A::cols == B::cols); // Check reduction dim is same

    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half>) ||
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, fp8e4m3> &&
            std::is_same_v<typename B::T, fp8e4m3>)
    );

    // Helper function template for compile-time MMA operations
    // First MMA operation with k=0
    using range_type_A = ducks::art::get_nth_range_t<typename A::register_ranges, N * A::width + K>;
    using range_type_B = ducks::art::get_nth_range_t<typename B::register_ranges, M * B::width + K>;
    using range_type_D = ducks::art::get_nth_range_t<typename D::register_ranges, N * D::width + M>;
    mma_ABt_base_zero_accum<typename D::shape, typename A::T, range_type_A, range_type_B, range_type_D>();
}

template<ducks::art::all D, ducks::art::all A, ducks::art::all B>
__device__ static inline void mma_ABt(D &d,
                                const A &a,
                                const B &b) {
                                
    static_assert(std::is_same_v<typename D::layout, ducks::rt_layout::col>, "D must be a col layout");
    static_assert(std::is_same_v<typename A::layout, ducks::rt_layout::row>, "A must be a row layout");
    static_assert(std::is_same_v<typename B::layout, ducks::rt_layout::row>, "B must be a row layout");

    static_assert(D::rows == A::rows && D::cols == B::rows); // Check D matches A, B
    static_assert(A::cols == B::cols); // Check reduction dim is same

    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half>) ||
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, fp8e4m3> &&
            std::is_same_v<typename B::T, fp8e4m3>)
    );

    // Helper function template for compile-time MMA operations
    auto perform_mma_at = []<int N, int M>() {
        // First MMA operation with k=0
        using range_type_A = ducks::art::get_nth_range_t<typename A::register_ranges, N * A::width>;
        using range_type_B = ducks::art::get_nth_range_t<typename B::register_ranges, M * B::width>;
        using range_type_D = ducks::art::get_nth_range_t<typename D::register_ranges, N * D::width + M>;
        mma_ABt_base_zero_accum<typename D::shape, typename A::T, range_type_A, range_type_B, range_type_D>();

        // Subsequent MMA operations for k=1 to A::width-1
        [&]<std::size_t... Ks>(std::index_sequence<Ks...>) {
            ([&] {
                constexpr int k = Ks + 1;
                if constexpr (k < A::width) {
                    using range_type_A = ducks::art::get_nth_range_t<typename A::register_ranges, k + N * A::width>;
                    using range_type_B = ducks::art::get_nth_range_t<typename B::register_ranges, k + M * B::width>;
                    using range_type_D = ducks::art::get_nth_range_t<typename D::register_ranges, N * D::width + M>;
                    mma_ABt_base<typename D::shape, typename A::T, range_type_A, range_type_B, range_type_D, range_type_D>();
                }
            }(), ...);
        }(std::make_index_sequence<A::width>{});
    };

    // Compile-time nested loops over N and M
    [&]<std::size_t... Ns>(std::index_sequence<Ns...>) {
        ([&]<std::size_t N>() {
            [&]<std::size_t... Ms>(std::index_sequence<Ms...>) {
                ([&]<std::size_t M>() {
                    perform_mma_at.template operator()<N, M>();
                }.template operator()<Ms>(), ...);
            }(std::make_index_sequence<D::width>{});
        }.template operator()<Ns>(), ...);
    }(std::make_index_sequence<D::height>{});
}

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
template<int N, int M, int K, ducks::art::all D, ducks::art::all A, ducks::art::all B, ducks::art::all C>
__device__ static inline void mma_AtB(D &d,
                                const A &a,
                                const B &b,
                                const C &c) {
    
    static_assert(std::is_same_v<typename D::layout, ducks::rt_layout::col>, "D must be a col layout");
    static_assert(std::is_same_v<typename A::layout, ducks::rt_layout::col>, "A must be a col layout");
    static_assert(std::is_same_v<typename B::layout, ducks::rt_layout::col>, "B must be a col layout");
    static_assert(std::is_same_v<typename C::layout, ducks::rt_layout::col>, "C must be a col layout");

    static_assert(D::rows == A::cols && D::cols == B::cols); // Check D matches A, B
    static_assert(A::rows == B::rows); // Check reduction dim is same
    static_assert(D::rows == C::rows && D::cols == C::cols); // Check D matches C

    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)
    );

    // Helper function template for compile-time MMA operations
    using range_type_A = ducks::art::get_nth_range_t<typename A::register_ranges, N + K * A::width>;
    using range_type_B = ducks::art::get_nth_range_t<typename B::register_ranges, M + K * B::width>;
    using range_type_C = ducks::art::get_nth_range_t<typename C::register_ranges, N * C::width + M>;
    using range_type_D = ducks::art::get_nth_range_t<typename D::register_ranges, N * D::width + M>;
    mma_AtB_base<typename D::shape, range_type_A, range_type_B, range_type_C, range_type_D>();
}

template<ducks::art::all D, ducks::art::all A, ducks::art::all B, ducks::art::all C>
__device__ static inline void mma_AtB(D &d,
                                const A &a,
                                const B &b,
                                const C &c) {
    
    static_assert(std::is_same_v<typename D::layout, ducks::rt_layout::col>, "D must be a col layout");
    static_assert(std::is_same_v<typename A::layout, ducks::rt_layout::col>, "A must be a col layout");
    static_assert(std::is_same_v<typename B::layout, ducks::rt_layout::col>, "B must be a col layout");
    static_assert(std::is_same_v<typename C::layout, ducks::rt_layout::col>, "C must be a col layout");

    static_assert(D::rows == A::cols && D::cols == B::cols); // Check D matches A, B
    static_assert(A::rows == B::rows); // Check reduction dim is same
    static_assert(D::rows == C::rows && D::cols == C::cols); // Check D matches C

    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)
    );

    // Helper function template for compile-time MMA operations
    auto perform_mma_at = []<int N, int M>() {
        // First MMA operation with k=0
        using range_type_A = ducks::art::get_nth_range_t<typename A::register_ranges, N>;
        using range_type_B = ducks::art::get_nth_range_t<typename B::register_ranges, M>;
        using range_type_C = ducks::art::get_nth_range_t<typename C::register_ranges, N * C::width + M>;
        using range_type_D = ducks::art::get_nth_range_t<typename D::register_ranges, N * D::width + M>;
        mma_AtB_base<typename D::shape, range_type_A, range_type_B, range_type_C, range_type_D>();

        // Subsequent MMA operations for k=1 to A::width-1
        [&]<std::size_t... Ks>(std::index_sequence<Ks...>) {
            ([&] {
                constexpr int k = Ks + 1;
                if constexpr (k < A::height) {
                    using range_type_A = ducks::art::get_nth_range_t<typename A::register_ranges, k * A::width + N>;
                    using range_type_B = ducks::art::get_nth_range_t<typename B::register_ranges, k * B::width + M>;
                    using range_type_C = ducks::art::get_nth_range_t<typename C::register_ranges, N * C::width + M>;
                    using range_type_D = ducks::art::get_nth_range_t<typename D::register_ranges, N * D::width + M>;
                    mma_AtB_base<typename D::shape, range_type_A, range_type_B, range_type_C, range_type_D>();
                }
            }(), ...);
        }(std::make_index_sequence<A::height>{});
    };

    // Compile-time nested loops over N and M
    [&]<std::size_t... Ns>(std::index_sequence<Ns...>) {
        ([&]<std::size_t N>() {
            [&]<std::size_t... Ms>(std::index_sequence<Ms...>) {
                ([&]<std::size_t M>() {
                    perform_mma_at.template operator()<N, M>();
                }.template operator()<Ms>(), ...);
            }(std::make_index_sequence<D::width>{});
        }.template operator()<Ns>(), ...);
    }(std::make_index_sequence<D::height>{});
}

template<int N, int M, int K, ducks::art::all D, ducks::art::all A, ducks::art::all B>
__device__ static inline void mma_AtB(D &d,
                                const A &a,
                                const B &b) {
                                    
    static_assert(std::is_same_v<typename D::layout, ducks::rt_layout::col>, "D must be a col layout");
    static_assert(std::is_same_v<typename A::layout, ducks::rt_layout::col>, "A must be a col layout");
    static_assert(std::is_same_v<typename B::layout, ducks::rt_layout::col>, "B must be a col layout");

    static_assert(D::rows == A::cols && D::cols == B::cols); // Check D matches A, B
    static_assert(A::rows == B::rows); // Check reduction dim is same

    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half>)
    );

    // Helper function template for compile-time MMA operations
    using range_type_A = ducks::art::get_nth_range_t<typename A::register_ranges, N + K * A::width>;
    using range_type_B = ducks::art::get_nth_range_t<typename B::register_ranges, M + K * B::width>;
    using range_type_D = ducks::art::get_nth_range_t<typename D::register_ranges, N * D::width + M>;
    mma_AtB_base_zero_accum<typename D::shape, range_type_A, range_type_B, range_type_D>();
}

template<ducks::art::all D, ducks::art::all A, ducks::art::all B>
__device__ static inline void mma_AtB(D &d,
                                const A &a,
                                const B &b) {
    
    static_assert(std::is_same_v<typename D::layout, ducks::rt_layout::col>, "D must be a col layout");
    static_assert(std::is_same_v<typename A::layout, ducks::rt_layout::col>, "A must be a col layout");
    static_assert(std::is_same_v<typename B::layout, ducks::rt_layout::col>, "B must be a col layout");

    static_assert(D::rows == A::cols && D::cols == B::cols); // Check D matches A, B
    static_assert(A::rows == B::rows); // Check reduction dim is same

    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half>)
    );

    // Helper function template for compile-time MMA operations
    auto perform_mma_at = []<int N, int M>() {
        // First MMA operation with k=0
        using range_type_A = ducks::art::get_nth_range_t<typename A::register_ranges, N>;
        using range_type_B = ducks::art::get_nth_range_t<typename B::register_ranges, M>;
        using range_type_D = ducks::art::get_nth_range_t<typename D::register_ranges, N * D::width + M>;
        mma_AtB_base_zero_accum<typename D::shape, range_type_A, range_type_B, range_type_D>();

        // Subsequent MMA operations for k=1 to A::width-1
        [&]<std::size_t... Ks>(std::index_sequence<Ks...>) {
            ([&] {
                constexpr int k = Ks + 1;
                if constexpr (k < A::height) {
                    using range_type_A = ducks::art::get_nth_range_t<typename A::register_ranges, k * A::width + N>;
                    using range_type_B = ducks::art::get_nth_range_t<typename B::register_ranges, k * B::width + M>;
                    using range_type_D = ducks::art::get_nth_range_t<typename D::register_ranges, N * D::width + M>;
                    mma_AtB_base<typename D::shape, range_type_A, range_type_B, range_type_D, range_type_D>();
                }
            }(), ...);
        }(std::make_index_sequence<A::height>{});
    };

    // Compile-time nested loops over N and M
    [&]<std::size_t... Ns>(std::index_sequence<Ns...>) {
        ([&]<std::size_t N>() {
            [&]<std::size_t... Ms>(std::index_sequence<Ms...>) {
                ([&]<std::size_t M>() {
                    perform_mma_at.template operator()<N, M>();
                }.template operator()<Ms>(), ...);
            }(std::make_index_sequence<D::width>{});
        }.template operator()<Ns>(), ...);
    }(std::make_index_sequence<D::height>{});
}


/**
 * @brief Scaled fp8 matrix multiply-accumulate on one 16x16 subtile of art tiles: D(N, M) = A(N) * B(M)^T + C(N, M),
 * with E8M0 block scales applied by V_MFMA_SCALE_F32_16X16X128_F8F6F4.
 *
 * @tparam N Row subtile of D (and of A).
 * @tparam M Column subtile of D (row subtile of B).
 * @tparam OPSEL_A Byte of scale_a for this MFMA.
 * @tparam OPSEL_B Byte of scale_b for this MFMA.
 */
template<int N, int M, int OPSEL_A, int OPSEL_B, ducks::art::all D, ducks::art::all A, ducks::art::all B, ducks::art::all C>
__device__ static inline void mma_ABt_scaled(D &d,
                                const A &a,
                                const B &b,
                                const C &c,
                                uint32_t scale_a,
                                uint32_t scale_b) {
    static_assert(std::is_same_v<typename D::layout, ducks::rt_layout::col>, "D must be a col layout");
    static_assert(std::is_same_v<typename A::layout, ducks::rt_layout::row>, "A must be a row layout");
    static_assert(std::is_same_v<typename B::layout, ducks::rt_layout::row>, "B must be a row layout");
    static_assert(std::is_same_v<typename C::layout, ducks::rt_layout::col>, "C must be a col layout");
    static_assert(std::is_same_v<typename D::shape, ducks::rt_shape::rt_16x16> &&
                  std::is_same_v<typename C::shape, ducks::rt_shape::rt_16x16>, "C and D must be rt_16x16 tiles");
    static_assert(std::is_same_v<typename A::shape, ducks::rt_shape::rt_16x128> &&
                  std::is_same_v<typename B::shape, ducks::rt_shape::rt_16x128>, "A and B must be rt_16x128 tiles");
    static_assert(std::is_same_v<typename D::T, float> && std::is_same_v<typename C::T, float> &&
                  std::is_same_v<typename A::T, fp8e4m3> && std::is_same_v<typename B::T, fp8e4m3>,
                  "scaled mma takes fp8e4m3 operands and fp32 accumulators");
    static_assert(D::rows == A::rows && D::cols == B::rows); // Check D matches A, B
    static_assert(A::cols == 128 && B::cols == 128, "one 128-wide reduction subtile per MFMA");
    static_assert(D::rows == C::rows && D::cols == C::cols); // Check D matches C
    static_assert(N < D::height && M < D::width, "subtile out of range");

    using range_type_A = ducks::art::get_nth_range_t<typename A::register_ranges, N>;
    using range_type_B = ducks::art::get_nth_range_t<typename B::register_ranges, M>;
    using range_type_C = ducks::art::get_nth_range_t<typename C::register_ranges, N * C::width + M>;
    using range_type_D = ducks::art::get_nth_range_t<typename D::register_ranges, N * D::width + M>;
    macros::mfma_scale_f32_16x16x128_f8f6f4<range_type_D::lo, range_type_A::lo, range_type_B::lo, range_type_C::lo,
                                            OPSEL_A, OPSEL_B>(scale_a, scale_b);
}

/**
 * @brief Scaled fp8 matrix multiply-accumulate on whole art tiles: D = A * B^T + C, subtile (N, M) using byte N of
 * scale_a and byte M of scale_b (the convention of the rt mma_ABt_scaled).
 */
template<ducks::art::all D, ducks::art::all A, ducks::art::all B, ducks::art::all C>
__device__ static inline void mma_ABt_scaled(D &d,
                                const A &a,
                                const B &b,
                                const C &c,
                                uint32_t scale_a,
                                uint32_t scale_b) {
    static_assert(D::height <= 4 && D::width <= 4, "one scale byte per subtile: at most 4 subtiles per dimension");
    [&]<std::size_t... Ns>(std::index_sequence<Ns...>) {
        ([&]<std::size_t N>() {
            [&]<std::size_t... Ms>(std::index_sequence<Ms...>) {
                (mma_ABt_scaled<N, Ms, N, Ms>(d, a, b, c, scale_a, scale_b), ...);
            }(std::make_index_sequence<D::width>{});
        }.template operator()<Ns>(), ...);
    }(std::make_index_sequence<D::height>{});
}

}