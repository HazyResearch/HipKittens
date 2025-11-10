#pragma once

/* testing_commons.cuh
 * 
 * This file contains a bunch of moderately test-specific utils.
 * For example, test_name constructors and __device__ kernel wrappers.
 * This file is distinguished from testing_utils.cuh in that you
 * might need to add to this file in order to add more tests,
 * but you shouldn't need to modify that testing_utils at all.
 */

#include "kittens.cuh"

#include "testing_utils.cuh"

/* ---------- TEST NAMES ---------- */

// This how we generate parameterized names for tests.
// test_id is defined by the test, like "reg_mma" --
// then these templates build the rest of the test name.
// Note use of concepts to prevent template arg collisions!

// 1D test names
template<int S, int NW> std::string generate_test_name(std::string test_id) {
    std::string label = test_id+"_["+std::to_string(S)+"]";
    if constexpr (NW > 1) {label += "_["+std::to_string(NW)+"warps]";}
    return label;
}

template<typename T2, typename U2> std::string generate_copy_name() {
    std::string label = "";
    if constexpr (std::is_same_v<U2, float>) label += "_[float->";
    else if constexpr (std::is_same_v<U2, kittens::bf16>) label += "_[bf16->";
    else label += "_[half->";
    if constexpr (std::is_same_v<T2, float>) label += "float]";
    else if constexpr (std::is_same_v<T2, kittens::bf16>) label += "bf16]";
    else label += "half]";
    return label;
}
/**
* @brief Generate a test name for a 1D test with a row or column layout for 16x32 shapes. 
*/
template<kittens::ducks::rt_shape::all RT_SHAPE, kittens::ducks::st_shape::all ST_SHAPE, int S, int NW> std::string generate_test_name(std::string test_id) {
    std::string label = generate_test_name<S,NW>(test_id);

    // do we want this? 
    static_assert(RT_SHAPE::cols / ST_SHAPE::cols >= 1 , "RT_SHAPE::cols must be a positive factor of ST_SHAPE::cols");
    static_assert(RT_SHAPE::rows / ST_SHAPE::rows >= 1 , "RT_SHAPE::rows must be a positive factor of ST_SHAPE::rows");

    // rt shapes
    if constexpr (std::is_same_v<typename kittens::ducks::rt_shape::rt_16x16, RT_SHAPE>) label += "_[rt_16x16]";
    else if constexpr (std::is_same_v<typename kittens::ducks::rt_shape::rt_32x32, RT_SHAPE>) label += "_[rt_32x32]";
    else if constexpr (std::is_same_v<typename kittens::ducks::rt_shape::rt_32x32_8, RT_SHAPE>) label += "_[rt_32x32_8]";
    else if constexpr (std::is_same_v<typename kittens::ducks::rt_shape::rt_16x32, RT_SHAPE>) label += "_[rt_16x32]";
    else if constexpr (std::is_same_v<typename kittens::ducks::rt_shape::rt_32x16, RT_SHAPE>) label += "_[rt_32x16]";
    else if constexpr (std::is_same_v<typename kittens::ducks::rt_shape::rt_32x16_4, RT_SHAPE>) label += "_[rt_32x16_4]";
    else if constexpr (std::is_same_v<typename kittens::ducks::rt_shape::rt_16x32_4, RT_SHAPE>) label += "_[rt_16x32_4]";
    else static_assert(false, "Unknown shape");

    // st shapes
    if constexpr (std::is_same_v<typename kittens::ducks::st_shape::st_16x16, ST_SHAPE>) label += "_[st_16x16]";
    else if constexpr (std::is_same_v<typename kittens::ducks::st_shape::st_16x16_swizzled, ST_SHAPE>) label += "_[st_16x16_swizzled]";
    else if constexpr (std::is_same_v<typename kittens::ducks::st_shape::st_32x32, ST_SHAPE>) label += "_[st_32x32]";
    else if constexpr (std::is_same_v<typename kittens::ducks::st_shape::st_16x32, ST_SHAPE>) label += "_[st_16x32]";
    else if constexpr (std::is_same_v<typename kittens::ducks::st_shape::st_32x16, ST_SHAPE>) label += "_[st_32x16]";
    else if constexpr (std::is_same_v<typename kittens::ducks::st_shape::st_8x32, ST_SHAPE>) label += "_[st_8x32]";
    else if constexpr (std::is_same_v<typename kittens::ducks::st_shape::st_16x128, ST_SHAPE>) label += "_[st_16x128]";
    else static_assert(false, "Unknown shape");
    return label;
}
/**
* @brief Generate a test name for a 1D test with a row or column layout for 16x32 shapes. 
*/
template<kittens::ducks::rt_shape::all RT_SHAPE, kittens::ducks::st_shape::all ST_SHAPE, int S, int NW, kittens::ducks::rv_layout::all L> std::string generate_test_name(std::string test_id) {
    std::string label = generate_test_name<RT_SHAPE,ST_SHAPE,S,NW>(test_id);

    if constexpr (std::is_same_v<L, kittens::naive_l>) label += "_[rv_naive_layout]";
    else if constexpr (std::is_same_v<L, kittens::ortho_l>) label += "_[rv_ortho_layout]";
    else label += "_[rv_align_layout]";
    return label;
}
template<kittens::ducks::rt_shape::all RT_SHAPE, kittens::ducks::st_shape::all ST_SHAPE, int S, int NW, kittens::ducks::rv_layout::all L1, typename T2, typename U2> std::string generate_test_name(std::string test_id) {
    std::string label = generate_test_name<RT_SHAPE,ST_SHAPE,S,NW,L1>(test_id);
    label += generate_copy_name<T2, U2>();
    return label;
}

// 2D test names

template<int H, int W, int NW> std::string generate_test_name(std::string test_id) {
    std::string label = test_id+"_["+std::to_string(H)+"x"+std::to_string(W)+"]";
    if constexpr (NW > 1) {
        label += "_["+std::to_string(NW)+"warps]";
    }
    return label;
}

template<kittens::ducks::rt_shape::all RT_SHAPE, kittens::ducks::st_shape::all ST_SHAPE, int H, int W, int NW> std::string generate_test_name(std::string test_id) {
    std::string label = generate_test_name<H,W,NW>(test_id);

    static_assert(RT_SHAPE::cols / ST_SHAPE::cols >= 1 , "RT_SHAPE::cols must be a positive factor of ST_SHAPE::cols");
    static_assert(RT_SHAPE::rows / ST_SHAPE::rows >= 1 , "RT_SHAPE::rows must be a positive factor of ST_SHAPE::rows");

    // rt shapes
    if constexpr (std::is_same_v<typename kittens::ducks::rt_shape::rt_16x16, RT_SHAPE>) label += "_[rt_16x16]";
    else if constexpr (std::is_same_v<typename kittens::ducks::rt_shape::rt_32x32, RT_SHAPE>) label += "_[rt_32x32]";
    else if constexpr (std::is_same_v<typename kittens::ducks::rt_shape::rt_32x32_8, RT_SHAPE>) label += "_[rt_32x32_8]";
    else if constexpr (std::is_same_v<typename kittens::ducks::rt_shape::rt_16x32, RT_SHAPE>) label += "_[rt_16x32]";
    else if constexpr (std::is_same_v<typename kittens::ducks::rt_shape::rt_32x16, RT_SHAPE>) label += "_[rt_32x16]";
    else if constexpr (std::is_same_v<typename kittens::ducks::rt_shape::rt_32x16_4, RT_SHAPE>) label += "_[rt_32x16_4]";
    else if constexpr (std::is_same_v<typename kittens::ducks::rt_shape::rt_16x32_4, RT_SHAPE>) label += "_[rt_16x32_4]";
    else static_assert(false, "Unknown shape");

    // st shapes
    if constexpr (std::is_same_v<typename kittens::ducks::st_shape::st_16x16, ST_SHAPE>) label += "_[st_16x16]";
    else if constexpr (std::is_same_v<typename kittens::ducks::st_shape::st_16x16_swizzled, ST_SHAPE>) label += "_[st_16x16_swizzled]";
    else if constexpr (std::is_same_v<typename kittens::ducks::st_shape::st_32x32, ST_SHAPE>) label += "_[st_32x32]";
    else if constexpr (std::is_same_v<typename kittens::ducks::st_shape::st_16x32, ST_SHAPE>) label += "_[st_16x32]";
    else if constexpr (std::is_same_v<typename kittens::ducks::st_shape::st_32x16, ST_SHAPE>) label += "_[st_32x16]";
    else if constexpr (std::is_same_v<typename kittens::ducks::st_shape::st_8x32, ST_SHAPE>) label += "_[st_8x32]";
    else if constexpr (std::is_same_v<typename kittens::ducks::st_shape::st_16x128, ST_SHAPE>) label += "_[st_16x128]";
    else static_assert(false, "Unknown shape");
    return label;
}

template<kittens::ducks::rt_shape::all RT_SHAPE, kittens::ducks::st_shape::all ST_SHAPE, int H, int W, int NW, kittens::ducks::rt_layout::all L> std::string generate_test_name(std::string test_id) {
    std::string label = generate_test_name<RT_SHAPE,ST_SHAPE,H,W,NW>(test_id);

    // layouts
    if constexpr (std::is_same_v<L, kittens::ducks::rt_layout::row>) label += "_[rt_row_layout]";
    else label += "_[rt_col_layout]";

    return label;
}

template<kittens::ducks::rt_shape::all RT_SHAPE, kittens::ducks::st_shape::all ST_SHAPE, int H, int W, int NW, kittens::ducks::rt_layout::all L1, kittens::ducks::rt_shape::all RT_SHAPE2, kittens::ducks::rt_layout::all L2> std::string generate_test_name(std::string test_id) {
    std::string label = generate_test_name<RT_SHAPE,ST_SHAPE,H,W,NW,L1>(test_id);

    // shapes
    if constexpr (std::is_same_v<typename kittens::ducks::rt_shape::rt_16x16, RT_SHAPE2>) label += "_[rt_16x16]";
    else if constexpr (std::is_same_v<typename kittens::ducks::rt_shape::rt_32x32, RT_SHAPE2>) label += "_[rt_32x32]";
    else if constexpr (std::is_same_v<typename kittens::ducks::rt_shape::rt_32x32_8, RT_SHAPE2>) label += "_[rt_32x32_8]";
    else if constexpr (std::is_same_v<typename kittens::ducks::rt_shape::rt_16x32, RT_SHAPE2>) label += "_[rt_16x32]";
    else if constexpr (std::is_same_v<typename kittens::ducks::rt_shape::rt_32x16, RT_SHAPE2>) label += "_[rt_32x16]";
    else if constexpr (std::is_same_v<typename kittens::ducks::rt_shape::rt_32x16_4, RT_SHAPE2>) label += "_[rt_32x16_4]";
    else if constexpr (std::is_same_v<typename kittens::ducks::rt_shape::rt_16x32_4, RT_SHAPE2>) label += "_[rt_16x32_4]";
    else static_assert(false, "Unknown shape");

    if constexpr (std::is_same_v<L2, kittens::ducks::rt_layout::row>) label += "_[rt_row_layout]";
    else label += "_[rt_col_layout]";

    return label;
}
template <typename T> concept integral_wrapper = std::is_integral_v<decltype(T::value)>;
template<kittens::ducks::rt_shape::all RT_SHAPE, kittens::ducks::st_shape::all ST_SHAPE, int H, int W, int NW, integral_wrapper _ST_H> std::string generate_test_name(std::string test_id) {
    constexpr int ST_H = _ST_H::value;
    std::string label = test_id+"_["+std::to_string(H)+"x"+std::to_string(W)+"_"+std::to_string(ST_H)+"x"+std::to_string(W)+"]";
    if constexpr (NW > 1) {
        label += "_["+std::to_string(NW)+"warps]";
    }
    return label;
}
template<kittens::ducks::rt_shape::all RT_SHAPE, kittens::ducks::st_shape::all ST_SHAPE, int H, int W, int NW, kittens::ducks::base_types::T1 T2, kittens::ducks::base_types::T1 U2> std::string generate_test_name(std::string test_id) {
    std::string label = generate_test_name<H,W,NW>(test_id);

    static_assert(RT_SHAPE::cols / ST_SHAPE::cols >= 1 , "RT_SHAPE::cols must be a positive factor of ST_SHAPE::cols");
    static_assert(RT_SHAPE::rows / ST_SHAPE::rows >= 1 , "RT_SHAPE::rows must be a positive factor of ST_SHAPE::rows");

    // shapes
    if constexpr (std::is_same_v<typename kittens::ducks::rt_shape::rt_16x16, RT_SHAPE>) label += "_[rt_16x16]";
    else if constexpr (std::is_same_v<typename kittens::ducks::rt_shape::rt_32x32, RT_SHAPE>) label += "_[rt_32x32]";
    else if constexpr (std::is_same_v<typename kittens::ducks::rt_shape::rt_32x32_8, RT_SHAPE>) label += "_[rt_32x32_8]";
    else if constexpr (std::is_same_v<typename kittens::ducks::rt_shape::rt_16x32, RT_SHAPE>) label += "_[rt_16x32]";
    else if constexpr (std::is_same_v<typename kittens::ducks::rt_shape::rt_32x16, RT_SHAPE>) label += "_[rt_32x16]";
    else if constexpr (std::is_same_v<typename kittens::ducks::rt_shape::rt_32x16_4, RT_SHAPE>) label += "_[rt_32x16_4]";
    else if constexpr (std::is_same_v<typename kittens::ducks::rt_shape::rt_16x32_4, RT_SHAPE>) label += "_[rt_16x32_4]";
    else static_assert(false, "Unknown shape");

    // copy
    label += generate_copy_name<T2, U2>();
    return label;
}


/* ---------- TEST WRAPPERS ---------- */

// These are wrappers to make it really easy to call and run tests.
// The basic wrappers:
// - Check if the test is valid and not compile it otherwise (the if constexpr)
// - Initialize input and output memory on both host and device
// - Call test functions on host and device
// - Validate outputs, append the result to test_data& results
// - Cleanup
// Additionally, the templated wrappers:
// - Loop through lots of template args in a grid to check validity.

template<typename T> concept has_dtype = requires { typename T::dtype; };
template<typename T>  struct gmem_wrapper    { using dtype = kittens::bf16; };
template<has_dtype T> struct gmem_wrapper<T> { using dtype = typename T::dtype; };
template<typename T> using gmem_dtype = typename gmem_wrapper<T>::dtype;

template<typename T> concept has_rt_shape = requires { typename T::rt_shape; };
template<typename T> struct rt_shape_wrapper { using rt_shape = kittens::ducks::rt_shape::rt_32x16; };
template<has_rt_shape T> struct rt_shape_wrapper<T> { using rt_shape = typename T::rt_shape; };
template<typename T> using rt_shape = typename rt_shape_wrapper<T>::rt_shape;

template<typename T> concept has_st_shape = requires { typename T::st_shape; };
template<typename T> struct st_shape_wrapper { using st_shape = kittens::ducks::st_shape::st_32x16; };
template<has_st_shape T> struct st_shape_wrapper<T> { using st_shape = typename T::st_shape; };
template<typename T> using st_shape = typename st_shape_wrapper<T>::st_shape;

// ----- 1D Wrappers -----

template<typename Ker, typename RT_SHAPE, typename ST_SHAPE, typename dtype, int S, int NW, kittens::ducks::gl::all GL, typename... args>
static __global__ void global_wrapper_1d(GL input, const GL output) {
    Ker::template device_func<RT_SHAPE, ST_SHAPE, dtype, S, NW, GL, args...>(input, output);
}
template<typename test, typename RT_SHAPE, typename ST_SHAPE, int S, int NUM_WORKERS, typename... args>
struct wrapper_1d {
    using dtype = gmem_dtype<test>; // defaults to bf16 in global memory if the test doesn't specify.
    static void run(test_data& results) {
        test_info this_result;
        this_result.label = generate_test_name<RT_SHAPE,ST_SHAPE,S,NUM_WORKERS,args...>(test::test_identifier);
        if constexpr (test::template valid<RT_SHAPE, ST_SHAPE, S, NUM_WORKERS, args...>::value) {
            constexpr int SIZE = S*RT_SHAPE::cols;
            // initialize
            dtype *d_i, *d_o;
            std::vector<float> i_ref(SIZE);
            std::vector<float> o_ref(SIZE);
            initialize(&d_i, &d_o, i_ref, o_ref);
            // make descriptors
            using GL = typename kittens::gl<dtype, 1, 1, 1, S*RT_SHAPE::cols>;
            GL input(d_i, nullptr, nullptr, nullptr, nullptr);
            GL output(d_o, nullptr, nullptr, nullptr, nullptr);
            // run kernel
            hipFuncSetAttribute(
                reinterpret_cast<void *>(global_wrapper_1d<test, RT_SHAPE, ST_SHAPE, dtype, S, NUM_WORKERS, GL, args...>),
                hipFuncAttributeMaxDynamicSharedMemorySize,
                kittens::MAX_SHARED_MEMORY / 2
            );
            global_wrapper_1d<test, RT_SHAPE, ST_SHAPE, dtype, S, NUM_WORKERS, GL, args...><<<1, NUM_WORKERS*kittens::WARP_THREADS, kittens::MAX_SHARED_MEMORY / 2>>>(input, output);
            // fill in correct results on cpu
            test::template host_func<RT_SHAPE, ST_SHAPE, S, NUM_WORKERS, GL, args...>(i_ref, o_ref);
            // check and cleanup
            this_result.result = validate(d_i, d_o, i_ref, o_ref, this_result.label, S*RT_SHAPE::cols);
        }
        else {
            this_result.result = test_result::INVALID;
        }
        results.push_back(this_result);
    }
};
template<typename test, typename RT_SHAPE, typename ST_SHAPE, int S, int NUM_WORKERS=1, typename... args> using wrapper_1d_warp      = wrapper_1d<test, RT_SHAPE, ST_SHAPE, S, NUM_WORKERS, args...>;
template<typename test, typename RT_SHAPE, typename ST_SHAPE, int S, int NUM_WORKERS=4, typename... args> using wrapper_1d_warpgroup = wrapper_1d<test, RT_SHAPE, ST_SHAPE, S, NUM_WORKERS, args...>;
template<typename test, typename RT_SHAPE, typename ST_SHAPE, int S, int NUM_WORKERS=8, typename... args> using wrapper_1d_block     = wrapper_1d<test, RT_SHAPE, ST_SHAPE, S, NUM_WORKERS, args...>;
template<typename test, typename RT_SHAPE, typename ST_SHAPE, int MAX_S=8, int NUM_WORKERS=1, typename... args> 
using sweep_size_1d = loop_s<wrapper_1d, test, RT_SHAPE, ST_SHAPE, MAX_S, NUM_WORKERS, MAX_S, args...>;
template<typename test, typename RT_SHAPE, typename ST_SHAPE, int MAX_S=8, int NUM_WORKERS=1, typename... args> using sweep_size_1d_warp = sweep_size_1d<test, RT_SHAPE, ST_SHAPE, MAX_S, NUM_WORKERS, args...>;


template<template<typename> typename test, typename RT_SHAPE, typename ST_SHAPE, int MAX_S=8, int NUM_WORKERS=1, typename... args>
struct sweep_gmem_type_1d {
    static void run(test_data &results) {
        sweep_size_1d<test<float>, RT_SHAPE, ST_SHAPE, MAX_S, NUM_WORKERS, args...>::run(results);
        sweep_size_1d<test<kittens::bf16>, RT_SHAPE, ST_SHAPE, MAX_S, NUM_WORKERS, args...>::run(results);
        sweep_size_1d<test<kittens::half>, RT_SHAPE, ST_SHAPE, MAX_S, NUM_WORKERS, args...>::run(results);
    }
};
template<template<typename> typename test, typename RT_SHAPE, typename ST_SHAPE, int MAX_S=8, typename... args> using sweep_gmem_type_1d_warp = sweep_gmem_type_1d<test, RT_SHAPE, ST_SHAPE, MAX_S, 1, args...>;

// ----- 2D Wrappers -----

template<typename Ker, typename RT_SHAPE, typename ST_SHAPE, typename dtype, int H, int W, int NW, typename G, typename... args>
static __global__ void global_wrapper_2d(const G input, const G output) {
    Ker::template device_func<RT_SHAPE, ST_SHAPE, dtype, H, W, NW, G, args...>(input, output);
}
template<typename test, typename RT_SHAPE, typename ST_SHAPE, int H, int W, int NUM_WORKERS=1, typename... args>
struct wrapper_2d {
    using dtype = gmem_dtype<test>; // defaults to bf16 in global memory if the test doesn't specify.
    static void run(test_data& results) {
        test_info this_result;
        this_result.label = generate_test_name<RT_SHAPE, ST_SHAPE, H, W, NUM_WORKERS, args...>(test::test_identifier);
        if constexpr (test::template valid<RT_SHAPE, ST_SHAPE, H, W, NUM_WORKERS, args...>::value) {
            constexpr int SIZE = H*W*RT_SHAPE::cols*RT_SHAPE::rows;
            // initialize
            dtype *d_i, *d_o;
            std::vector<float> i_ref(SIZE);
            std::vector<float> o_ref(SIZE);
            initialize(&d_i, &d_o, i_ref, o_ref);
            // make descriptors
            using GL = typename kittens::gl<dtype, 1, 1, H*RT_SHAPE::rows, W*RT_SHAPE::cols>;
            GL input(d_i, nullptr, nullptr, nullptr, nullptr);
            GL output(d_o, nullptr, nullptr, nullptr, nullptr);
            // run kernel
            hipFuncSetAttribute(
                reinterpret_cast<const void*>(global_wrapper_2d<test, RT_SHAPE, ST_SHAPE, dtype, H, W, NUM_WORKERS, GL, args...>),
                hipFuncAttributeMaxDynamicSharedMemorySize,
                kittens::MAX_SHARED_MEMORY / 2
            );
            global_wrapper_2d<test, RT_SHAPE, ST_SHAPE, dtype, H, W, NUM_WORKERS, GL, args...><<<1, NUM_WORKERS*kittens::WARP_THREADS, kittens::MAX_SHARED_MEMORY / 2>>>(input, output);
            // fill in correct results on cpu
            test::template host_func<RT_SHAPE, ST_SHAPE, H, W, NUM_WORKERS, GL, args...>(i_ref, o_ref);

            // check and cleanup
            int is_fp8 = (this_result.label.find("fp8") != std::string::npos) || (this_result.label.find("e4m3") != std::string::npos) || (this_result.label.find("e5m2") != std::string::npos);
            this_result.result = validate(d_i, d_o, i_ref, o_ref, this_result.label, W*RT_SHAPE::cols, is_fp8 ? 0.1 : 1e-2); // mma's sometimes produce small errors. this appears to be hardware.
        }
        else {
            this_result.result = test_result::INVALID;
        }
        results.push_back(this_result);
    }
};
template<typename test, typename RT_SHAPE, typename ST_SHAPE, int H, int W, int NUM_WORKERS=1, typename... args> using wrapper_2d_warp = wrapper_2d<test, RT_SHAPE, ST_SHAPE, H, W, NUM_WORKERS, args...>;

template<typename test, typename RT_SHAPE, typename ST_SHAPE, int MAX_H=8, int MAX_W=8, int NUM_WORKERS=1, typename... args> using sweep_size_2d = loop_h<wrapper_2d, test, RT_SHAPE, ST_SHAPE, MAX_H, MAX_W, NUM_WORKERS, MAX_H, args...>;
template<typename test, typename RT_SHAPE, typename ST_SHAPE, int MAX_H=8, int MAX_W=8, int NUM_WORKERS=1, typename... args> using sweep_size_2d_warp = sweep_size_2d<test, RT_SHAPE, ST_SHAPE, MAX_H, MAX_W, NUM_WORKERS, args...>;

template<template<typename> typename test, typename RT_SHAPE, typename ST_SHAPE, int MAX_H=8, int MAX_W=8, int NUM_WORKERS=1, typename... args>
struct sweep_gmem_type_2d {
    static void run(test_data &results) {
        sweep_size_2d<test<float>, RT_SHAPE, ST_SHAPE, MAX_H, MAX_W, NUM_WORKERS, args...>::run(results);
        sweep_size_2d<test<kittens::bf16>, RT_SHAPE, ST_SHAPE, MAX_H, MAX_W, NUM_WORKERS, args...>::run(results);
        sweep_size_2d<test<kittens::half>, RT_SHAPE, ST_SHAPE, MAX_H, MAX_W, NUM_WORKERS, args...>::run(results);
    }
};
template<template<typename> typename test, typename RT_SHAPE, typename ST_SHAPE, int MAX_H=8, int MAX_W=8, int NUM_WORKERS=1, typename... args> using sweep_gmem_type_2d_warp = sweep_gmem_type_2d<test, RT_SHAPE, ST_SHAPE, MAX_H, MAX_W, NUM_WORKERS, args...>;

template<typename T> concept gl_t = kittens::ducks::gl::all<T>;