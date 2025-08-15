#include "mma.cuh"

#ifdef TEST_WARP_REGISTER_TILE_MMA

#ifdef KITTENS_CDNA4
using accum_layout = kittens::ducks::rt_layout::accumulator;
#else
using accum_layout = kittens::ducks::rt_layout::col;
#endif

struct test_mma_AB {
    static constexpr int rt_rows = kittens::TILE_ROW_DIM<kittens::bf16>;
    static constexpr int rt_cols = kittens::TILE_COL_DIM<kittens::bf16>;
    template<int H, int W, int NW, typename K> using valid = std::bool_constant<NW == 1 && (2*W*H+W*K::value+H*K::value)<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_mma_AB";
    template<int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename _K> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        constexpr int K = _K::value;
        for(int i = 0; i < H*rt_rows; i++) {
            for(int j = 0; j < W*rt_rows; j++) {
                float sum = 0;
                for(int k = 0; k < K*rt_cols; k++) {
                    sum += i_ref[i*rt_cols*K + k]*i_ref[(rt_rows*rt_cols*H*K) + k*rt_rows*W + j];
                }
                o_ref[i*rt_rows*W + j] = sum;
            }
        }
    }
    template<int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename _K> __device__ static void device_func(const GTL_A &a_input, const GTL_B &b_input, const GTL_C &c_output) {
        constexpr int K = _K::value;
        kittens::rt_bf<rt_rows*H, rt_cols*K> a;
        kittens::rt_bf<rt_cols*K, rt_rows*W, kittens::ducks::rt_layout::col> b;
        kittens::rt_fl<rt_rows*H, rt_rows*W, accum_layout> c;
        kittens::load(a, a_input, {});
        kittens::load(b, b_input, {});
        kittens::zero(c);
        kittens::mma_AB(c, a, b, c);
        kittens::store(c_output, c, {});
    }
    template<int H, int W, typename K> using make_a_layout = typename kittens::gl<kittens::bf16, 1, 1, rt_rows*H, rt_cols*K::value>;
    template<int H, int W, typename K> using make_b_layout = typename kittens::gl<kittens::bf16, 1, 1, rt_cols*K::value, rt_rows*W>;
    template<int H, int W, typename K> using make_c_layout = typename kittens::gl<kittens::bf16, 1, 1, rt_rows*H, rt_rows*W>;
};
struct test_mma_ABt {
    static constexpr int rt_rows = kittens::TILE_ROW_DIM<kittens::bf16>;
    static constexpr int rt_cols = kittens::TILE_COL_DIM<kittens::bf16>;
    template<int H, int W, int NW, typename K> using valid = std::bool_constant<NW == 1 && (2*W*H+W*K::value+H*K::value)<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_mma_ABt";
    template<int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename _K> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        constexpr int K = _K::value;
        for(int i = 0; i < H*rt_rows; i++) {
            for(int j = 0; j < W*rt_rows; j++) {
                float sum = 0;
                for(int k = 0; k < K*rt_cols; k++) {
                    sum += i_ref[i*K*rt_cols+k]*i_ref[rt_rows*rt_cols*K*H + j*K*rt_cols+k];
                }
                o_ref[i*W*rt_rows+j] = sum;
            }
        }
    }
    template<int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename _K> __device__ static void device_func(const GTL_A &a_input, const GTL_B &b_input, const GTL_C &c_output) {
        constexpr int K = _K::value;
        kittens::rt_bf<rt_rows*H, rt_cols*K> a;
        kittens::rt_bf<rt_rows*W, rt_cols*K> b;
        kittens::rt_fl<rt_rows*H, rt_rows*W, accum_layout> c;
        kittens::load(a, a_input, {});
        kittens::load(b, b_input, {});
        kittens::zero(c);
        kittens::mma_ABt(c, a, b, c);
        kittens::store(c_output, c, {});
    }
    template<int H, int W, typename K> using make_a_layout = typename kittens::gl<kittens::bf16, 1, 1, rt_rows*H, rt_cols*K::value>;
    template<int H, int W, typename K> using make_b_layout = typename kittens::gl<kittens::bf16, 1, 1, rt_rows*W, rt_cols*K::value>;
    template<int H, int W, typename K> using make_c_layout = typename kittens::gl<kittens::bf16, 1, 1, rt_rows*H, rt_rows*W>;
};
struct test_mma_ABt_fp8 {
    static constexpr int rt_rows = kittens::TILE_ROW_DIM<kittens::fp8e4m3>;
    static constexpr int rt_cols = kittens::TILE_COL_DIM<kittens::fp8e4m3>;
    static constexpr int out_rt_rows = kittens::TILE_ROW_DIM<kittens::bf16>;
    static constexpr int out_rt_cols = kittens::TILE_COL_DIM<kittens::bf16>;
    template<int H, int W, int NW, typename K> using valid = std::bool_constant<NW == 1 && (2*W*H+W*K::value*rt_cols/32+H*K::value*rt_cols/32)<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_mma_ABt_fp8";
    template<int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename _K> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        constexpr int K = _K::value;
        for(int i = 0; i < H*rt_rows; i++) {
            for(int j = 0; j < W*rt_rows; j++) {
                float sum = 0;
                for(int k = 0; k < K*rt_cols; k++) {
                    sum += i_ref[i*K*rt_cols+k]*i_ref[rt_rows*rt_cols*K*H + j*K*rt_cols+k];
                }
                o_ref[i*W*rt_rows+j] = sum;
            }
        }
    }
    template<int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename _K> __device__ static void device_func(const GTL_A &a_input, const GTL_B &b_input, const GTL_C &c_output) {
        constexpr int K = _K::value;
        kittens::rt_fp8e4m3<rt_rows*H, rt_cols*K> a;
        kittens::rt_fp8e4m3<rt_rows*W, rt_cols*K> b;
        kittens::rt_fl<rt_rows*H, rt_rows*W, accum_layout> c;
        kittens::load(a, a_input, {});
        kittens::load(b, b_input, {});
        kittens::zero(c);
        kittens::mma_ABt(c, a, b, c);
        kittens::store(c_output, c, {});
    }
    template<int H, int W, typename K> using make_a_layout = typename kittens::gl<kittens::fp8e4m3, 1, 1, rt_rows*H, rt_cols*K::value>;
    template<int H, int W, typename K> using make_b_layout = typename kittens::gl<kittens::fp8e4m3, 1, 1, rt_rows*W, rt_cols*K::value>;
    template<int H, int W, typename K> using make_c_layout = typename kittens::gl<kittens::bf16, 1, 1, out_rt_rows*H, out_rt_cols*W>;
};
struct test_mma_AtB {
    static constexpr int rt_rows = kittens::TILE_ROW_DIM<kittens::bf16>;
    static constexpr int rt_cols = kittens::TILE_COL_DIM<kittens::bf16>;
    template<int H, int W, int NW, typename K> using valid = std::bool_constant<NW == 1 && (2*W*H+W*K::value+H*K::value)<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_mma_AtB";
    template<int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename _K> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        constexpr int K = _K::value;
        for(int i = 0; i < H*rt_rows; i++) {
            for(int j = 0; j < W*rt_rows; j++) {
                float sum = 0;
                for(int k = 0; k < K*rt_cols; k++) {
                    sum += i_ref[i + k*rt_rows*H]*i_ref[(rt_rows*rt_cols*H*K) + k*rt_rows*W + j];
                }
                o_ref[i*rt_rows*W + j] = sum;
            }
        }
    }
    template<int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename _K> __device__ static void device_func(const GTL_A &a_input, const GTL_B &b_input, const GTL_C &c_output) {
        constexpr int K = _K::value;
        kittens::rt_bf<rt_cols*K, rt_rows*H, kittens::ducks::rt_layout::col> a;
        kittens::rt_bf<rt_cols*K, rt_rows*W, kittens::ducks::rt_layout::col> b;
        kittens::rt_fl<rt_rows*H, rt_rows*W, accum_layout> c;
        kittens::load(a, a_input, {});
        kittens::load(b, b_input, {});
        kittens::zero(c);
        kittens::mma_AtB(c, a, b, c);
        kittens::store(c_output, c, {});
    }
    template<int H, int W, typename K> using make_a_layout = typename kittens::gl<kittens::bf16, 1, 1, rt_cols*K::value, rt_rows*H>;
    template<int H, int W, typename K> using make_b_layout = typename kittens::gl<kittens::bf16, 1, 1, rt_cols*K::value, rt_rows*W>;
    template<int H, int W, typename K> using make_c_layout = typename kittens::gl<kittens::bf16, 1, 1, rt_rows*H, rt_rows*W>;
};
struct test_mma_AtBt {
    static constexpr int rt_rows = kittens::TILE_ROW_DIM<kittens::bf16>;
    static constexpr int rt_cols = kittens::TILE_COL_DIM<kittens::bf16>;
    template<int H, int W, int NW, typename K> using valid = std::bool_constant<NW == 1 && (2*W*H+W*K::value+H*K::value)<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_mma_AtBt";
    template<int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename _K> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        constexpr int K = _K::value;
        for(int i = 0; i < H*rt_rows; i++) {
            for(int j = 0; j < W*rt_rows; j++) {
                float sum = 0;
                for(int k = 0; k < K*rt_cols; k++) {
                    sum += i_ref[i+k*H*rt_rows]*i_ref[rt_rows*rt_cols*K*H + j*K*rt_cols+k];
                }
                o_ref[i*W*rt_rows+j] = sum;
            }
        }
    }
    template<int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename _K> __device__ static void device_func(const GTL_A &a_input, const GTL_B &b_input, const GTL_C &c_output) {
        constexpr int K = _K::value;
        kittens::rt_bf<rt_cols*K, rt_rows*H, kittens::ducks::rt_layout::col> a;
        kittens::rt_bf<rt_rows*W, rt_cols*K> b;
        kittens::rt_fl<rt_rows*H, rt_rows*W, accum_layout> c;
        kittens::load(a, a_input, {});
        kittens::load(b, b_input, {});
        kittens::zero(c);
        kittens::mma_AtBt(c, a, b, c);
        kittens::store(c_output, c, {});
    }
    template<int H, int W, typename K> using make_a_layout = typename kittens::gl<kittens::bf16, 1, 1, rt_cols*K::value, rt_rows*H>;
    template<int H, int W, typename K> using make_b_layout = typename kittens::gl<kittens::bf16, 1, 1, rt_rows*W, rt_cols*K::value>;
    template<int H, int W, typename K> using make_c_layout = typename kittens::gl<kittens::bf16, 1, 1, rt_rows*H, rt_rows*W>;
};

// Due to the strange sizes instantiated, we need a custom base wrapper here
template<typename Ker, typename T, int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename... args>
static __global__ void mma_global_wrapper_2d(const GTL_A a_input, const GTL_B b_input, GTL_C c_output) {
    Ker::template device_func<H, W, NW, GTL_A, GTL_B, GTL_C, args...>(a_input, b_input, c_output);
}
// Type wrapper template to separate I_T from other parameters
template<typename I_T>
struct mma_type_wrapper {
    template<typename test, int H, int W, int NUM_WORKERS, typename _K, typename... args>
    struct mma_wrapper_2d {
        static void run(test_data& results) {
            using namespace kittens;
            constexpr int K = _K::value;
            test_info this_result;
            this_result.label = generate_test_name<H,W,NUM_WORKERS,_K,args...>(test::test_identifier);
            if constexpr (test::template valid<H, W, NUM_WORKERS, _K, args...>::value) {
                // initialize
                I_T *d_i;
                kittens::bf16 *d_o;
                constexpr int rt_rows = kittens::TILE_ROW_DIM<I_T>;
                constexpr int rt_cols = kittens::TILE_COL_DIM<I_T>;
                std::vector<float> i_ref((H+W)*K*rt_rows*rt_cols);
                std::vector<float> o_ref(H*W*rt_rows*rt_rows);
                initialize<I_T, kittens::bf16>(&d_i, &d_o, i_ref, o_ref);
                // make descriptors
                using GTL_A = test::template make_a_layout<H, W, _K>;
                using GTL_B = test::template make_b_layout<H, W, _K>;
                using GTL_C = test::template make_c_layout<H, W, _K>;
                GTL_A a_input (d_i,           nullptr, nullptr, nullptr, nullptr);
                GTL_B b_input (d_i + H*K*rt_rows*rt_cols, nullptr, nullptr, nullptr, nullptr);
                GTL_C c_output(d_o,           nullptr, nullptr, nullptr, nullptr);
                // run kernel
                hipFuncSetAttribute(
                    reinterpret_cast<void *>(mma_global_wrapper_2d<test, I_T, H, W, NUM_WORKERS, GTL_A, GTL_B, GTL_C, _K, args...>),
                    hipFuncAttributeMaxDynamicSharedMemorySize,
                    kittens::MAX_SHARED_MEMORY
                );
                mma_global_wrapper_2d<test, I_T, H, W, NUM_WORKERS, GTL_A, GTL_B, GTL_C, _K, args...><<<1, NUM_WORKERS*64, kittens::MAX_SHARED_MEMORY>>>(a_input, b_input, c_output);
                // fill in correct results on cpu
                test::template host_func<H, W, NUM_WORKERS, GTL_A, GTL_B, GTL_C, _K, args...>(i_ref, o_ref);
                // check and cleanup
                this_result.result = validate(d_i, d_o, i_ref, o_ref, this_result.label, W*rt_rows, std::is_same_v<I_T, kittens::fp8e4m3> ? 0.125 : 0.0625); 
            }
            else {
                this_result.result = test_result::INVALID;
            }
            results.push_back(this_result);
        }
    };
};
// Type-parameterized aliases for different data types
template<typename I_T, typename test, int MAX_H=8, int MAX_W=8, int NUM_WORKERS=1, typename... args>
using mma_sweep_size_typed = loop_h<mma_type_wrapper<I_T>::template mma_wrapper_2d, test, MAX_H, MAX_W, NUM_WORKERS, MAX_H, args...>;

template<typename I_T, typename test, int MAX_H=8, int MAX_W=8, typename... args>
using mma_sweep_size_warp_typed = mma_sweep_size_typed<I_T, test, MAX_H, MAX_W, 1, args...>;

template<typename test, int MAX_H=8, int MAX_W=8, int NUM_WORKERS=1, typename... args>
using mma_sweep_size_fp8 = mma_sweep_size_typed<kittens::fp8e4m3, test, MAX_H, MAX_W, NUM_WORKERS, args...>;

template<typename test, int MAX_H=8, int MAX_W=8, typename... args>
using mma_sweep_size_warp_fp8 = mma_sweep_size_warp_typed<kittens::fp8e4m3, test, MAX_H, MAX_W, args...>;

// Default is bf16
template<typename test, int MAX_H=8, int MAX_W=8, int NUM_WORKERS=1, typename... args>
using mma_sweep_size = mma_sweep_size_typed<kittens::bf16, test, MAX_H, MAX_W, NUM_WORKERS, args...>;

template<typename test, int MAX_H=8, int MAX_W=8, typename... args>
using mma_sweep_size_warp = mma_sweep_size_warp_typed<kittens::bf16, test, MAX_H, MAX_W, args...>;

void warp::reg::tile::mma::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/register/tile/mma tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
    // bf16
    mma_sweep_size_warp<test_mma_AB, SIZE, SIZE, std::integral_constant<int, 1>>::run(results);
    mma_sweep_size_warp<test_mma_AB, SIZE, SIZE, std::integral_constant<int, 2>>::run(results);
    mma_sweep_size_warp<test_mma_AB, SIZE, SIZE, std::integral_constant<int, 3>>::run(results);
    mma_sweep_size_warp<test_mma_AB, SIZE, SIZE, std::integral_constant<int, 4>>::run(results);
    mma_sweep_size_warp<test_mma_ABt, SIZE, SIZE, std::integral_constant<int, 1>>::run(results);
    mma_sweep_size_warp<test_mma_ABt, SIZE, SIZE, std::integral_constant<int, 2>>::run(results);
    mma_sweep_size_warp<test_mma_ABt, SIZE, SIZE, std::integral_constant<int, 3>>::run(results);
    mma_sweep_size_warp<test_mma_ABt, SIZE, SIZE, std::integral_constant<int, 4>>::run(results);
    mma_sweep_size_warp<test_mma_AtB, SIZE, SIZE, std::integral_constant<int, 1>>::run(results);
    mma_sweep_size_warp<test_mma_AtB, SIZE, SIZE, std::integral_constant<int, 2>>::run(results);
    mma_sweep_size_warp<test_mma_AtB, SIZE, SIZE, std::integral_constant<int, 3>>::run(results);
    mma_sweep_size_warp<test_mma_AtB, SIZE, SIZE, std::integral_constant<int, 4>>::run(results);
    mma_sweep_size_warp<test_mma_AtBt, SIZE, SIZE, std::integral_constant<int, 1>>::run(results);
    mma_sweep_size_warp<test_mma_AtBt, SIZE, SIZE, std::integral_constant<int, 2>>::run(results);
    mma_sweep_size_warp<test_mma_AtBt, SIZE, SIZE, std::integral_constant<int, 3>>::run(results);
    mma_sweep_size_warp<test_mma_AtBt, SIZE, SIZE, std::integral_constant<int, 4>>::run(results);
    // fp8e4m3
    mma_sweep_size_warp_fp8<test_mma_ABt_fp8, SIZE, SIZE, std::integral_constant<int, 1>>::run(results);
    mma_sweep_size_warp_fp8<test_mma_ABt_fp8, SIZE, SIZE, std::integral_constant<int, 2>>::run(results);
    mma_sweep_size_warp_fp8<test_mma_ABt_fp8, SIZE, SIZE, std::integral_constant<int, 3>>::run(results);
    mma_sweep_size_warp_fp8<test_mma_ABt_fp8, SIZE, SIZE, std::integral_constant<int, 4>>::run(results);
}

#endif