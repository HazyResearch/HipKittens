#include "mma.cuh"

#ifdef TEST_WARP_REGISTER_TILE_MMA

#ifdef KITTENS_CDNA4
#define ROWS 32
#define COLS 32
#else
#define ROWS 16
#define COLS 16
#endif

struct test_mma_AB {
    template<int H, int W, int NW, typename K> using valid = std::bool_constant<NW == 1 && (2*W*H+W*K::value+H*K::value)<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_mma_AB";
    template<int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename _K> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        constexpr int K = _K::value;
        for(int i = 0; i < H*ROWS; i++) {
            for(int j = 0; j < W*ROWS; j++) {
                float sum = 0;
                for(int k = 0; k < K*COLS; k++) {
                    sum += i_ref[i*COLS*K + k]*i_ref[(ROWS*COLS*H*K) + k*ROWS*W + j];
                }
                o_ref[i*ROWS*W + j] = sum;
            }
        }
    }
    template<int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename _K> __device__ static void device_func(const GTL_A &a_input, const GTL_B &b_input, const GTL_C &c_output) {
        constexpr int K = _K::value;
        kittens::rt_bf<ROWS*H, COLS*K> a;
        kittens::rt_bf<COLS*K, ROWS*W, kittens::ducks::rt_layout::col> b;
#ifdef KITTENS_CDNA4
        kittens::rt_fl<ROWS*H, ROWS*W, kittens::ducks::rt_layout::accumulator_col> c;
#else
        kittens::rt_fl<ROWS*H, ROWS*W, kittens::ducks::rt_layout::col> c;
#endif
        kittens::load(a, a_input, {});
        kittens::load(b, b_input, {});
        kittens::zero(c);
        kittens::mma_AB(c, a, b, c);
        kittens::store(c_output, c, {});
    }
    template<int H, int W, typename K> using make_a_layout = typename kittens::gl<kittens::bf16, 1, 1, ROWS*H, COLS*K::value>;
    template<int H, int W, typename K> using make_b_layout = typename kittens::gl<kittens::bf16, 1, 1, COLS*K::value, ROWS*W>;
    template<int H, int W, typename K> using make_c_layout = typename kittens::gl<kittens::bf16, 1, 1, ROWS*H, ROWS*W>;
};
struct test_mma_ABt {
    template<int H, int W, int NW, typename K> using valid = std::bool_constant<NW == 1 && (2*W*H+W*K::value+H*K::value)<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_mma_ABt";
    template<int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename _K> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        constexpr int K = _K::value;
        for(int i = 0; i < H*ROWS; i++) {
            for(int j = 0; j < W*ROWS; j++) {
                float sum = 0;
                for(int k = 0; k < K*COLS; k++) {
                    sum += i_ref[i*K*COLS+k]*i_ref[ROWS*COLS*K*H + j*K*COLS+k];
                }
                o_ref[i*W*ROWS+j] = sum;
            }
        }
    }
    template<int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename _K> __device__ static void device_func(const GTL_A &a_input, const GTL_B &b_input, const GTL_C &c_output) {
        constexpr int K = _K::value;
        kittens::rt_bf<ROWS*H, COLS*K> a;
        kittens::rt_bf<ROWS*W, COLS*K> b;
#ifdef KITTENS_CDNA4
        kittens::rt_fl<ROWS*H, ROWS*W, kittens::ducks::rt_layout::accumulator_col> c;
#else
        kittens::rt_fl<ROWS*H, ROWS*W, kittens::ducks::rt_layout::col> c;
#endif
        kittens::load(a, a_input, {});
        kittens::load(b, b_input, {});
        kittens::zero(c);
        kittens::mma_ABt(c, a, b, c);
        kittens::store(c_output, c, {});
    }
    template<int H, int W, typename K> using make_a_layout = typename kittens::gl<kittens::bf16, 1, 1, ROWS*H, COLS*K::value>;
    template<int H, int W, typename K> using make_b_layout = typename kittens::gl<kittens::bf16, 1, 1, ROWS*W, COLS*K::value>;
    template<int H, int W, typename K> using make_c_layout = typename kittens::gl<kittens::bf16, 1, 1, ROWS*H, ROWS*W>;
};
struct test_mma_AtB {
    template<int H, int W, int NW, typename K> using valid = std::bool_constant<NW == 1 && (2*W*H+W*K::value+H*K::value)<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_mma_AtB";
    template<int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename _K> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        constexpr int K = _K::value;
        for(int i = 0; i < H*ROWS; i++) {
            for(int j = 0; j < W*ROWS; j++) {
                float sum = 0;
                for(int k = 0; k < K*COLS; k++) {
                    sum += i_ref[i + k*COLS*H]*i_ref[(ROWS*COLS*H*K) + k*ROWS*W + j];
                }
                o_ref[i*ROWS*W + j] = sum;
            }
        }
    }
    template<int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename _K> __device__ static void device_func(const GTL_A &a_input, const GTL_B &b_input, const GTL_C &c_output) {
        constexpr int K = _K::value;
        kittens::rt_bf<COLS*K, ROWS*H, kittens::ducks::rt_layout::col> a;
        kittens::rt_bf<COLS*K, ROWS*W, kittens::ducks::rt_layout::col> b;
#ifdef KITTENS_CDNA4
        kittens::rt_fl<ROWS*H, ROWS*W, kittens::ducks::rt_layout::accumulator_col> c;
#else
        kittens::rt_fl<ROWS*H, ROWS*W, kittens::ducks::rt_layout::col> c;
#endif
        kittens::load(a, a_input, {});
        kittens::load(b, b_input, {});
        kittens::zero(c);
        kittens::mma_AtB(c, a, b, c);
        kittens::store(c_output, c, {});
    }
    template<int H, int W, typename K> using make_a_layout = typename kittens::gl<kittens::bf16, 1, 1, COLS*K::value, ROWS*H>;
    template<int H, int W, typename K> using make_b_layout = typename kittens::gl<kittens::bf16, 1, 1, COLS*K::value, ROWS*W>;
    template<int H, int W, typename K> using make_c_layout = typename kittens::gl<kittens::bf16, 1, 1, ROWS*H, ROWS*W>;
};
struct test_mma_AtBt {
    template<int H, int W, int NW, typename K> using valid = std::bool_constant<NW == 1 && (2*W*H+W*K::value+H*K::value)<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_mma_AtBt";
    template<int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename _K> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        constexpr int K = _K::value;
        for(int i = 0; i < H*ROWS; i++) {
            for(int j = 0; j < W*ROWS; j++) {
                float sum = 0;
                for(int k = 0; k < K*COLS; k++) {
                    sum += i_ref[i+k*H*ROWS]*i_ref[ROWS*COLS*K*H + j*K*COLS+k];
                }
                o_ref[i*W*ROWS+j] = sum;
            }
        }
    }
    template<int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename _K> __device__ static void device_func(const GTL_A &a_input, const GTL_B &b_input, const GTL_C &c_output) {
        constexpr int K = _K::value;
        kittens::rt_bf<COLS*K, ROWS*H, kittens::ducks::rt_layout::col> a;
        kittens::rt_bf<ROWS*W, COLS*K> b;
#ifdef KITTENS_CDNA4
        kittens::rt_fl<ROWS*H, ROWS*W, kittens::ducks::rt_layout::accumulator_col> c;
#else
        kittens::rt_fl<ROWS*H, ROWS*W, kittens::ducks::rt_layout::col> c;
#endif
        kittens::load(a, a_input, {});
        kittens::load(b, b_input, {});
        kittens::zero(c);
        kittens::mma_AtBt(c, a, b, c);
        kittens::store(c_output, c, {});
    }
    template<int H, int W, typename K> using make_a_layout = typename kittens::gl<kittens::bf16, 1, 1, COLS*K::value, ROWS*H>;
    template<int H, int W, typename K> using make_b_layout = typename kittens::gl<kittens::bf16, 1, 1, ROWS*W, COLS*K::value>;
    template<int H, int W, typename K> using make_c_layout = typename kittens::gl<kittens::bf16, 1, 1, ROWS*H, ROWS*W>;
};

// Due to the strange sizes instantiated, we need a custom base wrapper here
template<typename Ker, typename T, int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename... args>
static __global__ void mma_global_wrapper_2d(const GTL_A a_input, const GTL_B b_input, GTL_C c_output) {
    Ker::template device_func<H, W, NW, GTL_A, GTL_B, GTL_C, args...>(a_input, b_input, c_output);
}
template<typename test, int H, int W, int NUM_WORKERS, typename _K, typename... args>
struct mma_wrapper_2d {
    static void run(test_data& results) {
        using namespace kittens;
        constexpr int K = _K::value;
        test_info this_result;
        this_result.label = generate_test_name<H,W,NUM_WORKERS,_K,args...>(test::test_identifier);
        if constexpr (test::template valid<H, W, NUM_WORKERS, _K, args...>::value) {
            // initialize
            kittens::bf16 *d_i, *d_o;
            std::vector<float> i_ref((H+W)*K*ROWS*COLS);
            std::vector<float> o_ref(H*W*ROWS*ROWS);
            initialize(&d_i, &d_o, i_ref, o_ref);
            // make descriptors
            using GTL_A = test::template make_a_layout<H, W, _K>;
            using GTL_B = test::template make_b_layout<H, W, _K>;
            using GTL_C = test::template make_c_layout<H, W, _K>;
            GTL_A a_input (d_i,           nullptr, nullptr, nullptr, nullptr);
            GTL_B b_input (d_i + H*K*ROWS*COLS, nullptr, nullptr, nullptr, nullptr);
            GTL_C c_output(d_o,           nullptr, nullptr, nullptr, nullptr);
            // run kernel
            hipFuncSetAttribute(
                reinterpret_cast<void *>(mma_global_wrapper_2d<test, kittens::bf16, H, W, NUM_WORKERS, GTL_A, GTL_B, GTL_C, _K, args...>),
                hipFuncAttributeMaxDynamicSharedMemorySize,
                kittens::MAX_SHARED_MEMORY
            );
            mma_global_wrapper_2d<test, kittens::bf16, H, W, NUM_WORKERS, GTL_A, GTL_B, GTL_C, _K, args...><<<1, NUM_WORKERS*64, kittens::MAX_SHARED_MEMORY>>>(a_input, b_input, c_output);
            // fill in correct results on cpu
            test::template host_func<H, W, NUM_WORKERS, GTL_A, GTL_B, GTL_C, _K, args...>(i_ref, o_ref);
            // check and cleanup
            this_result.result = validate(d_i, d_o, i_ref, o_ref, this_result.label, W*ROWS, 0.10); // mma's sometimes produce small errors. this appears to be hardware.
        }
        else {
            this_result.result = test_result::INVALID;
        }
        results.push_back(this_result);
    }
};
template<typename test, int MAX_H=8, int MAX_W=8, int NUM_WORKERS=1, typename... args> using mma_sweep_size = loop_h<mma_wrapper_2d, test, MAX_H, MAX_W, NUM_WORKERS, MAX_H, args...>;
template<typename test, int MAX_H=8, int MAX_W=8, typename... args> using mma_sweep_size_warp = mma_sweep_size<test, MAX_H, MAX_W, 1, args...>;

void warp::reg::tile::mma::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/register/tile/mma tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_0 ? 1  :
                         INTENSITY_1 ? 2  :
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
}

#endif