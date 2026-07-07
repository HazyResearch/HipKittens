#include "mma.cuh"

#ifdef TEST_WARP_REGISTER_TILE_MMA

namespace gfx1250_mma_test {

// WMMA accumulator store (matches gfx1250_gemm::store_acc16 in common.h).
__device__ inline void store_acc16(
    kittens::bf16* __restrict__ c_global,
    int gr_base, int gc_base, int N,
    const kittens::rt_base<float, kittens::ducks::rt_layout::col,
                            kittens::ducks::rt_shape::rt_16x16>& tile)
{
    const int L    = kittens::laneid();
    const int half = L / 16;
    const int col  = L % 16;
    const int gc   = gc_base + col;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int gr0 = gr_base + 2 * k     + 8 * half;
        const int gr1 = gr_base + 2 * k + 1 + 8 * half;
        c_global[gr0 * N + gc] =
            kittens::base_types::convertor<kittens::bf16, float>::convert(tile.data[k].x);
        c_global[gr1 * N + gc] =
            kittens::base_types::convertor<kittens::bf16, float>::convert(tile.data[k].y);
    }
}

template<int H, int W, kittens::ducks::rt_shape::all RT_SHAPE_ACCUM>
__device__ inline void store_wmma_acc(
    kittens::bf16* __restrict__ c_global,
    const kittens::rt_fl<RT_SHAPE_ACCUM::rows * H, RT_SHAPE_ACCUM::cols * W,
                         kittens::ducks::rt_layout::col, RT_SHAPE_ACCUM>& c)
{
    constexpr int N = W * RT_SHAPE_ACCUM::cols;
    #pragma unroll
    for (int ti = 0; ti < H; ++ti) {
        #pragma unroll
        for (int tj = 0; tj < W; ++tj) {
            store_acc16(c_global, ti * RT_SHAPE_ACCUM::rows, tj * RT_SHAPE_ACCUM::cols,
                        N, c.tiles[ti][tj]);
        }
    }
}

} // namespace gfx1250_mma_test

struct test_mma_AB {
    template<typename RT_SHAPE_ACCUM, int K_DIM, int H, int W, int NW, typename K> using valid = std::bool_constant<NW == 1 && (2*W*H+W*K::value+H*K::value)<=64>;
    static inline const std::string test_identifier = "reg_mma_AB";
    template<typename RT_SHAPE_ACCUM, int K_DIM, int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename _K> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        constexpr int K = _K::value;
        for(int i = 0; i < H*RT_SHAPE_ACCUM::rows; i++) {
            for(int j = 0; j < W*RT_SHAPE_ACCUM::cols; j++) {
                float sum = 0;
                for(int k = 0; k < K*K_DIM; k++) {
                    sum += i_ref[i*K_DIM*K + k]*i_ref[(RT_SHAPE_ACCUM::rows*K_DIM*H*K) + k*RT_SHAPE_ACCUM::cols*W + j];
                }
                o_ref[i*RT_SHAPE_ACCUM::cols*W + j] = sum;
            }
        }
    }
    template<typename RT_SHAPE_ACCUM, int K_DIM, int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename _K> __device__ static void device_func(const GTL_A &a_input, const GTL_B &b_input, const GTL_C &c_output) {
        constexpr int K = _K::value;
        using A_SHAPE = std::conditional_t<std::is_same_v<RT_SHAPE_ACCUM, kittens::ducks::rt_shape::rt_32x32>, kittens::ducks::rt_shape::rt_32x16, kittens::ducks::rt_shape::rt_16x32>;
        using B_SHAPE = std::conditional_t<std::is_same_v<RT_SHAPE_ACCUM, kittens::ducks::rt_shape::rt_32x32>, kittens::ducks::rt_shape::rt_16x32, kittens::ducks::rt_shape::rt_32x16>;

        kittens::rt_bf<RT_SHAPE_ACCUM::rows*H, K_DIM*K, kittens::ducks::rt_layout::row, A_SHAPE> a;
        kittens::rt_bf<K_DIM*K, RT_SHAPE_ACCUM::cols*W, kittens::ducks::rt_layout::col, B_SHAPE> b;
        kittens::rt_fl<RT_SHAPE_ACCUM::rows*H, RT_SHAPE_ACCUM::cols*W, kittens::ducks::rt_layout::col, RT_SHAPE_ACCUM> c;

        kittens::load(a, a_input, {});
        kittens::load(b, b_input, {});
        kittens::sync::sync();
        kittens::zero(c);
        kittens::mma_AB(c, a, b, c);
        kittens::sched::wait_alu();
        kittens::store(c_output, c, {});
        kittens::sync::wait_store<0>();
    }
    template<typename RT_SHAPE_ACCUM, int K_DIM, int H, int W, typename K> using make_a_layout = typename kittens::gl<kittens::bf16, 1, 1, RT_SHAPE_ACCUM::rows*H, K_DIM*K::value>;
    template<typename RT_SHAPE_ACCUM, int K_DIM, int H, int W, typename K> using make_b_layout = typename kittens::gl<kittens::bf16, 1, 1, K_DIM*K::value, RT_SHAPE_ACCUM::cols*W>;
    template<typename RT_SHAPE_ACCUM, int K_DIM, int H, int W, typename K> using make_c_layout = typename kittens::gl<kittens::bf16, 1, 1, RT_SHAPE_ACCUM::rows*H, RT_SHAPE_ACCUM::cols*W>;
};

struct test_mma_ABt {
    template<typename RT_SHAPE_ACCUM, int K_DIM, int H, int W, int NW, typename K> using valid = std::bool_constant<
        NW == 1 && (2*W*H+W*K::value+H*K::value)<=64
        && std::is_same_v<RT_SHAPE_ACCUM, kittens::ducks::rt_shape::rt_16x16>
    >;
    static inline const std::string test_identifier = "reg_mma_ABt";
    template<typename RT_SHAPE_ACCUM, int K_DIM, int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename _K> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        constexpr int K = _K::value;
        for(int i = 0; i < H*RT_SHAPE_ACCUM::rows; i++) {
            for(int j = 0; j < W*RT_SHAPE_ACCUM::cols; j++) {
                float sum = 0;
                for(int k = 0; k < K*K_DIM; k++) {
                    sum += i_ref[i*K*K_DIM+k]*i_ref[RT_SHAPE_ACCUM::rows*K_DIM*K*H + j*K*K_DIM+k];
                }
                o_ref[i*W*RT_SHAPE_ACCUM::cols+j] = sum;
            }
        }
    }
    template<typename RT_SHAPE_ACCUM, int K_DIM, int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename _K> __device__ static void device_func(const GTL_A &a_input, const GTL_B &b_input, const GTL_C &c_output) {
        constexpr int K = _K::value;
        kittens::rt_bf<RT_SHAPE_ACCUM::rows * H, K_DIM * K, kittens::ducks::rt_layout::row,
                      kittens::ducks::rt_shape::rt_16x32> a;
        kittens::rt_bf<RT_SHAPE_ACCUM::cols * W, K_DIM * K, kittens::ducks::rt_layout::row,
                      kittens::ducks::rt_shape::rt_16x32> b;
        kittens::rt_fl<RT_SHAPE_ACCUM::rows * H, RT_SHAPE_ACCUM::cols * W,
                       kittens::ducks::rt_layout::col, RT_SHAPE_ACCUM> c;

        // Operands: gfx1250 WMMA row layout matches the buffer_load g2r path.
        kittens::load(a, a_input, {});
        kittens::load(b, b_input, {});
        kittens::sync::sync();
        kittens::zero(c);
        kittens::mma_ABt(c, a, b, c);
        kittens::sched::wait_alu();
        gfx1250_mma_test::store_wmma_acc<H, W, RT_SHAPE_ACCUM>(c_output.raw_ptr, c);
        kittens::sync::wait_store<0>();
    }
    template<typename RT_SHAPE_ACCUM, int K_DIM, int H, int W, typename K> using make_a_layout = typename kittens::gl<kittens::bf16, 1, 1, RT_SHAPE_ACCUM::rows*H, K_DIM*K::value>;
    template<typename RT_SHAPE_ACCUM, int K_DIM, int H, int W, typename K> using make_b_layout = typename kittens::gl<kittens::bf16, 1, 1, RT_SHAPE_ACCUM::cols*W, K_DIM*K::value>;
    template<typename RT_SHAPE_ACCUM, int K_DIM, int H, int W, typename K> using make_c_layout = typename kittens::gl<kittens::bf16, 1, 1, RT_SHAPE_ACCUM::rows*H, RT_SHAPE_ACCUM::cols*W>;
};

struct test_mma_AtB {
    template<typename RT_SHAPE_ACCUM, int K_DIM, int H, int W, int NW, typename K> using valid = std::bool_constant<NW == 1 && (2*W*H+W*K::value+H*K::value)<=64>;
    static inline const std::string test_identifier = "reg_mma_AtB";
    template<typename RT_SHAPE_ACCUM, int K_DIM, int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename _K> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        constexpr int K = _K::value;
        for(int i = 0; i < H*RT_SHAPE_ACCUM::rows; i++) {
            for(int j = 0; j < W*RT_SHAPE_ACCUM::cols; j++) {
                float sum = 0;
                for(int k = 0; k < K*K_DIM; k++) {
                    sum += i_ref[i + k*RT_SHAPE_ACCUM::rows*H]*i_ref[(RT_SHAPE_ACCUM::rows*K_DIM*H*K) + k*RT_SHAPE_ACCUM::cols*W + j];
                }
                o_ref[i*RT_SHAPE_ACCUM::cols*W + j] = sum;
            }
        }
    }
    template<typename RT_SHAPE_ACCUM, int K_DIM, int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename _K> __device__ static void device_func(const GTL_A &a_input, const GTL_B &b_input, const GTL_C &c_output) {
        constexpr int K = _K::value;
        using A_SHAPE = std::conditional_t<std::is_same_v<RT_SHAPE_ACCUM, kittens::ducks::rt_shape::rt_32x32>, kittens::ducks::rt_shape::rt_16x32, kittens::ducks::rt_shape::rt_32x16>;
        using B_SHAPE = std::conditional_t<std::is_same_v<RT_SHAPE_ACCUM, kittens::ducks::rt_shape::rt_32x32>, kittens::ducks::rt_shape::rt_16x32, kittens::ducks::rt_shape::rt_32x16>;

        kittens::rt_bf<K_DIM*K, RT_SHAPE_ACCUM::rows*H, kittens::ducks::rt_layout::col, A_SHAPE> a;
        kittens::rt_bf<K_DIM*K, RT_SHAPE_ACCUM::cols*W, kittens::ducks::rt_layout::col, B_SHAPE> b;
        kittens::rt_fl<RT_SHAPE_ACCUM::rows*H, RT_SHAPE_ACCUM::cols*W, kittens::ducks::rt_layout::col, RT_SHAPE_ACCUM> c;
        kittens::load(a, a_input, {});
        kittens::load(b, b_input, {});
        kittens::sync::sync();
        kittens::zero(c);
        kittens::mma_AtB(c, a, b, c);
        kittens::sched::wait_alu();
        kittens::store(c_output, c, {});
        kittens::sync::wait_store<0>();
    }
    template<typename RT_SHAPE_ACCUM, int K_DIM, int H, int W, typename K> using make_a_layout = typename kittens::gl<kittens::bf16, 1, 1, K_DIM*K::value, RT_SHAPE_ACCUM::rows*H>;
    template<typename RT_SHAPE_ACCUM, int K_DIM, int H, int W, typename K> using make_b_layout = typename kittens::gl<kittens::bf16, 1, 1, K_DIM*K::value, RT_SHAPE_ACCUM::cols*W>;
    template<typename RT_SHAPE_ACCUM, int K_DIM, int H, int W, typename K> using make_c_layout = typename kittens::gl<kittens::bf16, 1, 1, RT_SHAPE_ACCUM::rows*H, RT_SHAPE_ACCUM::cols*W>;
};

struct test_mma_AtBt {
    template<typename RT_SHAPE_ACCUM, int K_DIM, int H, int W, int NW, typename K> using valid = std::bool_constant<NW == 1 && (2*W*H+W*K::value+H*K::value)<=64>;
    static inline const std::string test_identifier = "reg_mma_AtBt";
    template<typename RT_SHAPE_ACCUM, int K_DIM, int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename _K> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        constexpr int K = _K::value;
        for(int i = 0; i < H*RT_SHAPE_ACCUM::rows; i++) {
            for(int j = 0; j < W*RT_SHAPE_ACCUM::cols; j++) {
                float sum = 0;
                for(int k = 0; k < K*K_DIM; k++) {
                    sum += i_ref[i+k*H*RT_SHAPE_ACCUM::rows]*i_ref[RT_SHAPE_ACCUM::rows*K_DIM*K*H + j*K*K_DIM+k];
                }
                o_ref[i*W*RT_SHAPE_ACCUM::cols+j] = sum;
            }
        }
    }
    template<typename RT_SHAPE_ACCUM, int K_DIM, int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename _K> __device__ static void device_func(const GTL_A &a_input, const GTL_B &b_input, const GTL_C &c_output) {
        constexpr int K = _K::value;
        using A_SHAPE = std::conditional_t<std::is_same_v<RT_SHAPE_ACCUM, kittens::ducks::rt_shape::rt_32x32>, kittens::ducks::rt_shape::rt_16x32, kittens::ducks::rt_shape::rt_32x16>;
        using B_SHAPE = std::conditional_t<std::is_same_v<RT_SHAPE_ACCUM, kittens::ducks::rt_shape::rt_32x32>, kittens::ducks::rt_shape::rt_32x16, kittens::ducks::rt_shape::rt_16x32>;

        kittens::rt_bf<K_DIM*K, RT_SHAPE_ACCUM::rows*H, kittens::ducks::rt_layout::col, A_SHAPE> a;
        kittens::rt_bf<RT_SHAPE_ACCUM::cols*W, K_DIM*K, kittens::ducks::rt_layout::row, B_SHAPE> b;
        kittens::rt_fl<RT_SHAPE_ACCUM::rows*H, RT_SHAPE_ACCUM::cols*W, kittens::ducks::rt_layout::col, RT_SHAPE_ACCUM> c;
        kittens::load(a, a_input, {});
        kittens::load(b, b_input, {});
        kittens::sync::sync();
        kittens::zero(c);
        kittens::mma_AtBt(c, a, b, c);
        kittens::sched::wait_alu();
        kittens::store(c_output, c, {});
        kittens::sync::wait_store<0>();
    }
    template<typename RT_SHAPE_ACCUM, int K_DIM, int H, int W, typename K> using make_a_layout = typename kittens::gl<kittens::bf16, 1, 1, K_DIM*K::value, RT_SHAPE_ACCUM::rows*H>;
    template<typename RT_SHAPE_ACCUM, int K_DIM, int H, int W, typename K> using make_b_layout = typename kittens::gl<kittens::bf16, 1, 1, RT_SHAPE_ACCUM::cols*W, K_DIM*K::value>;
    template<typename RT_SHAPE_ACCUM, int K_DIM, int H, int W, typename K> using make_c_layout = typename kittens::gl<kittens::bf16, 1, 1, RT_SHAPE_ACCUM::rows*H, RT_SHAPE_ACCUM::cols*W>;
};

template<typename Ker, typename T, typename RT_SHAPE_ACCUM, int K_DIM, int H, int W, int NW, gl_t GTL_A, gl_t GTL_B, gl_t GTL_C, typename... args>
static __global__ void mma_global_wrapper_2d(const GTL_A a_input, const GTL_B b_input, GTL_C c_output) {
    Ker::template device_func<RT_SHAPE_ACCUM, K_DIM, H, W, NW, GTL_A, GTL_B, GTL_C, args...>(a_input, b_input, c_output);
}

template<typename test, typename RT_SHAPE_ACCUM, typename ST_SHAPE, int H, int W, int NUM_WORKERS, typename _K, typename... args>
struct mma_wrapper_2d {
    static void run(test_data& results) {
        using namespace kittens;
        constexpr int K = _K::value;
        constexpr int K_DIM = std::is_same_v<RT_SHAPE_ACCUM, kittens::ducks::rt_shape::rt_32x32> ? 16 : 32;
        constexpr int MN_DIM = RT_SHAPE_ACCUM::rows;
        test_info this_result;

        this_result.label = generate_test_name<RT_SHAPE_ACCUM, H, W, NUM_WORKERS, _K>(test::test_identifier);
        this_result.label += "_[k=" + std::to_string(K) + "]";
        if constexpr (test::template valid<RT_SHAPE_ACCUM, K_DIM, H, W, NUM_WORKERS, _K, args...>::value) {
            kittens::bf16 *d_i, *d_o;
            std::vector<float> i_ref((H+W)*K*MN_DIM*K_DIM);
            std::vector<float> o_ref(H*W*MN_DIM*MN_DIM);
            initialize(&d_i, &d_o, i_ref, o_ref);
            using GTL_A = test::template make_a_layout<RT_SHAPE_ACCUM, K_DIM, H, W, _K>;
            using GTL_B = test::template make_b_layout<RT_SHAPE_ACCUM, K_DIM, H, W, _K>;
            using GTL_C = test::template make_c_layout<RT_SHAPE_ACCUM, K_DIM, H, W, _K>;
            GTL_A a_input (d_i,           nullptr, nullptr, nullptr, nullptr);
            GTL_B b_input (d_i + H*K*MN_DIM*K_DIM, nullptr, nullptr, nullptr, nullptr);
            GTL_C c_output(d_o,           nullptr, nullptr, nullptr, nullptr);
            hipFuncSetAttribute(
                reinterpret_cast<void *>(mma_global_wrapper_2d<test, kittens::bf16, RT_SHAPE_ACCUM, K_DIM, H, W, NUM_WORKERS, GTL_A, GTL_B, GTL_C, _K, args...>),
                hipFuncAttributeMaxDynamicSharedMemorySize,
                kittens::MAX_SHARED_MEMORY
            );
            mma_global_wrapper_2d<test, kittens::bf16, RT_SHAPE_ACCUM, K_DIM, H, W, NUM_WORKERS, GTL_A, GTL_B, GTL_C, _K, args...><<<1, NUM_WORKERS*kittens::WARP_THREADS, kittens::MAX_SHARED_MEMORY>>>(a_input, b_input, c_output);
            test::template host_func<RT_SHAPE_ACCUM, K_DIM, H, W, NUM_WORKERS, GTL_A, GTL_B, GTL_C, _K, args...>(i_ref, o_ref);
            this_result.result = validate(d_i, d_o, i_ref, o_ref, this_result.label, W*RT_SHAPE_ACCUM::cols, 0.10);
        }
        else {
            this_result.result = test_result::INVALID;
        }
        results.push_back(this_result);
    }
};

template<typename test, typename RT_SHAPE_ACCUM, typename ST_SHAPE=kittens::ducks::st_shape::st_16x16, int MAX_H=8, int MAX_W=8, int NUM_WORKERS=1, typename... args>
using mma_sweep_size = loop_h<mma_wrapper_2d, test, RT_SHAPE_ACCUM, ST_SHAPE, MAX_H, MAX_W, NUM_WORKERS, MAX_H, args...>;
template<typename test, typename RT_SHAPE_ACCUM, typename ST_SHAPE=kittens::ducks::st_shape::st_16x16, int MAX_H=8, int MAX_W=8, typename... args>
using mma_sweep_size_warp = mma_sweep_size<test, RT_SHAPE_ACCUM, ST_SHAPE, MAX_H, MAX_W, 1, args...>;

void warp::reg::tile::mma::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/register/tile/mma tests! -----\n" << std::endl;

    // gfx1250 native path: WMMA 16x16x32 via `mma_ABt` with rt_16x16 fp32
    // accumulators and rt_16x32 bf16 operands. The MFMA transpose variants
    // (AB/AtB/AtBt) target CDNA lane layouts and do not compile on gfx1250.
    constexpr int SIZE = INTENSITY_0 ? 1  :
                         INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  :
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;

    using RT = kittens::ducks::rt_shape::rt_16x16;
    using ST = kittens::ducks::st_shape::st_16x16;

    mma_sweep_size_warp<test_mma_ABt, RT, ST, SIZE, SIZE, std::integral_constant<int, 1>>::run(results);
    mma_sweep_size_warp<test_mma_ABt, RT, ST, SIZE, SIZE, std::integral_constant<int, 2>>::run(results);
    mma_sweep_size_warp<test_mma_ABt, RT, ST, SIZE, SIZE, std::integral_constant<int, 3>>::run(results);
    mma_sweep_size_warp<test_mma_ABt, RT, ST, SIZE, SIZE, std::integral_constant<int, 4>>::run(results);
}

#endif
