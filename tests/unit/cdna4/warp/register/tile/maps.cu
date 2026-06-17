#include "maps.cuh"

#ifdef TEST_WARP_REGISTER_TILE_MAPS

struct test_exp {
    template<typename RT_SHAPE, typename ST_SHAPE, int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_exp";
    template<typename RT_SHAPE, typename ST_SHAPE, int H, int W, int NW, kittens::ducks::gl::all GL, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < i_ref.size(); i++) o_ref[i] = ::expf(i_ref[i]);
    }
    template<typename RT_SHAPE, typename ST_SHAPE, typename dtype, int H, int W, int NW, kittens::ducks::gl::all GL, kittens::ducks::rt_layout::all L> __device__ static void device_func(const GL input, const GL output) {
        kittens::rt<dtype, RT_SHAPE::rows*H, RT_SHAPE::cols*W, L, RT_SHAPE> reg_tile;
        kittens::load(reg_tile, input, {});
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();
        kittens::exp(reg_tile, reg_tile);
        kittens::store(output, reg_tile, {});
    }
};
struct test_gelu {
    using dtype = float;
    template<typename RT_SHAPE, typename ST_SHAPE, int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>;
    static inline const std::string test_identifier = "reg_gelu";
    template<typename RT_SHAPE, typename ST_SHAPE, int H, int W, int NW, kittens::ducks::gl::all GL, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        constexpr float S2P = 0.7978845608028654f;
        constexpr float C   = 0.044715f;
        for(size_t i = 0; i < i_ref.size(); i++) {
            float x = i_ref[i];
            o_ref[i] = x * (0.5f + 0.5f * ::tanhf(S2P * x * (1.f + C * x * x)));
        }
    }
    template<typename RT_SHAPE, typename ST_SHAPE, typename dtype, int H, int W, int NW, kittens::ducks::gl::all GL, kittens::ducks::rt_layout::all L> __device__ static void device_func(const GL input, const GL output) {
        kittens::rt<dtype, RT_SHAPE::rows*H, RT_SHAPE::cols*W, L, RT_SHAPE> reg_tile;
        kittens::load(reg_tile, input, {});
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();
        kittens::gelu(reg_tile, reg_tile);
        kittens::store(output, reg_tile, {});
    }
};
struct test_dgelu {
    using dtype = float;
    template<typename RT_SHAPE, typename ST_SHAPE, int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>;
    static inline const std::string test_identifier = "reg_dgelu";
    template<typename RT_SHAPE, typename ST_SHAPE, int H, int W, int NW, kittens::ducks::gl::all GL, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        constexpr float S2P = 0.7978845608028654f;
        constexpr float C   = 0.044715f;
        constexpr float DC  = 3.0f * C * S2P;
        for(size_t i = 0; i < i_ref.size(); i++) {
            float x = i_ref[i];
            float t = ::tanhf(S2P * x * (1.f + C * x * x));
            o_ref[i] = 0.5f * x * ((1.f - t * t) * (S2P + DC * x * x)) + 0.5f * (1.f + t);
        }
    }
    template<typename RT_SHAPE, typename ST_SHAPE, typename dtype, int H, int W, int NW, kittens::ducks::gl::all GL, kittens::ducks::rt_layout::all L> __device__ static void device_func(const GL input, const GL output) {
        kittens::rt<dtype, RT_SHAPE::rows*H, RT_SHAPE::cols*W, L, RT_SHAPE> reg_tile;
        kittens::load(reg_tile, input, {});
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();
        kittens::dgelu(reg_tile, reg_tile);
        kittens::store(output, reg_tile, {});
    }
};

template<kittens::ducks::rt_shape::all RT_SHAPE, kittens::ducks::st_shape::all ST_SHAPE=kittens::ducks::st_shape::st_16x16>
void test_generator(test_data &results) {
    constexpr int SIZE = INTENSITY_0 ? 1  :
                         INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  :
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;

    sweep_size_2d_warp<test_exp,   RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<test_exp,   RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d_warp<test_gelu,  RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<test_gelu,  RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d_warp<test_dgelu, RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<test_dgelu, RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, kittens::ducks::rt_layout::col>::run(results);
}

void warp::reg::tile::maps::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/register/tile/maps tests! -----\n" << std::endl;

    test_generator<kittens::ducks::rt_shape::rt_16x32>(results);
    test_generator<kittens::ducks::rt_shape::rt_32x16>(results);
    test_generator<kittens::ducks::rt_shape::rt_16x16>(results);
    test_generator<kittens::ducks::rt_shape::rt_32x32>(results);
    test_generator<kittens::ducks::rt_shape::rt_32x32_8>(results);
    test_generator<kittens::ducks::rt_shape::rt_16x32_4>(results);
    test_generator<kittens::ducks::rt_shape::rt_32x16_4>(results);
}

#endif