#include "reductions.cuh"

#ifdef TEST_WARP_REGISTER_VEC_REDUCTIONS

struct vec_norm {
    using dtype = float;
    template<typename RT_SHAPE, typename ST_SHAPE, int S, int NW, kittens::ducks::rv_layout::all L>
    using valid = std::bool_constant<NW == 1 && S<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_vec_norm";
    template<typename RT_SHAPE, typename ST_SHAPE, int S, int NW, gl_t GL, kittens::ducks::rv_layout::all L>
    __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        float f = 1.f;
        for(int i = 0; i < o_ref.size(); i++) f += i_ref[i];
        for(int i = 0; i < o_ref.size(); i++) o_ref[i] = f;
    }
    template<typename RT_SHAPE, typename ST_SHAPE, typename dtype, int S, int NW, gl_t GL, kittens::ducks::rv_layout::all L>
    __device__ static void device_func(const GL &input, const GL &output) {
        kittens::rv<dtype, RT_SHAPE::cols*S, RT_SHAPE::cols, RT_SHAPE, L> vec;
        kittens::load(vec, input, {});
        dtype f = kittens::base_types::constants<dtype>::ones();
        kittens::sum(f, vec, f);
        kittens::zero(vec);
        kittens::add(vec, vec, f);
        kittens::store(output, vec, {});
    }
};

template<kittens::ducks::rt_shape::all RT_SHAPE, kittens::ducks::st_shape::all ST_SHAPE=kittens::ducks::st_shape::st_16x16>
void test_generator(test_data &results) {
    constexpr int SIZE = INTENSITY_0 ? 1  :
                         INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
                         
    sweep_size_1d_warp<vec_norm, RT_SHAPE, ST_SHAPE, SIZE, 1, kittens::ducks::rv_layout::align>::run(results);
    sweep_size_1d_warp<vec_norm, RT_SHAPE, ST_SHAPE, SIZE, 1, kittens::ducks::rv_layout::ortho>::run(results);
    sweep_size_1d_warp<vec_norm, RT_SHAPE, ST_SHAPE, SIZE, 1, kittens::ducks::rv_layout::naive>::run(results);
}

void warp::reg::vec::reductions::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/register/vec/reductions tests! -----\n" << std::endl;
    
    test_generator<kittens::ducks::rt_shape::rt_16x32>(results);
    test_generator<kittens::ducks::rt_shape::rt_32x16>(results);
    test_generator<kittens::ducks::rt_shape::rt_16x16>(results);
    test_generator<kittens::ducks::rt_shape::rt_32x32>(results);
    test_generator<kittens::ducks::rt_shape::rt_32x32_8>(results);
    test_generator<kittens::ducks::rt_shape::rt_16x32_4>(results);
    test_generator<kittens::ducks::rt_shape::rt_32x16_4>(results);
}

#endif