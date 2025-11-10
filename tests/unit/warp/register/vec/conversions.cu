#include "conversions.cuh"

#ifdef TEST_WARP_REGISTER_VEC_CONVERSIONS

struct vec_copy_convert {
    template<typename RT_SHAPE, typename ST_SHAPE, int S, int NW, kittens::ducks::rv_layout::all L, typename T2, typename U2>
    using valid = std::bool_constant<NW == 1 && S<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_vec_convert";
    template<typename RT_SHAPE, typename ST_SHAPE, int S, int NW, gl_t GL, kittens::ducks::rv_layout::all L1, typename T2, typename U2>
    __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<typename RT_SHAPE, typename ST_SHAPE, typename dtype, int S, int NW, gl_t GL, kittens::ducks::rv_layout::all L1, typename T2, typename U2>
    __device__ static void device_func(const GL &input, const GL &output) {
        kittens::rv<U2, RT_SHAPE::cols*S, RT_SHAPE::cols, RT_SHAPE, L1> vec1;
        kittens::rv<T2, RT_SHAPE::cols*S, RT_SHAPE::cols, RT_SHAPE, L1> vec2;
        kittens::load(vec1, input, {});
        kittens::copy(vec2, vec1);
        kittens::store(output, vec2, {});
    }
};

template<kittens::ducks::rt_shape::all RT_SHAPE, kittens::ducks::st_shape::all ST_SHAPE=kittens::ducks::st_shape::st_16x16>
void test_generator(test_data &results) {
    constexpr int SIZE = INTENSITY_0 ? 1  :
                         INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;

    sweep_size_1d_warp<vec_copy_convert, RT_SHAPE, ST_SHAPE, SIZE, 1, kittens::ducks::rv_layout::naive, float, kittens::bf16>::run(results);
    sweep_size_1d_warp<vec_copy_convert, RT_SHAPE, ST_SHAPE, SIZE, 1, kittens::ducks::rv_layout::naive, kittens::bf16, float>::run(results);
    sweep_size_1d_warp<vec_copy_convert, RT_SHAPE, ST_SHAPE, SIZE, 1, kittens::ducks::rv_layout::naive, float, kittens::half>::run(results);
    sweep_size_1d_warp<vec_copy_convert, RT_SHAPE, ST_SHAPE, SIZE, 1, kittens::ducks::rv_layout::naive, kittens::half, float>::run(results);
    sweep_size_1d_warp<vec_copy_convert, RT_SHAPE, ST_SHAPE, SIZE, 1, kittens::ducks::rv_layout::naive, kittens::half, kittens::bf16>::run(results);
    sweep_size_1d_warp<vec_copy_convert, RT_SHAPE, ST_SHAPE, SIZE, 1, kittens::ducks::rv_layout::naive, kittens::bf16, kittens::half>::run(results);

    sweep_size_1d_warp<vec_copy_convert, RT_SHAPE, ST_SHAPE, SIZE, 1, kittens::ducks::rv_layout::ortho, float, kittens::bf16>::run(results);
    sweep_size_1d_warp<vec_copy_convert, RT_SHAPE, ST_SHAPE, SIZE, 1, kittens::ducks::rv_layout::ortho, kittens::bf16, float>::run(results);
    sweep_size_1d_warp<vec_copy_convert, RT_SHAPE, ST_SHAPE, SIZE, 1, kittens::ducks::rv_layout::ortho, float, kittens::half>::run(results);
    sweep_size_1d_warp<vec_copy_convert, RT_SHAPE, ST_SHAPE, SIZE, 1, kittens::ducks::rv_layout::ortho, kittens::half, float>::run(results);
    sweep_size_1d_warp<vec_copy_convert, RT_SHAPE, ST_SHAPE, SIZE, 1, kittens::ducks::rv_layout::ortho, kittens::half, kittens::bf16>::run(results);
    sweep_size_1d_warp<vec_copy_convert, RT_SHAPE, ST_SHAPE, SIZE, 1, kittens::ducks::rv_layout::ortho, kittens::bf16, kittens::half>::run(results);

    sweep_size_1d_warp<vec_copy_convert, RT_SHAPE, ST_SHAPE, SIZE, 1, kittens::ducks::rv_layout::align, float, kittens::bf16>::run(results);
    sweep_size_1d_warp<vec_copy_convert, RT_SHAPE, ST_SHAPE, SIZE, 1, kittens::ducks::rv_layout::align, kittens::bf16, float>::run(results);
    sweep_size_1d_warp<vec_copy_convert, RT_SHAPE, ST_SHAPE, SIZE, 1, kittens::ducks::rv_layout::align, float, kittens::half>::run(results);
    sweep_size_1d_warp<vec_copy_convert, RT_SHAPE, ST_SHAPE, SIZE, 1, kittens::ducks::rv_layout::align, kittens::half, float>::run(results);
    sweep_size_1d_warp<vec_copy_convert, RT_SHAPE, ST_SHAPE, SIZE, 1, kittens::ducks::rv_layout::align, kittens::half, kittens::bf16>::run(results);
    sweep_size_1d_warp<vec_copy_convert, RT_SHAPE, ST_SHAPE, SIZE, 1, kittens::ducks::rv_layout::align, kittens::bf16, kittens::half>::run(results);
}

void warp::reg::vec::conversions::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/register/vec/conversions tests! -----\n" << std::endl;

    test_generator<kittens::ducks::rt_shape::rt_16x32>(results);
    test_generator<kittens::ducks::rt_shape::rt_32x16>(results);
    test_generator<kittens::ducks::rt_shape::rt_16x16>(results);
    test_generator<kittens::ducks::rt_shape::rt_32x32>(results);
    test_generator<kittens::ducks::rt_shape::rt_32x32_8>(results);
    test_generator<kittens::ducks::rt_shape::rt_16x32_4>(results);
    test_generator<kittens::ducks::rt_shape::rt_32x16_4>(results);
}

#endif