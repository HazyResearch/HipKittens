#include "maps.cuh"

#ifdef TEST_WARP_REGISTER_VEC_MAPS
// Every test file in this tree names its scaffolding the same way -- test_generator,
// and wrapper structs like vec_load_store. Those are templates with external
// linkage, so across translation units they are one entity with many definitions,
// and the linker keeps whichever it saw first. Internal linkage makes each file's
// version its own.
namespace {


struct vec_add1 {
    using dtype = float;
    template<typename RT_SHAPE, typename ST_SHAPE, int S, int NW, kittens::ducks::rv_layout::all L>
    using valid = std::bool_constant<NW == 1 && S<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_vec_add1";
    template<typename RT_SHAPE, typename ST_SHAPE, int S, int NW, gl_t GL, kittens::ducks::rv_layout::all L>
    __host__ static void host_func(const std::vector<dtype> &i_ref, std::vector<dtype> &o_ref) {
        for(int i = 0; i < o_ref.size(); i++) o_ref[i] = i_ref[i]+1.; // overwrite the whole thing
    }
    template<typename RT_SHAPE, typename ST_SHAPE, typename dtype, int S, int NW, gl_t GL, kittens::ducks::rv_layout::all L>
    __device__ static void device_func(const GL &input, const GL &output) {
        kittens::rv<dtype, RT_SHAPE::cols*S, L> vec;
        kittens::load(vec, input, {});
        kittens::add(vec, vec, kittens::base_types::constants<dtype>::one());
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
                         
    sweep_size_1d_warp<vec_add1, RT_SHAPE, ST_SHAPE, SIZE, 1, kittens::ducks::rv_layout::naive>::run(results);
    sweep_size_1d_warp<vec_add1, RT_SHAPE, ST_SHAPE, SIZE, 1, kittens::ducks::rv_layout::align>::run(results);
    sweep_size_1d_warp<vec_add1, RT_SHAPE, ST_SHAPE, SIZE, 1, kittens::ducks::rv_layout::ortho>::run(results);
}


} // anonymous namespace
void warp::reg::vec::maps::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/register/vec/maps tests! -----\n" << std::endl;
    
    test_generator<kittens::ducks::rt_shape::rt_16x16>(results);
}

#endif