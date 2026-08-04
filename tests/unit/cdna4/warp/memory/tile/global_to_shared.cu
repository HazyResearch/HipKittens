#include "global_to_shared.cuh"

#ifdef TEST_WARP_MEMORY_TILE_GLOBAL_TO_SHARED

template<typename T>
struct st_load_store {
    using dtype = T;
    template<typename RT_SHAPE, typename ST_SHAPE, int H, int W, int NW, typename axis> using valid = std::bool_constant<
        (NW == 1 && W*H<=64) 
        && (W*H*ST_SHAPE::cols*ST_SHAPE::rows*sizeof(T) <= kittens::MAX_SHARED_MEMORY)
        && ((W*H*ST_SHAPE::cols*ST_SHAPE::rows*sizeof(T)) % (kittens::WARP_THREADS * ST_SHAPE::template bytes_per_thread<T>()) == 0)
    >;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "shared_loadstore_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "shared_loadstore_gmem=half" :
                                                      std::is_same_v<T, kittens::fp8e4m3> ? "shared_loadstore_gmem=fp8e4m3" :
                                                                                              "shared_loadstore_gmem=float";
    template<int H, int W, int NW, kittens::ducks::gl::all GL, typename axis> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<typename RT_SHAPE, typename ST_SHAPE, typename dtype, int H, int W, int NW, kittens::ducks::gl::all GL, typename axis> __device__ static void device_func(const GL &input, const GL &output) {
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the HIP shared memory
        kittens::shared_allocator<1024> al((int*)&__shm[0]);
        using ST = kittens::st<T, ST_SHAPE::rows*H, ST_SHAPE::cols*W, ST_SHAPE>;
        ST &shared_tile = al.allocate<ST>();
        int num_batches = axis::value==0?((int)input.batch()/shared_tile.rows):(int)input.batch();
        int num_depths = axis::value==1?((int)input.depth()/shared_tile.rows):(int)input.depth();
        int num_rows = axis::value==2?((int)input.rows()/shared_tile.rows):(int)input.rows();
        for(int i = 0; i < num_batches; i++)
            for(int j = 0; j < num_depths; j++)
                for(int k = 0; k < num_rows; k++)
                    for(int l = 0; l < (input.cols()/shared_tile.cols); l++) {
            kittens::load <axis::value, false, ST, GL, kittens::coord<ST>>(shared_tile,  input, {i, j, k, l});
            __builtin_amdgcn_s_waitcnt(0);
            __builtin_amdgcn_s_barrier();
            kittens::store<axis::value, false, ST, GL, kittens::coord<ST>>(output, shared_tile, {i, j, k, l});
            __builtin_amdgcn_s_waitcnt(0);
            __builtin_amdgcn_s_barrier();
        }
    }
};

using I0_t = std::integral_constant<int, 0>;
using I1_t = std::integral_constant<int, 1>;
using I2_t = std::integral_constant<int, 2>;
template<kittens::ducks::st_shape::all ST_SHAPE, kittens::ducks::rt_shape::all RT_SHAPE=kittens::ducks::rt_shape::rt_16x16>
void test_generator(test_data &results) {
    constexpr int SIZE = INTENSITY_0 ? 1  :
                         INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  :
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;

    g2s_sweep_size_2d_warp<st_load_store<kittens::bf16>, RT_SHAPE, ST_SHAPE, SIZE, SIZE, I0_t>::run(results);
    g2s_sweep_size_2d_warp<st_load_store<kittens::bf16>, RT_SHAPE, ST_SHAPE, SIZE, SIZE, I1_t>::run(results);
    g2s_sweep_size_2d_warp<st_load_store<kittens::bf16>, RT_SHAPE, ST_SHAPE, SIZE, SIZE, I2_t>::run(results);

    g2s_sweep_size_2d_warp<st_load_store<kittens::half>, RT_SHAPE, ST_SHAPE, SIZE, SIZE, I0_t>::run(results);
    g2s_sweep_size_2d_warp<st_load_store<kittens::half>, RT_SHAPE, ST_SHAPE, SIZE, SIZE, I1_t>::run(results);
    g2s_sweep_size_2d_warp<st_load_store<kittens::half>, RT_SHAPE, ST_SHAPE, SIZE, SIZE, I2_t>::run(results);

    g2s_sweep_size_2d_warp<st_load_store<float>, RT_SHAPE, ST_SHAPE, SIZE, SIZE, I0_t>::run(results);
    g2s_sweep_size_2d_warp<st_load_store<float>, RT_SHAPE, ST_SHAPE, SIZE, SIZE, I1_t>::run(results);
    g2s_sweep_size_2d_warp<st_load_store<float>, RT_SHAPE, ST_SHAPE, SIZE, SIZE, I2_t>::run(results);
}



void warp::memory::tile::global_to_shared::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/memory/tile/global_to_shared tests! -----\n" << std::endl;

    test_generator<kittens::ducks::st_shape::st_16x16>(results);
    test_generator<kittens::ducks::st_shape::st_16x16_swizzled>(results);
    test_generator<kittens::ducks::st_shape::st_32x32>(results);
    test_generator<kittens::ducks::st_shape::st_16x32>(results);
    test_generator<kittens::ducks::st_shape::st_32x16>(results);
    test_generator<kittens::ducks::st_shape::st_8x32>(results);
    test_generator<kittens::ducks::st_shape::st_16x64>(results);
}
#endif
