#include "conversions.cuh"

#ifdef TEST_WARP_SHARED_TILE_CONVERSIONS

template<typename T>
struct test_subtile {
    using dtype = T;
    template<typename RT_SHAPE, typename ST_SHAPE, int H, int W, int NW, typename _ST_H, typename _ST_W> using valid = std::bool_constant<(
        NW == 1 && W*H<=64
        && (H % _ST_H::value == 0 && W % _ST_W::value == 0 ) 
        && (W*H*ST_SHAPE::cols*ST_SHAPE::rows*sizeof(T) <= kittens::MAX_SHARED_MEMORY / 2)
        && (W*H*ST_SHAPE::cols*ST_SHAPE::rows*sizeof(T)) % (kittens::WARP_THREADS * ST_SHAPE::template bytes_per_thread<T>()) == 0
        && sizeof(dtype) != 1
    )>;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "shared_subtile_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "shared_subtile_gmem=half" :
                                                                                         "shared_subtile_gmem=float";
    template<typename RT_SHAPE, typename ST_SHAPE, int H, int W, int NW, gl_t GL, typename _ST_H, typename _ST_W> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        constexpr int ST_H = _ST_H::value, ST_W = _ST_W::value;
        for(int i = 0; i < H*ST_SHAPE::rows; i++)
            for(int j = 0; j < W*ST_SHAPE::cols; j++)
                o_ref[i*W*ST_SHAPE::cols + j] = i_ref[i*W*ST_SHAPE::cols + j] * float(i/(ST_H*ST_SHAPE::rows)) + float(j/(ST_W*ST_SHAPE::cols));
    }
    template<typename RT_SHAPE, typename ST_SHAPE, typename dtype, int H, int W, int NW, gl_t GL, typename _ST_H, typename _ST_W> __device__ static void device_func(const GL &input, const GL &output) {
        constexpr int ST_H = _ST_H::value, ST_W = _ST_W::value;
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::shared_allocator al((int*)&__shm[0]); 
        kittens::st<dtype, ST_SHAPE::rows*H, ST_SHAPE::cols*W, ST_SHAPE> &t = al.allocate<kittens::st<dtype, ST_SHAPE::rows*H, ST_SHAPE::cols*W, ST_SHAPE>>();
        kittens::load(t, input, {});
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();
        for(int i = 0; i < H/ST_H; i++) {
            for(int j = 0; j < W/ST_W; j++) {
                auto ref = kittens::subtile_inplace<ST_SHAPE::rows*ST_H, ST_SHAPE::cols*ST_W>(t, {i, j});
                kittens::rt<dtype, ST_SHAPE::rows*ST_H, ST_SHAPE::cols*ST_W, kittens::ducks::rt_layout::row, RT_SHAPE> reg;
                kittens::load(reg, ref);
                __builtin_amdgcn_s_waitcnt(0);
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);
                kittens::mul(reg, reg, dtype(i));
                kittens::add(reg, reg, dtype(j));
                kittens::store(ref, reg);
                __builtin_amdgcn_s_waitcnt(0);
                __builtin_amdgcn_s_barrier();
            }
        }
        kittens::store(output, t, {});
    }
};

template<kittens::ducks::rt_shape::all RT_SHAPE, kittens::ducks::st_shape::all ST_SHAPE>
void test_generator(test_data &results) {
    constexpr int SIZE = INTENSITY_0 ? 1  :
                         INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;

    sweep_size_2d_warp<test_subtile<kittens::bf16>, RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, std::integral_constant<int, 1>, std::integral_constant<int, 1>>::run(results);
    sweep_size_2d_warp<test_subtile<kittens::bf16>, RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, std::integral_constant<int, 1>, std::integral_constant<int, 2>>::run(results);
    sweep_size_2d_warp<test_subtile<kittens::bf16>, RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, std::integral_constant<int, 1>, std::integral_constant<int, 3>>::run(results);
    sweep_size_2d_warp<test_subtile<kittens::bf16>, RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, std::integral_constant<int, 1>, std::integral_constant<int, 4>>::run(results);
    sweep_size_2d_warp<test_subtile<kittens::bf16>, RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, std::integral_constant<int, 2>, std::integral_constant<int, 1>>::run(results);
    sweep_size_2d_warp<test_subtile<kittens::bf16>, RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, std::integral_constant<int, 2>, std::integral_constant<int, 2>>::run(results);
    sweep_size_2d_warp<test_subtile<kittens::bf16>, RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, std::integral_constant<int, 2>, std::integral_constant<int, 3>>::run(results);
    sweep_size_2d_warp<test_subtile<kittens::bf16>, RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, std::integral_constant<int, 2>, std::integral_constant<int, 4>>::run(results);
    sweep_size_2d_warp<test_subtile<kittens::bf16>, RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, std::integral_constant<int, 3>, std::integral_constant<int, 1>>::run(results);
    sweep_size_2d_warp<test_subtile<kittens::bf16>, RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, std::integral_constant<int, 3>, std::integral_constant<int, 2>>::run(results);
    sweep_size_2d_warp<test_subtile<kittens::bf16>, RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, std::integral_constant<int, 3>, std::integral_constant<int, 3>>::run(results);
    sweep_size_2d_warp<test_subtile<kittens::bf16>, RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, std::integral_constant<int, 3>, std::integral_constant<int, 4>>::run(results);
    sweep_size_2d_warp<test_subtile<kittens::bf16>, RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, std::integral_constant<int, 4>, std::integral_constant<int, 1>>::run(results);
    sweep_size_2d_warp<test_subtile<kittens::bf16>, RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, std::integral_constant<int, 4>, std::integral_constant<int, 2>>::run(results);
    sweep_size_2d_warp<test_subtile<kittens::bf16>, RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, std::integral_constant<int, 4>, std::integral_constant<int, 3>>::run(results);
    sweep_size_2d_warp<test_subtile<kittens::bf16>, RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, std::integral_constant<int, 4>, std::integral_constant<int, 4>>::run(results);
}

void warp::shared::tile::conversions::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/shared/conversions tests! -----\n" << std::endl;

    test_generator<kittens::ducks::rt_shape::rt_16x16, kittens::ducks::st_shape::st_16x16>(results);
    test_generator<kittens::ducks::rt_shape::rt_16x16, kittens::ducks::st_shape::st_16x16_swizzled>(results);
    test_generator<kittens::ducks::rt_shape::rt_32x32_8, kittens::ducks::st_shape::st_32x32>(results);
    test_generator<kittens::ducks::rt_shape::rt_16x32, kittens::ducks::st_shape::st_16x32>(results);
    test_generator<kittens::ducks::rt_shape::rt_32x16, kittens::ducks::st_shape::st_32x16>(results);
}

#endif