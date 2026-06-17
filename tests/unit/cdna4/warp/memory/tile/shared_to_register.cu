#include "shared_to_register.cuh"

#ifdef TEST_WARP_MEMORY_TILE_SHARED_TO_REGISTER

template<typename T>
struct sharedreg_load_store {
    using dtype = T;

    // NOTE: 'valid' does NOT take dtype; it uses the enclosing T
    template<typename RT_SHAPE, typename ST_SHAPE, int H, int W, int NW, kittens::ducks::rt_layout::all RL>
    using valid = std::bool_constant<
        (NW == 1 && W*H <= 16)
        && (W*H*ST_SHAPE::cols*ST_SHAPE::rows*sizeof(T) <= kittens::MAX_SHARED_MEMORY / 2)
        && (W*H*ST_SHAPE::cols*ST_SHAPE::rows*sizeof(T)) % (kittens::WARP_THREADS * ST_SHAPE::template bytes_per_thread<T>()) == 0
        && (ST_SHAPE::cols % RT_SHAPE::cols == 0)
        && (ST_SHAPE::rows % RT_SHAPE::rows == 0)
    >;

    static inline const std::string test_identifier =
        std::is_same_v<T, kittens::bf16> ? "shared_reg_loadstore_gmem=bf16" :
        std::is_same_v<T, kittens::half> ? "shared_reg_loadstore_gmem=half" :
                                           "shared_reg_loadstore_gmem=float";

    template<typename RT_SHAPE, typename ST_SHAPE, int H, int W, int NW,
             kittens::ducks::gl::all GL, kittens::ducks::rt_layout::all RL>
    __host__ static void host_func(const std::vector<float>& i_ref, std::vector<float>& o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }

    template<typename RT_SHAPE, typename ST_SHAPE, typename DTYPE,
             int H, int W, int NW,
             kittens::ducks::gl::all GL, kittens::ducks::rt_layout::all RL>
    __device__ static void device_func(const GL input, const GL output) {
        static_assert(std::is_same_v<DTYPE, T>, "dtype mismatch");

        extern __shared__ kittens::alignment_dummy __shm[];
        kittens::shared_allocator<16> al((int*)&__shm[0]);

        using ST_TILE = kittens::st<T, ST_SHAPE::rows*H, ST_SHAPE::cols*W, ST_SHAPE>;
        ST_TILE& shared_tile = al.allocate<ST_TILE>();

        kittens::load(shared_tile, input, {0,0,0,0});
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();

        kittens::rt<T, ST_SHAPE::rows*H, ST_SHAPE::cols*W, RL, RT_SHAPE> reg_tile;
        kittens::load(reg_tile, shared_tile);
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();

        kittens::store(shared_tile, reg_tile);
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();

        kittens::store(output, shared_tile, {0,0,0,0});
    }
};

template<kittens::ducks::rt_shape::all RT_SHAPE, kittens::ducks::st_shape::all ST_SHAPE>
void test_generator(test_data &results) {
    constexpr int SIZE = INTENSITY_0 ? 1  :
                         INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;

    sweep_size_2d_warp<sharedreg_load_store<kittens::bf16>, RT_SHAPE, ST_SHAPE,
                       SIZE, SIZE, 1, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<sharedreg_load_store<kittens::bf16>, RT_SHAPE, ST_SHAPE,
                       SIZE, SIZE, 1, kittens::ducks::rt_layout::col>::run(results);

    sweep_size_2d_warp<sharedreg_load_store<kittens::half>, RT_SHAPE, ST_SHAPE,
                       SIZE, SIZE, 1, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<sharedreg_load_store<kittens::half>, RT_SHAPE, ST_SHAPE,
                       SIZE, SIZE, 1, kittens::ducks::rt_layout::col>::run(results);

    // sweep_size_2d_warp<sharedreg_load_store<float>, RT_SHAPE, ST_SHAPE,
    //                    SIZE, SIZE, 1, kittens::ducks::rt_layout::row>::run(results);
    // sweep_size_2d_warp<sharedreg_load_store<float>, RT_SHAPE, ST_SHAPE,
    //                    SIZE, SIZE, 1, kittens::ducks::rt_layout::col>::run(results);
}

template<kittens::ducks::rt_shape::all RT_SHAPE>
void nested_test_generator(test_data &results) {
    test_generator<RT_SHAPE, kittens::ducks::st_shape::st_16x16>(results);
    test_generator<RT_SHAPE, kittens::ducks::st_shape::st_16x16_swizzled>(results);
    test_generator<RT_SHAPE, kittens::ducks::st_shape::st_32x32>(results);
    test_generator<RT_SHAPE, kittens::ducks::st_shape::st_16x32>(results);
    test_generator<RT_SHAPE, kittens::ducks::st_shape::st_32x16>(results);
    test_generator<RT_SHAPE, kittens::ducks::st_shape::st_8x32>(results);
}

void warp::memory::tile::shared_to_register::tests(test_data& results) {
    std::cout << "\n ----- Starting ops/warp/memory/tile/shared_to_register tests! -----\n" << std::endl;

    nested_test_generator<kittens::ducks::rt_shape::rt_16x32>(results);
    nested_test_generator<kittens::ducks::rt_shape::rt_32x16>(results);
    nested_test_generator<kittens::ducks::rt_shape::rt_16x16>(results);
    nested_test_generator<kittens::ducks::rt_shape::rt_32x32>(results);
    nested_test_generator<kittens::ducks::rt_shape::rt_32x32_8>(results);
    nested_test_generator<kittens::ducks::rt_shape::rt_16x32_4>(results);
    nested_test_generator<kittens::ducks::rt_shape::rt_32x16_4>(results);
}
#endif
