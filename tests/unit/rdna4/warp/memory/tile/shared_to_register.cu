#include "shared_to_register.cuh"

#ifdef TEST_WARP_MEMORY_TILE_SHARED_TO_REGISTER

template<typename T>
struct sharedreg_load_store {
    using dtype = T;

    // Simple case only: one plain st_16x16 subtile (H=W=1), rt_16x16, row layout.
    // g2s `lds_offset()` and templated s2r agree only for this configuration.
    template<typename RT_SHAPE, typename ST_SHAPE, int H, int W, int NW, kittens::ducks::rt_layout::all RL>
    using valid = std::bool_constant<
        (NW == 1 && H == 1 && W == 1)
        && (ST_SHAPE::cols * ST_SHAPE::rows * sizeof(T) <= kittens::MAX_SHARED_MEMORY / 2)
        && (ST_SHAPE::cols * ST_SHAPE::rows * sizeof(T)) % (kittens::WARP_THREADS * ST_SHAPE::template bytes_per_thread<T>()) == 0
        && std::is_same_v<RL, kittens::ducks::rt_layout::row>
        && std::is_same_v<ST_SHAPE, kittens::ducks::st_shape::st_16x16>
        && std::is_same_v<RT_SHAPE, kittens::ducks::rt_shape::rt_16x16>
    >;

    static inline const std::string test_identifier =
        std::is_same_v<T, kittens::bf16> ? "shared_reg_loadstore_gmem=bf16" :
        std::is_same_v<T, kittens::half> ? "shared_reg_loadstore_gmem=half" :
                                           "shared_reg_loadstore_gmem=float";

    template<typename RT_SHAPE, typename ST_SHAPE, int H, int W, int NW,
             kittens::ducks::gl::all GL, kittens::ducks::rt_layout::all RL>
    __host__ static void host_func(const std::vector<float>& i_ref, std::vector<float>& o_ref) {
        o_ref = i_ref;
    }

    template<typename RT_SHAPE, typename ST_SHAPE, typename DTYPE,
             int H, int W, int NW,
             kittens::ducks::gl::all GL, kittens::ducks::rt_layout::all RL>
    __device__ static void device_func(const GL input, const GL output) {
        static_assert(std::is_same_v<DTYPE, T>, "dtype mismatch");

        extern __shared__ kittens::alignment_dummy __shm[];
        kittens::shared_allocator<16> al((int*)&__shm[0]);

        using ST_TILE = kittens::st<T, ST_SHAPE::rows, ST_SHAPE::cols, ST_SHAPE>;
        ST_TILE& shared_tile = al.allocate<ST_TILE>();

        const int row_stride = (int)input.cols();
        kittens::load<NW*kittens::WARP_THREADS>(shared_tile, input, {0, 0, 0, 0}, row_stride);
        kittens::sync::sync();

        kittens::rt<T, ST_SHAPE::rows, ST_SHAPE::cols, RL, RT_SHAPE> reg_tile;
        kittens::load(reg_tile, shared_tile);
        kittens::sync::wait_ds<0>();
        kittens::sync::sync();

        kittens::store(shared_tile, reg_tile);
        kittens::sync::wait_ds<0>();
        kittens::sync::sync();

        kittens::store(output, reg_tile, {0, 0, 0, 0});
        kittens::sync::wait_store<0>();
        kittens::sync::sync();
    }
};

void warp::memory::tile::shared_to_register::tests(test_data& results) {
    std::cout << "\n ----- Starting ops/warp/memory/tile/shared_to_register tests! -----\n" << std::endl;

    sweep_size_2d_warp<sharedreg_load_store<kittens::bf16>,
                       kittens::ducks::rt_shape::rt_16x16,
                       kittens::ducks::st_shape::st_16x16,
                       1, 1, 1, kittens::ducks::rt_layout::row>::run(results);

    sweep_size_2d_warp<sharedreg_load_store<kittens::half>,
                       kittens::ducks::rt_shape::rt_16x16,
                       kittens::ducks::st_shape::st_16x16,
                       1, 1, 1, kittens::ducks::rt_layout::row>::run(results);
}
#endif
