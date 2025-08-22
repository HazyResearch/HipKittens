#include "shared_to_register.cuh"

#ifdef TEST_WARP_MEMORY_TILE_SHARED_TO_REGISTER

template<typename T>
struct sharedreg_load_store {
    using dtype = T;
    template<int H, int W, int NW, kittens::ducks::st_layout::all SL, kittens::ducks::rt_layout::all RL> using valid = std::bool_constant<
      ( NW == 1 && W*H<=16 ) && (W*H*kittens::TILE_COL_DIM<T>*kittens::TILE_ROW_DIM<T>*sizeof(T) <= kittens::MAX_SHARED_MEMORY)
    >;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "shared_reg_loadstore_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "shared_reg_loadstore_gmem=half" :
                                                                                         "shared_reg_loadstore_gmem=float";
    template<int H, int W, int NW, kittens::ducks::gl::all GL, kittens::ducks::st_layout::all SL, kittens::ducks::rt_layout::all RL> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int H, int W, int NW, kittens::ducks::gl::all GL, kittens::ducks::st_layout::all SL, kittens::ducks::rt_layout::all RL> __device__ static void device_func(const GL input, const GL output) {
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::shared_allocator<16> al((int*)&__shm[0]); 
        kittens::st<T, kittens::TILE_ROW_DIM<T>*H, kittens::TILE_COL_DIM<T>*W, SL> &shared_tile = al.allocate<kittens::st<T, kittens::TILE_ROW_DIM<T>*H, kittens::TILE_COL_DIM<T>*W, SL>>();
        kittens::load(shared_tile, input, {0, 0, 0, 0});
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();
        kittens::rt<T, kittens::TILE_ROW_DIM<T>*H, kittens::TILE_COL_DIM<T>*W, RL> reg_tile;
        kittens::load(reg_tile, shared_tile);
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();
        kittens::store(shared_tile, reg_tile);
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();
        kittens::store(output, shared_tile, {0, 0, 0, 0});
    }
};

void warp::memory::tile::shared_to_register::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/memory/tile/shared_to_register tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_0 ? 1  :
                         INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;

    sweep_gmem_type_2d_warp<sharedreg_load_store, SIZE, SIZE, kittens::ducks::st_layout::row, kittens::ducks::rt_layout::row>::run(results);
    sweep_gmem_type_2d_warp<sharedreg_load_store, SIZE, SIZE, kittens::ducks::st_layout::col, kittens::ducks::rt_layout::col>::run(results);
    sweep_gmem_type_2d_warp<sharedreg_load_store, SIZE, SIZE, kittens::ducks::st_layout::accumulator_col, kittens::ducks::rt_layout::accumulator_col>::run(results);
}

#endif