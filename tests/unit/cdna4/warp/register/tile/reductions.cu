#include "reductions.cuh"

#ifdef TEST_WARP_REGISTER_TILE_REDUCTIONS

struct normalize_row {
    using dtype = float;
    template<typename RT_SHAPE, typename ST_SHAPE, int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_norm_row";
    template<typename RT_SHAPE, typename ST_SHAPE, int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < RT_SHAPE::rows*H; i++) {
            float row_sum = 0;
            for(int j = 0; j < RT_SHAPE::cols*W; j++) {
                o_ref[i*RT_SHAPE::cols*W+j]  = i_ref[i*RT_SHAPE::cols*W+j];
                row_sum         += i_ref[i*RT_SHAPE::cols*W+j];
            }
            for(int j = 0; j < RT_SHAPE::cols*W; j++) o_ref[i*RT_SHAPE::cols*W+j] /= row_sum;
        }
    }
    template<typename RT_SHAPE, typename ST_SHAPE, typename dtype, int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> __device__ static void device_func(const GLT &input, const GLT &output) {
        kittens::rt<dtype, RT_SHAPE::rows*H, RT_SHAPE::cols*W, L, RT_SHAPE> reg_tile;
        kittens::load(reg_tile, input, {});
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();
        typename kittens::rt<dtype, RT_SHAPE::rows*H, RT_SHAPE::cols*W, L, RT_SHAPE>::col_vec accum;
        kittens::row_sum(accum, reg_tile);
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();
        kittens::div_row(reg_tile, reg_tile, accum);
        kittens::store(output, reg_tile, {});
    }
};
struct normalize_col {
    using dtype = float;
    template<typename RT_SHAPE, typename ST_SHAPE, int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_norm_col";
    template<typename RT_SHAPE, typename ST_SHAPE, int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < RT_SHAPE::cols*W; i++) {
            float col_sum = 0;
            for(int j = 0; j < RT_SHAPE::rows*H; j++) {
                o_ref[i+j*RT_SHAPE::cols*W]  = i_ref[i+j*RT_SHAPE::cols*W];
                col_sum         += i_ref[i+j*RT_SHAPE::cols*W];
            }
            for(int j = 0; j < RT_SHAPE::rows*H; j++) o_ref[i+j*RT_SHAPE::cols*W] /= col_sum;
        }
    }
    template<typename RT_SHAPE, typename ST_SHAPE, typename dtype, int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> __device__ static void device_func(const GLT &input, const GLT &output) {
        kittens::rt<dtype, RT_SHAPE::rows*H, RT_SHAPE::cols*W, L, RT_SHAPE> reg_tile;
        kittens::load(reg_tile, input, {});
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();
        typename kittens::rt<dtype, RT_SHAPE::rows*H, RT_SHAPE::cols*W, L, RT_SHAPE>::row_vec accum;
        kittens::col_sum(accum, reg_tile);
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();
        kittens::div_col(reg_tile, reg_tile, accum);
        kittens::store(output, reg_tile, {});
    }
};
struct broadcast_row {
    using dtype = float;
    template<typename RT_SHAPE, typename ST_SHAPE, int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_broadcast_row";
    template<typename RT_SHAPE, typename ST_SHAPE, int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < RT_SHAPE::rows*H; i++) {
            float row_sum = 0;
            for(int j = 0; j < RT_SHAPE::cols*W; j++) {
                o_ref[i*RT_SHAPE::cols*W+j]  = i_ref[i*RT_SHAPE::cols*W+j];
                row_sum         += i_ref[i*RT_SHAPE::cols*W+j];
            }
            for(int j = 0; j < RT_SHAPE::cols*W; j++) o_ref[i*RT_SHAPE::cols*W+j] = row_sum;
        }
    }
    template<typename RT_SHAPE, typename ST_SHAPE, typename dtype, int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> __device__ static void device_func(const GLT &input, const GLT &output) {
        kittens::rt<dtype, RT_SHAPE::rows*H, RT_SHAPE::cols*W, L, RT_SHAPE> reg_tile;
        kittens::load(reg_tile, input, {});
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();
        typename kittens::rt<dtype, RT_SHAPE::rows*H, RT_SHAPE::cols*W, L, RT_SHAPE>::col_vec accum;
        kittens::row_sum(accum, reg_tile);
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();
        kittens::broadcast_row(reg_tile, accum);
        kittens::store(output, reg_tile, {});
    }
};
struct broadcast_col {
    using dtype = float;
    template<typename RT_SHAPE, typename ST_SHAPE, int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_broadcast_col";
    template<typename RT_SHAPE, typename ST_SHAPE, int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < RT_SHAPE::cols*W; i++) {
            float col_sum = 0;
            for(int j = 0; j < RT_SHAPE::rows*H; j++) {
                o_ref[i+j*RT_SHAPE::cols*W]  = i_ref[i+j*RT_SHAPE::cols*W];
                col_sum         += i_ref[i+j*RT_SHAPE::cols*W];
            }
            for(int j = 0; j < RT_SHAPE::rows*H; j++) o_ref[i+j*RT_SHAPE::cols*W] = col_sum;
        }
    }
    template<typename RT_SHAPE, typename ST_SHAPE, typename dtype, int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> __device__ static void device_func(const GLT &input, const GLT &output) {
        kittens::rt<dtype, RT_SHAPE::rows*H, RT_SHAPE::cols*W, L, RT_SHAPE> reg_tile;
        kittens::load(reg_tile, input, {});
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();
        typename kittens::rt<dtype, RT_SHAPE::rows*H, RT_SHAPE::cols*W, L, RT_SHAPE>::row_vec accum;
        kittens::col_sum(accum, reg_tile);
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();
        kittens::broadcast_col(reg_tile, accum);
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

    sweep_size_2d_warp<normalize_row, RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<normalize_row, RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d_warp<normalize_col, RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<normalize_col, RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d_warp<broadcast_row, RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<broadcast_row, RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d_warp<broadcast_col, RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<broadcast_col, RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, kittens::ducks::rt_layout::col>::run(results);
}
void warp::reg::tile::reductions::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/register/tile/reductions tests! -----\n" << std::endl;

    test_generator<kittens::ducks::rt_shape::rt_16x32>(results);
    test_generator<kittens::ducks::rt_shape::rt_32x16>(results);
    test_generator<kittens::ducks::rt_shape::rt_16x16>(results);
    test_generator<kittens::ducks::rt_shape::rt_32x32>(results);
    test_generator<kittens::ducks::rt_shape::rt_32x32_8>(results);
    test_generator<kittens::ducks::rt_shape::rt_16x32_4>(results);
    test_generator<kittens::ducks::rt_shape::rt_32x16_4>(results);
}

#endif