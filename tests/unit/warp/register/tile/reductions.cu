#include "reductions.cuh"

#ifdef TEST_WARP_REGISTER_TILE_REDUCTIONS

#ifdef KITTENS_CDNA4
#define ROWS 32
#define COLS 32
#else
#define ROWS 16
#define COLS 16
#endif

struct normalize_row {
    using dtype = float;
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_norm_row";
    template<int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < H*ROWS; i++) {
            float row_sum = 0;
            for(int j = 0; j < W*COLS; j++) {
                o_ref[i*W*COLS+j]  = i_ref[i*W*COLS+j];
                row_sum         += i_ref[i*W*COLS+j];
            }
            for(int j = 0; j < W*COLS; j++) o_ref[i*W*COLS+j] /= row_sum;
        }
    }
    template<int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> __device__ static void device_func(const GLT &input, const GLT &output) {
        kittens::rt_fl<ROWS*H, COLS*W, L> reg_tile;
        kittens::load(reg_tile, input, {});
        typename kittens::rt_fl<ROWS*H, COLS*W, L>::col_vec accum;
        kittens::row_sum(accum, reg_tile);
        kittens::div_row(reg_tile, reg_tile, accum);
        kittens::store(output, reg_tile, {});
    }
};
struct normalize_col {
    using dtype = float;
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_norm_col";
    template<int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < W*COLS; i++) {
            float col_sum = 0;
            for(int j = 0; j < H*ROWS; j++) {
                o_ref[i+j*W*COLS]  = i_ref[i+j*W*COLS];
                col_sum         += i_ref[i+j*W*COLS];
            }
            for(int j = 0; j < H*ROWS; j++) o_ref[i+j*W*COLS] /= col_sum;
        }
    }
    template<int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> __device__ static void device_func(const GLT &input, const GLT &output) {
        kittens::rt_fl<ROWS*H, COLS*W, L> reg_tile;
        kittens::load(reg_tile, input, {});
        typename kittens::rt_fl<ROWS*H, COLS*W, L>::row_vec accum;
        kittens::col_sum(accum, reg_tile);
        kittens::div_col(reg_tile, reg_tile, accum);
        kittens::store(output, reg_tile, {});
    }
};
struct broadcast_row {
    using dtype = float;
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_broadcast_row";
    template<int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < H*ROWS; i++) {
            float row_sum = 0;
            for(int j = 0; j < W*COLS; j++) {
                o_ref[i*W*COLS+j]  = i_ref[i*W*COLS+j];
                row_sum         += i_ref[i*W*COLS+j];
            }
            for(int j = 0; j < W*COLS; j++) o_ref[i*W*COLS+j] = row_sum;
        }
    }
    template<int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> __device__ static void device_func(const GLT &input, const GLT &output) {
        kittens::rt_fl<ROWS*H, COLS*W, L> reg_tile;
        kittens::load(reg_tile, input, {});
        typename kittens::rt_fl<ROWS*H, COLS*W, L>::col_vec accum;
        kittens::row_sum(accum, reg_tile);
        kittens::broadcast_row(reg_tile, accum);
        kittens::store(output, reg_tile, {});
    }
};
struct broadcast_col {
    using dtype = float;
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_broadcast_col";
    template<int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < W*COLS; i++) {
            float col_sum = 0;
            for(int j = 0; j < H*ROWS; j++) {
                o_ref[i+j*W*COLS]  = i_ref[i+j*W*COLS];
                col_sum         += i_ref[i+j*W*COLS];
            }
            for(int j = 0; j < H*ROWS; j++) o_ref[i+j*W*COLS] = col_sum;
        }
    }
    template<int H, int W, int NW, gl_t GLT, kittens::ducks::rt_layout::all L> __device__ static void device_func(const GLT &input, const GLT &output) {
        kittens::rt_fl<ROWS*H, COLS*W, L> reg_tile;
        kittens::load(reg_tile, input, {});
        typename kittens::rt_fl<ROWS*H, COLS*W, L>::row_vec accum;
        kittens::col_sum(accum, reg_tile);
        kittens::broadcast_col(reg_tile, accum);
        kittens::store(output, reg_tile, {});
    }
};

void warp::reg::tile::reductions::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/register/tile/reductions tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_0 ? 1  :
                         INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : 32;
    sweep_size_2d_warp<normalize_row, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<normalize_row, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d_warp<normalize_col, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<normalize_col, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d_warp<broadcast_row, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<broadcast_row, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d_warp<broadcast_col, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<broadcast_col, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);

    #ifdef KITTENS_CDNA4
    sweep_size_2d_warp<normalize_row, SIZE, SIZE, kittens::ducks::rt_layout::accumulator_row>::run(results);
    sweep_size_2d_warp<normalize_col, SIZE, SIZE, kittens::ducks::rt_layout::accumulator_row>::run(results);
    sweep_size_2d_warp<broadcast_row, SIZE, SIZE, kittens::ducks::rt_layout::accumulator_row>::run(results);
    sweep_size_2d_warp<broadcast_col, SIZE, SIZE, kittens::ducks::rt_layout::accumulator_row>::run(results);
    sweep_size_2d_warp<normalize_row, SIZE, SIZE, kittens::ducks::rt_layout::accumulator_col>::run(results);
    sweep_size_2d_warp<normalize_col, SIZE, SIZE, kittens::ducks::rt_layout::accumulator_col>::run(results);
    sweep_size_2d_warp<broadcast_row, SIZE, SIZE, kittens::ducks::rt_layout::accumulator_col>::run(results);
    sweep_size_2d_warp<broadcast_col, SIZE, SIZE, kittens::ducks::rt_layout::accumulator_col>::run(results);
    #endif
}

#endif