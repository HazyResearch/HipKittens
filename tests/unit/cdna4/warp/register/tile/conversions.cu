#include "conversions.cuh"

#ifdef TEST_WARP_REGISTER_TILE_CONVERSIONS

// Transpose happens to need its own wrapper, as it has a different shape input and output.
template<typename Ker, typename RT_SHAPE, typename ST_SHAPE, typename dtype, int H, int W, int NW, gl_t GTL_I, gl_t GTL_O, typename... args>
static __global__ void transpose_global_wrapper_2d(const GTL_I input, GTL_O output) {
    Ker::template device_func<RT_SHAPE, ST_SHAPE, dtype, H, W, NW, GTL_I, GTL_O, args...>(input, output);
}
template<typename test, typename RT_SHAPE, typename ST_SHAPE, int H, int W, int NUM_WORKERS, typename... args>
struct transpose_wrapper_2d {
    using dtype = gmem_dtype<test>; // defaults to bf16 in global memory if the test doesn't specify.
    static void run(test_data& results) {
        using namespace kittens;
        test_info this_result;
        this_result.label = generate_test_name<RT_SHAPE,ST_SHAPE,H,W,NUM_WORKERS,args...>(test::test_identifier);
        if constexpr (test::template valid<RT_SHAPE, ST_SHAPE, H, W, NUM_WORKERS, args...>::value) {
            constexpr int SIZE = H*W*RT_SHAPE::cols*RT_SHAPE::rows;
            // initialize
            dtype *d_i, *d_o;
            std::vector<float> i_ref(SIZE);
            std::vector<float> o_ref(SIZE);
            initialize(&d_i, &d_o, i_ref, o_ref);
            // make descriptors
            using GTL_I = typename kittens::gl<dtype, 1, 1, H*RT_SHAPE::rows, W*RT_SHAPE::cols>;
            using GTL_O = typename kittens::gl<dtype, 1, 1, W*RT_SHAPE::cols, H*RT_SHAPE::rows>;
            GTL_I input (d_i, nullptr, nullptr, nullptr, nullptr);
            GTL_O output(d_o, nullptr, nullptr, nullptr, nullptr);
            // run kernel
            hipFuncSetAttribute(
                reinterpret_cast<void *>(transpose_global_wrapper_2d<test, RT_SHAPE, ST_SHAPE, dtype, H, W, NUM_WORKERS, GTL_I, GTL_O, args...>),
                hipFuncAttributeMaxDynamicSharedMemorySize,
                kittens::MAX_SHARED_MEMORY / 2 // half the shared memory because permlane32_swap uses shared memory
            );
            transpose_global_wrapper_2d<test, RT_SHAPE, ST_SHAPE, dtype, H, W, NUM_WORKERS, GTL_I, GTL_O, args...><<<1, NUM_WORKERS*kittens::WARP_THREADS, kittens::MAX_SHARED_MEMORY / 2>>>(input, output);
            // fill in correct results on cpu
            test::template host_func<RT_SHAPE, ST_SHAPE, H, W, NUM_WORKERS, GTL_I, GTL_O, args...>(i_ref, o_ref);
            // check and cleanup
            this_result.result = validate(d_i, d_o, i_ref, o_ref, this_result.label, H*RT_SHAPE::rows, 1e-2); // mma's sometimes produce small errors. this appears to be hardware.
        }
        else {
            this_result.result = test_result::INVALID;
        }
        results.push_back(this_result);
    }
};
template<typename test, typename RT_SHAPE, typename ST_SHAPE, int MAX_H=8, int MAX_W=8, int NUM_WORKERS=1, typename... args> using transpose_sweep_size = loop_h<transpose_wrapper_2d, test, RT_SHAPE, ST_SHAPE, MAX_H, MAX_W, NUM_WORKERS, MAX_H, args...>;
template<typename test, typename RT_SHAPE, typename ST_SHAPE, int MAX_H=8, int MAX_W=8, int NUM_WORKERS=1, typename... args> using transpose_sweep_size_warp = transpose_sweep_size<test, RT_SHAPE, ST_SHAPE, MAX_H, MAX_W, NUM_WORKERS, args...>;

struct test_swap_layout {
    using dtype = kittens::bf16;
    template<typename RT_SHAPE, typename ST_SHAPE, int H, int W, int NW, kittens::ducks::rt_layout::all L1, typename RT_SHAPE2, kittens::ducks::rt_layout::all L2> using valid = std::bool_constant<NW == 1 && W*H<=64 && 
        ((RT_SHAPE2::cols >= RT_SHAPE::cols) ? ((W % (RT_SHAPE2::cols / RT_SHAPE::cols)) == 0) : ((RT_SHAPE::cols % RT_SHAPE2::cols) == 0)) && 
        ((RT_SHAPE2::rows >= RT_SHAPE::rows) ? ((H % (RT_SHAPE2::rows / RT_SHAPE::rows)) == 0) : ((RT_SHAPE::rows % RT_SHAPE2::rows) == 0))>; // this is warp-level
    static inline const std::string test_identifier = "reg_swaplayout";
    template<typename RT_SHAPE, typename ST_SHAPE, int H, int W, int NW, gl_t GL, kittens::ducks::rt_layout::all L1, typename RT_SHAPE2, kittens::ducks::rt_layout::all L2> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<typename RT_SHAPE, typename ST_SHAPE, typename dtype, int H, int W, int NW, gl_t GL, kittens::ducks::rt_layout::all L1, typename RT_SHAPE2, kittens::ducks::rt_layout::all L2> __device__ static void device_func(const GL input, const GL output) {
        kittens::rt<dtype, RT_SHAPE::rows*H, RT_SHAPE::cols*W, L1, RT_SHAPE> reg_tile;
        kittens::load(reg_tile, input, {});
        kittens::rt<dtype, RT_SHAPE::rows*H, RT_SHAPE::cols*W, L2, RT_SHAPE2> reg_tile_other_layout;
        kittens::swap_layout(reg_tile_other_layout, reg_tile);
        kittens::store(output, reg_tile_other_layout, {});
    }
};

struct test_transpose {
    using dtype = kittens::bf16;
    template<typename RT_SHAPE, typename ST_SHAPE, int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_transpose";
    template<typename RT_SHAPE, typename ST_SHAPE, int H, int W, int NW, gl_t GTL_I, gl_t GTL_O, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < H*RT_SHAPE::rows; i++)
            for(int j = 0; j < W*RT_SHAPE::cols; j++)
                o_ref[i+j*H*RT_SHAPE::rows] = i_ref[i*W*RT_SHAPE::cols+j];
    }
    template<typename RT_SHAPE, typename ST_SHAPE, typename dtype, int H, int W, int NW, gl_t GTL_I, gl_t GTL_O, kittens::ducks::rt_layout::all L> __device__ static void device_func(const GTL_I input, GTL_O output) {
        kittens::rt<dtype, RT_SHAPE::rows*H, RT_SHAPE::cols*W, L, RT_SHAPE> reg_tile;
        kittens::rt<dtype, RT_SHAPE::cols*W, RT_SHAPE::rows*H, typename kittens::ducks::rt_layout::transpose<L>::type, typename kittens::ducks::rt_shape::transpose<RT_SHAPE>::type> reg_tile_transpose;
        kittens::load(reg_tile, input, {});
        kittens::transpose(reg_tile_transpose, reg_tile);
        kittens::store(output, reg_tile_transpose, {});
    }
};
struct test_type_convert {
    template<typename RT_SHAPE, typename ST_SHAPE, int H, int W, int NW, typename T2, typename U2> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
        static inline const std::string test_identifier = "reg_typeconvert";
        template<typename RT_SHAPE, typename ST_SHAPE, int H, int W, int NW, gl_t GL, typename T2, typename U2> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<typename RT_SHAPE, typename ST_SHAPE, typename dtype, int H, int W, int NW, gl_t GL, typename T2, typename U2> __device__ static void device_func(const GL input, const GL output) {
        kittens::rt<U2, RT_SHAPE::rows*H, RT_SHAPE::cols*W, kittens::ducks::rt_layout::row, RT_SHAPE> reg_tile_U2;
        kittens::rt<T2, RT_SHAPE::rows*H, RT_SHAPE::cols*W, kittens::ducks::rt_layout::row, RT_SHAPE> reg_tile_T2;
        kittens::load(reg_tile_U2, input, {});
        kittens::copy(reg_tile_T2, reg_tile_U2);
        kittens::store(output, reg_tile_T2, {});
    }
};

struct test_subtile {
    using dtype = kittens::bf16;
    template<typename RT_SHAPE, typename ST_SHAPE, int H, int W, int NW, typename ST_H> using valid = std::bool_constant<NW == 1 && (H%ST_H::value)==0 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_subtile";
    template<typename RT_SHAPE, typename ST_SHAPE, int H, int W, int NW, gl_t GL, typename ST_H> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < H*RT_SHAPE::rows; i++)
            for(int j = 0; j < W*RT_SHAPE::cols; j++)
                o_ref[i*W*RT_SHAPE::cols + j] = i_ref[i*W*RT_SHAPE::cols + j] + float(i / (ST_H::value * RT_SHAPE::rows));
    }
    template<typename RT_SHAPE, typename ST_SHAPE, typename dtype, int H, int W, int NW, gl_t GL, typename ST_H> __device__ static void device_func(const GL input, const GL output) {
        kittens::rt<dtype, RT_SHAPE::rows*H, RT_SHAPE::cols*W, kittens::ducks::rt_layout::row, RT_SHAPE> reg_tile;
        kittens::load(reg_tile, input, {});
        #pragma unroll
        for(int i = 0; i < H/ST_H::value; i++) {
            auto &ref = kittens::subtile_inplace<ST_H::value*RT_SHAPE::rows>(reg_tile, i);
            kittens::add(ref, ref, dtype(i));
        }
        kittens::store(output, reg_tile, {});
    }
};

struct test_make_causal {
    using dtype = kittens::bf16;
    template<typename RT_SHAPE, typename ST_SHAPE, int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && H==W && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_make_causal";
    template<typename RT_SHAPE, typename ST_SHAPE, int H, int W, int NW, gl_t GL, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < H*RT_SHAPE::rows; i++)
            for(int j = 0; j < W*RT_SHAPE::cols; j++)
                o_ref[i*W*RT_SHAPE::cols + j] = j<=i ? i_ref[i*W*RT_SHAPE::cols + j] : 0;
    }
    template<typename RT_SHAPE, typename ST_SHAPE, typename dtype, int H, int W, int NW, gl_t GL, kittens::ducks::rt_layout::all L> __device__ static void device_func(const GL input, const GL output) {
        kittens::rt<dtype, RT_SHAPE::rows*H, RT_SHAPE::cols*W, L, RT_SHAPE> reg_tile;
        kittens::load(reg_tile, input, {});
        kittens::make_causal(reg_tile, reg_tile);
        kittens::store(output, reg_tile, {});
    }
};

struct test_tril {
    template<typename RT_SHAPE, typename ST_SHAPE, int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && H==W && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_tril";
    template<typename RT_SHAPE, typename ST_SHAPE, int H, int W, int NW, gl_t GL, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        // triangular lower, with diagonal starting at row_idx 4
        for(int i = 0; i < H*RT_SHAPE::rows; i++)
            for(int j = 0; j < W*RT_SHAPE::cols; j++)
                o_ref[i*W*RT_SHAPE::cols + j] = i>=j+(4*H) ? i_ref[i*W*RT_SHAPE::cols + j] : 0;
    }
    template<typename RT_SHAPE, typename ST_SHAPE, typename dtype, int H, int W, int NW, gl_t GL, kittens::ducks::rt_layout::all L> __device__ static void device_func(const GL input, const GL output) {
        kittens::rt<dtype, RT_SHAPE::rows*H, RT_SHAPE::cols*W, L, RT_SHAPE> reg_tile;
        kittens::load(reg_tile, input, {});
        kittens::tril(reg_tile, reg_tile, 4*H);
        kittens::store(output, reg_tile, {});
    }
};
struct test_triu {
    template<typename RT_SHAPE, typename ST_SHAPE, int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && H==W && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_triu";
    template<typename RT_SHAPE, typename ST_SHAPE, int H, int W, int NW, gl_t GL, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        // triangular upper, with diagonal starting at row_idx 4
        for(int i = 0; i < H*RT_SHAPE::rows; i++)
            for(int j = 0; j < W*RT_SHAPE::cols; j++)
                o_ref[i*W*RT_SHAPE::cols + j] = i<=j+(4*H) ? i_ref[i*W*RT_SHAPE::cols + j] : 0;
    }
    template<typename RT_SHAPE, typename ST_SHAPE, typename dtype, int H, int W, int NW, gl_t GL, kittens::ducks::rt_layout::all L> __device__ static void device_func(const GL input, const GL output) {
        kittens::rt<dtype, RT_SHAPE::rows*H, RT_SHAPE::cols*W, L, RT_SHAPE> reg_tile;
        kittens::load(reg_tile, input, {});
        kittens::triu(reg_tile, reg_tile, 4*H);
        kittens::store(output, reg_tile, {});
    }
};

struct test_right_fill {
    using dtype = kittens::bf16;
    template<typename RT_SHAPE, typename ST_SHAPE, int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && H==W && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_right_fill";
    template<typename RT_SHAPE, typename ST_SHAPE, int H, int W, int NW, gl_t GL, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        // here, set everything to from and right of col_idx 8 is set to zero
        for(int i = 0; i < H*RT_SHAPE::rows; i++) 
            for(int j = 0; j < W*RT_SHAPE::cols; j++) 
            o_ref[i*W*RT_SHAPE::cols + j] = (j < (8 * W)) ? i_ref[i*W*RT_SHAPE::cols + j] : 0;
    }
    template<typename RT_SHAPE, typename ST_SHAPE, typename dtype, int H, int W, int NW, gl_t GL, kittens::ducks::rt_layout::all L> __device__ static void device_func(const GL input, const GL output) {
        kittens::rt<dtype, RT_SHAPE::rows*H, RT_SHAPE::cols*W, L, RT_SHAPE> reg_tile;
        kittens::load(reg_tile, input, {});
        kittens::right_fill(reg_tile, reg_tile, 8 * W);
        kittens::store(output, reg_tile, {});
    }
};

struct test_left_fill {
    using dtype = kittens::bf16;
    template<typename RT_SHAPE, typename ST_SHAPE, int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && H==W && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_left_fill";
    template<typename RT_SHAPE, typename ST_SHAPE, int H, int W, int NW, gl_t GL, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        // here, set everything to from and left of col_idx 8 is set to zero
        for(int i = 0; i < H*RT_SHAPE::rows; i++) 
            for(int j = 0; j < W*RT_SHAPE::cols; j++) 
                o_ref[i*W*RT_SHAPE::cols + j] = (j >= (8 * W)) ? i_ref[i*W*RT_SHAPE::cols + j] : 0;
    }
    template<typename RT_SHAPE, typename ST_SHAPE, typename dtype, int H, int W, int NW, gl_t GL, kittens::ducks::rt_layout::all L> __device__ static void device_func(const GL input, const GL output) {
        kittens::rt<dtype, RT_SHAPE::rows*H, RT_SHAPE::cols*W, L, RT_SHAPE> reg_tile;
        kittens::load(reg_tile, input, {});
        kittens::left_fill(reg_tile, reg_tile, 8 * W);
        kittens::store(output, reg_tile, {});
    }
};

struct test_lower_fill {
    using dtype = kittens::bf16;
    template<typename RT_SHAPE, typename ST_SHAPE, int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && H==W && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_lower_fill";
    template<typename RT_SHAPE, typename ST_SHAPE, int H, int W, int NW, gl_t GL, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        // here, set everything to from and lower of row_idx 8 is set to zero
        for(int i = 0; i < H*RT_SHAPE::rows; i++) 
            for(int j = 0; j < W*RT_SHAPE::cols; j++) 
                o_ref[i*W*RT_SHAPE::cols + j] = (i < (8 * H)) ? i_ref[i*W*RT_SHAPE::cols + j] : 0;
    }
    template<typename RT_SHAPE, typename ST_SHAPE, typename dtype, int H, int W, int NW, gl_t GL, kittens::ducks::rt_layout::all L> __device__ static void device_func(const GL input, const GL output) {
        kittens::rt<dtype, RT_SHAPE::rows*H, RT_SHAPE::cols*W, L, RT_SHAPE> reg_tile;
        kittens::load(reg_tile, input, {});
        kittens::lower_fill(reg_tile, reg_tile, 8 * H);
        kittens::store(output, reg_tile, {});
    }
};
struct test_upper_fill {
    using dtype = kittens::bf16;
    template<typename RT_SHAPE, typename ST_SHAPE, int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && H==W && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_upper_fill";
    template<typename RT_SHAPE, typename ST_SHAPE, int H, int W, int NW, gl_t GL, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        // here, set everything to from and upper of row_idx 8 is set to zero
        for(int i = 0; i < H*RT_SHAPE::rows; i++) 
            for(int j = 0; j < W*RT_SHAPE::cols; j++) 
                o_ref[i*W*RT_SHAPE::cols + j] = (i >= ((8 * H))) ? i_ref[i*W*RT_SHAPE::cols + j] : 0;
    }
    template<typename RT_SHAPE, typename ST_SHAPE, typename dtype, int H, int W, int NW, gl_t GL, kittens::ducks::rt_layout::all L> __device__ static void device_func(const GL input, const GL output) {
        kittens::rt<dtype, RT_SHAPE::rows*H, RT_SHAPE::cols*W, L, RT_SHAPE> reg_tile;
        kittens::load(reg_tile, input, {});
        kittens::upper_fill(reg_tile, reg_tile, 8 * H);
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

    transpose_sweep_size_warp<test_transpose, RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, kittens::ducks::rt_layout::col>::run(results);
    transpose_sweep_size_warp<test_transpose, RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, kittens::ducks::rt_layout::row>::run(results);

    sweep_size_2d_warp<test_type_convert, RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, float, kittens::bf16>::run(results);
    sweep_size_2d_warp<test_type_convert, RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, kittens::bf16, float>::run(results);
    sweep_size_2d_warp<test_type_convert, RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, float, kittens::half>::run(results);
    sweep_size_2d_warp<test_type_convert, RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, kittens::half, float>::run(results);
    sweep_size_2d_warp<test_type_convert, RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, kittens::half, kittens::bf16>::run(results);
    sweep_size_2d_warp<test_type_convert, RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, kittens::bf16, kittens::half>::run(results);

    sweep_size_2d_warp<test_subtile, RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, std::integral_constant<int, 1>>::run(results);
    sweep_size_2d_warp<test_subtile, RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, std::integral_constant<int, 2>>::run(results);
    sweep_size_2d_warp<test_subtile, RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, std::integral_constant<int, 3>>::run(results);
    sweep_size_2d_warp<test_subtile, RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, std::integral_constant<int, 4>>::run(results);

    sweep_size_2d_warp<test_right_fill, RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<test_right_fill, RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d_warp<test_left_fill,  RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<test_left_fill,  RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d_warp<test_lower_fill, RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<test_lower_fill, RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d_warp<test_upper_fill, RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<test_upper_fill, RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, kittens::ducks::rt_layout::col>::run(results);

    sweep_size_2d_warp<test_tril, RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<test_tril, RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d_warp<test_triu, RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<test_triu, RT_SHAPE, ST_SHAPE, SIZE, SIZE, 1, kittens::ducks::rt_layout::col>::run(results);
}

void warp::reg::tile::conversions::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/register/tile/conversions tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_0 ? 1  :
                         INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;

    // Test layout swaps
    using DEFAULT_ST_SHAPE = kittens::ducks::st_shape::st_16x16;

    using FROM_RT_SHAPE_1 = kittens::ducks::rt_shape::rt_16x16;
    using TO_RT_SHAPE_1   = kittens::ducks::rt_shape::rt_16x32;

    using FROM_RT_LAYOUT_1 = kittens::ducks::rt_layout::col;
    using TO_RT_LAYOUT_1   = kittens::ducks::rt_layout::col;
    sweep_size_2d_warp<test_swap_layout, FROM_RT_SHAPE_1, DEFAULT_ST_SHAPE, SIZE, SIZE, 1, FROM_RT_LAYOUT_1, TO_RT_SHAPE_1, TO_RT_LAYOUT_1>::run(results);

    using FROM_RT_SHAPE_2 = kittens::ducks::rt_shape::rt_32x32;
    using TO_RT_SHAPE_2   = kittens::ducks::rt_shape::rt_16x32_4;

    using FROM_RT_LAYOUT_2 = kittens::ducks::rt_layout::col;
    using TO_RT_LAYOUT_2   = kittens::ducks::rt_layout::col;
    sweep_size_2d_warp<test_swap_layout, FROM_RT_SHAPE_2, DEFAULT_ST_SHAPE, SIZE, SIZE, 1, FROM_RT_LAYOUT_2, TO_RT_SHAPE_2, TO_RT_LAYOUT_2>::run(results);

    // Test make causal
    sweep_size_2d_warp<test_make_causal, kittens::ducks::rt_shape::rt_16x16, DEFAULT_ST_SHAPE, SIZE, SIZE, 1, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d_warp<test_make_causal, kittens::ducks::rt_shape::rt_32x32, DEFAULT_ST_SHAPE, SIZE, SIZE, 1, kittens::ducks::rt_layout::col>::run(results);

    // Test other conversions
    test_generator<kittens::ducks::rt_shape::rt_16x32>(results);
    test_generator<kittens::ducks::rt_shape::rt_32x16>(results);
    test_generator<kittens::ducks::rt_shape::rt_16x16>(results);
    test_generator<kittens::ducks::rt_shape::rt_32x32>(results);
    test_generator<kittens::ducks::rt_shape::rt_32x32_8>(results);
    test_generator<kittens::ducks::rt_shape::rt_16x32_4>(results);
    test_generator<kittens::ducks::rt_shape::rt_32x16_4>(results);
}

#endif