#include "global_to_shared.cuh"

#ifdef TEST_WARP_MEMORY_TILE_GLOBAL_TO_SHARED

// Mechanism selector for the gfx1250 global -> shared load under test.
enum class g2s_mech { reg_mediated, async, tdm };

template<typename T, g2s_mech MECH = g2s_mech::reg_mediated>
struct st_load_store {
    using dtype = T;
    // `load_async` issues one b128 (16B) per lane, so it requires a 2-byte dtype
    // and the gfx1250 padded production layout. `load_tdm` is the intended path
    // for that shape: the D# carries the shape's pad_interval/pad_amount fields.
    template<typename RT_SHAPE, typename ST_SHAPE, int H, int W, int NW, typename axis> using valid = std::bool_constant<
        (NW == 1 && W*H<=64) 
        && (W*H*ST_SHAPE::cols*ST_SHAPE::rows*sizeof(T) <= kittens::MAX_SHARED_MEMORY)
        && ((W*H*ST_SHAPE::cols*ST_SHAPE::rows*sizeof(T)) % (kittens::WARP_THREADS * ST_SHAPE::template bytes_per_thread<T>()) == 0)
        && (MECH == g2s_mech::reg_mediated
            || (MECH == g2s_mech::async
                && sizeof(T) == 2
                && std::is_same_v<ST_SHAPE, kittens::ducks::st_shape::st_16x32_padded<>>))
    >;
    static inline const std::string test_identifier =
        (MECH == g2s_mech::async ?
            (std::is_same_v<T, kittens::bf16> ? "shared_async_loadstore_gmem=bf16" :
             std::is_same_v<T, kittens::half> ? "shared_async_loadstore_gmem=half" :
                                                "shared_async_loadstore_gmem=other") :
         MECH == g2s_mech::tdm ?
            (std::is_same_v<T, kittens::bf16> ? "shared_tdm_loadstore_gmem=bf16" :
             std::is_same_v<T, kittens::half> ? "shared_tdm_loadstore_gmem=half" :
             std::is_same_v<T, kittens::fp8e4m3> ? "shared_tdm_loadstore_gmem=fp8e4m3" :
                                                "shared_tdm_loadstore_gmem=float") :
            (std::is_same_v<T, kittens::bf16> ? "shared_loadstore_gmem=bf16" :
             std::is_same_v<T, kittens::half> ? "shared_loadstore_gmem=half" :
             std::is_same_v<T, kittens::fp8e4m3> ? "shared_loadstore_gmem=fp8e4m3" :
                                                "shared_loadstore_gmem=float"));
    template<int H, int W, int NW, kittens::ducks::gl::all GL, typename axis> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<typename RT_SHAPE, typename ST_SHAPE, typename dtype, int H, int W, int NW, kittens::ducks::gl::all GL, typename axis> __device__ static void device_func(const GL &input, const GL &output) {
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the HIP shared memory
        kittens::shared_allocator<1024> al((int*)&__shm[0]);
        using ST = kittens::st<T, ST_SHAPE::rows*H, ST_SHAPE::cols*W, ST_SHAPE>;
        ST &shared_tile = al.allocate<ST>();

        // gfx1250 native global -> shared is row-tiled (axis 2 only). The
        // register-mediated `load` and async `load_async` scatter into the
        // subtile-major padded map via `lds_offset(flat)`; TDM lands row-major
        // padded data via the D# fields and is read back with `store_tdm`.
        const int row_stride = (int)input.cols();
        const int tensor_rows = (int)input.rows();
        const int tensor_cols = (int)input.cols();
        const int num_rows = tensor_rows / ST::rows;
        const int num_cols = tensor_cols / ST::cols;
        for (int i = 0; i < (int)input.batch(); i++)
            for (int j = 0; j < (int)input.depth(); j++)
                for (int k = 0; k < num_rows; k++)
                    for (int l = 0; l < num_cols; l++) {
            if constexpr (MECH == g2s_mech::async) {
                kittens::load_async<NW*kittens::WARP_THREADS>(shared_tile, input, {i, j, k, l}, row_stride);
                kittens::sync::wait_async();
            } else {
                kittens::load<NW*kittens::WARP_THREADS>(shared_tile, input, {i, j, k, l}, row_stride);
            }
            kittens::sync::sync();
            kittens::store<NW*kittens::WARP_THREADS>(output, shared_tile, {i, j, k, l}, row_stride);
            kittens::sync::wait_store<0>();
            kittens::sync::sync();
        }
    }
};

// TDM round-trip on a dense ST-sized GL (row_stride == tile width). FFM-Lite
// models TDM correctly only when the D# row stride matches the tile width; the
// axis-sweep g2s harness uses a wider row stride and is not used for TDM.
template<typename T>
struct st_tdm_load_store {
    using dtype = T;
    template<typename RT_SHAPE, typename ST_SHAPE, int H, int W, int NW, typename axis> using valid = std::bool_constant<
        (NW == 1 && W*H<=64)
        && (W*H*ST_SHAPE::cols*ST_SHAPE::rows*sizeof(T) <= kittens::MAX_SHARED_MEMORY)
        && (sizeof(T) == 2)
        && std::is_same_v<ST_SHAPE, kittens::ducks::st_shape::st_16x32_padded<>>
    >;
    static inline const std::string test_identifier =
        std::is_same_v<T, kittens::bf16> ? "shared_tdm_loadstore_gmem=bf16" :
        std::is_same_v<T, kittens::half> ? "shared_tdm_loadstore_gmem=half" :
                                           "shared_tdm_loadstore_gmem=other";
    template<int H, int W, int NW, kittens::ducks::gl::all GL, typename axis> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref;
    }
    template<typename RT_SHAPE, typename ST_SHAPE, typename dtype, int H, int W, int NW, kittens::ducks::gl::all GL, typename axis> __device__ static void device_func(const GL &input, const GL &output) {
        extern __shared__ kittens::alignment_dummy __shm[];
        kittens::shared_allocator<1024> al((int*)&__shm[0]);
        using ST = kittens::st<T, ST_SHAPE::rows*H, ST_SHAPE::cols*W, ST_SHAPE>;
        ST &shared_tile = al.allocate<ST>();

        const int tile_rows = ST::rows;
        const int tile_cols = ST::cols;
        const int row_stride = tile_cols;
        const int num_rows = (int)input.rows() / tile_rows;
        const int num_cols = (int)input.cols() / tile_cols;
        for (int k = 0; k < num_rows; k++)
            for (int l = 0; l < num_cols; l++) {
                if (kittens::laneid() == 0) {
                    kittens::load_tdm(shared_tile, input, {0, 0, k, l},
                                      tile_rows, tile_cols, row_stride);
                }
                kittens::sync::wait_tdm();
                kittens::sync::sync();
                kittens::store_tdm<NW*kittens::WARP_THREADS>(output, shared_tile, {0, 0, k, l}, row_stride);
                kittens::sync::wait_store<0>();
                kittens::sync::sync();
            }
    }
};

template<typename test, typename RT_SHAPE, typename ST_SHAPE, int H, int W, int NUM_WORKERS=1>
struct tdm_wrapper_2d {
    using dtype = gmem_dtype<test>;
    static void run(test_data& results) {
        test_info this_result;
        this_result.label = generate_test_name<RT_SHAPE, ST_SHAPE, H, W, NUM_WORKERS>(test::test_identifier);
        if constexpr (test::template valid<RT_SHAPE, ST_SHAPE, H, W, NUM_WORKERS, std::integral_constant<int, 0>>::value) {
            constexpr int SIZE = H*W*ST_SHAPE::cols*ST_SHAPE::rows;
            dtype *d_i, *d_o;
            std::vector<float> i_ref(SIZE);
            std::vector<float> o_ref(SIZE);
            initialize(&d_i, &d_o, i_ref, o_ref);
            using GL = typename kittens::gl<dtype, 1, 1, H*ST_SHAPE::rows, W*ST_SHAPE::cols>;
            GL input(d_i, nullptr, nullptr, nullptr, nullptr);
            GL output(d_o, nullptr, nullptr, nullptr, nullptr);
            hipFuncSetAttribute(
                reinterpret_cast<const void*>(global_wrapper_2d<test, RT_SHAPE, ST_SHAPE, dtype, H, W, NUM_WORKERS, GL, std::integral_constant<int, 0>>),
                hipFuncAttributeMaxDynamicSharedMemorySize,
                kittens::MAX_SHARED_MEMORY / 2
            );
            global_wrapper_2d<test, RT_SHAPE, ST_SHAPE, dtype, H, W, NUM_WORKERS, GL, std::integral_constant<int, 0>><<<1, NUM_WORKERS*kittens::WARP_THREADS, kittens::MAX_SHARED_MEMORY / 2>>>(input, output);
            test::template host_func<H, W, NUM_WORKERS, GL, std::integral_constant<int, 0>>(i_ref, o_ref);
            this_result.result = validate(d_i, d_o, i_ref, o_ref, this_result.label, W*ST_SHAPE::cols);
        } else {
            this_result.result = test_result::INVALID;
        }
        results.push_back(this_result);
    }
};

template<typename test, typename RT_SHAPE, typename ST_SHAPE, int MAX_H, int MAX_W, int NUM_WORKERS=1>
using tdm_sweep_size_2d = loop_h<tdm_wrapper_2d, test, RT_SHAPE, ST_SHAPE, MAX_H, MAX_W, NUM_WORKERS, MAX_H>;
template<typename test, typename RT_SHAPE, typename ST_SHAPE, int MAX_H, int MAX_W>
using tdm_sweep_size_2d_warp = tdm_sweep_size_2d<test, RT_SHAPE, ST_SHAPE, MAX_H, MAX_W, 1>;

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

    // gfx1250 `load(st, gl, idx, row_stride)` is row-tiled, so only the
    // axis-2 (row) sweep is meaningful.
    g2s_sweep_size_2d_warp<st_load_store<kittens::bf16>, RT_SHAPE, ST_SHAPE, SIZE, SIZE, I2_t>::run(results);
    g2s_sweep_size_2d_warp<st_load_store<kittens::half>, RT_SHAPE, ST_SHAPE, SIZE, SIZE, I2_t>::run(results);
    g2s_sweep_size_2d_warp<st_load_store<float>, RT_SHAPE, ST_SHAPE, SIZE, SIZE, I2_t>::run(results);

    // The async global_load_async_to_lds_b128 path (bf16/half + padded layout);
    // `valid` skips the combos it doesn't apply to.
    g2s_sweep_size_2d_warp<st_load_store<kittens::bf16, g2s_mech::async>, RT_SHAPE, ST_SHAPE, SIZE, SIZE, I2_t>::run(results);
    g2s_sweep_size_2d_warp<st_load_store<kittens::half, g2s_mech::async>, RT_SHAPE, ST_SHAPE, SIZE, SIZE, I2_t>::run(results);

    // Hardware TDM (dense ST-sized GL; bf16/half + st_16x32_padded only).
    tdm_sweep_size_2d_warp<st_tdm_load_store<kittens::bf16>, RT_SHAPE, ST_SHAPE, SIZE, SIZE>::run(results);
    tdm_sweep_size_2d_warp<st_tdm_load_store<kittens::half>, RT_SHAPE, ST_SHAPE, SIZE, SIZE>::run(results);
}



void warp::memory::tile::global_to_shared::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/memory/tile/global_to_shared tests! -----\n" << std::endl;

    test_generator<kittens::ducks::st_shape::st_16x32_padded<>>(results);
    test_generator<kittens::ducks::st_shape::st_16x16>(results);
    test_generator<kittens::ducks::st_shape::st_16x16_swizzled>(results);
    test_generator<kittens::ducks::st_shape::st_32x32>(results);
    test_generator<kittens::ducks::st_shape::st_16x32>(results);
    test_generator<kittens::ducks::st_shape::st_32x16>(results);
    test_generator<kittens::ducks::st_shape::st_8x32>(results);
}
#endif
