#include "shared_to_register.cuh"

#ifdef TEST_WARP_MEMORY_TILE_SHARED_TO_REGISTER

namespace {

// Mirrors gfx1250_gemm tile geometry from kernels/gemm/bf16fp32/gfx1250/common.h.
constexpr int GEMM_BLOCK_M  = 64;
constexpr int GEMM_BLOCK_N  = 64;
constexpr int GEMM_K_STEP   = 32;
constexpr int GEMM_WARP_M   = GEMM_BLOCK_M / 2;  // WARPS_M = 2
constexpr int GEMM_WARP_N   = GEMM_BLOCK_N / 2;  // WARPS_N = 2
constexpr int GEMM_WARP_SLICE_ORIGIN = GEMM_WARP_M * GEMM_K_STEP; // warp index 1 along M/N

/**
 * @brief Round-trip the production GEMM operand path on gfx1250:
 *   reg-mediated `load(st, gl, …, row_stride)` into `st_bf` (padded)
 *   → WMMA `load(rt_bf, st, warp_origin_flat)` (`ds_load_b128`)
 *   → `store(gl, rt_bf)` for verification.
 *
 * There is no padded register→shared store; g2r out is the check.
 */
template<int ST_ROWS, int ST_COLS, int WARP_ROWS, int WARP_COLS, int ORIGIN_FLAT>
struct gemm_padded_s2r {
    using dtype = kittens::bf16;

    static_assert(ST_ROWS % 16 == 0 && ST_COLS % 32 == 0);
    static_assert(WARP_ROWS % 16 == 0 && WARP_COLS % 32 == 0);
    static_assert(WARP_ROWS <= ST_ROWS && WARP_COLS <= ST_COLS);

    static inline const std::string test_identifier = "shared_reg_gemm_chain_gmem=bf16";

    __host__ static void host_func(const std::vector<float>& i_ref, std::vector<float>& o_ref) {
        const int origin_row = ORIGIN_FLAT / ST_COLS;
        const int origin_col = ORIGIN_FLAT % ST_COLS;
        for (int i = 0; i < WARP_ROWS; ++i) {
            for (int j = 0; j < WARP_COLS; ++j) {
                o_ref[i * WARP_COLS + j] =
                    i_ref[(origin_row + i) * ST_COLS + (origin_col + j)];
            }
        }
    }

    __device__ static void device_func(const kittens::gl<dtype, 1, 1, ST_ROWS, ST_COLS>& input,
                                       const kittens::gl<dtype, 1, 1, WARP_ROWS, WARP_COLS>& output_chain,
                                       const kittens::gl<dtype, 1, 1, WARP_ROWS, WARP_COLS>& output_ref) {
        extern __shared__ kittens::alignment_dummy __shm[];
        kittens::shared_allocator<1024> al(reinterpret_cast<int*>(&__shm[0]));

        kittens::st_bf<ST_ROWS, ST_COLS>& shared_tile = al.allocate<kittens::st_bf<ST_ROWS, ST_COLS>>();

        // Tile coords for the g2r reference load (matches gemm_naive warp slice indexing).
        const int tile_r = (ORIGIN_FLAT / ST_COLS) / WARP_ROWS;
        const int tile_c = (ORIGIN_FLAT % ST_COLS) / WARP_COLS;

        const int row_stride = (int)input.cols();
        kittens::load<kittens::WARP_THREADS>(shared_tile, input, {0, 0, 0, 0}, row_stride);
        kittens::sync::sync();

        kittens::rt_bf<WARP_ROWS, WARP_COLS, kittens::ducks::rt_layout::row,
                       kittens::ducks::rt_shape::rt_16x32> reg_chain;
        kittens::load(reg_chain, shared_tile, ORIGIN_FLAT);
        kittens::sync::wait_ds<0>();
        kittens::store(output_chain, reg_chain, {0, 0, 0, 0});
        kittens::sync::wait_store<0>();

        kittens::rt_bf<WARP_ROWS, WARP_COLS, kittens::ducks::rt_layout::row,
                       kittens::ducks::rt_shape::rt_16x32> reg_ref;
        kittens::load(reg_ref, input, {0, 0, tile_r, tile_c});
        kittens::sync::sync();
        kittens::store(output_ref, reg_ref, {0, 0, 0, 0});
        kittens::sync::wait_store<0>();
    }
};

template<int ST_ROWS, int ST_COLS, int WARP_ROWS, int WARP_COLS, int ORIGIN_FLAT>
__global__ void gemm_padded_s2r_kernel(
    const kittens::gl<kittens::bf16, 1, 1, ST_ROWS, ST_COLS> input,
    const kittens::gl<kittens::bf16, 1, 1, WARP_ROWS, WARP_COLS> output_chain,
    const kittens::gl<kittens::bf16, 1, 1, WARP_ROWS, WARP_COLS> output_ref)
{
    gemm_padded_s2r<ST_ROWS, ST_COLS, WARP_ROWS, WARP_COLS, ORIGIN_FLAT>::device_func(
        input, output_chain, output_ref);
}

test_result compare_device_bf16(kittens::bf16* d_a, kittens::bf16* d_b, int size, float atol = 1e-2f,
                               int* first_mismatch = nullptr) {
    std::vector<kittens::bf16> a(size), b(size);
    hipMemcpy(a.data(), d_a, size * sizeof(kittens::bf16), hipMemcpyDeviceToHost);
    hipMemcpy(b.data(), d_b, size * sizeof(kittens::bf16), hipMemcpyDeviceToHost);
    HipCheckError();
    for (int i = 0; i < size; ++i) {
        const float fa = __bfloat162float(a[i]);
        const float fb = __bfloat162float(b[i]);
        if (std::abs(fa - fb) > atol) {
            if (first_mismatch) *first_mismatch = i;
            return test_result::FAILED;
        }
    }
    return test_result::PASSED;
}

template<int ST_ROWS, int ST_COLS, int WARP_ROWS, int WARP_COLS, int ORIGIN_FLAT>
void run_gemm_padded_s2r(test_data& results) {
    using dtype = kittens::bf16;
    test_info this_result;
    this_result.label = gemm_padded_s2r<ST_ROWS, ST_COLS, WARP_ROWS, WARP_COLS, ORIGIN_FLAT>::test_identifier
        + "_[st=" + std::to_string(ST_ROWS) + "x" + std::to_string(ST_COLS) + "]"
        + "_[warp=" + std::to_string(WARP_ROWS) + "x" + std::to_string(WARP_COLS) + "]"
        + "_[origin=" + std::to_string(ORIGIN_FLAT) + "]";

    constexpr int in_size  = ST_ROWS * ST_COLS;
    constexpr int out_size = WARP_ROWS * WARP_COLS;
    static_assert(sizeof(kittens::st_bf<ST_ROWS, ST_COLS>) <= kittens::MAX_SHARED_MEMORY / 2,
        "shared tile too large for test harness");

    const int origin_row = ORIGIN_FLAT / ST_COLS;
    const int origin_col = ORIGIN_FLAT % ST_COLS;
    if (origin_row + WARP_ROWS > ST_ROWS || origin_col + WARP_COLS > ST_COLS) {
        this_result.result = test_result::INVALID;
        results.push_back(this_result);
        return;
    }

    dtype *d_i, *d_chain, *d_ref;
    std::vector<float> i_ref(in_size);
    std::vector<float> o_ref(out_size);
    initialize<dtype, dtype>(&d_i, &d_chain, i_ref, o_ref);
    hipMalloc(&d_ref, out_size * sizeof(dtype));
    HipCheckError();

    kittens::gl<dtype, 1, 1, ST_ROWS, ST_COLS> input(d_i, nullptr, nullptr, nullptr, nullptr);
    kittens::gl<dtype, 1, 1, WARP_ROWS, WARP_COLS> output_chain(d_chain, nullptr, nullptr, nullptr, nullptr);
    kittens::gl<dtype, 1, 1, WARP_ROWS, WARP_COLS> output_ref(d_ref, nullptr, nullptr, nullptr, nullptr);

    hipFuncSetAttribute(
        reinterpret_cast<const void*>(gemm_padded_s2r_kernel<ST_ROWS, ST_COLS, WARP_ROWS, WARP_COLS, ORIGIN_FLAT>),
        hipFuncAttributeMaxDynamicSharedMemorySize,
        kittens::MAX_SHARED_MEMORY / 2);
    gemm_padded_s2r_kernel<ST_ROWS, ST_COLS, WARP_ROWS, WARP_COLS, ORIGIN_FLAT>
        <<<1, kittens::WARP_THREADS, kittens::MAX_SHARED_MEMORY / 2>>>(input, output_chain, output_ref);

    gemm_padded_s2r<ST_ROWS, ST_COLS, WARP_ROWS, WARP_COLS, ORIGIN_FLAT>::host_func(i_ref, o_ref);

    int first_mismatch = -1;
    const test_result chain_vs_ref = compare_device_bf16(d_chain, d_ref, out_size, 1e-2f, &first_mismatch);
    const test_result ref_vs_cpu   = validate(d_i, d_ref, i_ref, o_ref, this_result.label + "_ref", WARP_COLS);
    hipFree(d_chain);

    std::cout << "test `" << this_result.label << "` ";
    if (chain_vs_ref == test_result::PASSED && ref_vs_cpu == test_result::PASSED) {
        std::cout << " -- PASSED" << std::endl;
        this_result.result = test_result::PASSED;
    } else {
        std::cout << " ----- ALERT! FAILED test `" << this_result.label << "` -----" << std::endl;
        if (chain_vs_ref != test_result::PASSED) {
            std::cout << "  g2s->s2r chain differs from g2r reference load";
            if (first_mismatch >= 0) {
                std::cout << " (first mismatch at index " << first_mismatch << ")";
            }
            std::cout << std::endl;
        }
        if (ref_vs_cpu != test_result::PASSED) {
            std::cout << "  g2r reference differs from CPU golden slice" << std::endl;
        }
        this_result.result = test_result::FAILED;
    }
    results.push_back(this_result);
}

template<typename T>
struct sharedreg_load_store {
    using dtype = T;

    // CDNA XOR-swizzle path: one plain st_16x16 subtile (H=W=1), rt_16x16, row layout.
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

} // namespace

void warp::memory::tile::shared_to_register::tests(test_data& results) {
    std::cout << "\n ----- Starting ops/warp/memory/tile/shared_to_register tests! -----\n" << std::endl;

    // Production gfx1250 GEMM operand chain (st_16x32_padded + WMMA ds_load_b128 s2r).
    run_gemm_padded_s2r<16, GEMM_K_STEP, 16, GEMM_K_STEP, 0>(results);
    run_gemm_padded_s2r<GEMM_WARP_M, GEMM_K_STEP, GEMM_WARP_M, GEMM_K_STEP, 0>(results);
    run_gemm_padded_s2r<GEMM_BLOCK_M, GEMM_K_STEP, GEMM_WARP_M, GEMM_K_STEP, 0>(results);
    run_gemm_padded_s2r<GEMM_BLOCK_M, GEMM_K_STEP, GEMM_WARP_M, GEMM_K_STEP, GEMM_WARP_SLICE_ORIGIN>(results);

    // CDNA XOR-swizzle templated path (st_16x16 only).
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
