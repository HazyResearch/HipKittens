#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

constexpr int B = 64;
constexpr int N = 8192;
constexpr int D = 128;
constexpr int BLOCK_SIZE = 32;

#define NUM_WORKERS (4)
#define NUM_THREADS (NUM_WORKERS*kittens::WARP_THREADS)

using namespace kittens;
using shape = kittens::rt_32x32_8_s;

template<int _D> struct softmax_globals {
    static constexpr int d = _D;

    using x_gl = gl<bf16, -1, -1, -1, -1>;
    using o_gl = gl<bf16, -1, -1, -1, -1>;

    x_gl x;
    o_gl o;

    // Each warp handles 32 rows. 4 warps per block = 128 rows per block.
    dim3 grid() { return dim3(N / (NUM_WORKERS * BLOCK_SIZE), B, 1); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 0; }
};

template<int _D>
__global__ void softmax_tk(const softmax_globals<_D> g) {
    auto warpid = kittens::warpid();
    const int batch = blockIdx.y;
    const int tile_idx = blockIdx.x * NUM_WORKERS + warpid;

    // 32 rows x 128 cols per warp
    using tile_t = rt<bf16, BLOCK_SIZE, _D, row_l, shape>;
    tile_t x_reg;
    // per-row reduction vectors — must use tile's col_vec type for layout compatibility
    typename tile_t::col_vec max_vec, sum_vec;

    load<2>(x_reg, g.x, {0, batch, tile_idx, 0});
    asm volatile("s_waitcnt vmcnt(0)");

    // Step 1: per-row max for numerical stability
    row_max(max_vec, x_reg);

    // Step 2: subtract max and exponentiate
    sub_row(x_reg, x_reg, max_vec);
    exp(x_reg, x_reg);

    // Step 3: per-row sum of exponentials
    row_sum(sum_vec, x_reg);

    // Step 4: normalize each row
    div_row(x_reg, x_reg, sum_vec);

    // Store result
    store(g.o, x_reg, {0, batch, tile_idx, 0});
}

template<int _D>
void dispatch_softmax(softmax_globals<_D> g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)softmax_tk<_D>, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    softmax_tk<_D><<<g.grid(), g.block(), mem_size>>>(g);
    hipDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "Softmax HipKittens kernel";
    py::bind_function<dispatch_softmax<D>>(m, "dispatch_softmax",
        &softmax_globals<D>::x,
        &softmax_globals<D>::o
    );
}
