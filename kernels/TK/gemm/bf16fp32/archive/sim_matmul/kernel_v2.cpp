#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

constexpr int BLOCK_SIZE_M  = 128; 
constexpr int BLOCK_SIZE_N  = 256; 
 
constexpr int BLOCK_WARPS_M = 2; 
constexpr int BLOCK_WARPS_N = 4;

constexpr int WARP_SIZE_M   = 64;
constexpr int WARP_SIZE_N   = 64;

constexpr int K_STEP        = 16;

// {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
            // num_warps=4, num_stages=2),

#define NUM_WARPS 8
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

#define M 8192
#define K 8192
#define N 8192

using _gl_A = gl<bf16, -1, -1, -1, -1>;
using _gl_B = gl<bf16, -1, -1, -1, -1>;
using _gl_C = gl<float, -1, -1, -1, -1>;

struct micro_globals {
    _gl_A a;
    _gl_B b;
    _gl_C c;
    float alpha;
    float beta;

    dim3 grid()  { return dim3(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M); } 
    dim3 block() { return dim3(NUM_THREADS); } 
    size_t dynamic_shared_memory() { return 1024; }
};

__global__ __launch_bounds__(NUM_THREADS, 4)
void micro_tk(const micro_globals g) {
    rt_bf<WARP_SIZE_M, K_STEP> a_reg;
    rt_bf<K_STEP, WARP_SIZE_N, ducks::rt_layout::col> b_reg;
    rt_fl<WARP_SIZE_M, WARP_SIZE_N, ducks::rt_layout::col> C_accum;
    zero(C_accum);

    int row = blockIdx.y;
    int col = blockIdx.x;

    int warp_id = kittens::warpid();
    int warp_row = warp_id / BLOCK_WARPS_N;  
    int warp_col = warp_id % BLOCK_WARPS_N; 

    int num_tiles = K / K_STEP;
    for (int tile = 0; tile < num_tiles; ++tile) {
        load(a_reg, g.a, {0, 0, row * BLOCK_WARPS_M + warp_row, tile});
        load(b_reg, g.b, {0, 0, tile, col * BLOCK_WARPS_N + warp_col});
        __syncthreads();

        mma_AB(C_accum, a_reg, b_reg, C_accum);
    }
    store(g.c, C_accum, {0, 0, row * BLOCK_WARPS_M + warp_row, col * BLOCK_WARPS_N + warp_col});
}

void dispatch_micro(micro_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute(
        (void*)micro_tk,
        hipFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );

    micro_tk<<<g.grid(), g.block(), mem_size>>>(g);
    hipDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module";
    py::bind_kernel<micro_tk>(m, "micro_tk", &micro_globals::a, &micro_globals::b, &micro_globals::c,
                              &micro_globals::alpha, &micro_globals::beta); 
    py::bind_function<dispatch_micro>(m, "dispatch_micro", &micro_globals::a, &micro_globals::b, &micro_globals::c,
                                      &micro_globals::alpha, &micro_globals::beta);
}


