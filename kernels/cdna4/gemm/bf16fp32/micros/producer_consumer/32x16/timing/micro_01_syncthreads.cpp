#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

#define NUM_WARPS 1
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)
#define ITERS 100

// problem dims
constexpr int b = 1;
constexpr int h = 1;
constexpr int d = 32;          
constexpr int n = 32;      
constexpr int BLOCK_SIZE = n;


using G = kittens::group<NUM_WARPS>;

struct micro_globals {
    gl<bf16, -1, -1, -1, -1> in;   
    gl<bf16, -1, -1, -1, -1> out;  
    gl<float, -1, -1, -1, -1> results;
    dim3 grid()  { return dim3(n / BLOCK_SIZE); }  
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY - 2048; }
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const micro_globals g) {
    const int tile_idx = blockIdx.x;
    
    uint64_t t0, t1;

    rt_bf<n, d, row_l> tile;

    for (int i = 0; i < ITERS; ++i) {
        load(tile, g.in, {0, 0, tile_idx, 0}); 
        if (threadIdx.x == 0) t0 = clock64();
        __syncthreads();
        if (threadIdx.x == 0) t1 = clock64();
        store(g.out, tile, {0, 0, tile_idx, 0});

        if (threadIdx.x == 0) {
            g.results[blockIdx.x * ITERS + i] = (t1 - t0);
        }
    }
}

void dispatch_micro(micro_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)micro_tk, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    micro_tk<<<g.grid(), g.block(), mem_size>>>(g);
    hipDeviceSynchronize();
}


PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module";
    py::bind_function<dispatch_micro>(m, "dispatch_micro", &micro_globals::in, &micro_globals::out, &micro_globals::results);
}
