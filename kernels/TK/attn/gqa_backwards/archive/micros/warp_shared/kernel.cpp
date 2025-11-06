#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

constexpr int b = 1;
constexpr int h = 1;
constexpr int n = 128;
constexpr int d = 32;

constexpr int BLOCK_SIZE = 32;

#define NUM_WARPS 4
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using G = kittens::group<NUM_WARPS>;

struct micro_globals {
    gl<bf16, -1, -1, -1, -1> in;
    gl<bf16, -1, -1, -1, -1> out;
    dim3 grid()  { return dim3(((n / BLOCK_SIZE + NUM_WARPS - 1) / NUM_WARPS)); } 
    dim3 block() { return dim3(NUM_THREADS); } 
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY-2048; }
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const micro_globals g) {

    int i = blockIdx.x;

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<n, d, ducks::st_layout::row> (&tile_smem)[NUM_WARPS] = al.allocate<st_bf<n, d, ducks::st_layout::row>, NUM_WARPS>();
    // load to smem
    load<2, false>(tile_smem[warpid()], g.in, {0, 0, i, 0});
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // load to registers
    rt_bf<n, d, row_l> tile;
    load(tile, tile_smem[warpid()]);
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // store output
    store(g.out, tile, {0, 0, i, 0});
}

void dispatch_micro(micro_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)micro_tk, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    micro_tk<<<g.grid(), g.block(), mem_size>>>(g);
    hipDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module";
    py::bind_function<dispatch_micro>(m, "dispatch_micro", &micro_globals::in, &micro_globals::out);
}

