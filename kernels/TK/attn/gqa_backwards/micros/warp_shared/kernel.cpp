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
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<n, d, ducks::st_layout::row> (&tile_smem)[NUM_WARPS] = al.allocate<st_bf<n, d, ducks::st_layout::row>, NUM_WARPS>();

    // preswizzling
    using T = typename st_bf<n, d>::dtype;
    constexpr int bytes_per_thread = 16;
    constexpr int bytes_per_memcpy = bytes_per_thread * WARP_THREADS;
    constexpr int memcpy_per_tile = n * d * sizeof(T) / bytes_per_memcpy;
    uint32_t swizzled_offsets[memcpy_per_tile];
    prefill_swizzled_offsets<2, false>(tile_smem[warpid()], g.in, swizzled_offsets);
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __syncthreads();

    // load to smem
    load<2, false>(tile_smem[warpid()], g.in, {0, 0, 0, 0}, swizzled_offsets);
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __syncthreads();

    // load to registers
    rt_bf<n, d, row_l> tile;
    load(tile, tile_smem[warpid()]);
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __syncthreads();

    // store output
    store(g.out, tile, {0, 0, 0, 0});
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __syncthreads();
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

