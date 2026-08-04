#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include "../../utils.cpp"
using namespace kittens;


constexpr int TILE_SIZE_ROWS = 32;

#define NUM_WARPS 1
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using _gl_A = gl<bf16, -1, -1, -1, -1>;
using _gl_B = gl<bf16, -1, -1, -1, -1>;
using _gl_C = gl<bf16, -1, -1, -1, -1>;

using G = kittens::group<NUM_WARPS>;

struct micro_globals {
    _gl_A in;
    _gl_C out;
    dim3 grid()  { return dim3(1); } 
    dim3 block() { return dim3(NUM_THREADS); } 
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const micro_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<TILE_SIZE_ROWS, TILE_SIZE_ROWS> (&In) = al.allocate<st_bf<TILE_SIZE_ROWS, TILE_SIZE_ROWS>>();

    rt_bf<TILE_SIZE_ROWS, TILE_SIZE_ROWS> tile;
    rt_bf<TILE_SIZE_ROWS, TILE_SIZE_ROWS, ducks::rt_layout::col> tile_col;
    zero(tile);
    zero(tile_col);

    // global to shared
    // load(tile, g.in, {0, 0, 0, 0});
    load_global_to_shared_direct<2, false, st_bf<TILE_SIZE_ROWS, TILE_SIZE_ROWS>, _gl_A, coord<st_bf<TILE_SIZE_ROWS, TILE_SIZE_ROWS>>, NUM_THREADS>(g.in, {0, 0, 0, 0}, In);
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);
    __syncthreads();

    // shared to registers
    load_lds_reg(tile, In);
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);
    __syncthreads();

    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);
    __syncthreads();    

    store(g.out, tile, {0, 0, 0, 0});
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