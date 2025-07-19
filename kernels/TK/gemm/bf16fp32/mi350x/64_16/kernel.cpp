#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include "../utils.cpp"
using namespace kittens;

constexpr int BLOCK_SIZE_ROWS = 64;
constexpr int BLOCK_SIZE_COLS = 16;

#define NUM_WARPS 1
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using _gl_A = gl<bf16, -1, -1, -1, -1>;
using _gl_B = gl<bf16, -1, -1, -1, -1>;
using _gl_C = gl<bf16, -1, -1, -1, -1>;

using G = kittens::group<NUM_WARPS>;

struct micro_globals {
    _gl_A A;
    _gl_B B;
    _gl_C C;
    dim3 grid()  { return dim3(1); } 
    dim3 block() { return dim3(NUM_THREADS); } 
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const micro_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<BLOCK_SIZE_ROWS, BLOCK_SIZE_COLS> (&A) = al.allocate<st_bf<BLOCK_SIZE_ROWS, BLOCK_SIZE_COLS>>();
    st_bf<BLOCK_SIZE_ROWS, BLOCK_SIZE_COLS> (&B) = al.allocate<st_bf<BLOCK_SIZE_ROWS, BLOCK_SIZE_COLS>>();

    rt_bf<BLOCK_SIZE_ROWS, BLOCK_SIZE_COLS> A_tile, B_tile;
    rt_fl<BLOCK_SIZE_ROWS, BLOCK_SIZE_ROWS, ducks::rt_layout::accumulator> C_accum;
    zero(C_accum);

    // global to shared
    load_global_to_shared_direct<2, false, st_bf<BLOCK_SIZE_ROWS, BLOCK_SIZE_COLS>, _gl_A, coord<st_bf<BLOCK_SIZE_ROWS, BLOCK_SIZE_COLS>>, NUM_THREADS>(g.A, {0, 0, 0, 0}, A);
    load_global_to_shared_direct<2, false, st_bf<BLOCK_SIZE_ROWS, BLOCK_SIZE_COLS>, _gl_B, coord<st_bf<BLOCK_SIZE_ROWS, BLOCK_SIZE_COLS>>, NUM_THREADS>(g.B, {0, 0, 0, 0}, B);
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);
    __syncthreads();

    // shared to registers
    load_lds_reg(A_tile, subtile_inplace<BLOCK_SIZE_ROWS, BLOCK_SIZE_COLS>(A, {0, 0}));
    load_lds_reg(B_tile, subtile_inplace<BLOCK_SIZE_ROWS, BLOCK_SIZE_COLS>(B, {0, 0}));
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);
    __syncthreads();

    // compute
    mma_ABt(C_accum, A_tile, B_tile, C_accum);
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);
    __syncthreads();

    // register to global
    store(g.C, C_accum, {0, 0, 0, 0});
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
    py::bind_function<dispatch_micro>(m, "dispatch_micro", &micro_globals::A, &micro_globals::B, &micro_globals::C);
}