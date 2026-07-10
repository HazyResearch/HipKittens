#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include "../utils.cpp"
using namespace kittens;

constexpr int BLOCK_SIZE = 128;  
constexpr int K_STEP     = 16;
constexpr int REG_BLOCK  = BLOCK_SIZE / 2;


#define M 256
#define K 256
#define N 256


#define NUM_WARPS 4
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using _gl_A = gl<bf16, -1, -1, -1, -1>;
using _gl_B = gl<bf16, -1, -1, -1, -1>;
using _gl_C = gl<bf16, -1, -1, -1, -1>;

using G = kittens::group<NUM_WARPS>;

struct micro_globals {
    _gl_A A;
    _gl_B B;
    _gl_C C;
    dim3 grid()  { return dim3(N / BLOCK_SIZE, M / BLOCK_SIZE); } 
    dim3 block() { return dim3(NUM_THREADS); } 
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const micro_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<BLOCK_SIZE, K_STEP> (&A) = al.allocate<st_bf<BLOCK_SIZE, K_STEP>>();
    st_bf<BLOCK_SIZE, K_STEP> (&B) = al.allocate<st_bf<BLOCK_SIZE, K_STEP>>();

    const int row = blockIdx.y;
    const int col = blockIdx.x;

    const int warp_id = kittens::warpid();
    const int warp_row = warp_id / 2;
    const int warp_col = warp_id % 2;


    rt_bf<REG_BLOCK, K_STEP> A_tile, B_tile;
    rt_fl<REG_BLOCK, REG_BLOCK, ducks::rt_layout::accumulator> C_accum;
    zero(C_accum);

    int num_tiles = K / K_STEP;

    for (int tile = 0; tile < num_tiles; ++tile) {

        load_global_to_shared_direct<2, false, st_bf<BLOCK_SIZE, K_STEP>, _gl_A, coord<st_bf<BLOCK_SIZE, K_STEP>>, NUM_THREADS>(g.A, {0, 0, row, tile}, A);
        load_global_to_shared_direct<2, false, st_bf<BLOCK_SIZE, K_STEP>, _gl_B, coord<st_bf<BLOCK_SIZE, K_STEP>>, NUM_THREADS>(g.B, {0, 0, col, tile}, B);
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
        __syncthreads();

        load_lds_reg(A_tile, subtile_inplace<REG_BLOCK, K_STEP>(A, {warp_row, 0}));
        load_lds_reg(B_tile, subtile_inplace<REG_BLOCK, K_STEP>(B, {warp_col, 0}));
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
        __syncthreads();

        // compute
        mma_ABt(C_accum, A_tile,  B_tile,  C_accum);
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
        __syncthreads();

    }

    int global_row = row * 2 + warp_row;  
    int global_col = col * 2 + warp_col;
    store(g.C, C_accum, {0, 0, global_row, global_col});
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