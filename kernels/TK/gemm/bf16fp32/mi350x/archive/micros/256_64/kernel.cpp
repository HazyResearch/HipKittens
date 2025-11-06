#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include "../utils.cpp"
using namespace kittens;

constexpr int BLOCK_SIZE       = 256;  
constexpr int K_STEP           = 64;
constexpr int REG_BLOCK        = BLOCK_SIZE / 4;
constexpr int DOT_SLICE        = 16;

#define NUM_WARPS 8
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

#define M 256
#define K 64
#define N 256

using _gl_A = gl<bf16, -1, -1, -1, -1>;
using _gl_B = gl<bf16, -1, -1, -1, -1>;
using _gl_C = gl<bf16, -1, -1, -1, -1>;

using G = kittens::group<NUM_WARPS>;

__host__ __device__ inline int ceil_div(int a, int b) {
  return (a + b - 1) / b;
}

struct micro_globals {
    _gl_A a;
    _gl_B b;
    _gl_C c;
    dim3 grid()  { return dim3((N / BLOCK_SIZE), (M / BLOCK_SIZE)); } 
    dim3 block() { return dim3(NUM_THREADS); } 
    size_t dynamic_shared_memory() { return 160000; } //MAX_SHARED_MEMORY; }
};

__global__ __launch_bounds__(NUM_THREADS, 2)
void micro_tk(const micro_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<BLOCK_SIZE, K_STEP> (&As) = al.allocate<st_bf<BLOCK_SIZE, K_STEP>>();
    st_bf<BLOCK_SIZE, K_STEP> (&Bs) = al.allocate<st_bf<BLOCK_SIZE, K_STEP>>();

    rt_bf<REG_BLOCK, DOT_SLICE> tiles[6];
    rt_fl<REG_BLOCK, REG_BLOCK, ducks::rt_layout::accumulator> C_accum[2];
    for (int i = 0; i < 2; i++) { zero(C_accum[i]); }

    const int row = blockIdx.x;
    const int col = blockIdx.y;

    // Info
    const int warp_id = kittens::warpid();
    const int warp_row = warp_id / 4;
    const int warp_col = warp_id % 4;
    const int num_tiles = K / K_STEP;

    // Load first tile into shared memory
    load_global_to_shared_direct<2, false, st_bf<BLOCK_SIZE, K_STEP>, _gl_A, coord<st_bf<BLOCK_SIZE, K_STEP>>, NUM_THREADS>(g.a, {0, 0, row, 0}, As);
    load_global_to_shared_direct<2, false, st_bf<BLOCK_SIZE, K_STEP>, _gl_B, coord<st_bf<BLOCK_SIZE, K_STEP>>, NUM_THREADS>(g.b, {0, 0, col, 0}, Bs);
    asm volatile("s_waitcnt vmcnt(0)");
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Epilogue
    // Cluster 0
    __builtin_amdgcn_sched_barrier(0);
    load_lds_reg(tiles[0], subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col, 0}));
    load_lds_reg(tiles[1], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 0}));
    load_lds_reg(tiles[2], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 0}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);
    

    // Cluster 1
    __builtin_amdgcn_s_setprio(1);
    mma_ABt(C_accum[0], tiles[1], tiles[0], C_accum[0]);
    mma_ABt(C_accum[1], tiles[2], tiles[0], C_accum[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 2
    load_lds_reg(tiles[3], subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col, 1}));
    load_lds_reg(tiles[4], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 1}));
    load_lds_reg(tiles[5], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 1}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 3
    __builtin_amdgcn_s_setprio(1);
    mma_ABt(C_accum[0], tiles[4], tiles[3], C_accum[0]);
    mma_ABt(C_accum[1], tiles[5], tiles[3], C_accum[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 4
    load_lds_reg(tiles[0], subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col, 2}));
    load_lds_reg(tiles[1], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 2}));
    load_lds_reg(tiles[2], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 2}));
    load_lds_reg(tiles[3], subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col, 3}));
    load_lds_reg(tiles[4], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 3}));
    load_lds_reg(tiles[5], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 3}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 5
    __builtin_amdgcn_s_setprio(1);
    mma_ABt(C_accum[0], tiles[1], tiles[0], C_accum[0]);
    mma_ABt(C_accum[1], tiles[2], tiles[0], C_accum[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 7
    __builtin_amdgcn_s_setprio(1);
    mma_ABt(C_accum[0], tiles[4], tiles[3], C_accum[0]);
    mma_ABt(C_accum[1], tiles[5], tiles[3], C_accum[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    store(g.c, C_accum[0], {0, 0, row * 4 + warp_row, col * 4 + warp_col});
    store(g.c, C_accum[1], {0, 0, row * 4 + warp_row + 2, col * 4 + warp_col});
}

void dispatch_micro(micro_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)micro_tk, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    micro_tk<<<g.grid(), g.block(), mem_size>>>(g);
    hipDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module";
    // py::bind_kernel<micro_tk>(m, "micro_tk", &micro_globals::a, &micro_globals::b, &micro_globals::c); 
    py::bind_function<dispatch_micro>(m, "dispatch_micro", &micro_globals::a, &micro_globals::b, &micro_globals::c);
}