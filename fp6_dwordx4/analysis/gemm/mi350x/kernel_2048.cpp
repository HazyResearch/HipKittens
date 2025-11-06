#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

constexpr int BLOCK_SIZE       = 128;  
constexpr int K_STEP           = 64;
constexpr int REG_BLOCK_M      = BLOCK_SIZE / 2;
constexpr int REG_BLOCK_N      = BLOCK_SIZE / 4;
constexpr int DOT_SLICE        = 32;

#define NUM_WARPS 8
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

#define M 2048
#define K 2048
#define N 2048

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
    dim3 grid()  { return dim3((N / BLOCK_SIZE) * (M / BLOCK_SIZE)); } 
    dim3 block() { return dim3(NUM_THREADS); } 
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY / 2; } 
};

__global__ __launch_bounds__(NUM_THREADS, 2)
void micro_tk(const micro_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<BLOCK_SIZE, K_STEP> (&As)[2] = al.allocate<st_bf<BLOCK_SIZE, K_STEP>, 2>();
    st_bf<BLOCK_SIZE, K_STEP> (&Bs)[2] = al.allocate<st_bf<BLOCK_SIZE, K_STEP>, 2>();

    rt_bf<REG_BLOCK_M, DOT_SLICE> A_tile;
    rt_bf<REG_BLOCK_N, DOT_SLICE> B_tile;
    rt_fl<REG_BLOCK_M, REG_BLOCK_N, ducks::rt_layout::accumulator_col> C_accum;
    zero(C_accum);

    // Original WGID.
    int wgid = (blockIdx.y * gridDim.x) + blockIdx.x;
    const int NUM_WGS = gridDim.x * gridDim.y;
    const int NUM_XCDS = 8;
    const int CUS_PER_XCD = 32;
    const int NUM_CUS = CUS_PER_XCD * NUM_XCDS;
    // Swizzle chiplet so that wgids are in the same XCD.
    wgid = (wgid % NUM_XCDS) * (NUM_WGS / NUM_XCDS) + (wgid / NUM_XCDS);
    // Swizzle for better L2 within the same XCD.
    const int WGM = 8;
    const int num_pid = ceil_div(M, BLOCK_SIZE);
    int num_wgid_in_group = WGM * num_pid;
    int group_id = wgid / num_wgid_in_group;
    int first_pid_m = group_id * WGM;
    int group_size_m = min(num_pid - first_pid_m, WGM);
    int pid_m = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    int pid_n = (wgid % num_wgid_in_group) / group_size_m;
    // Assign the tile's row/column based on the pid_m and pid_n.
    const int row = pid_m; 
    const int col = pid_n; 

    // Info
    const int warp_id = kittens::warpid();
    const int warp_row = warp_id / 4;
    const int warp_col = warp_id % 4;
    const int num_tiles = K / K_STEP;

    int tic = 0;
    int toc = 1;
    using T = typename st_bf<BLOCK_SIZE, K_STEP>::dtype;
    constexpr int bytes_per_thread = 16;
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_THREADS;
    constexpr int memcpy_per_tile = BLOCK_SIZE * K_STEP * sizeof(T) / bytes_per_memcpy;
    uint32_t swizzled_offsets_A[memcpy_per_tile];
    uint32_t swizzled_offsets_B[memcpy_per_tile];
    G::prefill_swizzled_offsets(As[tic], g.a, swizzled_offsets_A);
    G::prefill_swizzled_offsets(Bs[tic], g.b, swizzled_offsets_B);
    const lds_lane_ofs lane_ofs = prefill_swizzled_offsets(A_tile, As[tic]);

    // Load first tile into shared memory
    G::load(As[tic], g.a, {0, 0, row, 0}, swizzled_offsets_A);  
    G::load(Bs[tic], g.b, {0, 0, col, 0}, swizzled_offsets_B);
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();

    if (warp_row == 1) {
        __builtin_amdgcn_s_barrier();
    }

    #pragma unroll
    for (int tile = 0; tile < num_tiles - 1; ++tile, tic^=1, toc^=1) {

        // Cluster 0
        load(A_tile, subtile_inplace<REG_BLOCK_M, DOT_SLICE>(As[tic], {warp_row, 0}), lane_ofs);
        G::load(As[toc], g.a, {0, 0, row, tile+1}, swizzled_offsets_A);
        load(B_tile, subtile_inplace<REG_BLOCK_N, DOT_SLICE>(Bs[tic], {warp_col, 0}), lane_ofs);
        G::load(Bs[toc], g.b, {0, 0, col, tile+1}, swizzled_offsets_B);
        __builtin_amdgcn_s_barrier();

        // Cluster 1
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum, A_tile, B_tile, C_accum);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        // Cluster 2
        load(A_tile, subtile_inplace<REG_BLOCK_M, DOT_SLICE>(As[tic], {warp_row, 1}), lane_ofs);
        load(B_tile, subtile_inplace<REG_BLOCK_N, DOT_SLICE>(Bs[tic], {warp_col, 1}), lane_ofs);
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();

        // Cluster 3
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum, A_tile, B_tile, C_accum);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier(); 
    }

    // Epilogue
    // Cluster 0
    __builtin_amdgcn_sched_barrier(0);
    load(A_tile, subtile_inplace<REG_BLOCK_M, DOT_SLICE>(As[tic], {warp_row, 0}), lane_ofs);
    load(B_tile, subtile_inplace<REG_BLOCK_N, DOT_SLICE>(Bs[tic], {warp_col, 0}), lane_ofs);
    __builtin_amdgcn_s_barrier();    

    // Cluster 1
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_setprio(1);
    mma_ABt(C_accum, A_tile, B_tile, C_accum);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();

    // Cluster 2
    load(A_tile, subtile_inplace<REG_BLOCK_M, DOT_SLICE>(As[tic], {warp_row, 1}), lane_ofs);
    load(B_tile, subtile_inplace<REG_BLOCK_N, DOT_SLICE>(Bs[tic], {warp_col, 1}), lane_ofs);
    __builtin_amdgcn_s_barrier();

    // Cluster 3
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_setprio(1);
    mma_ABt(C_accum, A_tile, B_tile, C_accum);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();

    store(g.c, C_accum, {0, 0, row * 2 + warp_row, col * 4 + warp_col});
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
