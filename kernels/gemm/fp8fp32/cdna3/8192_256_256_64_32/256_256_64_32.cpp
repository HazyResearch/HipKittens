#include "kittens.cuh" 
#include "pyutils/pyutils.cuh"
using namespace kittens;

#define NUM_WARPS 8
#define M 8192
#define N 8192
#define K 8192

constexpr int BLOCK_M = 256;
constexpr int BLOCK_N = 256;
constexpr int BLOCK_K = 128;
constexpr int REG_MN  =  64;
constexpr int REG_K   =  32;

using G = kittens::group<NUM_WARPS>;
using _gl_A = gl<fp8e4m3,-1,-1,-1,-1>;
using _gl_B = gl<fp8e4m3,-1,-1,-1,-1>;
using _gl_C = gl<float,-1,-1,-1,-1>;

struct micro_globals {
    _gl_A A;
    _gl_B B;
    _gl_C C;
    hipStream_t stream;
    dim3 grid()  { return dim3((N / BLOCK_N) * (M / BLOCK_M)); }
    dim3 block() { return dim3(NUM_WARPS * WARP_THREADS); }
    size_t dynamic_shared_memory() { return 65536; }
};

__global__ __launch_bounds__(NUM_WARPS * WARP_THREADS, 2)  // launch_bounds(max_threads_per_block, min_warps_per_simd)
void micro_tk(const micro_globals g) {
    constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    auto (&As) = al.allocate<st<fp8e4m3, BLOCK_M, BLOCK_K>>();
    auto (&Bs) = al.allocate<st<fp8e4m3, BLOCK_N, BLOCK_K>>();
    rt<fp8e4m3, REG_MN, REG_K> tiles[8];
    rt_fl<REG_MN, REG_MN, ducks::rt_layout::col> C_accum[2];
    for (int i = 0; i < 2; i++) { zero(C_accum[i]); }

    int wgid = (blockIdx.y * gridDim.x) + blockIdx.x;
    const int NUM_WGS = gridDim.x * gridDim.y;
    constexpr int WGM = 4;
    wgid = chiplet_transform_chunked(wgid, NUM_WGS, NUM_XCDS, WGM*WGM);
    const int num_pid_m = ceil_div(M, BLOCK_M);
    const int num_pid_n = ceil_div(N, BLOCK_N);
    int num_wgid_in_group = WGM * num_pid_n;
    int group_id = wgid / num_wgid_in_group;
    int first_pid_m = group_id * WGM;
    int group_size_m = min(num_pid_m - first_pid_m, WGM);
    int output_m = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    int output_n = (wgid % num_wgid_in_group) / group_size_m;
    
    const int warp_id = warpid();
    const int warp_row = warp_id / 4, warp_col = warp_id % 4;
    const int k_iters = g.A.cols() / BLOCK_K;

    G::load(As, g.A, {0, 0, output_m, 0});
    G::load(Bs, g.B, {0, 0, output_n, 0});
    __builtin_amdgcn_s_barrier();

    if (warp_row == 1) {
        __builtin_amdgcn_s_barrier();
    }

    for (int K_TILE = 0; K_TILE < k_iters - 1; ++K_TILE) {
        constexpr int BUFFER_SIZE_A = (BLOCK_M * BLOCK_K) / NUM_THREADS / sizeof(float4) / sizeof(fp8e4m3);
        constexpr int BUFFER_SIZE_B = (BLOCK_N * BLOCK_K) / NUM_THREADS / sizeof(float4) / sizeof(fp8e4m3);
        float4 a_buffer_next[BUFFER_SIZE_A];
        float4 b_buffer_next[BUFFER_SIZE_B];

        // Cluster 0
        load_global_to_register_buffer<2, false, NUM_THREADS>(a_buffer_next, BUFFER_SIZE_A, g.A, {0, 0, output_m, K_TILE + 1}, As);
        load(tiles[1], subtile_inplace<REG_MN, REG_K>(As, {warp_row, 0}));
        load(tiles[2], subtile_inplace<REG_MN, REG_K>(As, {warp_row + 2, 0}));
        load(tiles[0], subtile_inplace<REG_MN, REG_K>(Bs, {warp_col, 0}));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 1
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0], tiles[1], tiles[0], C_accum[0]);
        mma_ABt(C_accum[1], tiles[2], tiles[0], C_accum[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 2
        load(tiles[3], subtile_inplace<REG_MN, REG_K>(Bs, {warp_col, 1}));
        load(tiles[4], subtile_inplace<REG_MN, REG_K>(As, {warp_row, 1}));
        load(tiles[5], subtile_inplace<REG_MN, REG_K>(As, {warp_row + 2, 1}));
        load(tiles[0], subtile_inplace<REG_MN, REG_K>(Bs, {warp_col, 2}));
        load(tiles[1], subtile_inplace<REG_MN, REG_K>(As, {warp_row, 2}));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 3
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0], tiles[4], tiles[3], C_accum[0]);
        mma_ABt(C_accum[1], tiles[5], tiles[3], C_accum[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 4
        load_global_to_register_buffer<2, false, NUM_THREADS>(b_buffer_next, BUFFER_SIZE_B, g.B, {0, 0, output_n, K_TILE + 1}, Bs);
        load(tiles[2], subtile_inplace<REG_MN, REG_K>(As, {warp_row + 2, 2}));
        load(tiles[6], subtile_inplace<REG_MN, REG_K>(Bs, {warp_col, 3}));
        load(tiles[7], subtile_inplace<REG_MN, REG_K>(As, {warp_row, 3}));
        load(tiles[5], subtile_inplace<REG_MN, REG_K>(As, {warp_row + 2, 3}));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 5
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0], tiles[1], tiles[0], C_accum[0]);
        mma_ABt(C_accum[1], tiles[2], tiles[0], C_accum[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 6
        asm volatile("s_waitcnt lgkmcnt(0)");
        store_register_buffer_to_shared<NUM_THREADS>(As, a_buffer_next);
        store_register_buffer_to_shared<NUM_THREADS>(Bs, b_buffer_next);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 7
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0], tiles[7], tiles[6], C_accum[0]);
        mma_ABt(C_accum[1], tiles[5], tiles[6], C_accum[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    }
    // epilogue
    __builtin_amdgcn_sched_barrier(0);
    load(tiles[0], subtile_inplace<REG_MN, REG_K>(Bs, {warp_col, 0}));
    load(tiles[1], subtile_inplace<REG_MN, REG_K>(As, {warp_row, 0}));
    load(tiles[2], subtile_inplace<REG_MN, REG_K>(As, {warp_row + 2, 0}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    __builtin_amdgcn_s_setprio(1);
    mma_ABt(C_accum[0], tiles[1], tiles[0], C_accum[0]);
    mma_ABt(C_accum[1], tiles[2], tiles[0], C_accum[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    load(tiles[3], subtile_inplace<REG_MN, REG_K>(Bs, {warp_col, 1}));
    load(tiles[4], subtile_inplace<REG_MN, REG_K>(As, {warp_row, 1}));
    load(tiles[5], subtile_inplace<REG_MN, REG_K>(As, {warp_row + 2, 1}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    __builtin_amdgcn_s_setprio(1);
    mma_ABt(C_accum[0], tiles[4], tiles[3], C_accum[0]);
    mma_ABt(C_accum[1], tiles[5], tiles[3], C_accum[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    load(tiles[0], subtile_inplace<REG_MN, REG_K>(Bs, {warp_col, 2}));
    load(tiles[1], subtile_inplace<REG_MN, REG_K>(As, {warp_row, 2}));
    load(tiles[2], subtile_inplace<REG_MN, REG_K>(As, {warp_row + 2, 2}));
    load(tiles[3], subtile_inplace<REG_MN, REG_K>(Bs, {warp_col, 3}));
    load(tiles[4], subtile_inplace<REG_MN, REG_K>(As, {warp_row, 3}));
    load(tiles[5], subtile_inplace<REG_MN, REG_K>(As, {warp_row + 2, 3}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    __builtin_amdgcn_s_setprio(1);
    mma_ABt(C_accum[0], tiles[1], tiles[0], C_accum[0]);
    mma_ABt(C_accum[1], tiles[2], tiles[0], C_accum[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    __builtin_amdgcn_s_setprio(1);
    mma_ABt(C_accum[0], tiles[4], tiles[3], C_accum[0]);
    mma_ABt(C_accum[1], tiles[5], tiles[3], C_accum[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    if (warp_row == 0) {
        __builtin_amdgcn_s_barrier();
    }
    store(g.C, C_accum[0], {0, 0, output_m * 4 + warp_row,     output_n * 4 + warp_col});
    store(g.C, C_accum[1], {0, 0, output_m * 4 + warp_row + 2, output_n * 4 + warp_col});
}

void dispatch_micro(micro_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)micro_tk, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    micro_tk<<<g.grid(), g.block(), mem_size, g.stream>>>(g);
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module";
    py::bind_kernel<micro_tk>(m, "micro_tk", &micro_globals::A, &micro_globals::B, &micro_globals::C);
    py::bind_function<dispatch_micro>(m, "dispatch_micro", &micro_globals::A, &micro_globals::B, &micro_globals::C);
}
