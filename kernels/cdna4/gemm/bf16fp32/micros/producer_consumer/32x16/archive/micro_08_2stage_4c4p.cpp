#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

/*******************************************************************************/

constexpr int BLOCK_SIZE_MN = 64;
constexpr int K_STEP = 128;
constexpr int M_BLOCK = 1;
constexpr int N_BLOCK = 4;
constexpr int DOT_SLICE = 32;

constexpr int NEW_ROW_BLOCK_SIZE = BLOCK_SIZE_MN * M_BLOCK;
constexpr int NEW_COL_BLOCK_SIZE = BLOCK_SIZE_MN * N_BLOCK;

#define NUM_PRODUCER_WORKERS (4)
#define NUM_CONSUMER_WORKERS (M_BLOCK * 4)
#define NUM_THREADS ((NUM_PRODUCER_WORKERS + NUM_CONSUMER_WORKERS) * kittens::WARP_THREADS)
#define NUM_PRODUCER_THREADS (NUM_PRODUCER_WORKERS * kittens::WARP_THREADS)

using G = kittens::group<NUM_PRODUCER_WORKERS>;
using A_slice = rt_bf<BLOCK_SIZE_MN, DOT_SLICE, row_l, rt_32x16_s>;
using B_slice = rt_bf<BLOCK_SIZE_MN, DOT_SLICE, row_l, rt_32x16_s>;


#define M 8192  
#define K 8192
#define N 8192

struct micro_globals {
    gl<bf16, -1, -1, -1, -1> a, b;
    gl<bf16, -1, -1, -1, -1> c;
    dim3 grid()  { return dim3(ceil_div(N / NEW_COL_BLOCK_SIZE) * ceil_div( M / NEW_ROW_BLOCK_SIZE)); } 
    dim3 block() { return dim3(NUM_THREADS); } 
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; } 
};

__global__ __launch_bounds__(NUM_THREADS, 2)
void micro_tk(const micro_globals g) {

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<BLOCK_SIZE_MN, K_STEP, st_32x32_s> (&As)[2][M_BLOCK] = al.allocate<st_bf<BLOCK_SIZE_MN, K_STEP, st_32x32_s>, 2, M_BLOCK>();
    st_bf<BLOCK_SIZE_MN, K_STEP, st_32x32_s> (&Bs)[2][N_BLOCK] = al.allocate<st_bf<BLOCK_SIZE_MN, K_STEP, st_32x32_s>, 2, N_BLOCK>();

    rt_fl<BLOCK_SIZE_MN, BLOCK_SIZE_MN, accum_col_l> C_accum;
    zero(C_accum);

    int wgid = (blockIdx.y * gridDim.x) + blockIdx.x;
    const int NUM_WGS  = gridDim.x * gridDim.y;
    const int NUM_XCDS = 8;  
    wgid = (wgid % NUM_XCDS) * (NUM_WGS / NUM_XCDS) + (wgid / NUM_XCDS);
    const int WGM = 4;  
    const int num_pid_m = ceil_div(M, NEW_ROW_BLOCK_SIZE); // 7680 / 192 = 40
    const int num_pid_n = ceil_div(N, NEW_COL_BLOCK_SIZE); // 7680 / 256 = 30
    const int num_wgid_in_group = WGM * num_pid_n;
    const int group_id     = wgid / num_wgid_in_group;
    const int first_pid_m  = group_id * WGM;
    const int group_size_m = min(num_pid_m - first_pid_m, WGM);
    const int pid_m = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    const int pid_n = (wgid % num_wgid_in_group) / group_size_m;
    int row = pid_m * M_BLOCK;  
    int col = pid_n * N_BLOCK;  

    int warp_id = kittens::warpid();
    const int local_warp_id = warp_id % 4;
    int warp_group_id = kittens::warpgroupid();
    bool is_producer = (warp_group_id == 0);
    bool is_consumer = (warp_group_id > 0 && warp_group_id <= M_BLOCK);
    int consumer_idx = is_consumer ? warp_group_id - 1 : 0;

    using T = typename st_bf<BLOCK_SIZE_MN, K_STEP, st_32x32_s>::dtype;
    constexpr int bytes_per_thread = st_32x32_s::template bytes_per_thread<T>();
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_PRODUCER_THREADS;
    constexpr int memcpy_per_tile = BLOCK_SIZE_MN * K_STEP * sizeof(T) / bytes_per_memcpy;
    uint32_t swizzled_offsets_A[memcpy_per_tile];
    uint32_t swizzled_offsets_B[memcpy_per_tile];
    G::prefill_swizzled_offsets(As[0][0], g.a, swizzled_offsets_A);
    G::prefill_swizzled_offsets(Bs[0][0], g.b, swizzled_offsets_B);

    int tic = 0; int toc = 1;
    if (is_producer) {
        #pragma unroll
        for (int m=0; m<M_BLOCK; ++m) G::load<2,false>(As[tic][m],  g.a, {0,0, row+m, 0}, swizzled_offsets_A);
        #pragma unroll
        for (int n=0; n<N_BLOCK; ++n) G::load<2,false>(Bs[tic][n],  g.b, {0,0, col+n, 0}, swizzled_offsets_B);
    }
    __builtin_amdgcn_s_waitcnt(0);
    __syncthreads();

    const int num_tiles = K / K_STEP;
    #pragma unroll
    for (int tile = 0; tile < num_tiles-1; ++tile, tic^=1, toc^=1) {
        if (is_producer) {
            #pragma unroll
            for (int m=0; m<M_BLOCK; ++m) G::load<2,false>(As[toc][m], g.a, {0,0, row+m, tile+1}, swizzled_offsets_A);
            #pragma unroll
            for (int n=0; n<N_BLOCK; ++n) G::load<2,false>(Bs[toc][n], g.b, {0,0, col+n, tile+1}, swizzled_offsets_B);
            __builtin_amdgcn_s_waitcnt(0);
        } 
        __syncthreads();
        if (is_consumer) {
            A_slice a0;
            B_slice b0;

            load(a0, subtile_inplace<BLOCK_SIZE_MN, DOT_SLICE>(As[tic][consumer_idx], {0,0}));
            load(b0, subtile_inplace<BLOCK_SIZE_MN, DOT_SLICE>(Bs[tic][local_warp_id], {0,0}));
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(C_accum, a0, b0, C_accum);
            __builtin_amdgcn_s_setprio(0);

            load(a0, subtile_inplace<BLOCK_SIZE_MN, DOT_SLICE>(As[tic][consumer_idx], {0,1}));
            load(b0, subtile_inplace<BLOCK_SIZE_MN, DOT_SLICE>(Bs[tic][local_warp_id], {0,1}));
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(C_accum, a0, b0, C_accum);
            __builtin_amdgcn_s_setprio(0);

            load(a0, subtile_inplace<BLOCK_SIZE_MN, DOT_SLICE>(As[tic][consumer_idx], {0,2}));
            load(b0, subtile_inplace<BLOCK_SIZE_MN, DOT_SLICE>(Bs[tic][local_warp_id], {0,2}));
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(C_accum, a0, b0, C_accum);
            __builtin_amdgcn_s_setprio(0);

            // K in [96:128]
            load(a0, subtile_inplace<BLOCK_SIZE_MN, DOT_SLICE>(As[tic][consumer_idx], {0,3}));
            load(b0, subtile_inplace<BLOCK_SIZE_MN, DOT_SLICE>(Bs[tic][local_warp_id], {0,3}));
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(C_accum, a0, b0, C_accum);
            __builtin_amdgcn_s_setprio(0);
        }
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
    }

    if (is_consumer) {
        A_slice a0;
        B_slice b0;
        load(a0, subtile_inplace<BLOCK_SIZE_MN, DOT_SLICE>(As[tic][consumer_idx], {0,0}));
        load(b0, subtile_inplace<BLOCK_SIZE_MN, DOT_SLICE>(Bs[tic][local_warp_id], {0,0}));
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum, a0, b0, C_accum);
        __builtin_amdgcn_s_setprio(0);

        load(a0, subtile_inplace<BLOCK_SIZE_MN, DOT_SLICE>(As[tic][consumer_idx], {0,1}));
        load(b0, subtile_inplace<BLOCK_SIZE_MN, DOT_SLICE>(Bs[tic][local_warp_id], {0,1}));
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum, a0, b0, C_accum);
        __builtin_amdgcn_s_setprio(0);

        load(a0, subtile_inplace<BLOCK_SIZE_MN, DOT_SLICE>(As[tic][consumer_idx], {0,2}));
        load(b0, subtile_inplace<BLOCK_SIZE_MN, DOT_SLICE>(Bs[tic][local_warp_id], {0,2}));
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum, a0, b0, C_accum);
        __builtin_amdgcn_s_setprio(0);
        
        load(a0, subtile_inplace<BLOCK_SIZE_MN, DOT_SLICE>(As[tic][consumer_idx], {0,3}));
        load(b0, subtile_inplace<BLOCK_SIZE_MN, DOT_SLICE>(Bs[tic][local_warp_id], {0,3}));
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum, a0, b0, C_accum);
        __builtin_amdgcn_s_setprio(0);
    }

    if (is_consumer) {
        store(g.c, C_accum, {0, 0, row + consumer_idx, col + local_warp_id});
    }
}

void dispatch_micro(micro_globals g) {
    const unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)micro_tk, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    hipEvent_t start, stop;
    hipEventCreate(&start); hipEventCreate(&stop);
    hipEventRecord(start);
    micro_tk<<<g.grid(), g.block(), mem_size>>>(g);
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    float ms=0.f; hipEventElapsedTime(&ms, start, stop);
    hipEventDestroy(start); hipEventDestroy(stop);
    hipDeviceSynchronize();
  }

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module";
    py::bind_function<dispatch_micro>(m, "dispatch_micro", &micro_globals::a, &micro_globals::b, &micro_globals::c);
}

