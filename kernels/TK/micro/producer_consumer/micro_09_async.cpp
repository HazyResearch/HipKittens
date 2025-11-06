#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

constexpr int BLOCK_SIZE = 64;
constexpr int M_BLOCK = 3;
constexpr int N_BLOCK = 4;
constexpr int DOT_SLICE = 32;

constexpr int NEW_ROW_BLOCK_SIZE = BLOCK_SIZE * M_BLOCK;
constexpr int NEW_COL_BLOCK_SIZE = BLOCK_SIZE * N_BLOCK;

#define NUM_PRODUCER_WORKERS (4)
#define NUM_CONSUMER_WORKERS (M_BLOCK * 4)
#define NUM_THREADS ((NUM_PRODUCER_WORKERS + NUM_CONSUMER_WORKERS) * kittens::WARP_THREADS)
#define NUM_PRODUCER_THREADS (NUM_PRODUCER_WORKERS * kittens::WARP_THREADS)

using G = kittens::group<NUM_PRODUCER_WORKERS>;
using A_slice = rt_bf<BLOCK_SIZE, DOT_SLICE, row_l>;
using B_slice = rt_bf<BLOCK_SIZE, DOT_SLICE, row_l>;

#define M 192*40
#define N 192*40
#define K 192*40

__host__ __device__ inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

struct micro_globals {
    gl<bf16, -1, -1, -1, -1> a, b;
    gl<bf16, -1, -1, -1, -1> c;
    dim3 grid()  { return dim3((N / NEW_COL_BLOCK_SIZE) * ( M / NEW_ROW_BLOCK_SIZE)); } 
    dim3 block() { return dim3(NUM_THREADS); } 
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; } 
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const micro_globals g) {

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<BLOCK_SIZE, BLOCK_SIZE, ducks::st_layout::row> (&As)[2][M_BLOCK] = al.allocate<st_bf<BLOCK_SIZE, BLOCK_SIZE, ducks::st_layout::row>, 2, M_BLOCK>();
    st_bf<BLOCK_SIZE, BLOCK_SIZE, ducks::st_layout::row> (&Bs)[2][N_BLOCK] = al.allocate<st_bf<BLOCK_SIZE, BLOCK_SIZE, ducks::st_layout::row>, 2, N_BLOCK>();
    rt_fl<BLOCK_SIZE, BLOCK_SIZE, accum_col_l> C_accum;

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
    // int row = blockIdx.y * M_BLOCK; // works better for large matrices
    // int col = blockIdx.x * N_BLOCK;

    int warp_id = kittens::warpid();
    int local_warp_id = warp_id % 4;
    int warp_group_id = (warp_id - NUM_PRODUCER_WORKERS) / 4;
    // kittens::warpgroupid();
    bool is_producer = (warp_id < NUM_PRODUCER_WORKERS);
    bool is_consumer = (warp_id >= NUM_PRODUCER_WORKERS && warp_group_id <= M_BLOCK);
    int consumer_idx = is_consumer ? warp_group_id : 0;

    using T = typename st_bf<BLOCK_SIZE, BLOCK_SIZE>::dtype;
    constexpr int bytes_per_thread = 16;
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_PRODUCER_THREADS;
    constexpr int memcpy_per_tile = BLOCK_SIZE * BLOCK_SIZE * sizeof(T) / bytes_per_memcpy;
    uint32_t swizzled_offsets_A[memcpy_per_tile];
    uint32_t swizzled_offsets_B[memcpy_per_tile];
    G::prefill_swizzled_offsets(As[0][0], g.a, swizzled_offsets_A);
    G::prefill_swizzled_offsets(Bs[0][0], g.b, swizzled_offsets_B);
    A_slice A_tile;
    const lds_lane_ofs lane_ofs = prefill_swizzled_offsets(A_tile, As[0][0]);

    const bool warp_leader = (threadIdx.x % kittens::WARP_THREADS) == 0;

    // Volatile LDS flags/counters
    __shared__ __align__(16) unsigned int ready[2], done[2], prod_cnt[2];

    int tic = 0;
    int toc = 0;
    if (is_producer) {
        #pragma unroll
        for (int m=0; m<M_BLOCK; ++m) 
            G::load<2,false>(As[tic][m], g.a, {0, 0, row+m, 0}, swizzled_offsets_A);
        #pragma unroll
        for (int n=0; n<N_BLOCK; ++n) 
            G::load<2,false>(Bs[tic][n], g.b, {0, 0, col+n, 0}, swizzled_offsets_B);
        if (warp_leader) atomicAdd((int*)&prod_cnt[0], 1);  
    }
    if (is_consumer) {  
        zero(C_accum); 
    }
    if (is_consumer && consumer_idx == 0) {   
        #pragma unroll
        for (int n=0; n<N_BLOCK; ++n) 
            G::load<2,false>(Bs[1][n], g.b, {0,0, col+n, 1}, swizzled_offsets_B);
        #pragma unroll
        for (int m=0; m<M_BLOCK; ++m) 
            G::load<2,false>(As[1][m], g.a, {0,0, row+m, 1}, swizzled_offsets_A);
        if (warp_leader) atomicAdd((int*)&prod_cnt[1], 1);
    }
    asm volatile("s_waitcnt vmcnt(1)");
    __syncthreads();
    if (threadIdx.x == 0) {
        while (prod_cnt[0] < NUM_PRODUCER_WORKERS) { __builtin_amdgcn_s_sleep(0); } 
        while (prod_cnt[1] < NUM_PRODUCER_WORKERS) { __builtin_amdgcn_s_sleep(0); } 
        __threadfence_block(); 
        ready[0] = 0; 
        ready[1] = 1;
        done[0]  = 0; 
        done[1]  = 0; 
        prod_cnt[0] = 0; 
        prod_cnt[1] = 0;
    }
    __syncthreads();

    int num_tiles = K / BLOCK_SIZE;
    constexpr int sleep_time = 0;

    if (is_producer) {
        #pragma unroll
        for (int tile = 2; tile < num_tiles; ++tile, toc^=1) {
        
            // Wait for consumers to finish with buffer
            const int target_1 = NUM_CONSUMER_WORKERS*(tile/2);
            while (done[toc] < target_1) { 
                __builtin_amdgcn_s_sleep(sleep_time); 
            }

            // Load next tile
            #pragma unroll
            for (int n=0; n<N_BLOCK; ++n) 
                G::load<2,false>(Bs[toc][n], g.b, {0,0, col+n, tile}, swizzled_offsets_B);
            #pragma unroll
            for (int m=0; m<M_BLOCK; ++m) 
                G::load<2,false>(As[toc][m], g.a, {0,0, row+m, tile}, swizzled_offsets_A);
            
            if (warp_leader) atomicAdd((int*)&prod_cnt[toc], 1);
            asm volatile("s_waitcnt vmcnt(1)");

            const int target_2 = NUM_PRODUCER_WORKERS*(tile/2);
            while (prod_cnt[toc] < target_2) { 
                __builtin_amdgcn_s_sleep(sleep_time);  
            } 
            if (warp_leader && warp_id == 0) { // First warp leader only
                atomicExch((int*)&ready[toc], tile);
            }
            
        }
    }

    if (is_consumer) {
        #pragma unroll
        for (int tile = 0; tile < num_tiles; ++tile, tic^=1) {
            unsigned int* ready_ptr = &ready[tic];
            while (*ready_ptr < (unsigned)tile) { 
                __builtin_amdgcn_s_sleep(sleep_time); 
            }

            A_slice a0; 
            B_slice b0;

            load(a0, subtile_inplace<BLOCK_SIZE, DOT_SLICE>(As[tic][consumer_idx], {0,0}), lane_ofs);
            load(b0, subtile_inplace<BLOCK_SIZE, DOT_SLICE>(Bs[tic][local_warp_id], {0,0}), lane_ofs);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(C_accum, a0, b0, C_accum);
            __builtin_amdgcn_s_setprio(0);

            load(a0, subtile_inplace<BLOCK_SIZE, DOT_SLICE>(As[tic][consumer_idx], {0,1}), lane_ofs);
            load(b0, subtile_inplace<BLOCK_SIZE, DOT_SLICE>(Bs[tic][local_warp_id], {0,1}), lane_ofs);
            asm volatile("s_waitcnt lgkmcnt(0)");

            if (warp_leader) atomicAdd((int*)&done[tic], 1);

            __builtin_amdgcn_s_setprio(1);
            mma_ABt(C_accum, a0, b0, C_accum);
            __builtin_amdgcn_s_setprio(0);
        }

        store(g.c, C_accum, {0,0, row + consumer_idx, col + local_warp_id});
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