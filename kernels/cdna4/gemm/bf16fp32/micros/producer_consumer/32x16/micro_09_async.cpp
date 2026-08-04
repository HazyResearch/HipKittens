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
using A_slice = rt_bf<BLOCK_SIZE, DOT_SLICE, row_l, rt_32x16_s>;
using B_slice = rt_bf<BLOCK_SIZE, DOT_SLICE, row_l, rt_32x16_s>;

#define M 192*40
#define N 8192
#define K 8192


struct micro_globals {
    gl<bf16, -1, -1, -1, -1> a, b;
    gl<bf16, -1, -1, -1, -1> c;
    hipStream_t stream;
    dim3 grid()  { return dim3((N / NEW_COL_BLOCK_SIZE) * ( M / NEW_ROW_BLOCK_SIZE)); } 
    dim3 block() { return dim3(NUM_THREADS); } 
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; } 
};

__global__ __launch_bounds__(NUM_THREADS, 2)
void micro_tk(const micro_globals g) {

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    using ST_A = st_bf<BLOCK_SIZE, BLOCK_SIZE, st_32x16_s>;
    using ST_B = st_bf<BLOCK_SIZE, BLOCK_SIZE, st_32x16_s>;
    ST_A (&As)[2][M_BLOCK] = al.allocate<ST_A, 2, M_BLOCK>();
    ST_B (&Bs)[2][N_BLOCK] = al.allocate<ST_B, 2, N_BLOCK>();
    rt_fl<BLOCK_SIZE, BLOCK_SIZE, col_l, rt_32x32_s> C_accum;

    /// Original WGID.
    int wgid = (blockIdx.y * gridDim.x) + blockIdx.x;
    const int NUM_WGS  = gridDim.x * gridDim.y;
    const int WGM = 4;
    // Swizzle chiplet so that wgids are in the same XCD.
    wgid = chiplet_transform_chunked(wgid, NUM_WGS, NUM_XCDS, WGM*WGM);
    // Swizzle for better L2 within the same XCD.
    const int num_pid_m = ceil_div(M, NEW_ROW_BLOCK_SIZE); // 7680 / 192 = 40
    const int num_pid_n = ceil_div(N, NEW_COL_BLOCK_SIZE); // 7680 / 256 = 30
    const int num_wgid_in_group = WGM * num_pid_n;
    int group_id = wgid / num_wgid_in_group;
    int first_pid_m = group_id * WGM;
    int group_size_m = min(num_pid_m - first_pid_m, WGM);
    int pid_m = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    int pid_n = (wgid % num_wgid_in_group) / group_size_m;
    // Assign the tile's row/column based on the pid_m and pid_n.
    int row = pid_m * M_BLOCK; 
    int col = pid_n * N_BLOCK; 
    // int row = blockIdx.y * M_BLOCK; // works better for large matrices
    // int col = blockIdx.x * N_BLOCK;

    int warp_id = kittens::warpid();
    int local_warp_id = warp_id % 4;
    int warp_group_id = (warp_id - NUM_PRODUCER_WORKERS) / 4;
    bool is_producer = (warp_id < NUM_PRODUCER_WORKERS);
    bool is_consumer = (warp_id >= NUM_PRODUCER_WORKERS && warp_group_id <= M_BLOCK);
    int consumer_idx = is_consumer ? warp_group_id : -1;


    using T = typename st_bf<BLOCK_SIZE, BLOCK_SIZE, st_32x16_s>::dtype;
    constexpr int bytes_per_thread = st_32x16_s::template bytes_per_thread<T>();
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_PRODUCER_THREADS;
    constexpr int memcpy_per_tile = BLOCK_SIZE * BLOCK_SIZE * sizeof(T) / bytes_per_memcpy;
    uint32_t swizzled_offsets_A[memcpy_per_tile];
    uint32_t swizzled_offsets_B[memcpy_per_tile];
    G::prefill_swizzled_offsets(As[0][0], g.a, swizzled_offsets_A);
    G::prefill_swizzled_offsets(Bs[0][0], g.b, swizzled_offsets_B);

    const bool warp_leader = (threadIdx.x % kittens::WARP_THREADS) == 0;

    // Volatile LDS flags/counters
    __shared__ __align__(16) unsigned int ready[2], done[2], prod_cnt[2], init_done;

    if (threadIdx.x == 0) {
        ready[0] = 0; 
        ready[1] = 0;
        done[0]  = 0; 
        done[1]  = 0; 
        prod_cnt[0] = 0; 
        prod_cnt[1] = 0;
        init_done = 0;
    }

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
        asm volatile("s_waitcnt vmcnt(0)");
    }
    if (threadIdx.x == 0) {
        while (prod_cnt[0] < NUM_PRODUCER_WORKERS) { __builtin_amdgcn_s_sleep(0); } 
        prod_cnt[0] = 0; 
        init_done = 1;
    }
    if (consumer_idx == 0) {   
        #pragma unroll
        for (int n=0; n<N_BLOCK; ++n) 
            G::load<2,false>(Bs[1][n], g.b, {0,0, col+n, 1}, swizzled_offsets_B);
        #pragma unroll
        for (int m=0; m<M_BLOCK; ++m) 
            G::load<2,false>(As[1][m], g.a, {0,0, row+m, 1}, swizzled_offsets_A);
        if (warp_leader) atomicAdd((int*)&prod_cnt[1], 1);
        asm volatile("s_waitcnt vmcnt(4)");
    }
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    if (threadIdx.x == 0) {
        // while (prod_cnt[0] < NUM_PRODUCER_WORKERS) { __builtin_amdgcn_s_sleep(0); } 
        while (prod_cnt[1] < NUM_PRODUCER_WORKERS) { __builtin_amdgcn_s_sleep(0); } 
        __threadfence_block(); 
        ready[0] = 0; 
        ready[1] = 1;
        done[0]  = 0; 
        done[1]  = 0; 
        prod_cnt[0] = 0; 
        prod_cnt[1] = 0;
        init_done = 1;
    }

    int num_tiles = K / BLOCK_SIZE;
    constexpr int sleep_time = 0;

    if (is_producer) {
        while (!init_done) { __builtin_amdgcn_s_sleep(0); } 
        #pragma unroll
        for (int tile = 2; tile < num_tiles; ++tile, toc^=1) {
        
            // Wait for consumers to finish with buffer
            const int target_1 = NUM_CONSUMER_WORKERS*(tile/2);
            const int target_2 = NUM_PRODUCER_WORKERS*(tile/2);
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
            
            asm volatile("s_waitcnt vmcnt(4)");
            if (warp_leader) atomicAdd((int*)&prod_cnt[toc], 1);
            while (prod_cnt[toc] < target_2) { __builtin_amdgcn_s_sleep(sleep_time); } 
            if (warp_leader && warp_id == 0) { atomicExch((int*)&ready[toc], tile); }
            
        }
    }

    if (is_consumer) {
        zero(C_accum);
        while (!init_done) { __builtin_amdgcn_s_sleep(0); } 
        unsigned go = 0;
        // #pragma unroll
        for (int tile = 0; tile < num_tiles; ++tile, tic^=1) {
            do {
                unsigned leader_sees_ready = 0;
                // acquire load so LDS writes published by producers are visible
                if (laneid() == 0) { leader_sees_ready = (__atomic_load_n(&ready[tic], __ATOMIC_ACQUIRE) >= (unsigned)tile); }
                // broadcast lane0's decision to the whole warp
                go = __builtin_amdgcn_readfirstlane(leader_sees_ready);
                if (!go) __builtin_amdgcn_s_sleep(0);  
            } while (!go);

            A_slice a0; 
            B_slice b0;

            load(a0, subtile_inplace<BLOCK_SIZE, DOT_SLICE>(As[tic][consumer_idx], {0,0}));
            load(b0, subtile_inplace<BLOCK_SIZE, DOT_SLICE>(Bs[tic][local_warp_id], {0,0}));
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(C_accum, a0, b0, C_accum);
            __builtin_amdgcn_s_setprio(0);

            load(a0, subtile_inplace<BLOCK_SIZE, DOT_SLICE>(As[tic][consumer_idx], {0,1}));
            load(b0, subtile_inplace<BLOCK_SIZE, DOT_SLICE>(Bs[tic][local_warp_id], {0,1}));
            asm volatile("s_waitcnt lgkmcnt(0)");

            __builtin_amdgcn_s_setprio(1);
            mma_ABt(C_accum, a0, b0, C_accum);
            __builtin_amdgcn_s_setprio(0);

            if (warp_leader) atomicAdd((int*)&done[tic], 1);
        }

        store(g.c, C_accum, {0,0, row + consumer_idx, col + local_warp_id});
    }
}

void dispatch_micro(micro_globals g) {
    const unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)micro_tk, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    micro_tk<<<g.grid(), g.block(), mem_size, g.stream>>>(g);
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module";
    py::bind_function<dispatch_micro>(m, "dispatch_micro", &micro_globals::a, &micro_globals::b, &micro_globals::c);
}