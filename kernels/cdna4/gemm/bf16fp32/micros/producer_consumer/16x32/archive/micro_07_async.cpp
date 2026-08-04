#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

constexpr int BLOCK_SIZE = 64;
constexpr int M_BLOCK = 3;
constexpr int N_BLOCK = 4;
constexpr int DOT_SLICE = 32;
constexpr int HALF_BLOCK_SIZE = BLOCK_SIZE / 2; // 32

constexpr int NEW_ROW_BLOCK_SIZE = BLOCK_SIZE * M_BLOCK;
constexpr int NEW_COL_BLOCK_SIZE = BLOCK_SIZE * N_BLOCK;

#define NUM_PRODUCER_WORKERS (4)
#define NUM_CONSUMER_WORKERS (M_BLOCK * 4)
#define NUM_THREADS ((NUM_PRODUCER_WORKERS + NUM_CONSUMER_WORKERS) * kittens::WARP_THREADS)
#define NUM_PRODUCER_THREADS (NUM_PRODUCER_WORKERS * kittens::WARP_THREADS)

using G = kittens::group<NUM_PRODUCER_WORKERS>;
using A_slice = rt_bf<HALF_BLOCK_SIZE, BLOCK_SIZE, row_l, rt_16x32_s>;
using B_slice = rt_bf<HALF_BLOCK_SIZE, BLOCK_SIZE, row_l, rt_16x32_s>;

#define M 192*40
#define K 8192
#define N 8192

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
    using ST_A = st_bf<HALF_BLOCK_SIZE, BLOCK_SIZE, st_16x32_s>;
    using ST_B = st_bf<HALF_BLOCK_SIZE, BLOCK_SIZE, st_16x32_s>;
    ST_A (&As)[2][M_BLOCK][2] = al.allocate<ST_A, 2, M_BLOCK, 2>();
    ST_B (&Bs)[2][N_BLOCK][2] = al.allocate<ST_B, 2, N_BLOCK, 2>();
    rt_fl<HALF_BLOCK_SIZE, HALF_BLOCK_SIZE, col_l, rt_16x16_s> C_accum[2][2];

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
    bool is_consumer = (warp_id >= NUM_PRODUCER_WORKERS) && (warp_group_id < M_BLOCK);
    int consumer_idx = is_consumer ? warp_group_id : -1;

    using T = typename st_bf<BLOCK_SIZE, BLOCK_SIZE, st_16x32_s>::dtype;
    constexpr int bytes_per_thread = st_16x32_s::template bytes_per_thread<T>();
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_PRODUCER_THREADS;
    constexpr int memcpy_per_tile = BLOCK_SIZE * BLOCK_SIZE * sizeof(T) / bytes_per_memcpy;
    uint32_t swizzled_offsets_A[memcpy_per_tile];
    uint32_t swizzled_offsets_B[memcpy_per_tile];
    G::prefill_swizzled_offsets(As[0][0][0], g.a, swizzled_offsets_A);
    G::prefill_swizzled_offsets(Bs[0][0][0], g.b, swizzled_offsets_B);

    const bool warp_leader = (threadIdx.x % kittens::WARP_THREADS) == 0;

    // Volatile LDS flags/counters
    __shared__ __align__(16) unsigned int readyA[2][M_BLOCK], readyB[2][N_BLOCK], doneA[2][M_BLOCK], doneB[2][N_BLOCK], prod_cntA[2][M_BLOCK], prod_cntB[2][N_BLOCK];
    if (threadIdx.x == 0) {
        for (int i=0; i<M_BLOCK; ++i) {
            readyA[0][i]=readyA[1][i]=0;
            doneA[0][i]=doneA[1][i]=0;
            prod_cntA[0][i]=prod_cntA[1][i]=0;
        }
        for (int j=0; j<N_BLOCK; ++j) {
            readyB[0][j]=readyB[1][j]=0;
            doneB[0][j]=doneB[1][j]=0;
            prod_cntB[0][j]=prod_cntB[1][j]=0;
        }
    }

    int tic = 0;
    int toc = 0;
    if (warp_id < 3) {
        int m = warp_id, n = warp_id;
        load<2,false>(As[tic][m][0], g.a, {0, 0, row*2 + 2*m + 0, 0});
        load<2,false>(Bs[tic][n][0], g.b, {0, 0, col*2 + 2*n + 0, 0});
        load<2,false>(As[tic][m][1], g.a, {0, 0, row*2 + 2*m + 1, 0});
        load<2,false>(Bs[tic][n][1], g.b, {0, 0, col*2 + 2*n + 1, 0});
        if (warp_leader) atomicAdd((int*)&prod_cntA[0][warp_id], 1);  
        if (warp_leader) atomicAdd((int*)&prod_cntB[0][warp_id], 1);  
    }
    if (warp_id == 6) {
        int n = 3;
        load<2,false>(Bs[tic][n][0], g.b, {0, 0, col*2 + 2*n + 0, 0});
        load<2,false>(Bs[tic][n][1], g.b, {0, 0, col*2 + 2*n + 1, 0});
        if (warp_leader) atomicAdd((int*)&prod_cntB[0][3], 1);  
    }
    // if (warp_id >= 3 && warp_id < 6) {
    //     int m = warp_id % 3, n = warp_id % 3;
    //     load<2,false>(As[1][m][0], g.a, {0, 0, row*2 + 2*m + 0, 1});
    //     load<2,false>(Bs[1][n][0], g.b, {0, 0, col*2 + 2*n + 0, 1});
    //     load<2,false>(As[1][m][1], g.a, {0, 0, row*2 + 2*m + 1, 1});
    //     load<2,false>(Bs[1][n][1], g.b, {0, 0, col*2 + 2*n + 1, 1});
    //     if (warp_leader) atomicAdd((int*)&prod_cntA[1][warp_id % 3], 1);  
    //     if (warp_leader) atomicAdd((int*)&prod_cntB[1][warp_id % 3], 1);  
    // }
    // if (warp_id == 7) {
    //     int n = 3;
    //     load<2,false>(Bs[1][n][0], g.b, {0, 0, col*2 + 2*n + 0, 1});
    //     load<2,false>(Bs[1][n][1], g.b, {0, 0, col*2 + 2*n + 1, 1});
    //     if (warp_leader) atomicAdd((int*)&prod_cntB[1][3], 1);  
    // }
    
    if (warp_leader && warp_id < 3) {
        while (prod_cntA[0][warp_id] < 1) { __builtin_amdgcn_s_sleep(0); }
        while (prod_cntB[0][warp_id] < 1) { __builtin_amdgcn_s_sleep(0); }
        // while (prod_cntA[1][warp_id] < 1) { __builtin_amdgcn_s_sleep(0); }
        // while (prod_cntB[1][warp_id] < 1) { __builtin_amdgcn_s_sleep(0); }
        prod_cntA[0][warp_id] = prod_cntB[0][warp_id] = 0;
        // prod_cntA[1][warp_id] = prod_cntB[1][warp_id] = 0;
        __atomic_store_n(&readyA[0][warp_id], 0, __ATOMIC_RELEASE);
        __atomic_store_n(&readyB[0][warp_id], 0, __ATOMIC_RELEASE);
        // __atomic_store_n(&readyA[1][warp_id], 1, __ATOMIC_RELEASE);
        // __atomic_store_n(&readyB[1][warp_id], 1, __ATOMIC_RELEASE);
    }
    if (warp_leader && warp_id == 3) {
        while (prod_cntB[0][3] < 1) { __builtin_amdgcn_s_sleep(0); }
        // while (prod_cntB[1][3] < 1) { __builtin_amdgcn_s_sleep(0); }
        prod_cntB[0][3] = prod_cntB[1][3] = 0;
        __atomic_store_n(&readyB[0][3], 1, __ATOMIC_RELEASE);
        // __atomic_store_n(&readyB[1][3], 1, __ATOMIC_RELEASE);
    }
    asm volatile("s_waitcnt vmcnt(0)");
    __syncthreads();

    int num_tiles = K / BLOCK_SIZE;
    constexpr int sleep_time = 0;

    if (warp_id < 3) {
        #pragma unroll
        for (int tile = 1; tile < num_tiles; ++tile) {
            // Wait for consumers to finish with buffer
            const int tiles_in_that_buffer = tile;
            const int needA = tiles_in_that_buffer * N_BLOCK; // 3 * tiles_in_that_buffer
            const int needB = tiles_in_that_buffer * M_BLOCK; // 3 * tiles_in_that_buffer
            while (doneA[toc][warp_id] < needA) { __builtin_amdgcn_s_sleep(sleep_time); }
            while (doneB[toc][warp_id] < needB) { __builtin_amdgcn_s_sleep(sleep_time); }
            load<2,false>(As[toc][warp_id][0], g.a, {0,0, row*2 + 2*warp_id + 0, tile});
            load<2,false>(As[toc][warp_id][1], g.a, {0,0, row*2 + 2*warp_id + 1, tile});
            load<2,false>(Bs[toc][warp_id][0], g.b, {0,0, col*2 + 2*warp_id + 0, tile});
            load<2,false>(Bs[toc][warp_id][1], g.b, {0,0, col*2 + 2*warp_id + 1, tile});
            asm volatile("s_waitcnt vmcnt(8)");
            if (warp_leader) { __atomic_store_n(&readyA[toc][warp_id], tile, __ATOMIC_RELEASE); }
            if (warp_leader) { __atomic_store_n(&readyB[toc][warp_id], tile, __ATOMIC_RELEASE); }
        }
    }
    if (warp_id == 3) {
        #pragma unroll
        for (int tile = 1; tile < num_tiles; ++tile) {
            // Wait for consumers to finish with buffer
            const int tiles_in_that_buffer = tile;
            const int needB = tiles_in_that_buffer * M_BLOCK; // 3 * tiles_in_that_buffer
            while (doneB[toc][warp_id] < needB) { __builtin_amdgcn_s_sleep(sleep_time); }
            load<2,false>(Bs[toc][warp_id][0], g.b, {0,0, col*2 + 2*warp_id + 0, tile});
            load<2,false>(Bs[toc][warp_id][1], g.b, {0,0, col*2 + 2*warp_id + 1, tile});
            asm volatile("s_waitcnt vmcnt(8)");
            if (warp_leader) { __atomic_store_n(&readyB[toc][warp_id], tile, __ATOMIC_RELEASE); }
        }
    }

    if (is_consumer) {
        zero(C_accum[0][0]); 
        zero(C_accum[0][1]); 
        zero(C_accum[1][0]); 
        zero(C_accum[1][1]); 
        unsigned go = 0;
        // #pragma unroll
        for (int tile = 0; tile < num_tiles; ++tile) {
            do {
                unsigned okA = 0, okB = 0;
                // acquire load so LDS writes published by producers are visible
                if (laneid() == 0) { okA = (__atomic_load_n(&readyA[tic][consumer_idx], __ATOMIC_ACQUIRE) >= (unsigned)tile); }
                if (laneid() == 0) { okB = (__atomic_load_n(&readyB[tic][local_warp_id], __ATOMIC_ACQUIRE) >= (unsigned)tile); }
                // broadcast lane0's decision to the whole warp
                go = __builtin_amdgcn_readfirstlane(okA & okB);
                if (!go) __builtin_amdgcn_s_sleep(0);  // polite spin
            } while (!go);

            A_slice a0; 
            B_slice b0, b1;
            auto st_subtile_b = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(Bs[tic][local_warp_id][0], {0,0});
            auto st_subtile_a = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(As[tic][consumer_idx][0], {0,0});
            auto st_subtile_b_next = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(Bs[tic][local_warp_id][1], {0,0});
            load(a0, st_subtile_a);
            load(b0, st_subtile_b);
            load(b1, st_subtile_b_next);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(C_accum[0][0], a0, b0, C_accum[0][0]);
            mma_ABt(C_accum[0][1], a0, b1, C_accum[0][1]);
            __builtin_amdgcn_s_setprio(0);

            st_subtile_a = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(As[tic][consumer_idx][1], {0,0});
            load(a0, st_subtile_a);
            asm volatile("s_waitcnt lgkmcnt(0)");
            if (warp_leader) atomicAdd((int*)&doneB[tic][local_warp_id], 1);
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(C_accum[1][0], a0, b0, C_accum[1][0]);
            mma_ABt(C_accum[1][1], a0, b1, C_accum[1][1]);
            __builtin_amdgcn_s_setprio(0);
            if (warp_leader) atomicAdd((int*)&doneA[tic][consumer_idx], 1);
        }

        store(g.c, C_accum[0][0], {0, 0,
            (row + consumer_idx) * 2 + 0,
            (col + local_warp_id) * 2 + 0});
        
        store(g.c, C_accum[0][1], {0, 0,
            (row + consumer_idx) * 2 + 0,
            (col + local_warp_id) * 2 + 1});
        
        store(g.c, C_accum[1][0], {0, 0,
            (row + consumer_idx) * 2 + 1,
            (col + local_warp_id) * 2 + 0});
        
        store(g.c, C_accum[1][1], {0, 0,
            (row + consumer_idx) * 2 + 1,
            (col + local_warp_id) * 2 + 1});
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