#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

constexpr int BLOCK_SIZE = 64;
constexpr int M_BLOCK = 2;
constexpr int N_BLOCK = 4;
constexpr int DOT_SLICE = 32;
constexpr int HALF_BLOCK_SIZE = BLOCK_SIZE / 2; // 32
constexpr int NUM_STAGES = 3;

constexpr int NEW_ROW_BLOCK_SIZE = BLOCK_SIZE * M_BLOCK;
constexpr int NEW_COL_BLOCK_SIZE = BLOCK_SIZE * N_BLOCK;

#define NUM_PRODUCER_WORKERS (4)
#define NUM_CONSUMER_WORKERS (M_BLOCK * 4)
#define NUM_THREADS ((NUM_PRODUCER_WORKERS + NUM_CONSUMER_WORKERS) * kittens::WARP_THREADS)
#define NUM_PRODUCER_THREADS (NUM_PRODUCER_WORKERS * kittens::WARP_THREADS)

using G = kittens::group<NUM_PRODUCER_WORKERS>;
using A_slice = rt_bf<HALF_BLOCK_SIZE, BLOCK_SIZE, row_l, rt_16x32_s>;
using B_slice = rt_bf<HALF_BLOCK_SIZE, BLOCK_SIZE, row_l, rt_16x32_s>;

#define M 7680
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
    ST_A (&As)[NUM_STAGES][M_BLOCK][2] = al.allocate<ST_A, NUM_STAGES, M_BLOCK, 2>();
    ST_B (&Bs)[NUM_STAGES][N_BLOCK][2] = al.allocate<ST_B, NUM_STAGES, N_BLOCK, 2>();
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
    bool is_consumer = (warp_id >= NUM_PRODUCER_WORKERS && warp_group_id <= M_BLOCK);
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
    __shared__ __align__(16) unsigned int ready[NUM_STAGES], done[NUM_STAGES], prod_cnt[NUM_STAGES], init_done;
    if (threadIdx.x == 0) {
        ready[0] = 0; 
        ready[1] = 1;
        ready[2] = 0;
        done[0]  = 0; 
        done[1]  = 0; 
        done[2]  = 0; 
        prod_cnt[0] = 0; 
        prod_cnt[1] = 0;
        prod_cnt[2] = 0;
        init_done = 0;
    }

    int s = 0, n1 = 1, n2 = 2;
    if (is_producer) {
        #pragma unroll
        for (int m=0; m<M_BLOCK; ++m) {
            G::load<2,false>(As[s][m][0], g.a, {0, 0, row*2 + 2*m + 0, 0}, swizzled_offsets_A);
            G::load<2,false>(As[s][m][1], g.a, {0, 0, row*2 + 2*m + 1, 0}, swizzled_offsets_A);
        }
        #pragma unroll
        for (int n=0; n<N_BLOCK; ++n) {
            G::load<2,false>(Bs[s][n][0], g.b, {0, 0, col*2 + 2*n + 0, 0}, swizzled_offsets_B);
            G::load<2,false>(Bs[s][n][1], g.b, {0, 0, col*2 + 2*n + 1, 0}, swizzled_offsets_B);
        }
        if (warp_leader) atomicAdd((int*)&prod_cnt[0], 1);  
        asm volatile("s_waitcnt vmcnt(0)");
    }
    if (threadIdx.x == 0) {
        while (prod_cnt[0] < NUM_PRODUCER_WORKERS) { __builtin_amdgcn_s_sleep(0); } 
        prod_cnt[0] = 0; 
    }
    if (consumer_idx == 0) {   
        #pragma unroll
        for (int n=0; n<N_BLOCK; ++n) {
            G::load<2,false>(Bs[1][n][0], g.b, {0,0, col*2 + 2*n + 0, 1}, swizzled_offsets_B);
            G::load<2,false>(Bs[1][n][1], g.b, {0,0, col*2 + 2*n + 1, 1}, swizzled_offsets_B);
        }
        #pragma unroll
        for (int m=0; m<M_BLOCK; ++m) {
            G::load<2,false>(As[1][m][0], g.a, {0,0, row*2 + 2*m + 0, 1}, swizzled_offsets_A);
            G::load<2,false>(As[1][m][1], g.a, {0,0, row*2 + 2*m + 1, 1}, swizzled_offsets_A);
        }
        if (warp_leader) atomicAdd((int*)&prod_cnt[1], 1);
        asm volatile("s_waitcnt vmcnt(0)");
    }
    if (is_consumer) {  
        zero(C_accum[0][0]); 
        zero(C_accum[0][1]); 
        zero(C_accum[1][0]); 
        zero(C_accum[1][1]); 
    }
    if (threadIdx.x == 0) {
        while (prod_cnt[1] < NUM_PRODUCER_WORKERS) { __builtin_amdgcn_s_sleep(0); } 
        prod_cnt[1] = 0;
        init_done = 1;
    }
    __syncthreads();

    int num_tiles = K / BLOCK_SIZE;
    constexpr int sleep_time = 0;

    if (is_producer) {
        while (!init_done) { __builtin_amdgcn_s_sleep(0); } 
        // #pragma unroll
        for (int tile = 2; tile < num_tiles; ++tile) {
        
            // Wait for consumers to finish with buffer
            const int target_1 = NUM_CONSUMER_WORKERS*(tile/NUM_STAGES);
            const int target_2 = NUM_PRODUCER_WORKERS*(tile/NUM_STAGES);
            while (done[n2] < target_1) { 
                __builtin_amdgcn_s_sleep(sleep_time); 
            }
            // Load next tile
            #pragma unroll
            for (int n=0; n<N_BLOCK; ++n) {
                G::load<2,false>(Bs[n2][n][0], g.b, {0,0, col*2 + 2*n + 0, tile}, swizzled_offsets_B);
                G::load<2,false>(Bs[n2][n][1], g.b, {0,0, col*2 + 2*n + 1, tile}, swizzled_offsets_B);
            }
            #pragma unroll
            for (int m=0; m<M_BLOCK; ++m) {
                G::load<2,false>(As[n2][m][0], g.a, {0,0, row*2 + 2*m + 0, tile}, swizzled_offsets_A);
                G::load<2,false>(As[n2][m][1], g.a, {0,0, row*2 + 2*m + 1, tile}, swizzled_offsets_A);
            }
            asm volatile("s_waitcnt vmcnt(0)");

            if (warp_leader) atomicAdd((int*)&prod_cnt[n2], 1);
            while (prod_cnt[n2] < target_2) { __builtin_amdgcn_s_sleep(sleep_time); } 
            if (warp_leader && warp_id == 0) { atomicExch((int*)&ready[n2], tile); }
            int tmp = s;
            s = n1; n1 = n2; n2 = tmp;
        }
    }

    if (is_consumer) {
        while (!init_done) { __builtin_amdgcn_s_sleep(0); } 
        unsigned go = 0;

        // #pragma unroll
        for (int tile = 0; tile < num_tiles; ++tile) {
            do {
                unsigned leader_sees_ready = 0;
                // acquire load so LDS writes published by producers are visible
                if (laneid() == 0) { leader_sees_ready = (__atomic_load_n(&ready[s], __ATOMIC_ACQUIRE) == (unsigned)tile); }
                // broadcast lane0's decision to the whole warp
                go = __builtin_amdgcn_readfirstlane(leader_sees_ready);
                if (!go) __builtin_amdgcn_s_sleep(0);  // polite spin
            } while (!go);

            A_slice a0; 
            B_slice b0, b1;
            auto st_subtile_b = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(Bs[s][local_warp_id][0], {0,0});
            auto st_subtile_a = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(As[s][consumer_idx][0], {0,0});
            auto st_subtile_b_next = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(Bs[s][local_warp_id][1], {0,0});
            load(a0, st_subtile_a);
            load(b0, st_subtile_b);
            load(b1, st_subtile_b_next);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(C_accum[0][0], a0, b0, C_accum[0][0]);
            mma_ABt(C_accum[0][1], a0, b1, C_accum[0][1]);
            __builtin_amdgcn_s_setprio(0);

            st_subtile_a = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(As[s][consumer_idx][1], {0,0});
            load(a0, st_subtile_a);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(C_accum[1][0], a0, b0, C_accum[1][0]);
            mma_ABt(C_accum[1][1], a0, b1, C_accum[1][1]);
            __builtin_amdgcn_s_setprio(0);

            if (warp_leader) atomicAdd((int*)&done[s], 1);
            int tmp = s;
            s = n1; n1 = n2; n2 = tmp;
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