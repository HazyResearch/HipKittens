#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;


constexpr int BLOCK_SIZE = 64;
constexpr int M_BLOCK = 2;
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

__host__ __device__ inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
  }

#define M 8192
#define K 8192
#define N 8192

struct micro_globals {
    gl<bf16, -1, -1, -1, -1> a, b;
    gl<bf16, -1, -1, -1, -1> c;
    dim3 grid()  { return dim3((N / NEW_COL_BLOCK_SIZE) * ( M / NEW_ROW_BLOCK_SIZE)); } 
    dim3 block() { return dim3(NUM_THREADS); } 
    size_t dynamic_shared_memory() { return 3 * (M_BLOCK + N_BLOCK) * BLOCK_SIZE * BLOCK_SIZE * sizeof(bf16); } 
};

__global__ __launch_bounds__(NUM_THREADS, 2)
void micro_tk(const micro_globals g) {

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<BLOCK_SIZE, BLOCK_SIZE, ducks::st_layout::row> (&As)[3][M_BLOCK] =
    al.allocate<st_bf<BLOCK_SIZE, BLOCK_SIZE, ducks::st_layout::row>, 3, M_BLOCK>();
    st_bf<BLOCK_SIZE, BLOCK_SIZE, ducks::st_layout::row> (&Bs)[3][N_BLOCK] =
    al.allocate<st_bf<BLOCK_SIZE, BLOCK_SIZE, ducks::st_layout::row>, 3, N_BLOCK>();

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

    using T = typename st_bf<BLOCK_SIZE, BLOCK_SIZE>::dtype;
    constexpr int bytes_per_thread = 16;
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_PRODUCER_THREADS;
    constexpr int memcpy_per_tile = BLOCK_SIZE * BLOCK_SIZE * sizeof(T) / bytes_per_memcpy;
    uint32_t swizzled_offsets_A[memcpy_per_tile];
    uint32_t swizzled_offsets_B[memcpy_per_tile];
    G::prefill_swizzled_offsets(As[0][0], g.a, swizzled_offsets_A);
    G::prefill_swizzled_offsets(Bs[0][0], g.b, swizzled_offsets_B);

    __shared__ int ready[3];      // published epoch per stage (release store)
    __shared__ int done[3];       // number of consumer warps finished with a stage
    __shared__ int done_epoch[3];  // epoch of last consumer warp to finish with a stage
    __shared__ int prod_cnt[3];   // producer quorum per stage
    __shared__ int go[3][M_BLOCK];
    const bool warp_leader = (threadIdx.x % kittens::WARP_THREADS) == 0;
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int s=0;s<3;++s) {
            ready[s] = 0;
            done[s]       = 0;   
            done_epoch[s] = 0;  
            prod_cnt[s] = 0;
        }
    }
    __syncthreads();
    
    int s = 0, n1 = 1, n2 = 2;
    if (is_producer) {
        // preload tile 0 into stage s
        #pragma unroll
        for (int m=0; m<M_BLOCK; ++m) G::load<2,false>(As[s][m],  g.a, {0,0, row+m, 0}, swizzled_offsets_A);
        #pragma unroll
        for (int n=0; n<N_BLOCK; ++n) G::load<2,false>(Bs[s][n],  g.b, {0,0, col+n, 0}, swizzled_offsets_B);

        __builtin_amdgcn_s_waitcnt(0);
        if (warp_leader) atomicAdd(&prod_cnt[s], 1);
        if (threadIdx.x == 0) {
            while (__atomic_load_n(&prod_cnt[s], __ATOMIC_ACQUIRE) < NUM_PRODUCER_WORKERS)
                __builtin_amdgcn_s_sleep(1);
            __atomic_store_n(&ready[s], 1, __ATOMIC_RELEASE);
            atomicExch(&prod_cnt[s], 0); // reset
        }
        
        // preload tile 1 into stage n1
        #pragma unroll
        for (int m=0; m<M_BLOCK; ++m) G::load<2,false>(As[n1][m], g.a, {0,0, row+m, 1}, swizzled_offsets_A);
        #pragma unroll
        for (int n=0; n<N_BLOCK; ++n) G::load<2,false>(Bs[n1][n], g.b, {0,0, col+n, 1}, swizzled_offsets_B);

        __builtin_amdgcn_s_waitcnt(0);
        if (warp_leader) atomicAdd(&prod_cnt[n1], 1);
        if (threadIdx.x == 0) {
            while (__atomic_load_n(&prod_cnt[n1], __ATOMIC_ACQUIRE) < NUM_PRODUCER_WORKERS)
                __builtin_amdgcn_s_sleep(1);
            __atomic_store_n(&ready[n1], 2, __ATOMIC_RELEASE);
            atomicExch(&prod_cnt[n1], 0); // reset
        } 
    }

    rt_fl<BLOCK_SIZE, BLOCK_SIZE, accum_col_l> C_accum;
    if (is_consumer) {zero(C_accum);}
    const int num_tiles = K / BLOCK_SIZE;
    #pragma unroll
    for (int tile = 0; tile < num_tiles; ++tile) {
        int s  =  tile      % 3;
        int n2 = (tile + 2) % 3;
        int need_epoch = tile + 1;
        int next_epoch = tile + 2;
        int next_next_epoch = tile + 3;
        bool has_next2 = (tile + 2) < num_tiles;

        if (is_consumer) {
            while (__atomic_load_n(&ready[s], __ATOMIC_ACQUIRE) < need_epoch)
                __builtin_amdgcn_s_sleep(1);
            
            A_slice a0; B_slice b0;
            load(a0, subtile_inplace<BLOCK_SIZE, DOT_SLICE>(As[s][consumer_idx],  {0,0}));
            load(b0, subtile_inplace<BLOCK_SIZE, DOT_SLICE>(Bs[s][local_warp_id], {0,0}));
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(C_accum, a0, b0, C_accum);
            __builtin_amdgcn_s_setprio(0);

            load(a0, subtile_inplace<BLOCK_SIZE, DOT_SLICE>(As[s][consumer_idx],  {0,1}));
            load(b0, subtile_inplace<BLOCK_SIZE, DOT_SLICE>(Bs[s][local_warp_id], {0,1}));
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(C_accum, a0, b0, C_accum);
            __builtin_amdgcn_s_setprio(0);

            if (warp_leader) {
                int v = atomicAdd(&done[s], 1) + 1;
                if (v == NUM_CONSUMER_WORKERS) {
                    __atomic_store_n(&done_epoch[s], need_epoch, __ATOMIC_RELEASE);
                    atomicExch(&done[s], 0);
                }
            }
        }

        if (is_producer && has_next2) {
            int prev = __atomic_load_n(&ready[n2], __ATOMIC_ACQUIRE);
            while (__atomic_load_n(&done_epoch[n2], __ATOMIC_ACQUIRE) < prev)
                __builtin_amdgcn_s_sleep(1);
        
            #pragma unroll
            for (int m=0; m<M_BLOCK; ++m)
                G::load<2,false>(As[n2][m], g.a, {0,0,row+m,tile+2}, swizzled_offsets_A);
            #pragma unroll
            for (int n=0; n<N_BLOCK; ++n)
                G::load<2,false>(Bs[n2][n], g.b, {0,0,col+n,tile+2}, swizzled_offsets_B);
        
            __builtin_amdgcn_s_waitcnt(0);  
            if (warp_leader)
                __atomic_fetch_add(&prod_cnt[n2], 1, __ATOMIC_RELEASE); 
        
            if (threadIdx.x == 0) {
                while (__atomic_load_n(&prod_cnt[n2], __ATOMIC_ACQUIRE) < NUM_PRODUCER_WORKERS) 
                    __builtin_amdgcn_s_sleep(1);
                __atomic_store_n(&ready[n2], next_next_epoch, __ATOMIC_RELEASE);
                atomicExch(&prod_cnt[n2], 0);
            }
        }
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
    // printf("kernel_ms=%.3f\n", ms);
    hipDeviceSynchronize();
  }

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module";
    py::bind_function<dispatch_micro>(m, "dispatch_micro", &micro_globals::a, &micro_globals::b, &micro_globals::c);
}

