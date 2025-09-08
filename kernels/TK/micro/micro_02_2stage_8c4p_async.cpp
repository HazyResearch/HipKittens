#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

constexpr int BLOCK_SIZE = 64;
constexpr int M_BLOCK = 2;
constexpr int N_BLOCK = 4;

constexpr int NEW_ROW_BLOCK_SIZE = BLOCK_SIZE * M_BLOCK;
constexpr int NEW_COL_BLOCK_SIZE = BLOCK_SIZE * N_BLOCK;

#define NUM_PRODUCER_WORKERS (4)
#define NUM_CONSUMER_WORKERS (M_BLOCK * 4)
#define NUM_THREADS ((NUM_PRODUCER_WORKERS + NUM_CONSUMER_WORKERS) * kittens::WARP_THREADS)

using G = kittens::group<NUM_PRODUCER_WORKERS>;

#define M 8192
#define K 8192
#define N 8192

struct micro_globals {
    gl<bf16, -1, -1, -1, -1> a, b;
    gl<bf16, -1, -1, -1, -1> c;
    dim3 grid()  { return dim3((N / NEW_COL_BLOCK_SIZE) * ( M / NEW_ROW_BLOCK_SIZE)); } 
    dim3 block() { return dim3(NUM_THREADS); } 
    size_t dynamic_shared_memory() { return 98304; } 
};

__global__ __launch_bounds__(NUM_THREADS, 2)
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

    int warp_id = kittens::warpid();
    int local_warp_id = warp_id % 4;
    int warp_group_id = kittens::warpgroupid();
    bool is_producer = (warp_group_id == 0);
    bool is_consumer = (warp_group_id > 0 && warp_group_id <= M_BLOCK);
    int consumer_idx = is_consumer ? warp_group_id - 1 : 0;

    const bool warp_leader = (threadIdx.x % kittens::WARP_THREADS) == 0;

    __shared__ int ready[2];     // epoch counters for stage readiness
    __shared__ int done[2];      // number of consumer warps finished with a stage
    __shared__ int prod_cnt[2];  // number of producer warps finished with a stage
    if (threadIdx.x == 0) {
        ready[0] = ready[1] = 0;
        done[0]  = done[1]  = NUM_CONSUMER_WORKERS;
        prod_cnt[0] = prod_cnt[1] = 0;
    }
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
 
    int tic = 0;
    int toc = 1;
    if (is_producer) {
        for (int m=0; m<M_BLOCK; ++m) G::load<2,false>(As[tic][m], g.a, {0,0, row+m, 0});
        for (int n=0; n<N_BLOCK; ++n) G::load<2,false>(Bs[tic][n], g.b, {0,0, col+n, 0});
        __builtin_amdgcn_s_waitcnt(0);
        if (warp_leader) atomicAdd(&prod_cnt[tic], 1);       // 1 tick per producer warp
        if (threadIdx.x == 0) {
            while (__atomic_load_n(&prod_cnt[tic], __ATOMIC_ACQUIRE) < NUM_PRODUCER_WORKERS)
                __builtin_amdgcn_s_sleep(2);
            __atomic_store_n(&ready[tic], 1, __ATOMIC_RELEASE);
            atomicExch(&prod_cnt[tic], 0);                   // reset for next reuse
        }
    }

    if (is_consumer) {zero(C_accum);}
    const int num_tiles = K / BLOCK_SIZE;
    // #pragma unroll // warning this causes a hang :( 
    for (int tile = 0; tile < num_tiles; ++tile, tic^=1, toc^=1) {
        const int need_epoch = tile + 1;
        const int next_epoch = tile + 2;
        const bool has_next  = (tile + 1) < num_tiles;

        if (is_consumer) {
            while (__atomic_load_n(&ready[tic], __ATOMIC_ACQUIRE) < need_epoch)
                __builtin_amdgcn_s_sleep(2);

            rt_bf<BLOCK_SIZE,BLOCK_SIZE,row_l> a_reg, b_reg;
            load(a_reg, As[tic][consumer_idx]);
            load(b_reg, Bs[tic][local_warp_id]);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(C_accum, a_reg, b_reg, C_accum);
            __builtin_amdgcn_s_setprio(0);
            if (warp_leader) atomicAdd(&done[tic], 1);
        }
        if (is_producer && has_next) {
            // make sure the 'toc' stage finished with prior epoch
            while (__atomic_load_n(&done[toc], __ATOMIC_ACQUIRE) < NUM_CONSUMER_WORKERS)
                __builtin_amdgcn_s_sleep(2);
            if (threadIdx.x == 0) atomicExch(&done[toc], 0); // reset for next round

            for (int m=0; m<M_BLOCK; ++m) G::load<2,false>(As[toc][m], g.a, {0,0, row+m, tile+1});
            for (int n=0; n<N_BLOCK; ++n) G::load<2,false>(Bs[toc][n], g.b, {0,0, col+n, tile+1});

            __builtin_amdgcn_s_waitcnt(0);
            if (warp_leader) atomicAdd(&prod_cnt[toc], 1);
            if (threadIdx.x == 0) {
                while (__atomic_load_n(&prod_cnt[toc], __ATOMIC_ACQUIRE) < NUM_PRODUCER_WORKERS)
                    __builtin_amdgcn_s_sleep(2);
                __atomic_store_n(&ready[toc], next_epoch, __ATOMIC_RELEASE);
                atomicExch(&prod_cnt[toc], 0);
            }
        }
    }

    if (is_consumer) {
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
    printf("kernel_ms=%.3f\n", ms);
    hipDeviceSynchronize();
  }

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module";
    py::bind_function<dispatch_micro>(m, "dispatch_micro", &micro_globals::a, &micro_globals::b, &micro_globals::c);
}

