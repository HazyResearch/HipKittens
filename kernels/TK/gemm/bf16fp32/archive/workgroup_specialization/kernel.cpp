#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

constexpr int BLOCK_SIZE = 32;
constexpr int M_BLOCK = 2;
constexpr int N_BLOCK = 4;

constexpr int NEW_ROW_BLOCK_SIZE = BLOCK_SIZE * M_BLOCK;
constexpr int NEW_COL_BLOCK_SIZE = BLOCK_SIZE * N_BLOCK;

#define NUM_PRODUCER_WORKERS (4)
#define NUM_CONSUMER_WORKERS (M_BLOCK * 4)
#define NUM_THREADS ((NUM_PRODUCER_WORKERS + NUM_CONSUMER_WORKERS) * kittens::WARP_THREADS)

#define M 8192
#define K 8192
#define N 8192

using _gl_A = gl<bf16, -1, -1, -1, -1>;
using _gl_B = gl<bf16, -1, -1, -1, -1>;
using _gl_C = gl<float, -1, -1, -1, -1>;

struct micro_globals {
    _gl_A a;
    _gl_B b;
    _gl_C c;
    float alpha;
    float beta;

    // grid - number of thread blocks we are launching
    dim3 grid()  { return dim3(N / NEW_COL_BLOCK_SIZE, M / NEW_ROW_BLOCK_SIZE); } 

    // block - number of threads in a thread block
    dim3 block() { return dim3(NUM_THREADS); } 

    // Safe shared memory size for MI325X
    size_t dynamic_shared_memory() { return 32768; } 
};

__global__
void micro_tk(const micro_globals g) {

    // shared memory
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<BLOCK_SIZE, BLOCK_SIZE> (&As)[2][M_BLOCK] = al.allocate<st_bf<BLOCK_SIZE, BLOCK_SIZE>, 2, M_BLOCK>();
    st_bf<BLOCK_SIZE, BLOCK_SIZE> (&Bs)[2][N_BLOCK] = al.allocate<st_bf<BLOCK_SIZE, BLOCK_SIZE>, 2, N_BLOCK>();

    st_bf<BLOCK_SIZE, BLOCK_SIZE> (&C_tiles)[M_BLOCK][N_BLOCK] = al.allocate<st_bf<BLOCK_SIZE, BLOCK_SIZE>, M_BLOCK, N_BLOCK>();

    int tic = 0;
    int toc = 1;

    // Accumulator for each consumer warp group
    rt_fl<BLOCK_SIZE, BLOCK_SIZE, ducks::rt_layout::col> C_accum;

    int row = blockIdx.y * M_BLOCK;
    int col = blockIdx.x * N_BLOCK;

    int warp_id = kittens::warpid();
    int local_warp_id = warp_id % 4;
    int warp_group_id = kittens::warpgroupid();

    bool is_producer = (warp_group_id == 0);
    bool is_consumer = (warp_group_id > 0 && warp_group_id <= M_BLOCK);

    int consumer_idx = is_consumer ? warp_group_id - 1 : 0;
    
    // Pre-load first iteration into buffer 0
    if (is_producer && warp_id == 0) {

        for (int m = 0; m < M_BLOCK; m++) {
            load(As[tic][m], g.a, {0, 0, row + m, 0});
        }

        for (int n = 0; n < N_BLOCK; n++) {
            load(Bs[tic][n], g.b, {0, 0, 0, col + n});
        }
    }
    __syncthreads();

    if (is_consumer) {
        zero(C_accum);
    }
    
    // loop over K with double buffering
    int num_tiles = K / BLOCK_SIZE;
    for (int tile = 0; tile < num_tiles; ++tile, tic ^= 1, toc ^= 1) {

        if (is_producer && tile + 1 < num_tiles && warp_id == 0) {
            for (int m = 0; m < M_BLOCK; m++) {
                load(As[toc][m], g.a, {0, 0, row + m, tile + 1});
            }

            for (int n = 0; n < N_BLOCK; n++) {
                load(Bs[toc][n], g.b, {0, 0, tile + 1, col + n});
            }
        } else if (is_consumer) {

            // load from SMEM to registers for current iteration
            rt_bf<BLOCK_SIZE, BLOCK_SIZE> a_reg;
            rt_bf<BLOCK_SIZE, BLOCK_SIZE, ducks::rt_layout::col> b_reg;

            load(a_reg, As[tic][consumer_idx]);
            load(b_reg, Bs[tic][local_warp_id]);

            // accumulate
            mma_AB(C_accum, a_reg, b_reg, C_accum);
        }
        __syncthreads();
    }

    // Store result back to HBM
    if (is_consumer) {
        store(g.c, C_accum, {0, 0, row + consumer_idx, col + local_warp_id});
    }
}

// Launch Kernel
void dispatch_micro(micro_globals g) {

    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute(
        (void*)micro_tk,
        hipFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );

    micro_tk<<<g.grid(), g.block(), mem_size>>>(g);
    hipDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module";
    // For wrapping kernels directly.
    // Old: BIND_KERNEL(m, "micro_tk", micro_tk, micro_globals, x, o); 
    py::bind_kernel<micro_tk>(m, "micro_tk", &micro_globals::a, &micro_globals::b, &micro_globals::c,
                              &micro_globals::alpha, &micro_globals::beta); 

    // For host functions that wrap the kernel, this will be called from Python
    // Old: BIND_FUNCTION(m, "dispatch_micro", dispatch_micro, micro_globals, x, o); 
    py::bind_function<dispatch_micro>(m, "dispatch_micro", &micro_globals::a, &micro_globals::b, &micro_globals::c,
                                      &micro_globals::alpha, &micro_globals::beta);
}