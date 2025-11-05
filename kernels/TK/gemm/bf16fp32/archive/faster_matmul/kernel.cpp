#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

#define NUM_THREADS (kittens::WARPGROUP_THREADS)

#define M 8192
#define K 8192
#define N 8192

#define BM 128
#define BN 128
#define BK 16

#define WM 64
#define WN 64
#define WNITER 4
#define TM 4
#define TN 1

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
    dim3 grid()  { return dim3(N / BN, M / BM); } 

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
    st_bf<WM, BK> (&As)[4] = al.allocate<st_bf<WM, BK>, 4>();
    st_bf<BK, WN> (&Bs)[4] = al.allocate<st_bf<BK, WN>, 4>();

    int row = blockIdx.x;
    int col = blockIdx.y;

    int warp_id = warpid();
    int warp_row = warp_id / 2;
    int warp_col = warp_id % 2;

    // register tiles for accumulation
    rt_fl<WM, WN, ducks::rt_layout::col> c_reg;
    zero(c_reg);
    
    // Pre-load first iteration into buffer 0
    if (warp_id == 0) {
        load(As[0], g.a, {0, 0, row * 2, 0});
        load(As[1], g.a, {0, 0, row * 2 + 1, 0});
        load(Bs[0], g.b, {0, 0, 0, col * 2});
        load(Bs[1], g.b, {0, 0, 0, col * 2 + 1});
    }
    __syncthreads();
    
    // loop over K with double buffering
    for (int bkIdx = 0; bkIdx < K / BK; bkIdx++) {
        // Determine which buffer to use for current and next iterations
        int curr_buf = (bkIdx % 2) * 2;      // 0 or 2
        int next_buf = ((bkIdx + 1) % 2) * 2; // 2 or 0
        
        // Prefetch next iteration while computing current (if not last iteration)
        if (bkIdx + 1 < K / BK && warp_id == 0) {
            load(As[next_buf], g.a, {0, 0, row * 2, bkIdx + 1});
            load(As[next_buf + 1], g.a, {0, 0, row * 2 + 1, bkIdx + 1});
            load(Bs[next_buf], g.b, {0, 0, bkIdx + 1, col * 2});
            load(Bs[next_buf + 1], g.b, {0, 0, bkIdx + 1, col * 2 + 1});
        }

        // load from SMEM to registers for current iteration
        rt_bf<WM, BK> a_reg;
        rt_bf<BK, WN, ducks::rt_layout::col> b_reg;

        load(a_reg, As[curr_buf + warp_row]);
        load(b_reg, Bs[curr_buf + warp_col]);

        // accumulate
        mma_AB(c_reg, a_reg, b_reg, c_reg);
        __syncthreads();
    }

    // Store result back to HBM
    store(g.c, c_reg, {0, 0, row * 2 + warp_row, col * 2 + warp_col});
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