#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

#define NUM_THREADS (kittens::WARPGROUP_THREADS)

#define M 8192
#define K 8192
#define N 8192

#define BM 16
#define BN 16
#define BK 16

using _gl_A = gl<bf16, -1, -1, -1, -1>;
using _gl_B = gl<bf16, -1, -1, -1, -1>;
using _gl_C = gl<float, -1, -1, -1, -1>;

struct micro_globals {
    _gl_A a;
    _gl_B b;
    _gl_C c;

    // grid - number of thread blocks we are launching
    dim3 grid()  { return dim3(256, 256); } 

    // block - number of threads in a thread block
    dim3 block() { return dim3(NUM_THREADS); } 

    // Safe shared memory size for MI325X
    size_t dynamic_shared_memory() { return 32768; } 
};

__global__
void micro_tk(const micro_globals g) {

    // // shared memory
    // extern __shared__ alignment_dummy __shm[];
    // shared_allocator al((int*)&__shm[0]);
    // st_bf<BM, BK> (&As) = al.allocate<st_bf<BM, BK>>();
    // st_bf<BK, BN> (&Bs) = al.allocate<st_bf<BK, BN>>();

    int row = blockIdx.x;
    int col = blockIdx.y;

    int warp_id = warpid();
    int warp_row = warp_id / 2;
    int warp_col = warp_id % 2;

    // // register tiles for accumulation
    rt_fl<BM, BN, ducks::rt_layout::col> c_reg;
    zero(c_reg);
    
    // loop over K
    for (int bkIdx = 0; bkIdx < K / BK; bkIdx++) {
        // // load from HBM to SMEM
        // load(As, g.a, {0, 0, C_row, bkIdx});
        // load(Bs, g.b, {0, 0, bkIdx, C_col});
        // __syncthreads();

        // load from HBM to registers
        rt_bf<BM, BK> a_reg;
        rt_bf<BK, BN> b_reg;

        load(a_reg, g.a, {0, 0, warp_row + row * 2, bkIdx});
        load(b_reg, g.b, {0, 0, warp_col + col * 2, bkIdx});

        // accumulate
        mma_ABt(c_reg, a_reg, b_reg, c_reg);
        __syncthreads();
    }

    // Store result back to HBM
    store(g.c, c_reg, {0, 0, warp_row + row * 2, warp_col + col * 2});
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
    py::bind_kernel<micro_tk>(m, "micro_tk", &micro_globals::a, &micro_globals::b, &micro_globals::c); 
    py::bind_function<dispatch_micro>(m, "dispatch_micro", &micro_globals::a, &micro_globals::b, &micro_globals::c);
}