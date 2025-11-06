#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

constexpr int BLOCK_SIZE = 256;  
constexpr int K_STEP     = 256;
constexpr int DOT_SLICE_SHARED = K_STEP / 4;
constexpr int REG_BLOCK  = BLOCK_SIZE / 4; 
constexpr int DOT_SLICE  = 16;


#define NUM_WARPS 8
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

#define M 8192
#define K 8192
#define N 8192

using _gl_A = gl<bf16, -1, -1, -1, -1>;
using _gl_B = gl<bf16, -1, -1, -1, -1>;
using _gl_C = gl<float, -1, -1, -1, -1>;

using G = kittens::group<8>;

struct micro_globals {
    _gl_A a;
    _gl_B b;
    _gl_C c;
    dim3 grid()  { return dim3(N / BLOCK_SIZE, M / BLOCK_SIZE); } 
    dim3 block() { return dim3(NUM_THREADS); } 
    size_t dynamic_shared_memory() { return 65356; }
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const micro_globals g) {

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<BLOCK_SIZE, DOT_SLICE_SHARED> (&As) = al.allocate<st_bf<BLOCK_SIZE, DOT_SLICE_SHARED>>();
    st_bf<BLOCK_SIZE, DOT_SLICE_SHARED> (&Bs) = al.allocate<st_bf<BLOCK_SIZE, DOT_SLICE_SHARED>>();

    rt_bf<REG_BLOCK, DOT_SLICE> a_reg_0, b_reg_0, b_reg_1;
    rt_fl<REG_BLOCK, REG_BLOCK, ducks::rt_layout::col> C_accum[2];
    for (int i = 0; i < 2; i++) { zero(C_accum[i]); }

    const int row = blockIdx.y;
    const int col = blockIdx.x;

    const int warp_id = kittens::warpid();
    const int warp_row = warp_id / 2;
    const int warp_col = warp_id % 2;

    const int num_tiles = K / K_STEP;
    const int num_shared_slices = K_STEP / DOT_SLICE_SHARED;
    const int num_slices = DOT_SLICE_SHARED / DOT_SLICE;
    #pragma unroll
    for (int tile = 0; tile < num_tiles; ++tile) {

        #pragma unroll
        for (int shared_slice = 0; shared_slice < num_shared_slices; ++shared_slice) {
            const int k_offset = tile * num_shared_slices + shared_slice;
            G::load(As, g.a, {0, 0, row, k_offset});
            G::load(Bs, g.b, {0, 0, col, k_offset});
            __syncthreads();

            #pragma unroll // important to unroll this loop
            for (int slice = 0; slice < num_slices; ++slice) {

                load(a_reg_0, subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, slice}));
                load(b_reg_0, subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col, slice}));
                mma_ABt(C_accum[0], a_reg_0, b_reg_0, C_accum[0]);
                load(b_reg_1, subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col + 2, slice}));
                mma_ABt(C_accum[1], a_reg_0, b_reg_1, C_accum[1]);

            }
        }
    }

    store(g.c, C_accum[0], {0, 0, row * 4 + warp_row, col * 4 + warp_col});
    store(g.c, C_accum[1], {0, 0, row * 4 + warp_row, col * 4 + warp_col + 2});
}

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