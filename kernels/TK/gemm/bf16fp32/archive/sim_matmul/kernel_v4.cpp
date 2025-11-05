#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

constexpr int BLOCK_SIZE = 128;  
constexpr int K_STEP     = 32;
constexpr int REG_BLOCK  = BLOCK_SIZE / 4; 

#define NUM_THREADS (kittens::WARPGROUP_THREADS)

#define M 8192
#define K 8192
#define N 8192

using _gl_A = gl<bf16, -1, -1, -1, -1>;
using _gl_B = gl<bf16, -1, -1, -1, -1>;
using _gl_C = gl<float, -1, -1, -1, -1>;

using G = kittens::group<4>;

struct micro_globals {
    _gl_A a;
    _gl_B b;
    _gl_C c;
    float alpha;
    float beta;

    dim3 grid()  { return dim3(N / BLOCK_SIZE, M / BLOCK_SIZE); } 
    dim3 block() { return dim3(NUM_THREADS); } 
    size_t dynamic_shared_memory() { return 16384*2; }
};

__global__ __launch_bounds__(NUM_THREADS, 2)
void micro_tk(const micro_globals g) {

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<BLOCK_SIZE, K_STEP> (&As) = al.allocate<st_bf<BLOCK_SIZE, K_STEP>>();
    st_bf<K_STEP, BLOCK_SIZE> (&Bs) = al.allocate<st_bf<K_STEP, BLOCK_SIZE>>();
    (void)__shm[0];

    rt_fl<REG_BLOCK, REG_BLOCK, ducks::rt_layout::col> C_accum[4];
    for (int i = 0; i < 4; i++) {
        zero(C_accum[i]);
    }

    rt_bf<REG_BLOCK, K_STEP> a_reg_0, a_reg_1;
    rt_bf<K_STEP, REG_BLOCK, ducks::rt_layout::col> b_reg_0, b_reg_1;


    int row = blockIdx.y;
    int col = blockIdx.x;

    int warp_id = kittens::warpid();
    int warp_row = warp_id / 2;
    int warp_col = warp_id % 2;

    int num_tiles = K / K_STEP;
    for (int tile = 0; tile < num_tiles; ++tile) {

        G::load(As, g.a, {0, 0, row, tile});
        G::load(Bs, g.b, {0, 0, tile, col});
        __syncthreads();

        // Row/Col identity of current thread
        int warp_id   = kittens::warpid();
        int warp_row  = warp_id / 2;
        int warp_col  = warp_id % 2;

        // Subtile A loads
        auto A0 = subtile_inplace<REG_BLOCK, K_STEP>(As, {0 * 2 + warp_row, 0});
        auto A1 = subtile_inplace<REG_BLOCK, K_STEP>(As, {1 * 2 + warp_row, 0});
        load(a_reg_0, A0);
        load(a_reg_1, A1);

        // Subtile B loads
        auto B0 = subtile_inplace<K_STEP, REG_BLOCK>(Bs, {0, 0 * 2 + warp_col});
        auto B1 = subtile_inplace<K_STEP, REG_BLOCK>(Bs, {0, 1 * 2 + warp_col});
        load(b_reg_0, B0);
        load(b_reg_1, B1);

        // Matrix multiplication
        mma_AB(C_accum[0], a_reg_0, b_reg_0, C_accum[0]);
        mma_AB(C_accum[1], a_reg_0, b_reg_1, C_accum[1]);
        mma_AB(C_accum[2], a_reg_1, b_reg_0, C_accum[2]);
        mma_AB(C_accum[3], a_reg_1, b_reg_1, C_accum[3]);

    }

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int offset_row = i / 2;
        int offset_col = i % 2;
        
        int global_row = row * 4 + offset_row * 2 + warp_row;  
        int global_col = col * 4 + offset_col * 2 + warp_col;
        store(g.c, C_accum[i], {0, 0, global_row, global_col});
    }
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
    py::bind_kernel<micro_tk>(m, "micro_tk", &micro_globals::a, &micro_globals::b, &micro_globals::c,
                              &micro_globals::alpha, &micro_globals::beta); 
    py::bind_function<dispatch_micro>(m, "dispatch_micro", &micro_globals::a, &micro_globals::b, &micro_globals::c,
                                      &micro_globals::alpha, &micro_globals::beta);
}


