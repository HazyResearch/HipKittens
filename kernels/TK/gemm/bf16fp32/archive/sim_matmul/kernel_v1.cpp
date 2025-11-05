#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

constexpr int BLOCK_SIZE = 64;   // Back to working size
constexpr int K_STEP     = 64;
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

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const micro_globals g) {

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<BLOCK_SIZE, K_STEP> (&As) = al.allocate<st_bf<BLOCK_SIZE, K_STEP>>();
    st_bf<K_STEP, BLOCK_SIZE> (&Bs) = al.allocate<st_bf<K_STEP, BLOCK_SIZE>>();

    rt_fl<REG_BLOCK, REG_BLOCK, ducks::rt_layout::col> C_accum[4];
    // Initialize accumulators to zero
    for (int i = 0; i < 4; i++) {
        zero(C_accum[i]);
    }
    rt_bf<REG_BLOCK, REG_BLOCK> a_reg;
    rt_bf<REG_BLOCK, REG_BLOCK, ducks::rt_layout::col> b_reg;


    int row = blockIdx.y;
    int col = blockIdx.x;

    int warp_id = kittens::warpid();

    int warp_row = warp_id / 2;  // Keep original logic
    int warp_col = warp_id % 2;  // Keep original logic

    int num_tiles = K / K_STEP;
    for (int tile = 0; tile < num_tiles; ++tile) {

        G::load(As, g.a, {0, 0, row, tile});
        G::load(Bs, g.b, {0, 0, tile, col});
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < 4; i++) {

            int offset_row = i / 2;  // 0,0,1,1
            int offset_col = i % 2;  // 0,1,0,1

            int subtile_row = offset_row * 2 + warp_row;  
            int subtile_col = offset_col * 2 + warp_col;

            #pragma unroll 4
            for (int j = 0; j < 4; j++) {

                auto st_sub_a = subtile_inplace<REG_BLOCK, REG_BLOCK>(As, {subtile_row, j});
                auto st_sub_b = subtile_inplace<REG_BLOCK, REG_BLOCK>(Bs, {j, subtile_col});
                load(a_reg, st_sub_a);
                load(b_reg, st_sub_b);

                mma_AB(C_accum[i], a_reg, b_reg, C_accum[i]);
            }
        }
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


