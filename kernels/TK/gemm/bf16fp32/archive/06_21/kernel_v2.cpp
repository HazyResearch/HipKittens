#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include "utils.cpp"
using namespace kittens;

constexpr int BLOCK_SIZE       = 128;  
constexpr int K_STEP           = 64;
constexpr int DOT_SLICE_SHARED = 64;
constexpr int REG_BLOCK        = BLOCK_SIZE / 4; 
constexpr int DOT_SLICE        = 16;

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
    size_t dynamic_shared_memory() { return 32768; }
};

__global__ __launch_bounds__(NUM_THREADS, 2)
void micro_tk(const micro_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<BLOCK_SIZE, DOT_SLICE_SHARED> (&As) = al.allocate<st_bf<BLOCK_SIZE, DOT_SLICE_SHARED>>();
    st_bf<BLOCK_SIZE, DOT_SLICE_SHARED> (&Bs) = al.allocate<st_bf<BLOCK_SIZE, DOT_SLICE_SHARED>>();

    rt_bf<REG_BLOCK, DOT_SLICE> a_reg_0, b_reg;
    rt_fl<REG_BLOCK, REG_BLOCK, ducks::rt_layout::col> C_accum[2];
    for (int i = 0; i < 2; i++) { zero(C_accum[i]); }

    // Small register buffers for pipelining
    constexpr int BUFFER_SIZE = 64;
    float4 a_buffer_next[BUFFER_SIZE];
    float4 b_buffer_next[BUFFER_SIZE];
    int a_metadata[3], b_metadata[3];

    const int row = blockIdx.y;
    const int col = blockIdx.x;

    const int warp_id = kittens::warpid();
    const int warp_row = warp_id / 2;
    const int warp_col = warp_id % 2;

    const int num_tiles = K / K_STEP;
    const int num_shared_slices = K_STEP / DOT_SLICE_SHARED;
    const int num_slices = DOT_SLICE_SHARED / DOT_SLICE;

    // Load first tile into shared memory
    G::load(As, g.a, {0, 0, row, 0});
    G::load(Bs, g.b, {0, 0, col, 0});
    __syncthreads();

    // Pre-load second tile into register bufers 
    int next_k_offset = (num_shared_slices > 1) ? 1 : num_shared_slices;
    load_global_to_registers<2, false, st_bf<BLOCK_SIZE, DOT_SLICE_SHARED>, _gl_A, coord<st_bf<BLOCK_SIZE, DOT_SLICE_SHARED>>, NUM_THREADS>(
        a_buffer_next, BUFFER_SIZE, g.a, {0, 0, row, next_k_offset}, As, a_metadata);
    load_global_to_registers<2, false, st_bf<BLOCK_SIZE, DOT_SLICE_SHARED>, _gl_B, coord<st_bf<BLOCK_SIZE, DOT_SLICE_SHARED>>, NUM_THREADS>(
        b_buffer_next, BUFFER_SIZE, g.b, {0, 0, col, next_k_offset}, Bs, b_metadata);
    
    if (warp_col / 8 == 0) {
        __builtin_amdgcn_s_barrier();
    }

    for (int tile = 0; tile < num_tiles; ++tile) {

        for (int shared_slice = 0; shared_slice < num_shared_slices; ++shared_slice) {
            const int current_k_offset = tile * num_shared_slices + shared_slice;
            const int next_k_offset = (shared_slice + 1 < num_shared_slices) ? 
                                      tile * num_shared_slices + shared_slice + 1 :
                                      (tile + 1 < num_tiles) ? (tile + 1) * num_shared_slices : -1;

            // Start loading NEXT data to registers 
            if (next_k_offset >= 0 && next_k_offset < num_tiles * num_shared_slices) {
                load_global_to_registers<2, false, st_bf<BLOCK_SIZE, DOT_SLICE_SHARED>, _gl_A, coord<st_bf<BLOCK_SIZE, DOT_SLICE_SHARED>>, NUM_THREADS>(
                    a_buffer_next, BUFFER_SIZE, g.a, {0, 0, row, next_k_offset}, As, a_metadata);
                
                load_global_to_registers<2, false, st_bf<BLOCK_SIZE, DOT_SLICE_SHARED>, _gl_B, coord<st_bf<BLOCK_SIZE, DOT_SLICE_SHARED>>, NUM_THREADS>(
                    b_buffer_next, BUFFER_SIZE, g.b, {0, 0, col, next_k_offset}, Bs, b_metadata);
            }
            __builtin_amdgcn_s_barrier();
            

            // Compute on CURRENT data in shared memory (while loads happen in background)
            for (int slice = 0; slice < num_slices; ++slice) {
                kittens::load(a_reg_0, subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, slice}));
                kittens::load(b_reg, subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col, slice}));
                mma_ABt(C_accum[0], a_reg_0, b_reg, C_accum[0]);
                kittens::load(b_reg, subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col + 2, slice}));
                mma_ABt(C_accum[1], a_reg_0, b_reg, C_accum[1]);
                __builtin_amdgcn_s_barrier();
            }

            // Now wait for loads and write to shared memory
            if (next_k_offset >= 0 && next_k_offset < num_tiles * num_shared_slices) {
                store_registers_to_shared<st_bf<BLOCK_SIZE, DOT_SLICE_SHARED>, NUM_THREADS>(
                    a_buffer_next, As, a_metadata);
                store_registers_to_shared<st_bf<BLOCK_SIZE, DOT_SLICE_SHARED>, NUM_THREADS>(
                    b_buffer_next, Bs, b_metadata);
            }
            // __builtin_amdgcn_s_barrier();
        }
    }

    if (warp_col / 8 == 1) {
        __builtin_amdgcn_s_barrier();
    }

    store(g.c, C_accum[0], {0, 0, row * 4 + warp_row, col * 4 + warp_col});
    store(g.c, C_accum[1], {0, 0, row * 4 + warp_row, col * 4 + warp_col + 2});
}

void dispatch_micro(micro_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)micro_tk, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    micro_tk<<<g.grid(), g.block(), mem_size>>>(g);
    hipDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module";
    py::bind_kernel<micro_tk>(m, "micro_tk", &micro_globals::a, &micro_globals::b, &micro_globals::c); 
    py::bind_function<dispatch_micro>(m, "dispatch_micro", &micro_globals::a, &micro_globals::b, &micro_globals::c);
}