#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include "utils.cpp"
using namespace kittens;

constexpr int BLOCK_SIZE       = 256;  
constexpr int K_STEP           = 512; 
constexpr int DOT_SLICE_SHARED = 64;
constexpr int REG_BLOCK        = BLOCK_SIZE / 8; 
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
    size_t dynamic_shared_memory() { return 65536; } // Increased for double buffering
};

__global__ __launch_bounds__(NUM_THREADS, 2)
void micro_tk(const micro_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<BLOCK_SIZE, DOT_SLICE_SHARED> (&As) = al.allocate<st_bf<BLOCK_SIZE, DOT_SLICE_SHARED>>();
    st_bf<BLOCK_SIZE, DOT_SLICE_SHARED> (&Bs) = al.allocate<st_bf<BLOCK_SIZE, DOT_SLICE_SHARED>>();

    rt_fl<REG_BLOCK, REG_BLOCK, ducks::rt_layout::col> C_accum[8];
    for (int i = 0; i < 8; i++) { zero(C_accum[i]); }

    // Reduced buffer size to minimize register pressure
    constexpr int BUFFER_SIZE = 4; 
    float4 a_buffer_next[BUFFER_SIZE];
    float4 b_buffer_next[BUFFER_SIZE];

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

    // Pre-load second tile into register buffers 
    if (num_tiles > 1) {
        load_global_to_registers<2, false, st_bf<BLOCK_SIZE, DOT_SLICE_SHARED>, _gl_A, coord<st_bf<BLOCK_SIZE, DOT_SLICE_SHARED>>, NUM_THREADS>(
            a_buffer_next, BUFFER_SIZE, g.a, {0, 0, row, 1}, As);
        load_global_to_registers<2, false, st_bf<BLOCK_SIZE, DOT_SLICE_SHARED>, _gl_B, coord<st_bf<BLOCK_SIZE, DOT_SLICE_SHARED>>, NUM_THREADS>(
            b_buffer_next, BUFFER_SIZE, g.b, {0, 0, col, 1}, Bs);
    }
    asm volatile("s_waitcnt vmcnt(0)");

    for (int tile = 0; tile < num_tiles; ++tile) {

        for (int shared_slice = 0; shared_slice < num_shared_slices; ++shared_slice) {
            const int next_k_offset = tile * num_shared_slices + shared_slice + 1;
            const bool should_load = (next_k_offset < num_tiles * num_shared_slices);

            // Start loading NEXT data to registers 
            if (should_load) {
                load_global_to_registers<2, false, st_bf<BLOCK_SIZE, DOT_SLICE_SHARED>, _gl_A, coord<st_bf<BLOCK_SIZE, DOT_SLICE_SHARED>>, NUM_THREADS>(
                    a_buffer_next, BUFFER_SIZE, g.a, {0, 0, row, next_k_offset}, As);
                
                load_global_to_registers<2, false, st_bf<BLOCK_SIZE, DOT_SLICE_SHARED>, _gl_B, coord<st_bf<BLOCK_SIZE, DOT_SLICE_SHARED>>, NUM_THREADS>(
                    b_buffer_next, BUFFER_SIZE, g.b, {0, 0, col, next_k_offset}, Bs);
            }

            rt_bf<REG_BLOCK, DOT_SLICE> a_reg_0, a_reg_1;
            rt_bf<REG_BLOCK, DOT_SLICE> b_reg_0, b_reg_1, b_reg_2, b_reg_3;
            
            // Compute on CURRENT data in shared memory with optimized MMA scheduling
            #pragma unroll 
            for (int slice = 0; slice < num_slices; ++slice) {
                load_async_shared_to_register(a_reg_0, subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, slice}));
                asm volatile("s_waitcnt lgkmcnt(0)\n");
                load_async_shared_to_register(a_reg_1, subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 4, slice}));
                asm volatile("s_waitcnt lgkmcnt(0)\n");

                load_async_shared_to_register(b_reg_0, subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col, slice}));
                mma_ABt(C_accum[0], a_reg_0, b_reg_0, C_accum[0]);
                asm volatile("s_waitcnt lgkmcnt(0)\n");

                load_async_shared_to_register(b_reg_1, subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col + 2, slice}));  
                mma_ABt(C_accum[4], a_reg_1, b_reg_0, C_accum[4]);   
                asm volatile("s_waitcnt lgkmcnt(0)\n");        
                
                load_async_shared_to_register(b_reg_2, subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col + 4, slice}));
                mma_ABt(C_accum[1], a_reg_0, b_reg_1, C_accum[1]);
                mma_ABt(C_accum[5], a_reg_1, b_reg_1, C_accum[5]);
                asm volatile("s_waitcnt lgkmcnt(0)\n");

                load_async_shared_to_register(b_reg_3, subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col + 6, slice}));
                mma_ABt(C_accum[2], a_reg_0, b_reg_2, C_accum[2]); 
                mma_ABt(C_accum[6], a_reg_1, b_reg_2, C_accum[6]);  
                asm volatile("s_waitcnt lgkmcnt(0)\n");
                              
                mma_ABt(C_accum[3], a_reg_0, b_reg_3, C_accum[3]);               
                mma_ABt(C_accum[7], a_reg_1, b_reg_3, C_accum[7]);                
            }
            asm volatile("s_waitcnt vmcnt(0)");
            __syncthreads();

            // Wait for loads and write to shared memory
            if (should_load) {
                store_registers_to_shared<st_bf<BLOCK_SIZE, DOT_SLICE_SHARED>, NUM_THREADS>(
                    a_buffer_next, As);
                store_registers_to_shared<st_bf<BLOCK_SIZE, DOT_SLICE_SHARED>, NUM_THREADS>(
                    b_buffer_next, Bs);
            }
        }
    }

    for (int i = 0; i < 8; ++i) {
        const int row_offset = (i < 4) ? 0 : 4;
        const int col_offset = (i % 4) * 2;
        store(g.c, C_accum[i], {
            0, 0,
            row * 8 + warp_row + row_offset,
            col * 8 + warp_col + col_offset
        });
    }
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