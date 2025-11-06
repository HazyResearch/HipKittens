#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include "utils.cpp"
using namespace kittens;

constexpr int BLOCK_SIZE       = 256;  
constexpr int K_STEP           = 64;
constexpr int SUB_K_STEP       = 32;
constexpr int REG_BLOCK        = BLOCK_SIZE / 8;
constexpr int DOT_SLICE        = 16;

#define NUM_WARPS 8
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)
#define NUM_THREADS_IN_GROUP NUM_THREADS / 2

#define M 8192
#define K 8192
#define N 8192

using _gl_A = gl<bf16, -1, -1, -1, -1>;
using _gl_B = gl<bf16, -1, -1, -1, -1>;
using _gl_C = gl<float, -1, -1, -1, -1>;

using G = kittens::group<NUM_WARPS>;

struct micro_globals {
    _gl_A a;
    _gl_B b;
    _gl_C c;
    dim3 grid()  { return dim3(N / BLOCK_SIZE, M / BLOCK_SIZE); } 
    dim3 block() { return dim3(NUM_THREADS); } 
    size_t dynamic_shared_memory() { return 65536; }
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const micro_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<BLOCK_SIZE, K_STEP> (&As) = al.allocate<st_bf<BLOCK_SIZE, K_STEP>>();
    st_bf<BLOCK_SIZE, K_STEP> (&Bs) = al.allocate<st_bf<BLOCK_SIZE, K_STEP>>();

    rt_bf<REG_BLOCK, DOT_SLICE> a_reg_0, a_reg_1, a_reg_2, a_reg_3;
    rt_bf<REG_BLOCK, DOT_SLICE> b_reg_0, b_reg_1;
    rt_fl<REG_BLOCK, REG_BLOCK, ducks::rt_layout::col> C_accum[8];
    for (int i = 0; i < 8; i++) { zero(C_accum[i]); }

    // Small register buffers for pipelining
    constexpr int BUFFER_SIZE = 128;
    float4 a_buffer_next[BUFFER_SIZE];
    float4 b_buffer_next[BUFFER_SIZE];

    const int row = blockIdx.y;
    const int col = blockIdx.x;

    const int warp_id = kittens::warpid();
    const int warp_row = warp_id / 4;
    const int warp_col = warp_id % 4;

    const int num_tiles = K / K_STEP;
    const int num_slices = K_STEP / DOT_SLICE;

    auto Ast = subtile_inplace<BLOCK_SIZE, SUB_K_STEP>(As, {0, warp_row});
    auto Bst = subtile_inplace<BLOCK_SIZE, SUB_K_STEP>(Bs, {0, warp_row});

    // Load first tile into shared memory
    G::load(As, g.a, {0, 0, row, 0});
    G::load(Bs, g.b, {0, 0, col, 0});
    __builtin_amdgcn_s_barrier();

    if (warp_row == 1) {
        __builtin_amdgcn_s_barrier();
    }

    for (int tile = 0; tile < num_tiles; ++tile) {
        const bool loading = tile + 1 < num_tiles;
        // Start loading NEXT data to registers

        if (loading) {
            load_global_to_registers<2, false, st_subtile<st_bf<BLOCK_SIZE, K_STEP>, BLOCK_SIZE, SUB_K_STEP>, _gl_A, coord<st_subtile<st_bf<BLOCK_SIZE, K_STEP>, BLOCK_SIZE, SUB_K_STEP>>, NUM_THREADS_IN_GROUP>(
                a_buffer_next, BUFFER_SIZE, g.a, {0, 0, row, (tile + 1) * 2 + warp_row}, Ast);
        }

        // Compute on CURRENT data in shared memory
        // #pragma unroll 
        for (int slice = 0; slice < num_slices; ++slice) {
            // if (loading) {
            //     if (slice == 0) {
            //         load_global_to_registers<2, false, st_subtile<st_bf<BLOCK_SIZE, K_STEP>, BLOCK_SIZE, SUB_K_STEP>, _gl_A, coord<st_subtile<st_bf<BLOCK_SIZE, K_STEP>, BLOCK_SIZE, SUB_K_STEP>>, NUM_THREADS_IN_GROUP>(
            //             a_buffer_next, BUFFER_SIZE, g.a, {0, 0, row, (tile + 1) * 2 + warp_row}, Ast);
            //     } else if (slice == 1) {
            //         load_global_to_registers<2, false, st_subtile<st_bf<BLOCK_SIZE, K_STEP>, BLOCK_SIZE, SUB_K_STEP>, _gl_B, coord<st_subtile<st_bf<BLOCK_SIZE, K_STEP>, BLOCK_SIZE, SUB_K_STEP>>, NUM_THREADS_IN_GROUP>(
            //             b_buffer_next, BUFFER_SIZE, g.b, {0, 0, col, (tile + 1) * 2 + warp_row}, Bst);
            //     }
            // }

            if (loading && slice == 2) {
                load_global_to_registers<2, false, st_subtile<st_bf<BLOCK_SIZE, K_STEP>, BLOCK_SIZE, SUB_K_STEP>, _gl_B, coord<st_subtile<st_bf<BLOCK_SIZE, K_STEP>, BLOCK_SIZE, SUB_K_STEP>>, NUM_THREADS_IN_GROUP>(
                    b_buffer_next, BUFFER_SIZE, g.b, {0, 0, col, (tile + 1) * 2 + warp_row}, Bst);
            }
            __builtin_amdgcn_sched_barrier(0);

            load_async_shared_to_register(b_reg_0, subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col, slice}));
            asm volatile("s_waitcnt lgkmcnt(0)\n");
            load_async_shared_to_register(b_reg_1, subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col + 4, slice}));
            asm volatile("s_waitcnt lgkmcnt(0)\n");
            load_async_shared_to_register(a_reg_0, subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, slice}));
            asm volatile("s_waitcnt lgkmcnt(0)\n");
            load_async_shared_to_register(a_reg_1, subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, slice}));
            asm volatile("s_waitcnt lgkmcnt(0)\n");
            load_async_shared_to_register(a_reg_2, subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 4, slice}));
            asm volatile("s_waitcnt lgkmcnt(0)\n");
            load_async_shared_to_register(a_reg_3, subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 6, slice}));
            asm volatile("s_waitcnt lgkmcnt(0)\n");

            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            __builtin_amdgcn_s_setprio(1);
            mma_ABt(C_accum[0], a_reg_0, b_reg_0, C_accum[0]);
            mma_ABt(C_accum[1], a_reg_0, b_reg_1, C_accum[1]);
            mma_ABt(C_accum[2], a_reg_1, b_reg_0, C_accum[2]);
            mma_ABt(C_accum[3], a_reg_1, b_reg_1, C_accum[3]);
            mma_ABt(C_accum[4], a_reg_2, b_reg_0, C_accum[4]);
            mma_ABt(C_accum[5], a_reg_2, b_reg_1, C_accum[5]);
            mma_ABt(C_accum[6], a_reg_3, b_reg_0, C_accum[6]);
            mma_ABt(C_accum[7], a_reg_3, b_reg_1, C_accum[7]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);
        }

        // Now wait for loads and write to shared memory
        if (loading) {
            asm volatile("s_waitcnt vmcnt(0)");
            store_registers_to_shared<st_subtile<st_bf<BLOCK_SIZE, K_STEP>, BLOCK_SIZE, SUB_K_STEP>, NUM_THREADS_IN_GROUP>(
                a_buffer_next, Ast);
            store_registers_to_shared<st_subtile<st_bf<BLOCK_SIZE, K_STEP>, BLOCK_SIZE, SUB_K_STEP>, NUM_THREADS_IN_GROUP>(
                b_buffer_next, Bst);
        }
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

    }

    if (warp_row == 0) {
        __builtin_amdgcn_s_barrier();
    }

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row_offset = (i / 2) * 2;
        const int col_offset = (i % 2) * 4;
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