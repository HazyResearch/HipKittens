#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include "utils.cpp"
using namespace kittens;

constexpr int BLOCK_SIZE       = 256;  
constexpr int K_STEP           = 64;
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

using G = kittens::group<NUM_WARPS>;

struct micro_globals {
    _gl_A a;
    _gl_B b;
    _gl_C c;
    dim3 grid()  { return dim3(N / BLOCK_SIZE, M / BLOCK_SIZE); } 
    dim3 block() { return dim3(NUM_THREADS); } 
    size_t dynamic_shared_memory() { return 65536; }
};

__global__ __launch_bounds__(NUM_THREADS, 2)
void micro_tk(const micro_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<BLOCK_SIZE, K_STEP> (&As) = al.allocate<st_bf<BLOCK_SIZE, K_STEP>>();
    st_bf<BLOCK_SIZE, K_STEP> (&Bs) = al.allocate<st_bf<BLOCK_SIZE, K_STEP>>();

    rt_bf<REG_BLOCK, DOT_SLICE> tiles[10];
    rt_fl<REG_BLOCK, REG_BLOCK, ducks::rt_layout::col> C_accum[2];
    for (int i = 0; i < 2; i++) { zero(C_accum[i]); }

    const int row = blockIdx.y;
    const int col = blockIdx.x;

    const int warp_id = kittens::warpid();
    const int warp_row = warp_id / 4;
    const int warp_col = warp_id % 4;

    const int num_tiles = K / K_STEP;

    auto a_tile_0 = subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, warp_col});
    auto a_tile_1 = subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, warp_col});
    auto b_tile_0 = subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_row, warp_col});
    auto b_tile_1 = subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_row + 2, warp_col});

    // Load first tile into shared memory
    G::load(As, g.a, {0, 0, row, 0});
    G::load(Bs, g.b, {0, 0, col, 0});
    __builtin_amdgcn_s_barrier();

    if (warp_row == 1) {
        __builtin_amdgcn_s_barrier();
    }

    for (int tile = 0; tile < num_tiles; ++tile) {
        const bool loading = tile + 1 < num_tiles;

        // Cluster 0
        __builtin_amdgcn_sched_barrier(0);
        if (loading) {
            load(tiles[0], g.a, {0, 0, row * 4 + warp_row, (tile + 1) * 4 + warp_col});
            load(tiles[1], g.a, {0, 0, row * 4 + warp_row + 2, (tile + 1) * 4 + warp_col});
        }
        __builtin_amdgcn_sched_barrier(0);

        load(tiles[2], subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col, 0}));
        load(tiles[3], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 0}));
        load(tiles[4], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 0}));
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 1
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0], tiles[3], tiles[2], C_accum[0]);
        mma_ABt(C_accum[1], tiles[4], tiles[2], C_accum[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 2
        if (loading) {
            load(tiles[2], g.b, {0, 0, col * 4 + warp_row, (tile + 1) * 4 + warp_col});
            load(tiles[3], g.b, {0, 0, col * 4 + warp_row + 2, (tile + 1) * 4 + warp_col});
        }
        __builtin_amdgcn_sched_barrier(0);

        load(tiles[4], subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col, 1}));
        load(tiles[5], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 1}));
        load(tiles[6], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 1}));
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 3
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0], tiles[5], tiles[4], C_accum[0]);
        mma_ABt(C_accum[1], tiles[6], tiles[4], C_accum[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 4
        load(tiles[4], subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col, 2}));
        load(tiles[5], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 2}));
        load(tiles[6], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 2}));
        load(tiles[7], subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col, 3}));
        load(tiles[8], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 3}));
        load(tiles[9], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 3}));
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 5
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0], tiles[5], tiles[4], C_accum[0]);
        mma_ABt(C_accum[1], tiles[6], tiles[4], C_accum[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 6
        if (loading) {
            asm volatile("s_waitcnt vmcnt(0)");
            store(a_tile_0, tiles[0]);
            store(a_tile_1, tiles[1]);
            store(b_tile_0, tiles[2]);
            store(b_tile_1, tiles[3]);
        }
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 7
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0], tiles[8], tiles[7], C_accum[0]);
        mma_ABt(C_accum[1], tiles[9], tiles[7], C_accum[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    }

    if (warp_row == 0) {
        __builtin_amdgcn_s_barrier();
    }

    store(g.c, C_accum[0], {0, 0, row * 4 + warp_row, col * 4 + warp_col});
    store(g.c, C_accum[1], {0, 0, row * 4 + warp_row + 2, col * 4 + warp_col});
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