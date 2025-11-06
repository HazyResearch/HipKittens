#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

constexpr int ATTN_N = 1024; // sequence length
constexpr int ATTN_D = 128; // dimension
constexpr int BLOCK_SIZE_QO = 16; // block size for QO
constexpr int BLOCK_SIZE_KV = 256; // block size for KV
constexpr int WARP_SIZE_QO = 16; // warp size for QO
constexpr int WARP_SIZE_KV = 32; // warp size for KV

template<typename T=bf16, typename L=row_l, typename M=mfma_16x16x32> using kv_tile_dq = rt<T, 256, 16, L, M>;

#define NUM_WARPS 1
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using G = kittens::group<NUM_WARPS>;

struct micro_globals {
    gl<bf16, -1, -1, -1, -1> in;
    gl<bf16, -1, -1, -1, -1> out;
    dim3 grid()  { return dim3(1, 1, ATTN_N / BLOCK_SIZE_KV); } 
    dim3 block() { return dim3(NUM_THREADS); } 
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY-2048; }
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const micro_globals g) {

    const int seq_idx = blockIdx.z;
    const int warpid = kittens::warpid();

    // load to shared
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<256, 128, ducks::st_layout::accumulator, ducks::st_matrix::mfma_16x16x32> (&K_j_smem) = al.allocate<st_bf<256, 128, ducks::st_layout::accumulator, ducks::st_matrix::mfma_16x16x32>>();

    // load to shared
    load(K_j_smem, g.in, {0, 0, seq_idx, 0});
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    kv_tile_dq<bf16, col_l, mfma_16x16x32> K_j;

    load(K_j, subtile_inplace<256, 16>(K_j_smem, {0, warpid}));
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // store output
    store(g.out, K_j, {0, 0, seq_idx, warpid});
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
}

void dispatch_micro(micro_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)micro_tk, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    micro_tk<<<g.grid(), g.block(), mem_size>>>(g);
    hipDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module";
    py::bind_function<dispatch_micro>(m, "dispatch_micro", &micro_globals::in, &micro_globals::out);
}

