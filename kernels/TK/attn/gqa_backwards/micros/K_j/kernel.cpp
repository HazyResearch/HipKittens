#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

constexpr int ATTN_D = 128; // dimension
constexpr int BLOCK_SIZE_KV = 256; // block size for KV
constexpr int WARP_SIZE_KV = 32; // warp size for KV

template<int D, typename T=bf16, typename L=row_l, typename M=mfma_16x16x32> using kv_tile = rt<T, WARP_SIZE_KV, D, L, M>;
template<int D, typename T=bf16, typename L=row_l, typename M=mfma_16x16x32> using kv_tile_dq = rt<T, 256, 16, L, M>;

#define NUM_WARPS 8
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using G = kittens::group<NUM_WARPS>;

template<int D> struct micro_globals {
    gl<bf16, -1, -1, -1, -1> in;
    gl<bf16, -1, -1, -1, -1> out;
    dim3 grid()  { return dim3(1); } 
    dim3 block() { return dim3(NUM_THREADS); } 
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

template<int D> __global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const micro_globals<D> g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    // K_j_smem is organized in 16x16 tiles
    st_bf<BLOCK_SIZE_KV, D, ducks::st_layout::accumulator, ducks::st_matrix::mfma_16x16x32> (&K_j_smem) = al.allocate<st_bf<BLOCK_SIZE_KV, D, ducks::st_layout::accumulator, ducks::st_matrix::mfma_16x16x32>>();

    // Register tiles
    kv_tile<D, bf16, row_l, mfma_16x16x32> K_j;
    kv_tile_dq<D, bf16, col_l> K_j_col; // for dq

    const int warpid = kittens::warpid();

    // Load KV data using the KV head index
    G::load<1, false>(K_j_smem, g.in, {0, 0, 0, 0});
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Load K_j from SMEM to registers  
    // load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
    load(K_j_col, subtile_inplace<256, 16>(K_j_smem, {0, warpid}));
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Store K_j to output
    // store(g.out, K_j, {0, 0, warpid, 0});
    store(g.out, K_j_col, {0, 0, 0, warpid});
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);
}

template<int D>
void dispatch_micro(micro_globals<D> g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)micro_tk<D>, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    micro_tk<D><<<g.grid(), g.block(), mem_size>>>(g);
    hipDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module";
    py::bind_function<dispatch_micro<ATTN_D>>(m, "dispatch_micro", &micro_globals<ATTN_D>::in, &micro_globals<ATTN_D>::out);
}

