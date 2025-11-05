#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

constexpr int ATTN_N = 1024; // sequence length
constexpr int ATTN_D = 128; // dimension
constexpr int BLOCK_SIZE_QO = 16; // block size for QO
constexpr int BLOCK_SIZE_KV = 256; // block size for KV
constexpr int WARP_SIZE_QO = 16; // warp size for QO
constexpr int WARP_SIZE_KV = 32; // warp size for KV

template<typename T=bf16, typename L=row_l, typename M=mfma_16x16x32> using kv_tile_dq = rt<T, 32, 16, L, M>;
template<int D, typename T=bf16, typename L=col_l, typename M=mfma_16x16x32> using attn_tile_T = rt<T, WARP_SIZE_KV, WARP_SIZE_QO, L, M>;
template<int D, typename T=bf16, typename L=col_l, typename M=mfma_16x16x32> using attn_tile_T_dq = rt<T, 32, 16, L, M>;

#define NUM_WARPS 1
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using G = kittens::group<NUM_WARPS>;

struct micro_globals {
    gl<bf16, -1, -1, -1, -1> in;
    gl<bf16, -1, -1, -1, -1> out;
    dim3 grid()  { return dim3(1, 1, 1); } 
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
    st_bf<32, BLOCK_SIZE_QO, ducks::st_layout::classical, ducks::st_matrix::mfma_32x32x16> (&attn_i_smem) = al.allocate<st_bf<32, BLOCK_SIZE_QO, ducks::st_layout::classical, ducks::st_matrix::mfma_32x32x16>>();

    // load to registers
    attn_tile_T<ATTN_D, bf16, accum_row_l> dP_ij_bf16_accum_row;
    load(dP_ij_bf16_accum_row, g.in, {0, 0, 0, 0});
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // store to shared
    store(attn_i_smem, dP_ij_bf16_accum_row);
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // load to registers
    attn_tile_T_dq<ATTN_D, bf16, col_l> dP_ij_bf16_col_T; // for dq
    load(dP_ij_bf16_col_T, attn_i_smem);
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // store output
    store(g.out, dP_ij_bf16_col_T, {0, 0, 0, 0});
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

