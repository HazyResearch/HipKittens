#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

constexpr int ATTN_D = 128; // dimension
constexpr int BLOCK_SIZE_KV = 256; // block size for KV
constexpr int DOT_SLICE_QO = 16;
constexpr int WARP_SIZE_KV = 32; // warp size for KV

template<int D, typename T=bf16, typename L=col_l, typename M=mfma_16x16x32> using attn_tile_T = rt<T, WARP_SIZE_KV, DOT_SLICE_QO, L, M>;
template<int D, typename T=bf16, typename L=col_l, typename M=mfma_16x16x32> using attn_tile_T_dq = rt<T, BLOCK_SIZE_KV, DOT_SLICE_QO, L, M>;

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

    // We parameterize this using mfma_32x32x16 because we want the base tile for it to be 32x16. Not that it uses that intrinsic.
    st_bf<BLOCK_SIZE_KV, DOT_SLICE_QO, ducks::st_layout::classical, ducks::st_matrix::mfma_32x32x16> (&attn_i_smem) = al.allocate<st_bf<BLOCK_SIZE_KV, DOT_SLICE_QO, ducks::st_layout::classical, ducks::st_matrix::mfma_32x32x16>>();

    // Register tiles
    attn_tile_T<D, bf16, accum_row_l> dP_ij_bf16_accum_row;
    attn_tile_T_dq<D, bf16, col_l> dP_ij_bf16_col_T; // for dq

    const int warpid = kittens::warpid();

    // Load dP_ij_bf16_accum_row from global to registers
    load(dP_ij_bf16_accum_row, g.in, {0, 0, warpid, 0});
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Write dP_ij_bf16_accum_row to SMEM
    auto attn_i_smem_subtile = subtile_inplace<WARP_SIZE_KV, DOT_SLICE_QO>(attn_i_smem, {warpid, 0});
    store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Load dP_ij_bf16_col_T from SMEM to registers
    load(dP_ij_bf16_col_T, attn_i_smem);
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Write dP_ij_bf16_col_T to output
    store(g.out, dP_ij_bf16_col_T, {0, 0, 0, 0});
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

