#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

constexpr int ATTN_B = 1; // batch size
constexpr int ATTN_H = 1; // number of heads
constexpr int ATTN_N = 1024; // sequence length
constexpr int ATTN_D = 128; // dimension
constexpr int BLOCK_SIZE_QO = 16; // block size for QO
constexpr int BLOCK_SIZE_KV = 256; // block size for KV
constexpr int WARP_SIZE_QO = 16; // warp size for QO
constexpr int WARP_SIZE_KV = 64; // warp size for KV

#define NUM_WARPS 4
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using namespace kittens;

template<int D, typename T=bf16, typename L=row_l, typename M=mfma_16x16x32> using qo_tile = rt<T, WARP_SIZE_QO, D, L, M>;
template<int D, typename T=bf16, typename L=row_l, typename M=mfma_16x16x32> using kv_tile = rt<T, WARP_SIZE_KV, D, L, M>;
template<int D, typename T=float, typename L=accum_col_l, typename M=mfma_16x16x32> using attn_tile = rt<T, WARP_SIZE_QO, WARP_SIZE_KV, L, M>;


template<int D> struct attn_prep_globals { 
    gl<bf16, -1, -1, -1, -1> Og;
    gl<float, -1, -1, -1, -1> dOg; 
    gl<float, -1, -1, -1, -1> delta;
    dim3 grid() { return dim3(ATTN_B, ATTN_H, ATTN_N / (WARP_SIZE_QO * NUM_WARPS)); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY - 32000; }
};

template<int D> __launch_bounds__(NUM_THREADS, 1)
__global__ void attend_prep_ker(const attn_prep_globals<D> g) {
    
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int seq_idx = blockIdx.z;

    const int warpid = kittens::warpid();

    qo_tile<D, float, row_l> dO, O;
    typename qo_tile<D, float, row_l>::col_vec delta_vec;

    load(dO, g.dOg, {batch_idx, head_idx, seq_idx * NUM_WARPS + warpid,0});
    load(O,  g.Og,  {batch_idx, head_idx, seq_idx * NUM_WARPS + warpid,0});
    
    // Δ_i = row_sum(dO ⊙ O) 
    mul(dO, dO, O);
    row_sum(delta_vec, dO); 
    store(g.delta, delta_vec, {batch_idx, head_idx, 0, seq_idx * NUM_WARPS + warpid});
}

template<int D>
void dispatch_prep(attn_prep_globals<D> g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)attend_prep_ker<D>, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    attend_prep_ker<D><<<g.grid(), g.block(), mem_size>>>(g);
    hipDeviceSynchronize();
}

template<int D> struct attn_bwd_combined_globals { 
    gl<bf16, -1, -1, -1, -1> Q, K, V, O, dS_ij;
    gl<float, -1, -1, -1, -1> dOg, dQg, dKg, dVg;
    gl<float, -1, -1, -1, -1> m_vec, l_vec, delta_vec;
    dim3 grid() { return dim3(ATTN_B, ATTN_H, ATTN_N / BLOCK_SIZE_KV); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY - 32000; }
};

template<int axis, ducks::rt::accumulator_col_layout RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void atomic_store(const GL &dst, const RT &src, const COORD &idx) { 
    using T = base_types::packing<typename RT::dtype>::unpacked_type;
    using U = typename GL::dtype;

    U *dst_ptr = (U*)&dst[(idx.template unit_coord<axis, 3>())];
    const int row_stride = dst.template stride<axis>();
    int laneid = kittens::laneid();

    const uint32_t buffer_size = row_stride * RT::rows * sizeof(U); 
    std::uintptr_t as_int = reinterpret_cast<std::uintptr_t>(dst_ptr);
    std::uint64_t  as_u64 = static_cast<std::uint64_t>(as_int);
    buffer_resource br = make_buffer_resource(as_u64, buffer_size, 0x00020000);

    int col_offset = laneid%(src.tile_size_col);
    int row_offset = laneid/(src.tile_size_col);

    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = src.tile_size_col*j + col_offset;
            int row = src.tile_size_row*i + row_offset * 4;
            const U val_0 = base_types::convertor<U, T>::convert(src.tiles[i][j].data[0].x);
            const U val_1 = base_types::convertor<U, T>::convert(src.tiles[i][j].data[0].y);
            const U val_2 = base_types::convertor<U, T>::convert(src.tiles[i][j].data[1].x);
            const U val_3 = base_types::convertor<U, T>::convert(src.tiles[i][j].data[1].y);

            uint32_t byte_offset_0 = static_cast<uint32_t>(((row + 0) * row_stride + col) * sizeof(U));
            uint32_t byte_offset_1 = static_cast<uint32_t>(((row + 1) * row_stride + col) * sizeof(U));
            uint32_t byte_offset_2 = static_cast<uint32_t>(((row + 2) * row_stride + col) * sizeof(U));
            uint32_t byte_offset_3 = static_cast<uint32_t>(((row + 3) * row_stride + col) * sizeof(U));

            uint32_t val_0_bits = *reinterpret_cast<const uint32_t*>(&val_0);
            uint32_t val_1_bits = *reinterpret_cast<const uint32_t*>(&val_1);
            uint32_t val_2_bits = *reinterpret_cast<const uint32_t*>(&val_2);
            uint32_t val_3_bits = *reinterpret_cast<const uint32_t*>(&val_3);

            asm volatile(
                "buffer_atomic_add_f32 %0, %1, %8, 0 offen\n"
                "buffer_atomic_add_f32 %2, %3, %8, 0 offen\n"
                "buffer_atomic_add_f32 %4, %5, %8, 0 offen\n"
                "buffer_atomic_add_f32 %6, %7, %8, 0 offen\n"
                "s_waitcnt vmcnt(0)\n"
                :
                : "v"(val_0_bits), "v"(byte_offset_0),      // %0, %1
                  "v"(val_1_bits), "v"(byte_offset_1),      // %2, %3
                  "v"(val_2_bits), "v"(byte_offset_2),      // %4, %5
                  "v"(val_3_bits), "v"(byte_offset_3),      // %6, %7
                  "s"(*(i32x4*)&br)                         // %8
                : "memory"
            );
        }
    }
}


template<int D> __launch_bounds__(NUM_THREADS, 1)
__global__ void attend_bwd_combined_ker(const attn_bwd_combined_globals<D> g) {
    
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int seq_idx = blockIdx.z;

    const int warpid = kittens::warpid();

    const float scale_factor = 1.0f / sqrt(D);

    // Register tiles
    kv_tile<D, bf16, row_l, mfma_16x16x32> K_j, V_j;
    kv_tile<D, float, accum_col_l, mfma_32x32x16> dK_j, dV_j;

    // 6. Load K_j and V_j from HBM to registers
    const int j = seq_idx * NUM_WARPS + warpid;
    load(K_j, g.K, {batch_idx, head_idx, j, 0});
    load(V_j, g.V, {batch_idx, head_idx, j, 0});
    
    // 7. Initialize dK_j = 0 and dV_j = 0
    zero(dK_j);
    zero(dV_j);

    // 8. for 1 <= i <= T_r (1024 / 32 = 32)
    for (int i = 0; i < ATTN_N / BLOCK_SIZE_QO; ++i) {

        // 9. Load Q_i, O_i, dO_i, dQ_i, l_i, m_i, delta_i from HBM to registers
        qo_tile<D, bf16, row_l, mfma_16x16x32> Q_i;
        qo_tile<D, bf16, col_l, mfma_32x32x16> dO_i;
        attn_tile<D, float, accum_col_l, mfma_16x16x32>::col_vec l_i, m_i, delta_i;
        load(Q_i, g.Q, {batch_idx, head_idx, i, 0});
        load(dO_i, g.dOg, {batch_idx, head_idx, i, 0});
        load(l_i, g.l_vec, {batch_idx, head_idx, 0, i});
        load(m_i, g.m_vec, {batch_idx, head_idx, 0, i});
        load(delta_i, g.delta_vec, {batch_idx, head_idx, 0, i});

        // 10. S_ij = Q_i K_j^T * scale
        attn_tile<D, float, accum_col_l, mfma_16x16x32> S_ij;
        zero(S_ij);
        mma_ABt(S_ij, Q_i, K_j, S_ij);
        mul(S_ij, S_ij, scale_factor);

        // 11. P_ij = exp(S_ij - m_i) / l_i
        sub_row(S_ij, S_ij, m_i);
        exp(S_ij, S_ij);
        div_row(S_ij, S_ij, l_i);

        // 12. dV_j += P_ij^T @ dO_i
        attn_tile<D, bf16, accum_col_l, mfma_16x16x32> P_ij_bf16_acc_col;
        copy(P_ij_bf16_acc_col, S_ij);
        attn_tile<D, bf16, col_l, mfma_32x32x16> P_ij_bf16_col;
        swap_layout(P_ij_bf16_col, P_ij_bf16_acc_col);
        mma_AtB(dV_j, P_ij_bf16_col, dO_i, dV_j);

        // // 13. dP_ij = dO_i @ V_j^T
        attn_tile<D, float, accum_col_l>dP_ij;
        zero(dP_ij);
        qo_tile<D, bf16, row_l, mfma_16x16x32> dO_i_row;
        load(dO_i_row, g.dOg, {batch_idx, head_idx, i, 0}); // TODO: replace with SMEM load
        mma_ABt(dP_ij, dO_i_row, V_j, dP_ij);

        // 14. dS_ij = P_ij o (dP_ij - delta_i)
        sub_row(dP_ij, dP_ij, delta_i);
        mul(dP_ij, dP_ij, S_ij);
        mul(dP_ij, dP_ij, scale_factor);
        store(g.dS_ij, dP_ij, {batch_idx,head_idx,i,j}); // TODO: replace with SMEM store
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();

        // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
        qo_tile<D, float, accum_col_l> dQ_i;
        zero(dQ_i);
        // load(dQ_i, g.dQg, {batch_idx,head_idx,i,0});
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        attn_tile<D, bf16, row_l> dS_ij_bf16_row;  
        load(dS_ij_bf16_row, g.dS_ij, {batch_idx,head_idx,i,j});
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        kv_tile<D, bf16, col_l> K_j_col;
        load(K_j_col, g.K, {batch_idx, head_idx, j, 0});  // TODO: replace with SMEM load
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        mma_AB(dQ_i, dS_ij_bf16_row, K_j_col, dQ_i);
        atomic_store<2>(g.dQg, dQ_i, {batch_idx,head_idx,i,0});
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();

        // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
        attn_tile<D,bf16,col_l, mfma_32x32x16> dS_ij_bf16_col; // TODO: replace with SMEM load
        load(dS_ij_bf16_col, g.dS_ij, {batch_idx,head_idx,i,j});
        qo_tile<D, bf16, col_l, mfma_32x32x16> Q_i_col; // TODO: replace with SMEM load
        load(Q_i_col, g.Q, {batch_idx, head_idx, i, 0});
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        mma_AtB(dK_j, dS_ij_bf16_col, Q_i_col, dK_j);
    }

    // 18. Write dK_j and dV_j back to HBM
    store(g.dKg, dK_j, {batch_idx, head_idx, seq_idx * NUM_WARPS + warpid, 0});
    store(g.dVg, dV_j, {batch_idx, head_idx, seq_idx * NUM_WARPS + warpid, 0});
}

template<int D>
void dispatch_bwd_combined(attn_bwd_combined_globals<D> g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)attend_bwd_combined_ker<D>, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    attend_bwd_combined_ker<D><<<g.grid(), g.block(), mem_size>>>(g);
    hipDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module";

    py::bind_function<dispatch_prep<ATTN_D>>(m, "dispatch_prep", 
        &attn_prep_globals<ATTN_D>::Og, 
        &attn_prep_globals<ATTN_D>::dOg,
        &attn_prep_globals<ATTN_D>::delta
    );

    py::bind_function<dispatch_bwd_combined<ATTN_D>>(m, "dispatch_bwd_combined", 
        &attn_bwd_combined_globals<ATTN_D>::Q, 
        &attn_bwd_combined_globals<ATTN_D>::K, 
        &attn_bwd_combined_globals<ATTN_D>::V, 
        &attn_bwd_combined_globals<ATTN_D>::O, 
        &attn_bwd_combined_globals<ATTN_D>::dS_ij,
        &attn_bwd_combined_globals<ATTN_D>::dOg, 
        &attn_bwd_combined_globals<ATTN_D>::dQg,
        &attn_bwd_combined_globals<ATTN_D>::dKg,
        &attn_bwd_combined_globals<ATTN_D>::dVg,
        &attn_bwd_combined_globals<ATTN_D>::m_vec, 
        &attn_bwd_combined_globals<ATTN_D>::l_vec,
        &attn_bwd_combined_globals<ATTN_D>::delta_vec
    );
}