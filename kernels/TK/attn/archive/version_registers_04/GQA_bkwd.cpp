#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

constexpr int ATTN_B = 16; // batch size
constexpr int ATTN_H = 16; // number of heads
constexpr int ATTN_N = 1024; // sequence length
constexpr int ATTN_D = 128; // dimension
constexpr int BLOCK_SIZE = 32; // block size

#define NUM_WARPS 1
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using namespace kittens;

template<int D, typename T=bf16, typename L=row_l> using qkvo_tile = rt<T, BLOCK_SIZE, D, L>;
template<int D, typename T=bf16, typename L=col_l> using qkvo_tile_transposed = rt<T, D, BLOCK_SIZE, L>;
template<int D, typename T=float, typename L=row_l> using attn_tile = rt<T, BLOCK_SIZE, BLOCK_SIZE, L>;


template<int D> struct attn_prep_globals { 
    gl<bf16, -1, -1, -1, -1> Og;
    gl<float, -1, -1, -1, -1> dOg; 
    gl<float, -1, -1, -1, -1> delta;
    dim3 grid() { return dim3(ATTN_B, ATTN_H, ATTN_N / BLOCK_SIZE); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY - 32000; }
};

template<int D> __launch_bounds__(NUM_THREADS, 1)
__global__ void attend_prep_ker(const attn_prep_globals<D> g) {
    
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int i = blockIdx.z;

    qkvo_tile<D, float, row_l> dO, O;
    load(dO, g.dOg, {b,h,i,0});
    load(O,  g.Og,  {b,h,i,0});
    
    // Δ_i = row_sum(dO ⊙ O) 
    mul(dO, dO, O);
    attn_tile<D,float,row_l>::col_vec delta_vec;
    row_sum(delta_vec, dO); 
    store(g.delta, delta_vec, {b,h,0,i});
}

template<int D>
void dispatch_prep(attn_prep_globals<D> g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)attend_prep_ker<D>, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    attend_prep_ker<D><<<g.grid(), g.block(), mem_size>>>(g);
    hipDeviceSynchronize();
}

template<int D> struct attn_bwd_combined_globals { 
    gl<bf16, -1, -1, -1, -1> Q, K, V, O;
    gl<float, -1, -1, -1, -1> dOg, dQg, dKg, dVg;
    gl<float, -1, -1, -1, -1> m_vec, l_vec, delta_vec;
    dim3 grid() { return dim3(ATTN_B, ATTN_H, ATTN_N / BLOCK_SIZE); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY-32000; }
};

template<int D> __launch_bounds__(NUM_THREADS, 1)
__global__ void attend_bwd_combined_ker(const attn_bwd_combined_globals<D> g) {
    
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int j = blockIdx.z;

    const float scale_factor = 1.0f / sqrt(D);

    // Register tiles
    qkvo_tile<D, bf16, row_l> K_j, V_j;
    qkvo_tile<D, float, accum_col_l> dK_j, dV_j;

    // 6. Load K_j and V_j from HBM to registers
    load(K_j, g.K, {b,h,j,0});
    load(V_j, g.V, {b,h,j,0});
    
    // 7. Initialize dK_j = 0 and dV_j = 0
    zero(dK_j);
    zero(dV_j);

    // 8. for 1 <= i <= T_r (1024 / 32 = 32)
    for (int i = 0; i < ATTN_N / BLOCK_SIZE; ++i) {

        // 9. Load Q_i, O_i, dO_i, dQ_i, l_i, m_i, delta_i from HBM to registers
        qkvo_tile<D, bf16, row_l> Q_i;
        qkvo_tile<D, bf16, accum_col_l> dO_i;
        attn_tile<D, float, accum_col_l>::col_vec l_i, m_i, delta_i;
        load(Q_i, g.Q, {b,h,i,0});
        load(dO_i, g.dOg, {b,h,i,0});
        load(l_i, g.l_vec, {b,h,0,i});
        load(m_i, g.m_vec, {b,h,0,i});
        load(delta_i, g.delta_vec, {b,h,0,i});

        // 10. S_ij = Q_i K_j^T * scale
        attn_tile<D,float,accum_col_l> S_ij;
        zero(S_ij);
        mma_ABt(S_ij, Q_i, K_j, S_ij);
        mul(S_ij, S_ij, scale_factor);

        // 11. P_ij = exp(S_ij - m_i) / l_i
        sub_row(S_ij, S_ij, m_i);
        exp(S_ij, S_ij);
        div_row(S_ij, S_ij, l_i);

        // 12. dV_j += P_ij^T @ dO_i
        attn_tile<D,bf16,accum_col_l> P_ij_bf16_acc_col;
        copy(P_ij_bf16_acc_col, S_ij);
        mma_AtB(dV_j, P_ij_bf16_acc_col, dO_i, dV_j);

        // 13. dP_ij = dO_i @ V_j^T
        attn_tile<D,float,accum_col_l> dP_ij;
        zero(dP_ij);
        qkvo_tile<D,bf16,row_l> dO_i_bf16_row = swap_layout_inplace<row_l>(dO_i);
        mma_ABt(dP_ij, dO_i_bf16_row, V_j, dP_ij);

        // 14. dS_ij = P_ij o (dP_ij - delta_i)
        sub_row(dP_ij, dP_ij, delta_i);
        mul(dP_ij, dP_ij, S_ij);
        mul(dP_ij, dP_ij, scale_factor);

        // 15. dQ_i += dS_ij @ K_j (load from HBM and write back)
        qkvo_tile<D, float, accum_col_l> dQ_i;
        load(dQ_i, g.dQg, {b,h,i,0});
        attn_tile<D, bf16, accum_col_l> dS_ij_bf16_acc_col;
        copy(dS_ij_bf16_acc_col, dP_ij);
        attn_tile<D, bf16, row_l> dS_ij_bf16_row = swap_layout_inplace<row_l>(dS_ij_bf16_acc_col);
        qkvo_tile<D, bf16, col_l> K_j_col;
        swap_layout(K_j_col, K_j);
        mma_AB(dQ_i, dS_ij_bf16_row, K_j_col, dQ_i);
        store(g.dQg, dQ_i, {b,h,i,0});

        // 16. dK_j += dS_ij^T @ Q_i
        attn_tile<D,bf16,col_l> dS_ij_bf16_col = swap_layout_inplace<col_l>(dS_ij_bf16_row);
        qkvo_tile<D, bf16, col_l> Q_i_col = swap_layout_inplace<col_l>(Q_i);
        mma_AtB(dK_j, dS_ij_bf16_col, Q_i_col, dK_j);
    }

    // 18. Write dK_j and dV_j back to HBM
    store(g.dKg, dK_j, {b,h,j,0});
    store(g.dVg, dV_j, {b,h,j,0});
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
        &attn_bwd_combined_globals<ATTN_D>::dOg, 
        &attn_bwd_combined_globals<ATTN_D>::dQg,
        &attn_bwd_combined_globals<ATTN_D>::dKg,
        &attn_bwd_combined_globals<ATTN_D>::dVg,
        &attn_bwd_combined_globals<ATTN_D>::m_vec, 
        &attn_bwd_combined_globals<ATTN_D>::l_vec,
        &attn_bwd_combined_globals<ATTN_D>::delta_vec
    );
}