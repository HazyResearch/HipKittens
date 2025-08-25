#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

constexpr int ATTN_B = 1; // batch size
constexpr int ATTN_H = 1; // number of heads
constexpr int ATTN_N = 32; // sequence length
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
    gl<float, -1, -1, -1, 1> delta;
    dim3 grid() { return dim3(ATTN_B, ATTN_H, ATTN_N / BLOCK_SIZE); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY-32000; }
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
    store(g.delta, delta_vec, {b,h,i,0});
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
    gl<float, -1, -1, -1, 1> m_vec, l_vec, delta_vec;
    dim3 grid() { return dim3(ATTN_B, ATTN_H, ATTN_N / BLOCK_SIZE); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY-32000; }
};

template<int D> __launch_bounds__(NUM_THREADS, 1)
__global__ void attend_bwd_combined_ker(const attn_bwd_combined_globals<D> g) {
    
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int i = blockIdx.z;  // This is the block index we're processing

    const float scale_factor = 1.0f / sqrt(D);

    // Register tiles
    qkvo_tile<D, bf16, row_l> qi_reg, ki_reg, vi_reg;
    qkvo_tile<D, bf16, col_l> ki_reg_col, vi_reg_col;
    qkvo_tile<D, float, row_l> dOi_reg, Oi_reg;
    qkvo_tile<D, float, accum_col_l> dQ_acc, dK_acc, dV_acc;
    
    // Initialize accumulators
    zero(dQ_acc);
    zero(dK_acc);
    zero(dV_acc);

    // Load this block's data (block i)
    load(qi_reg,  g.Q,  {b,h,i,0});
    load(ki_reg,  g.K,  {b,h,i,0});
    load(vi_reg,  g.V,  {b,h,i,0});
    load(dOi_reg, g.dOg, {b,h,i,0});
    load(Oi_reg,  g.O,  {b,h,i,0});
    
    // Load statistics for block i
    typename attn_tile<D,float,accum_col_l>::col_vec mi_vec, li_vec;
    load(mi_vec, g.m_vec, {b,h,i,0});
    load(li_vec, g.l_vec, {b,h,i,0});
    typename attn_tile<D,float,accum_col_l>::col_vec deltai_vec;
    load(deltai_vec, g.delta_vec, {b,h,i,0});

    // Convert layouts
    swap_layout(ki_reg_col, ki_reg);

    int num_blocks = ATTN_N / BLOCK_SIZE;
    
    // Loop over all blocks j
    for (int j = 0; j < num_blocks; ++j) {
        
        // ============ Compute dQ_i contribution from block j ============
        // Load K_j and V_j
        qkvo_tile<D, bf16, row_l> kj_reg, vj_reg;
        load(kj_reg, g.K, {b,h,j,0});
        load(vj_reg, g.V, {b,h,j,0});
        qkvo_tile<D, bf16, col_l> kj_reg_col;
        swap_layout(kj_reg_col, kj_reg);

        // S_ij = (Q_i K_j^T) * scale
        attn_tile<D,float,accum_col_l> S_ij; 
        zero(S_ij);
        mma_ABt(S_ij, qi_reg, kj_reg, S_ij);
        mul(S_ij, S_ij, scale_factor);

        // P_ij = exp(S_ij - m_i) / l_i
        sub_row(S_ij, S_ij, mi_vec);
        exp(S_ij, S_ij);
        div_row(S_ij, S_ij, li_vec);

        // dS_ij = P_ij ⊙ (dO_i V_j^T - Delta_i)
        attn_tile<D,float,accum_col_l> dOVt; 
        zero(dOVt);
        qkvo_tile<D,bf16,row_l> dOi_reg_bf16;
        copy(dOi_reg_bf16, dOi_reg);
        mma_ABt(dOVt, dOi_reg_bf16, vj_reg, dOVt);
        sub_row(dOVt, dOVt, deltai_vec);
        mul(dOVt, dOVt, S_ij);

        // dQ_i += dS_ij K_j * scale
        attn_tile<D,float,row_l> dOVt_row;
        swap_layout(dOVt_row, dOVt);
        mul(dOVt_row, dOVt_row, scale_factor);
        attn_tile<D,bf16,row_l> dOVt_bf16_row;
        copy(dOVt_bf16_row, dOVt_row);
        mma_AB(dQ_acc, dOVt_bf16_row, kj_reg_col, dQ_acc);

        // ============ Compute dK_i and dV_i contribution from block j ============
        // Load Q_j, dO_j, O_j and their statistics
        qkvo_tile<D, bf16, row_l> qj_reg;
        qkvo_tile<D, float, row_l> dOj_reg, Oj_reg;
        typename attn_tile<D,float,accum_col_l>::col_vec mj_vec, lj_vec;
        typename attn_tile<D,float,accum_col_l>::col_vec deltaj_vec;
        
        load(qj_reg,  g.Q,    {b,h,j,0});
        load(dOj_reg, g.dOg,  {b,h,j,0});
        load(Oj_reg,  g.O,    {b,h,j,0});
        load(mj_vec,  g.m_vec, {b,h,j,0});
        load(lj_vec,  g.l_vec, {b,h,j,0});
        load(deltaj_vec, g.delta_vec, {b,h,j,0});
        
        // P_ji = exp(Q_j K_i^T * scale - m_j) / l_j
        attn_tile<D,float,accum_col_l> S_ji; 
        zero(S_ji);
        mma_ABt(S_ji, qj_reg, ki_reg, S_ji);
        mul(S_ji, S_ji, scale_factor);
        sub_row(S_ji, S_ji, mj_vec);
        exp(S_ji, S_ji);
        div_row(S_ji, S_ji, lj_vec); 

        // dV_i += P_ji^T dO_j
        attn_tile<D,bf16,accum_col_l> P_ji_bf16; 
        copy(P_ji_bf16, S_ji);
        attn_tile<D,bf16,col_l> P_ji_bf16_col;
        swap_layout(P_ji_bf16_col, P_ji_bf16);
        
        qkvo_tile<D,bf16,row_l> dOj_bf16;
        copy(dOj_bf16, dOj_reg);
        qkvo_tile<D,bf16,col_l> dOj_bf16_col;
        swap_layout(dOj_bf16_col, dOj_bf16);
        mma_AtB(dV_acc, P_ji_bf16_col, dOj_bf16_col, dV_acc); 
        
        // dS_ji = P_ji ⊙ (dO_j V_i^T − Delta_j)
        attn_tile<D,float,accum_col_l> dOVt_ji; 
        zero(dOVt_ji);
        mma_ABt(dOVt_ji, dOj_bf16, vi_reg, dOVt_ji); 
        sub_row(dOVt_ji, dOVt_ji, deltaj_vec);
        mul(dOVt_ji, dOVt_ji, S_ji);
        
        // dK_i += dS_ji^T Q_j * scale
        mul(dOVt_ji, dOVt_ji, scale_factor);
        attn_tile<D,bf16,accum_col_l> dS_ji_bf16; 
        copy(dS_ji_bf16, dOVt_ji);
        attn_tile<D,bf16,col_l> dS_ji_bf16_col;
        swap_layout(dS_ji_bf16_col, dS_ji_bf16);
        qkvo_tile<D,bf16,col_l> qj_bf16_col;
        swap_layout(qj_bf16_col, qj_reg);
        mma_AtB(dK_acc, dS_ji_bf16_col, qj_bf16_col, dK_acc);
    }

    // Store results for block i
    store(g.dQg, dQ_acc, {b,h,i,0});
    store(g.dKg, dK_acc, {b,h,i,0});
    store(g.dVg, dV_acc, {b,h,i,0});
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