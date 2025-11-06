#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

constexpr int ATTN_B = 16; // batch size
constexpr int ATTN_H = 16; // number of heads
constexpr int ATTN_N = 1024; // sequence length
constexpr int ATTN_D = 128; // dimension
constexpr int BLOCK_SIZE = 32; // block size

#define NUM_WARPS_PREP 4
#define NUM_THREADS_PREP (kittens::WARP_THREADS * NUM_WARPS_PREP)
#define NUM_WARPS_BWD 8
#define NUM_THREADS_BWD (kittens::WARP_THREADS * NUM_WARPS_BWD)
using G = kittens::group<NUM_WARPS_BWD>;

using namespace kittens;

template<int D, typename T=bf16, typename L=row_l> using qkvo_tile = rt<T, BLOCK_SIZE, D, L>;
template<int D, typename T=bf16, typename L=col_l> using qkvo_tile_transposed = rt<T, D, BLOCK_SIZE, L>;
template<int D, typename T=float, typename L=row_l> using attn_tile = rt<T, BLOCK_SIZE, BLOCK_SIZE, L>;

template<int D> struct attn_prep_globals { 
    gl<bf16, -1, -1, -1, -1> Og;
    gl<float, -1, -1, -1, -1> dOg; 
    gl<float, -1, -1, -1, -1> delta;
    dim3 grid() { return dim3(ATTN_B, ATTN_H, ((ATTN_N / BLOCK_SIZE + NUM_WARPS_PREP - 1) / NUM_WARPS_PREP)); }
    dim3 block() { return dim3(NUM_THREADS_PREP); }
    size_t dynamic_shared_memory() { return 1024; }
};

template<int D> __launch_bounds__(NUM_THREADS_PREP, 2)
__global__ void attend_prep_ker(const attn_prep_globals<D> g) {

    const int warp_idx = warpid();
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int i = blockIdx.z * NUM_WARPS_PREP + warp_idx;

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
    dim3 grid() { return dim3(ATTN_B, ATTN_H, ((ATTN_N / BLOCK_SIZE + NUM_WARPS_BWD - 1) / NUM_WARPS_BWD)); }
    dim3 block() { return dim3(NUM_THREADS_BWD); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY-16000; }
};

template<int D> __launch_bounds__(NUM_THREADS_BWD, 1)
__global__ void attend_bwd_combined_ker(const attn_bwd_combined_globals<D> g) {
    
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int i = blockIdx.z * NUM_WARPS_BWD + warpid();

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<BLOCK_SIZE, ATTN_D, ducks::st_layout::row> (&q_smem)[2] = al.allocate<st_bf<BLOCK_SIZE, ATTN_D, ducks::st_layout::row>, 2>();
    st_bf<BLOCK_SIZE, ATTN_D, ducks::st_layout::row> (&k_smem)[2] = al.allocate<st_bf<BLOCK_SIZE, ATTN_D, ducks::st_layout::row>, 2>();
    st_bf<BLOCK_SIZE, ATTN_D, ducks::st_layout::row> (&v_smem)[2] = al.allocate<st_bf<BLOCK_SIZE, ATTN_D, ducks::st_layout::row>, 2>();

    const float scale_factor = 1.0f / sqrt(D);

    // Register tiles
    qkvo_tile<D, bf16, row_l> q_reg, k_reg, v_reg;
    qkvo_tile<D, float, row_l> dOi_reg;
    qkvo_tile<D, float, accum_col_l> dQ_acc, dK_acc, dV_acc;

    qkvo_tile<D, bf16, row_l> qj_reg, kj_reg, vj_reg;
    qkvo_tile<D, float, row_l> dOj_reg;
    qkvo_tile<D, bf16, col_l> kj_reg_col;
    qkvo_tile<D,bf16,col_l> q_bf16_col;
    qkvo_tile<D,bf16,col_l> dO_bf16_col;
    qkvo_tile<D,bf16,row_l> dO_reg_bf16;

    typename attn_tile<D,float,accum_col_l>::col_vec mi_vec, li_vec;
    typename attn_tile<D,float,accum_col_l>::col_vec deltai_vec;
    typename attn_tile<D,float,accum_col_l>::col_vec mj_vec, lj_vec;
    typename attn_tile<D,float,accum_col_l>::col_vec deltaj_vec;

    attn_tile<D,float,accum_col_l> S; 
    attn_tile<D,float,accum_col_l> dOVt;
    attn_tile<D,float,row_l> dOVt_row;
    attn_tile<D,bf16,col_l> P_bf16_col;
    attn_tile<D,bf16,accum_col_l> P_bf16;  
    
    // Initialize accumulators
    zero(dQ_acc);
    zero(dK_acc);
    zero(dV_acc);

    // Load this block's data (block i)
    load(q_reg,  g.Q,  {b,h,i,0});
    load(k_reg,  g.K,  {b,h,i,0});
    load(v_reg,  g.V,  {b,h,i,0});
    load(dOi_reg, g.dOg, {b,h,i,0});
    
    // Load statistics for block i
    load(mi_vec, g.m_vec, {b,h,0,i});
    load(li_vec, g.l_vec, {b,h,0,i});
    load(deltai_vec, g.delta_vec, {b,h,0,i});
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();

    // preswizzling for j blocks
    using T = typename st_bf<BLOCK_SIZE, ATTN_D>::dtype;
    constexpr int bytes_per_thread = 16;
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_THREADS_BWD;
    constexpr int memcpy_per_tile = BLOCK_SIZE * ATTN_D * sizeof(T) / bytes_per_memcpy;
    uint32_t swizzled_offsets_Q[memcpy_per_tile];
    uint32_t swizzled_offsets_V[memcpy_per_tile];
    uint32_t swizzled_offsets_K[memcpy_per_tile];
    G::prefill_swizzled_offsets<2, false>(q_smem[0], g.Q, swizzled_offsets_Q);
    G::prefill_swizzled_offsets<2, false>(k_smem[0], g.K, swizzled_offsets_K);
    G::prefill_swizzled_offsets<2, false>(v_smem[0], g.V, swizzled_offsets_V);

    // Initial load for K_j and V_j
    int tic = 0, toc = 1;
    G::load<2, false>(q_smem[tic], g.Q, {b,h,0,0}, swizzled_offsets_Q);
    G::load<2, false>(k_smem[tic], g.K, {b,h,0,0}, swizzled_offsets_K);
    G::load<2, false>(v_smem[tic], g.V, {b,h,0,0}, swizzled_offsets_V);
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();

    int num_blocks = ATTN_N / BLOCK_SIZE;
    int condition = (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0);
    
    // Loop over all blocks j
    for (int j = 0; j < num_blocks-1; ++j, tic^=1, toc^=1) {
        
        // ============ Compute dQ_i contribution from block j ============
        // Load K_j and V_j
        G::load<2, false>(k_smem[toc], g.K, {b,h,j+1,0}, swizzled_offsets_K);
        G::load<2, false>(v_smem[toc], g.V, {b,h,j+1,0}, swizzled_offsets_V);
        load(kj_reg, k_smem[tic]);
        load(vj_reg, v_smem[tic]);
        swap_layout(kj_reg_col, kj_reg);
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();

        // S_ij = (Q_i K_j^T) * scale
        zero(S);
        mma_ABt(S, q_reg, kj_reg, S);
        mul(S, S, scale_factor);

        // P_ij = exp(S_ij - m_i) / l_i
        sub_row(S, S, mi_vec);
        exp(S, S);
        div_row(S, S, li_vec);

        // dS_ij = P_ij ⊙ (dO_i V_j^T - Delta_i)
        zero(dOVt);
        copy(dO_reg_bf16, dOi_reg);
        mma_ABt(dOVt, dO_reg_bf16, vj_reg, dOVt);
        sub_row(dOVt, dOVt, deltai_vec);
        mul(dOVt, dOVt, S);

        // dQ_i += dS_ij K_j * scale
        swap_layout(dOVt_row, dOVt);
        mul(dOVt_row, dOVt_row, scale_factor);
        attn_tile<D,bf16,row_l> dOVt_bf16_row;
        copy(dOVt_bf16_row, dOVt_row);
        mma_AB(dQ_acc, dOVt_bf16_row, kj_reg_col, dQ_acc);

        // ============ Compute dK_i and dV_i contribution from block j ============
        // Load Q_j, dO_j, O_j and their statistics
        G::load<2, false>(q_smem[toc], g.Q, {b,h,j+1,0}, swizzled_offsets_Q);
        load(qj_reg, q_smem[tic]);
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();

        // Other loads
        load(dOj_reg, g.dOg,  {b,h,j,0});
        load(mj_vec,  g.m_vec, {b,h,0,j});
        load(lj_vec,  g.l_vec, {b,h,0,j});
        load(deltaj_vec, g.delta_vec, {b,h,0,j});
        
        // P_ji = exp(Q_j K_i^T * scale - m_j) / l_j
        zero(S);
        mma_ABt(S, qj_reg, k_reg, S);
        mul(S, S, scale_factor);
        sub_row(S, S, mj_vec);
        exp(S, S);
        div_row(S, S, lj_vec); 

        // dV_i += P_ji^T dO_j
        copy(P_bf16, S);
        swap_layout(P_bf16_col, P_bf16);
        
        copy(dO_reg_bf16, dOj_reg);
        swap_layout(dO_bf16_col, dO_reg_bf16);
        mma_AtB(dV_acc, P_bf16_col, dO_bf16_col, dV_acc); 
        
        // dS_ji = P_ji ⊙ (dO_j V_i^T − Delta_j) 
        zero(dOVt);
        mma_ABt(dOVt, dO_reg_bf16, v_reg, dOVt); 
        sub_row(dOVt, dOVt, deltaj_vec);
        mul(dOVt, dOVt, S);
        
        // dK_i += dS_ji^T Q_j * scale
        mul(dOVt, dOVt, scale_factor);
        copy(P_bf16, dOVt);
        swap_layout(P_bf16_col, P_bf16);
        swap_layout(q_bf16_col, qj_reg);
        mma_AtB(dK_acc, P_bf16_col, q_bf16_col, dK_acc);
    }

    // ============ Compute dQ_i contribution from block j ============
    // Load K_j and V_j
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_sched_barrier(0);
    load(kj_reg, k_smem[tic]);
    load(vj_reg, v_smem[tic]);
    swap_layout(kj_reg_col, kj_reg);

    // S_ij = (Q_i K_j^T) * scale
    zero(S);
    mma_ABt(S, q_reg, kj_reg, S);
    mul(S, S, scale_factor);

    // P_ij = exp(S_ij - m_i) / l_i
    sub_row(S, S, mi_vec);
    exp(S, S);
    div_row(S, S, li_vec);

    // dS_ij = P_ij ⊙ (dO_i V_j^T - Delta_i) 
    zero(dOVt);
    copy(dO_reg_bf16, dOi_reg);
    mma_ABt(dOVt, dO_reg_bf16, vj_reg, dOVt);
    sub_row(dOVt, dOVt, deltai_vec);
    mul(dOVt, dOVt, S);

    // dQ_i += dS_ij K_j * scale
    swap_layout(dOVt_row, dOVt);
    mul(dOVt_row, dOVt_row, scale_factor);
    attn_tile<D,bf16,row_l> dOVt_bf16_row;
    copy(dOVt_bf16_row, dOVt_row);
    mma_AB(dQ_acc, dOVt_bf16_row, kj_reg_col, dQ_acc);

    // ============ Compute dK_i and dV_i contribution from block j ============
    // Load Q_j, dO_j, O_j and their statistics
    load(qj_reg,  q_smem[tic]);
    load(dOj_reg, g.dOg,  {b,h,num_blocks-1,0});
    load(mj_vec,  g.m_vec, {b,h,0,num_blocks-1});
    load(lj_vec,  g.l_vec, {b,h,0,num_blocks-1});
    load(deltaj_vec, g.delta_vec, {b,h,0,num_blocks-1});
    
    // P_ji = exp(Q_j K_i^T * scale - m_j) / l_j
    zero(S);
    mma_ABt(S, qj_reg, k_reg, S);
    mul(S, S, scale_factor);
    sub_row(S, S, mj_vec);
    exp(S, S);
    div_row(S, S, lj_vec); 

    // dV_i += P_ji^T dO_j
    copy(P_bf16, S);
    swap_layout(P_bf16_col, P_bf16);
    
    copy(dO_reg_bf16, dOj_reg);
    swap_layout(dO_bf16_col, dO_reg_bf16);
    mma_AtB(dV_acc, P_bf16_col, dO_bf16_col, dV_acc); 
    
    // dS_ji = P_ji ⊙ (dO_j V_i^T − Delta_j)
    zero(dOVt);
    mma_ABt(dOVt, dO_reg_bf16, v_reg, dOVt); 
    sub_row(dOVt, dOVt, deltaj_vec);
    mul(dOVt, dOVt, S);
    
    // dK_i += dS_ji^T Q_j * scale
    mul(dOVt, dOVt, scale_factor);
    copy(P_bf16, dOVt);
    swap_layout(P_bf16_col, P_bf16);
    swap_layout(q_bf16_col, qj_reg);
    mma_AtB(dK_acc, P_bf16_col, q_bf16_col, dK_acc);

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