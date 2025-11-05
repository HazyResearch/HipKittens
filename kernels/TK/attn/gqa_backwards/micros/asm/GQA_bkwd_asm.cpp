#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include "utils.cpp"

constexpr int ATTN_B = 16; // batch size
constexpr int ATTN_H_Q = 16; // number of query heads
constexpr int ATTN_H_KV = 16; // number of key/value heads (for GQA)
constexpr int GROUP_SIZE = ATTN_H_Q / ATTN_H_KV; // queries per KV head group
constexpr int ATTN_N = 8192; // sequence length
constexpr int ATTN_D = 128; // dimension
constexpr int STEP_QO = 64; // block size for QO
constexpr int BLOCK_SIZE_KV = 256; // block size for KV
constexpr int SLICE_QO = 32;
constexpr int DOT_SLICE_QO = 16;
constexpr int WARP_SIZE_KV = 64; // warp size for KV

#define NUM_WARPS 4
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)
#define ASSEMBLY

using G = kittens::group<NUM_WARPS>;

using namespace kittens;

template<int D, typename T=bf16, typename L=row_l, typename M=mfma_16x16x32> using qo_tile = rt<T, DOT_SLICE_QO, D, L, M>;
template<int D, typename T=bf16, typename L=row_l, typename M=mfma_16x16x32> using kv_tile = rt<T, WARP_SIZE_KV, D, L, M>;
template<int D, typename T=bf16, typename L=row_l, typename M=mfma_16x16x32> using qo_tile_T_dq = rt<T, 32, 16, L, M>;
template<int D, typename T=bf16, typename L=row_l, typename M=mfma_16x16x32> using qo_tile_dq = rt<T, 16, 32, L, M>;
template<int D, typename T=bf16, typename L=row_l, typename M=mfma_16x16x32> using kv_tile_T = rt<T, D, WARP_SIZE_KV, L, M>;
template<int D, typename T=float, typename L=accum_col_l, typename M=mfma_16x16x32> using attn_tile = rt<T, DOT_SLICE_QO, WARP_SIZE_KV, L, M>;
template<int D, typename T=bf16, typename L=col_l, typename M=mfma_16x16x32> using attn_tile_T = rt<T, WARP_SIZE_KV, DOT_SLICE_QO, L, M>;

template<int D, typename T=bf16, typename L=col_l, typename M=mfma_16x16x32> using attn_tile_T_dq = rt<T, 256, 16, L, M>;
template<int D, typename T=bf16, typename L=row_l, typename M=mfma_16x16x32> using kv_tile_dq = rt<T, 256, 32, L, M>;

template<int D> struct attn_prep_globals { 
  gl<bf16, -1, -1, -1, -1> Og;
  gl<bf16, -1, -1, -1, -1> dOg; 
  gl<float, -1, -1, -1, -1> delta;
  dim3 grid() { return dim3(ATTN_B, ATTN_H_Q, ATTN_N / (DOT_SLICE_QO * NUM_WARPS)); }
  dim3 block() { return dim3(NUM_THREADS); }
  size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

template<int D> __launch_bounds__(NUM_THREADS, 1)
__global__ void attend_prep_ker(const attn_prep_globals<D> g) {
    
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int seq_idx = blockIdx.z;

    const int warpid = kittens::warpid();

    qo_tile<D, bf16, row_l> dO, O;
    qo_tile<D, float, row_l> dO_float, O_float;
    typename qo_tile<D, float, row_l>::col_vec delta_vec;

    load<1>(dO, g.dOg, {batch_idx, seq_idx * NUM_WARPS + warpid, head_idx, 0});
    load<1>(O,  g.Og,  {batch_idx, seq_idx * NUM_WARPS + warpid, head_idx, 0});
    copy(O_float, O);
    copy(dO_float, dO);
    
    // Δ_i = row_sum(dO ⊙ O) 
    mul(dO_float, dO_float, O_float);
    row_sum(delta_vec, dO_float); 
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
  gl<bf16, -1, -1, -1, -1> Q, K, V;
  gl<bf16, -1, -1, -1, -1> dOg, dQg, dKg, dVg;
  gl<float, -1, -1, -1, -1> L_vec, delta_vec;
  dim3 grid() { return dim3((ATTN_N / BLOCK_SIZE_KV), ATTN_H_KV, ATTN_B); }
  dim3 block() { return dim3(NUM_THREADS); }
  size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

#ifdef ASSEMBLY
template<int D> __launch_bounds__(NUM_THREADS, 1)
__global__ void attend_bwd_combined_ker(const attn_bwd_combined_globals<D> g) {

  const int seq_idx = blockIdx.x;
  const int kv_head_idx = blockIdx.y; // This is the KV head index
  const int batch_idx = blockIdx.z;
  const int first_q_head = kv_head_idx * GROUP_SIZE;

  const int warpid = kittens::warpid();
  const int j = seq_idx * NUM_WARPS + warpid;

  const int num_steps_per_head = ATTN_N / STEP_QO;
  const int num_steps = num_steps_per_head * GROUP_SIZE;
  const float scale_factor = 1.0f / sqrt(D);

  // Shared tiles
  extern __shared__ alignment_dummy __shm[];
  shared_allocator al((int*)&__shm[0]);

  // K_j_smem is organized in 16x16 tiles
  st_bf<BLOCK_SIZE_KV, D, ducks::st_layout::accumulator, ducks::st_matrix::mfma_16x16x32> (&K_j_smem) = al.allocate<st_bf<BLOCK_SIZE_KV, D, ducks::st_layout::accumulator, ducks::st_matrix::mfma_16x16x32>>();
  st_bf<SLICE_QO, D, ducks::st_layout::classical, ducks::st_matrix::mfma_16x16x32> (&Q_i_smem)[2][2] = al.allocate<st_bf<SLICE_QO, D, ducks::st_layout::classical, ducks::st_matrix::mfma_16x16x32>, 2, 2>();
  st_bf<SLICE_QO, D, ducks::st_layout::classical, ducks::st_matrix::mfma_16x16x32> (&dO_i_smem)[2][2] = al.allocate<st_bf<SLICE_QO, D, ducks::st_layout::classical, ducks::st_matrix::mfma_16x16x32>, 2, 2>();
  // We parameterize this using mfma_32x32x16 because we want the base tile for it to be 32x16. Not that it uses that intrinsic.
  st_bf<BLOCK_SIZE_KV, DOT_SLICE_QO, ducks::st_layout::classical, ducks::st_matrix::mfma_32x32x16> (&attn_i_smem) = al.allocate<st_bf<BLOCK_SIZE_KV, DOT_SLICE_QO, ducks::st_layout::classical, ducks::st_matrix::mfma_32x32x16>>();
  sv_fl<STEP_QO> (&L_smem)[2] = al.allocate<sv_fl<STEP_QO>, 2>();
  sv_fl<STEP_QO> (&delta_smem)[2] = al.allocate<sv_fl<STEP_QO>, 2>();

  // Register tiles
  using Q_ranges = ducks::rt_asm::split_many_t<ducks::rt_asm::type_list<ducks::rt_asm::range<368, 383>>, 4>; // 16 registers
  using dO_ranges = ducks::rt_asm::split_many_t<ducks::rt_asm::type_list<ducks::rt_asm::range<96, 111>>, 4>; // 16 registers
  using K_ranges = ducks::rt_asm::split_many_t<ducks::rt_asm::type_list<ducks::rt_asm::range<112, 127>, ducks::rt_asm::range<256, 303>>, 4>; // 64 registers
  using V_ranges = ducks::rt_asm::split_many_t<ducks::rt_asm::type_list<ducks::rt_asm::range<304, 367>>, 4>; // 64 registers
  using P_ranges = ducks::rt_asm::split_many_t<ducks::rt_asm::type_list<ducks::rt_asm::range<80, 95>>, 4>; // 16 registers
  using dP_ranges = ducks::rt_asm::split_many_t<ducks::rt_asm::type_list<ducks::rt_asm::range<64, 79>>, 4>; // 16 registers
  using P_bf16_ranges = ducks::rt_asm::split_many_t<ducks::rt_asm::type_list<ducks::rt_asm::range<56, 63>>, 2>; // 8 registers
  using dP_bf16_ranges = ducks::rt_asm::split_many_t<ducks::rt_asm::type_list<ducks::rt_asm::range<48, 55>>, 2>; // 8 registers
  using P_bf16_col_ranges = ducks::rt_asm::split_many_t<ducks::rt_asm::type_list<ducks::rt_asm::range<56, 63>>, 4>; // 8 registers
  using dP_bf16_col_ranges = ducks::rt_asm::split_many_t<ducks::rt_asm::type_list<ducks::rt_asm::range<48, 55>>, 4>; // 8 registers
  using dS_ranges = ducks::rt_asm::split_many_t<ducks::rt_asm::type_list<ducks::rt_asm::range<32, 63>>, 4>; // 32 registers
  using dQ_ranges = ducks::rt_asm::split_many_t<ducks::rt_asm::type_list<ducks::rt_asm::range<24, 31>>, 4>; // 8 registers  
  ducks::rt_asm::clobber<Q_ranges>();
  ducks::rt_asm::clobber<dO_ranges>();
  ducks::rt_asm::clobber<K_ranges>();
  ducks::rt_asm::clobber<V_ranges>();
  ducks::rt_asm::clobber<P_ranges>();
  ducks::rt_asm::clobber<dP_ranges>();
  ducks::rt_asm::clobber<P_bf16_ranges>();
  ducks::rt_asm::clobber<dP_bf16_ranges>();
  ducks::rt_asm::clobber<dS_ranges>();
  ducks::rt_asm::clobber<dQ_ranges>();

  using dV_ranges = ducks::rt_asm::split_many_t<ducks::rt_asm::type_list<ducks::rt_asm::range<128, 255>>, 16>; // 128 registers
  using dK_ranges = ducks::rt_asm::split_many_t<ducks::rt_asm::type_list<ducks::rt_asm::range<384, 511>>, 16>; // 128 registers
  ducks::rt_asm::clobber<dV_ranges>();
  ducks::rt_asm::clobber<dK_ranges>();

  rt_asm<bf16, DOT_SLICE_QO, D, row_l, mfma_16x16x32, Q_ranges> Q_i; // 16 registers
  rt_asm<bf16, DOT_SLICE_QO, D, row_l, mfma_16x16x32, dO_ranges> dO_i; // 16 registers
  rt_asm<bf16, DOT_SLICE_QO, D, col_l, mfma_32x32x16, Q_ranges> Q_i_col; // 16 registers
  rt_asm<bf16, DOT_SLICE_QO, D, col_l, mfma_32x32x16, dO_ranges> dO_i_col; // 16 registers
  rt_asm<bf16, WARP_SIZE_KV, D, row_l, mfma_16x16x32, K_ranges> K_j; // 64 registers
  rt_asm<bf16, WARP_SIZE_KV, D, row_l, mfma_16x16x32, V_ranges> V_j; // 64 registers
  rt<float, DOT_SLICE_QO, WARP_SIZE_KV, accum_col_l, mfma_16x16x32>::col_vec L_i, delta_i;

  rt_asm<float, DOT_SLICE_QO, WARP_SIZE_KV, accum_col_l, mfma_16x16x32, P_ranges> P_ij; // 16 registers
  rt_asm<float, DOT_SLICE_QO, WARP_SIZE_KV, accum_col_l, mfma_16x16x32, dP_ranges> dP_ij; // 16 registers
  rt_asm<bf16, DOT_SLICE_QO, WARP_SIZE_KV, accum_col_l, mfma_16x16x32, P_bf16_ranges> P_ij_bf16; // 8 registers
  rt_asm<bf16, DOT_SLICE_QO, WARP_SIZE_KV, accum_col_l, mfma_16x16x32, dP_bf16_ranges> dP_ij_bf16; // 8 registers
  rt_asm<bf16, WARP_SIZE_KV, DOT_SLICE_QO, accum_row_l, mfma_16x16x32, ducks::rt_asm::transpose_2d<dP_bf16_ranges, 1, 4>> dP_ij_bf16_accum_row; // 8 registers

  rt_asm<bf16, DOT_SLICE_QO, WARP_SIZE_KV, col_l, mfma_32x32x16, P_bf16_col_ranges> P_ij_bf16_col; // 8 registers
  rt_asm<bf16, DOT_SLICE_QO, WARP_SIZE_KV, col_l, mfma_32x32x16, dP_bf16_col_ranges> dP_ij_bf16_col; // 8 registers

  rt_asm<bf16, 256, 32, col_l, mfma_16x16x32, K_ranges> K_j_col; // 64 registers // for dq
  rt_asm<bf16, 256, 16, col_l, mfma_16x16x32, dS_ranges> dP_ij_bf16_col_T; // 32 registers // for dq

  rt_asm<float, D, WARP_SIZE_KV, accum_col_l, mfma_32x32x16, dK_ranges> dK_j_T; // 128 registers
  rt_asm<float, D, WARP_SIZE_KV, accum_col_l, mfma_32x32x16, dV_ranges> dV_j_T; // 128 registers
  rt_asm<float, 32, 16, accum_col_l, mfma_16x16x32, dQ_ranges> dQ_i_T; // 8 registers // for dq
  rt_asm<float, 16, 32, accum_row_l, mfma_16x16x32, ducks::rt_asm::transpose_2d<dQ_ranges, 2, 1>> dQ_i; // 8 registers // for dq

  // This is used for both dK_j_T and dV_j_T
  rt_asm<float, WARP_SIZE_KV, D, accum_row_l, mfma_32x32x16, ducks::rt_asm::transpose_2d<dV_ranges, 4, 2>> dV_j;

  int tic = 0, toc = 1;

  // Load K_j from HBM to shared memory
  G::load<1, false>(K_j_smem, g.K, {batch_idx, seq_idx, kv_head_idx, 0});

  // Load V_j from HBM to registers
  load<1>(V_j, g.V, {batch_idx, 0, kv_head_idx, 0}, {0, j, 0, 0});

  // Load Q, dO, L, delta for this specific query head
  load(L_smem[tic], g.L_vec, {batch_idx, first_q_head, 0, 0});
  load(delta_smem[tic], g.delta_vec, {batch_idx, first_q_head, 0, 0});
  G::load<1, false>(Q_i_smem[tic][0], g.Q, {batch_idx, 0, first_q_head, 0});
  G::load<1, false>(dO_i_smem[tic][0], g.dOg, {batch_idx, 0, first_q_head, 0});
  G::load<1, false>(Q_i_smem[tic][1], g.Q, {batch_idx, 1, first_q_head, 0});
  G::load<1, false>(dO_i_smem[tic][1], g.dOg, {batch_idx, 1, first_q_head, 0});
  __builtin_amdgcn_s_waitcnt(0);
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_sched_barrier(0);

  // Prologue
  {
    const int q_head_idx = 0 / num_steps_per_head + first_q_head;
    const int q_seq_idx = 0 % num_steps_per_head;

    const int next_q_head_idx = (0 + 1) / num_steps_per_head + first_q_head;
    const int next_q_seq_idx = (0 + 1) % num_steps_per_head;

    // dot slice 0
    {
      load(L_smem[toc], g.L_vec, {batch_idx, next_q_head_idx, 0, next_q_seq_idx});
      G::load<1, false>(Q_i_smem[toc][0], g.Q, {batch_idx, next_q_seq_idx * 2, next_q_head_idx, 0});
      load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
      load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
      load(L_i, subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 0));
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);

      // 10. S_ij = Q_i K_j^T * scale
      // 11. P_ij = exp(S_ij - L_i)
      // 13. dP_ij = dO_i @ V_j^T
      // 14. dS_ij = P_ij o (dP_ij - delta_i)
      load(delta_i, subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 0));
      load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
      mma_ABt(P_ij, Q_i, K_j);
      mul(P_ij, P_ij, scale_factor);
      sub_row(P_ij, P_ij, L_i);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);

      load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
      mma_ABt(dP_ij, dO_i, V_j);
      load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
      exp(P_ij, P_ij);
      copy(P_ij_bf16, P_ij);
      sub_row(dP_ij, dP_ij, delta_i);
      mul(dP_ij, dP_ij, scale_factor);
      mul(dP_ij, dP_ij, P_ij);
      copy(dP_ij_bf16, dP_ij);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);

      // 12. dV_j += P_ij^T @ dO_i
      // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
      auto attn_i_smem_subtile = subtile_inplace<WARP_SIZE_KV, DOT_SLICE_QO>(attn_i_smem, {warpid, 0});
      store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
      load(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}));
      swap_layout_inplace(P_ij_bf16_col, P_ij_bf16);
      mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col); 
      swap_layout_inplace(dP_ij_bf16_col, dP_ij_bf16);
      mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);
    
      load(dP_ij_bf16_col_T, attn_i_smem);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);
    }

    // dot slice 1
    {
      // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
      mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T);
      atomic_pk_add_bf16_with_warpid<2>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4, 0}, warpid);

      load(delta_smem[toc], g.delta_vec, {batch_idx, next_q_head_idx, 0, next_q_seq_idx});
      G::load<1, false>(dO_i_smem[toc][0], g.dOg, {batch_idx, next_q_seq_idx * 2, next_q_head_idx, 0});
      load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
      load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {1, 0}));
      load(L_i, subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 1));
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);

      // 10. S_ij = Q_i K_j^T * scale
      // 11. P_ij = exp(S_ij - L_i)
      // 13. dP_ij = dO_i @ V_j^T
      // 14. dS_ij = P_ij o (dP_ij - delta_i)
      load(delta_i, subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 1));
      load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {1, 0}));
      mma_ABt(P_ij, Q_i, K_j);
      mul(P_ij, P_ij, scale_factor);
      sub_row(P_ij, P_ij, L_i);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);

      load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {1, 0}));
      mma_ABt(dP_ij, dO_i, V_j);
      load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {1, 0}));
      exp(P_ij, P_ij);
      copy(P_ij_bf16, P_ij);
      sub_row(dP_ij, dP_ij, delta_i);
      mul(dP_ij, dP_ij, scale_factor);
      mul(dP_ij, dP_ij, P_ij);
      copy(dP_ij_bf16, dP_ij);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);

      // 12. dV_j += P_ij^T @ dO_i
      // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
      auto attn_i_smem_subtile = subtile_inplace<WARP_SIZE_KV, DOT_SLICE_QO>(attn_i_smem, {warpid, 0});
      store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
      load(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}));
      swap_layout_inplace(P_ij_bf16_col, P_ij_bf16);
      mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
      swap_layout_inplace(dP_ij_bf16_col, dP_ij_bf16);
      mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);

      load(dP_ij_bf16_col_T, attn_i_smem);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);
    }

    // dot slice 2
    {
      // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
      mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T);
      atomic_pk_add_bf16_with_warpid<2>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4 + 1, 0}, warpid);

      G::load<1, false>(Q_i_smem[toc][1], g.Q, {batch_idx, next_q_seq_idx * 2 + 1, next_q_head_idx, 0});
      load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
      load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {0, 0}));
      load(L_i, subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 2));
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);

      // 10. S_ij = Q_i K_j^T * scale
      // 11. P_ij = exp(S_ij - L_i)
      // 13. dP_ij = dO_i @ V_j^T
      // 14. dS_ij = P_ij o (dP_ij - delta_i)
      load(delta_i, subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 2));
      load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {0, 0}));
      mma_ABt(P_ij, Q_i, K_j);
      mul(P_ij, P_ij, scale_factor);
      sub_row(P_ij, P_ij, L_i);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);

      load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {0, 0}));
      mma_ABt(dP_ij, dO_i, V_j);
      load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {0, 0}));
      exp(P_ij, P_ij);
      copy(P_ij_bf16, P_ij);
      sub_row(dP_ij, dP_ij, delta_i);
      mul(dP_ij, dP_ij, scale_factor);
      mul(dP_ij, dP_ij, P_ij);
      copy(dP_ij_bf16, dP_ij);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);

      // 12. dV_j += P_ij^T @ dO_i
      // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
      auto attn_i_smem_subtile = subtile_inplace<WARP_SIZE_KV, DOT_SLICE_QO>(attn_i_smem, {warpid, 0});
      store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
      load(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}));
      swap_layout_inplace(P_ij_bf16_col, P_ij_bf16);
      mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
      swap_layout_inplace(dP_ij_bf16_col, dP_ij_bf16);
      mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);

      load(dP_ij_bf16_col_T, attn_i_smem);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);
    }

    // dot slice 3
    {
      // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
      mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T);
      atomic_pk_add_bf16_with_warpid<2>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4 + 2, 0}, warpid);

      G::load<1, false>(dO_i_smem[toc][1], g.dOg, {batch_idx, next_q_seq_idx * 2 + 1, next_q_head_idx, 0});
      load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
      load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {1, 0}));
      load(L_i, subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 3));
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);

      // 10. S_ij = Q_i K_j^T * scale
      // 11. P_ij = exp(S_ij - L_i)
      // 13. dP_ij = dO_i @ V_j^T
      // 14. dS_ij = P_ij o (dP_ij - delta_i)
      load(delta_i, subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 3));
      load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {1, 0}));
      mma_ABt(P_ij, Q_i, K_j);
      mul(P_ij, P_ij, scale_factor);
      sub_row(P_ij, P_ij, L_i);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);

      load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {1, 0}));
      mma_ABt(dP_ij, dO_i, V_j);
      load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {1, 0}));
      exp(P_ij, P_ij);
      copy(P_ij_bf16, P_ij);
      sub_row(dP_ij, dP_ij, delta_i);
      mul(dP_ij, dP_ij, scale_factor);
      mul(dP_ij, dP_ij, P_ij);
      copy(dP_ij_bf16, dP_ij);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);

      auto attn_i_smem_subtile = subtile_inplace<WARP_SIZE_KV, DOT_SLICE_QO>(attn_i_smem, {warpid, 0});
      store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
      load(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}));
      // 12. dV_j += P_ij^T @ dO_i
      // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
      swap_layout_inplace(P_ij_bf16_col, P_ij_bf16);
      mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
      swap_layout_inplace(dP_ij_bf16_col, dP_ij_bf16);
      mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      __builtin_amdgcn_s_waitcnt(0);
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);

      load(dP_ij_bf16_col_T, attn_i_smem);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);
    }
    tic ^= 1; toc ^= 1;
  }

  // 9. for 1 <= i <= T_r (1024 / 32 = 32)  
  for (int i = 1; i < num_steps - 1; ++i, tic ^= 1, toc ^= 1) {
    const int last_q_head_idx = (i - 1) / num_steps_per_head + first_q_head;
    const int last_q_seq_idx = (i - 1) % num_steps_per_head;

    const int q_head_idx = i / num_steps_per_head + first_q_head;
    const int q_seq_idx = i % num_steps_per_head;

    const int next_q_head_idx = (i + 1) / num_steps_per_head + first_q_head;
    const int next_q_seq_idx = (i + 1) % num_steps_per_head;

    // dot slice 0
    {
      // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
      mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T);
      atomic_pk_add_bf16_with_warpid<2>(g.dQg, dQ_i, {batch_idx, last_q_head_idx, last_q_seq_idx * 4 + 3, 0}, warpid);

      load(L_smem[toc], g.L_vec, {batch_idx, next_q_head_idx, 0, next_q_seq_idx});
      G::load<1, false>(Q_i_smem[toc][0], g.Q, {batch_idx, next_q_seq_idx * 2, next_q_head_idx, 0});
      load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
      load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
      load(L_i, subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 0));
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);

      // 10. S_ij = Q_i K_j^T * scale
      // 11. P_ij = exp(S_ij - L_i)
      // 13. dP_ij = dO_i @ V_j^T
      // 14. dS_ij = P_ij o (dP_ij - delta_i)
      load(delta_i, subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 0));
      load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
      mma_ABt(P_ij, Q_i, K_j);
      mul(P_ij, P_ij, scale_factor);
      sub_row(P_ij, P_ij, L_i);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);

      load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
      mma_ABt(dP_ij, dO_i, V_j);
      load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
      exp(P_ij, P_ij);
      copy(P_ij_bf16, P_ij);
      sub_row(dP_ij, dP_ij, delta_i);
      mul(dP_ij, dP_ij, scale_factor);
      mul(dP_ij, dP_ij, P_ij);
      copy(dP_ij_bf16, dP_ij);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);

      // 12. dV_j += P_ij^T @ dO_i
      // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
      auto attn_i_smem_subtile = subtile_inplace<WARP_SIZE_KV, DOT_SLICE_QO>(attn_i_smem, {warpid, 0});
      store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
      load(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}));
      swap_layout_inplace(P_ij_bf16_col, P_ij_bf16);
      mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T); 
      swap_layout_inplace(dP_ij_bf16_col, dP_ij_bf16);
      mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);
    
      load(dP_ij_bf16_col_T, attn_i_smem);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);
    }

    // dot slice 1
    {
      // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
      mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T);
      atomic_pk_add_bf16_with_warpid<2>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4, 0}, warpid);

      load(delta_smem[toc], g.delta_vec, {batch_idx, next_q_head_idx, 0, next_q_seq_idx});
      G::load<1, false>(dO_i_smem[toc][0], g.dOg, {batch_idx, next_q_seq_idx * 2, next_q_head_idx, 0});
      load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
      load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {1, 0}));
      load(L_i, subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 1));
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);

      // 10. S_ij = Q_i K_j^T * scale
      // 11. P_ij = exp(S_ij - L_i)
      // 13. dP_ij = dO_i @ V_j^T
      // 14. dS_ij = P_ij o (dP_ij - delta_i)
      load(delta_i, subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 1));
      load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {1, 0}));
      mma_ABt(P_ij, Q_i, K_j);
      mul(P_ij, P_ij, scale_factor);
      sub_row(P_ij, P_ij, L_i);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);

      load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {1, 0}));
      mma_ABt(dP_ij, dO_i, V_j);
      load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {1, 0}));
      exp(P_ij, P_ij);
      copy(P_ij_bf16, P_ij);
      sub_row(dP_ij, dP_ij, delta_i);
      mul(dP_ij, dP_ij, scale_factor);
      mul(dP_ij, dP_ij, P_ij);
      copy(dP_ij_bf16, dP_ij);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);

      // 12. dV_j += P_ij^T @ dO_i
      // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
      auto attn_i_smem_subtile = subtile_inplace<WARP_SIZE_KV, DOT_SLICE_QO>(attn_i_smem, {warpid, 0});
      store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
      load(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}));
      swap_layout_inplace(P_ij_bf16_col, P_ij_bf16);
      mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
      swap_layout_inplace(dP_ij_bf16_col, dP_ij_bf16);
      mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);

      load(dP_ij_bf16_col_T, attn_i_smem);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);
    }

    // dot slice 2
    {
      // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
      mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T);
      atomic_pk_add_bf16_with_warpid<2>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4 + 1, 0}, warpid);

      G::load<1, false>(Q_i_smem[toc][1], g.Q, {batch_idx, next_q_seq_idx * 2 + 1, next_q_head_idx, 0});
      load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
      load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {0, 0}));
      load(L_i, subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 2));
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);

      // 10. S_ij = Q_i K_j^T * scale
      // 11. P_ij = exp(S_ij - L_i)
      // 13. dP_ij = dO_i @ V_j^T
      // 14. dS_ij = P_ij o (dP_ij - delta_i)
      load(delta_i, subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 2));
      load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {0, 0}));
      mma_ABt(P_ij, Q_i, K_j);
      mul(P_ij, P_ij, scale_factor);
      sub_row(P_ij, P_ij, L_i);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);

      load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {0, 0}));
      mma_ABt(dP_ij, dO_i, V_j);
      load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {0, 0}));
      exp(P_ij, P_ij);
      copy(P_ij_bf16, P_ij);
      sub_row(dP_ij, dP_ij, delta_i);
      mul(dP_ij, dP_ij, scale_factor);
      mul(dP_ij, dP_ij, P_ij);
      copy(dP_ij_bf16, dP_ij);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);

      // 12. dV_j += P_ij^T @ dO_i
      // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
      auto attn_i_smem_subtile = subtile_inplace<WARP_SIZE_KV, DOT_SLICE_QO>(attn_i_smem, {warpid, 0});
      store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
      load(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}));
      swap_layout_inplace(P_ij_bf16_col, P_ij_bf16);
      mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
      swap_layout_inplace(dP_ij_bf16_col, dP_ij_bf16);
      mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);

      load(dP_ij_bf16_col_T, attn_i_smem);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);
    }

    // dot slice 3
    {
      // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
      mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T);
      atomic_pk_add_bf16_with_warpid<2>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4 + 2, 0}, warpid);

      G::load<1, false>(dO_i_smem[toc][1], g.dOg, {batch_idx, next_q_seq_idx * 2 + 1, next_q_head_idx, 0});
      load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
      load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {1, 0}));
      load(L_i, subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 3));
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);

      // 10. S_ij = Q_i K_j^T * scale
      // 11. P_ij = exp(S_ij - L_i)
      // 13. dP_ij = dO_i @ V_j^T
      // 14. dS_ij = P_ij o (dP_ij - delta_i)
      load(delta_i, subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 3));
      load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {1, 0}));
      mma_ABt(P_ij, Q_i, K_j);
      mul(P_ij, P_ij, scale_factor);
      sub_row(P_ij, P_ij, L_i);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);

      load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {1, 0}));
      mma_ABt(dP_ij, dO_i, V_j);
      load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {1, 0}));
      exp(P_ij, P_ij);
      copy(P_ij_bf16, P_ij);
      sub_row(dP_ij, dP_ij, delta_i);
      mul(dP_ij, dP_ij, scale_factor);
      mul(dP_ij, dP_ij, P_ij);
      copy(dP_ij_bf16, dP_ij);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);

      auto attn_i_smem_subtile = subtile_inplace<WARP_SIZE_KV, DOT_SLICE_QO>(attn_i_smem, {warpid, 0});
      store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
      load(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}));
      // 12. dV_j += P_ij^T @ dO_i
      // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
      swap_layout_inplace(P_ij_bf16_col, P_ij_bf16);
      mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
      swap_layout_inplace(dP_ij_bf16_col, dP_ij_bf16);
      mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
      __builtin_amdgcn_s_waitcnt(0);
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);

      load(dP_ij_bf16_col_T, attn_i_smem);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);
    }
  }

  // Epilogue
  {
    const int last_q_head_idx = (num_steps - 2) / num_steps_per_head + first_q_head;
    const int last_q_seq_idx = (num_steps - 2) % num_steps_per_head;

    const int q_head_idx = (num_steps - 1) / num_steps_per_head + first_q_head;
    const int q_seq_idx = (num_steps - 1) % num_steps_per_head;

    // dot slice 0
    {
      // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
      mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T);
      atomic_pk_add_bf16_with_warpid<2>(g.dQg, dQ_i, {batch_idx, last_q_head_idx, last_q_seq_idx * 4 + 3, 0}, warpid);

      load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
      load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
      load(L_i, subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 0));
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);

      // 10. S_ij = Q_i K_j^T * scale
      // 11. P_ij = exp(S_ij - L_i)
      // 13. dP_ij = dO_i @ V_j^T
      // 14. dS_ij = P_ij o (dP_ij - delta_i)
      load(delta_i, subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 0));
      load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
      mma_ABt(P_ij, Q_i, K_j);
      mul(P_ij, P_ij, scale_factor);
      sub_row(P_ij, P_ij, L_i);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);

      load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
      mma_ABt(dP_ij, dO_i, V_j);
      load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
      exp(P_ij, P_ij);
      copy(P_ij_bf16, P_ij);
      sub_row(dP_ij, dP_ij, delta_i);
      mul(dP_ij, dP_ij, scale_factor);
      mul(dP_ij, dP_ij, P_ij);
      copy(dP_ij_bf16, dP_ij);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);

      // 12. dV_j += P_ij^T @ dO_i
      // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
      auto attn_i_smem_subtile = subtile_inplace<WARP_SIZE_KV, DOT_SLICE_QO>(attn_i_smem, {warpid, 0});
      store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
      load(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}));
      swap_layout_inplace(P_ij_bf16_col, P_ij_bf16);
      mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T); 
      swap_layout_inplace(dP_ij_bf16_col, dP_ij_bf16);
      mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);
    
      load(dP_ij_bf16_col_T, attn_i_smem);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);
    }

    // dot slice 1
    {
      // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
      mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T);
      atomic_pk_add_bf16_with_warpid<2>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4, 0}, warpid);

      load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
      load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {1, 0}));
      load(L_i, subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 1));
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);

      // 10. S_ij = Q_i K_j^T * scale
      // 11. P_ij = exp(S_ij - L_i)
      // 13. dP_ij = dO_i @ V_j^T
      // 14. dS_ij = P_ij o (dP_ij - delta_i)
      load(delta_i, subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 1));
      load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {1, 0}));
      mma_ABt(P_ij, Q_i, K_j);
      mul(P_ij, P_ij, scale_factor);
      sub_row(P_ij, P_ij, L_i);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);

      load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {1, 0}));
      mma_ABt(dP_ij, dO_i, V_j);
      load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {1, 0}));
      exp(P_ij, P_ij);
      copy(P_ij_bf16, P_ij);
      sub_row(dP_ij, dP_ij, delta_i);
      mul(dP_ij, dP_ij, scale_factor);
      mul(dP_ij, dP_ij, P_ij);
      copy(dP_ij_bf16, dP_ij);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);

      // 12. dV_j += P_ij^T @ dO_i
      // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
      auto attn_i_smem_subtile = subtile_inplace<WARP_SIZE_KV, DOT_SLICE_QO>(attn_i_smem, {warpid, 0});
      store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
      load(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}));
      swap_layout_inplace(P_ij_bf16_col, P_ij_bf16);
      mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
      swap_layout_inplace(dP_ij_bf16_col, dP_ij_bf16);
      mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);

      load(dP_ij_bf16_col_T, attn_i_smem);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);
    }

    // dot slice 2
    {
      // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
      mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T);
      atomic_pk_add_bf16_with_warpid<2>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4 + 1, 0}, warpid);

      load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
      load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {0, 0}));
      load(L_i, subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 2));
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);

      // 10. S_ij = Q_i K_j^T * scale
      // 11. P_ij = exp(S_ij - L_i)
      // 13. dP_ij = dO_i @ V_j^T
      // 14. dS_ij = P_ij o (dP_ij - delta_i)
      load(delta_i, subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 2));
      load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {0, 0}));
      mma_ABt(P_ij, Q_i, K_j);
      mul(P_ij, P_ij, scale_factor);
      sub_row(P_ij, P_ij, L_i);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);

      load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {0, 0}));
      mma_ABt(dP_ij, dO_i, V_j);
      load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {0, 0}));
      exp(P_ij, P_ij);
      copy(P_ij_bf16, P_ij);
      sub_row(dP_ij, dP_ij, delta_i);
      mul(dP_ij, dP_ij, scale_factor);
      mul(dP_ij, dP_ij, P_ij);
      copy(dP_ij_bf16, dP_ij);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);

      // 12. dV_j += P_ij^T @ dO_i
      // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
      auto attn_i_smem_subtile = subtile_inplace<WARP_SIZE_KV, DOT_SLICE_QO>(attn_i_smem, {warpid, 0});
      store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
      load(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}));
      swap_layout_inplace(P_ij_bf16_col, P_ij_bf16);
      mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
      swap_layout_inplace(dP_ij_bf16_col, dP_ij_bf16);
      mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);

      load(dP_ij_bf16_col_T, attn_i_smem);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);
    }

    // dot slice 3
    {
      // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
      mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T);
      atomic_pk_add_bf16_with_warpid<2>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4 + 2, 0}, warpid);

      load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
      load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {1, 0}));
      load(L_i, subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 3));
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);

      // 10. S_ij = Q_i K_j^T * scale
      // 11. P_ij = exp(S_ij - L_i)
      // 13. dP_ij = dO_i @ V_j^T
      // 14. dS_ij = P_ij o (dP_ij - delta_i)
      load(delta_i, subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 3));
      load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {1, 0}));
      mma_ABt(P_ij, Q_i, K_j);
      mul(P_ij, P_ij, scale_factor);
      sub_row(P_ij, P_ij, L_i);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);

      load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {1, 0}));
      mma_ABt(dP_ij, dO_i, V_j);
      load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {1, 0}));
      exp(P_ij, P_ij);
      copy(P_ij_bf16, P_ij);
      sub_row(dP_ij, dP_ij, delta_i);
      mul(dP_ij, dP_ij, scale_factor);
      mul(dP_ij, dP_ij, P_ij);
      copy(dP_ij_bf16, dP_ij);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);

      auto attn_i_smem_subtile = subtile_inplace<WARP_SIZE_KV, DOT_SLICE_QO>(attn_i_smem, {warpid, 0});
      store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
      load(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}));
      // 12. dV_j += P_ij^T @ dO_i
      // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
      swap_layout_inplace(P_ij_bf16_col, P_ij_bf16);
      mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
      swap_layout_inplace(dP_ij_bf16_col, dP_ij_bf16);
      mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
      __builtin_amdgcn_s_waitcnt(0);
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);

      load(dP_ij_bf16_col_T, attn_i_smem);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_sched_barrier(0);

      mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T);
      atomic_pk_add_bf16_with_warpid<2>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4 + 3, 0}, warpid);
    }
  }

  store<1>(g.dVg, dV_j, {batch_idx, 0, kv_head_idx, 0}, {0, j, 0, 0});
  __builtin_amdgcn_s_waitcnt(0);
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_sched_barrier(0);

  // We first copy dV_j_T from accumulator GPRs to vector GPRs and then perform the store
  accvgpr_read(dV_j_T, dK_j_T);
  store<1>(g.dKg, dV_j, {batch_idx, 0, kv_head_idx, 0}, {0, j, 0, 0});
}
#else
template<int D> __launch_bounds__(NUM_THREADS, 1)
__global__ void attend_bwd_combined_ker(const attn_bwd_combined_globals<D> g) {
    
    const int seq_idx = blockIdx.x;
    const int kv_head_idx = blockIdx.y; // This is the KV head index
    const int batch_idx = blockIdx.z;
    const int first_q_head = kv_head_idx * GROUP_SIZE;

    const int warpid = kittens::warpid();
    const int j = seq_idx * NUM_WARPS + warpid;

    const int num_steps_per_head = ATTN_N / STEP_QO;
    const int num_steps = num_steps_per_head * GROUP_SIZE;
    const float scale_factor = 1.0f / sqrt(D);

    // Shared tiles
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    // K_j_smem is organized in 16x16 tiles
    st_bf<BLOCK_SIZE_KV, D, ducks::st_layout::accumulator, ducks::st_matrix::mfma_16x16x32> (&K_j_smem) = al.allocate<st_bf<BLOCK_SIZE_KV, D, ducks::st_layout::accumulator, ducks::st_matrix::mfma_16x16x32>>();
    st_bf<SLICE_QO, D, ducks::st_layout::classical, ducks::st_matrix::mfma_16x16x32> (&Q_i_smem)[2][2] = al.allocate<st_bf<SLICE_QO, D, ducks::st_layout::classical, ducks::st_matrix::mfma_16x16x32>, 2, 2>();
    st_bf<SLICE_QO, D, ducks::st_layout::classical, ducks::st_matrix::mfma_16x16x32> (&dO_i_smem)[2][2] = al.allocate<st_bf<SLICE_QO, D, ducks::st_layout::classical, ducks::st_matrix::mfma_16x16x32>, 2, 2>();
    // We parameterize this using mfma_32x32x16 because we want the base tile for it to be 32x16. Not that it uses that intrinsic.
    st_bf<BLOCK_SIZE_KV, DOT_SLICE_QO, ducks::st_layout::classical, ducks::st_matrix::mfma_32x32x16> (&attn_i_smem) = al.allocate<st_bf<BLOCK_SIZE_KV, DOT_SLICE_QO, ducks::st_layout::classical, ducks::st_matrix::mfma_32x32x16>>();
    sv_fl<STEP_QO> (&L_smem)[2] = al.allocate<sv_fl<STEP_QO>, 2>();
    sv_fl<STEP_QO> (&delta_smem)[2] = al.allocate<sv_fl<STEP_QO>, 2>();

    // Register tiles
    kv_tile<D, bf16, row_l, mfma_16x16x32> K_j, V_j;
    kv_tile_dq<D, bf16, col_l> K_j_col; // for dq
    qo_tile_T_dq<D, float, accum_col_l> dQ_i_T; // for dq
    kv_tile_T<D, float, accum_col_l, mfma_32x32x16> dK_j_T, dV_j_T;
    qo_tile<D, bf16, row_l, mfma_16x16x32> Q_i, dO_i;
    qo_tile<D, bf16, col_l, mfma_32x32x16> Q_i_col, dO_i_col;
    qo_tile_dq<D, float, accum_row_l> dQ_i;
    attn_tile<D, float, accum_col_l, mfma_16x16x32>::col_vec L_i, delta_i;

    attn_tile<D, float, accum_col_l> P_ij;
    attn_tile<D, bf16, accum_col_l> P_ij_bf16;
    attn_tile<D, float, accum_col_l> dP_ij;
    attn_tile<D, bf16, accum_col_l> dP_ij_bf16;
    attn_tile_T<D, bf16, accum_row_l> dP_ij_bf16_accum_row;

    attn_tile<D, bf16, col_l, mfma_32x32x16> P_ij_bf16_col;
    attn_tile<D, bf16, col_l, mfma_32x32x16> dP_ij_bf16_col;
    attn_tile_T_dq<D, bf16, col_l> dP_ij_bf16_col_T; // for dq

    int tic = 0, toc = 1;
    // Load KV data using the KV head index
    G::load<1, false>(K_j_smem, g.K, {batch_idx, seq_idx, kv_head_idx, 0});
    // 6. Load K_j and V_j from HBM to registers  
    load<1>(V_j, g.V, {batch_idx, j, kv_head_idx, 0});
    // 7. Initialize dK_j = 0 and dV_j = 0
    zero(dK_j_T);
    zero(dV_j_T);

    // Load Q, dO, L, delta for this specific query head
    load(L_smem[tic], g.L_vec, {batch_idx, first_q_head, 0, 0});
    load(delta_smem[tic], g.delta_vec, {batch_idx, first_q_head, 0, 0});
    G::load<1, false>(Q_i_smem[tic][0], g.Q, {batch_idx, 0, first_q_head, 0});
    G::load<1, false>(dO_i_smem[tic][0], g.dOg, {batch_idx, 0, first_q_head, 0});
    G::load<1, false>(Q_i_smem[tic][1], g.Q, {batch_idx, 1, first_q_head, 0});
    G::load<1, false>(dO_i_smem[tic][1], g.dOg, {batch_idx, 1, first_q_head, 0});
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    {
        const int q_head_idx = 0 / num_steps_per_head + first_q_head;
        const int q_seq_idx = 0 % num_steps_per_head;

        const int next_q_head_idx = (0 + 1) / num_steps_per_head + first_q_head;
        const int next_q_seq_idx = (0 + 1) % num_steps_per_head;

        // dot slice 0
        {
            load(L_smem[toc], g.L_vec, {batch_idx, next_q_head_idx, 0, next_q_seq_idx});
            G::load<1, false>(Q_i_smem[toc][0], g.Q, {batch_idx, next_q_seq_idx * 2, next_q_head_idx, 0});
            load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
            load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
            load(L_i, subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 0));
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // 10. S_ij = Q_i K_j^T * scale
            // 11. P_ij = exp(S_ij - L_i)
            // 13. dP_ij = dO_i @ V_j^T
            // 14. dS_ij = P_ij o (dP_ij - delta_i)
            load(delta_i, subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 0));
            load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
            zero(P_ij);
            mma_ABt(P_ij, Q_i, K_j, P_ij);
            mul(P_ij, P_ij, scale_factor);
            sub_row(P_ij, P_ij, L_i);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
            load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
            exp(P_ij, P_ij);
            copy(P_ij_bf16, P_ij);
            zero(dP_ij);
            mma_ABt(dP_ij, dO_i, V_j, dP_ij);
            sub_row(dP_ij, dP_ij, delta_i);
            mul(dP_ij, dP_ij, scale_factor);
            mul(dP_ij, dP_ij, P_ij);
            copy(dP_ij_bf16, dP_ij);
            swap_layout_and_transpose(dP_ij_bf16_accum_row, dP_ij_bf16);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // 12. dV_j += P_ij^T @ dO_i
            // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
            auto attn_i_smem_subtile = subtile_inplace<WARP_SIZE_KV, DOT_SLICE_QO>(attn_i_smem, {warpid, 0});
            store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
            load(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}));
            __builtin_amdgcn_s_setprio(1);
            P_ij_bf16_col = swap_layout_inplace<col_l, mfma_32x32x16>(P_ij_bf16);
            mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
            dP_ij_bf16_col = swap_layout_inplace<col_l, mfma_32x32x16>(dP_ij_bf16);
            mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
            __builtin_amdgcn_s_setprio(0);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            load(dP_ij_bf16_col_T, attn_i_smem);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);
        }
        // dot slice 1
        {
            load(delta_smem[toc], g.delta_vec, {batch_idx, next_q_head_idx, 0, next_q_seq_idx});
            G::load<1, false>(dO_i_smem[toc][0], g.dOg, {batch_idx, next_q_seq_idx * 2, next_q_head_idx, 0});
            load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
            load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {1, 0}));
            load(L_i, subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 1));
            // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
            zero(dQ_i_T);
            mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T,  dQ_i_T);
            swap_layout_and_transpose(dQ_i, dQ_i_T);
            atomic_pk_add_bf16_with_warpid<2>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4, 0}, warpid);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // 10. S_ij = Q_i K_j^T * scale
            // 11. P_ij = exp(S_ij - L_i)
            // 13. dP_ij = dO_i @ V_j^T
            // 14. dS_ij = P_ij o (dP_ij - delta_i)
            load(delta_i, subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 1));
            load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {1, 0}));
            zero(P_ij);
            mma_ABt(P_ij, Q_i, K_j, P_ij);
            mul(P_ij, P_ij, scale_factor);
            sub_row(P_ij, P_ij, L_i);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {1, 0}));
            load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {1, 0}));
            exp(P_ij, P_ij);
            copy(P_ij_bf16, P_ij);
            zero(dP_ij);
            mma_ABt(dP_ij, dO_i, V_j, dP_ij);
            sub_row(dP_ij, dP_ij, delta_i);
            mul(dP_ij, dP_ij, scale_factor);
            mul(dP_ij, dP_ij, P_ij);
            copy(dP_ij_bf16, dP_ij);
            swap_layout_and_transpose(dP_ij_bf16_accum_row, dP_ij_bf16);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // 12. dV_j += P_ij^T @ dO_i
            // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
            auto attn_i_smem_subtile = subtile_inplace<WARP_SIZE_KV, DOT_SLICE_QO>(attn_i_smem, {warpid, 0});
            store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
            load(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}));
            P_ij_bf16_col = swap_layout_inplace<col_l, mfma_32x32x16>(P_ij_bf16);
            mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
            dP_ij_bf16_col = swap_layout_inplace<col_l, mfma_32x32x16>(dP_ij_bf16);
            mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            load(dP_ij_bf16_col_T, attn_i_smem);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);
        }
        // dot slice 2
        {
            G::load<1, false>(Q_i_smem[toc][1], g.Q, {batch_idx, next_q_seq_idx * 2 + 1, next_q_head_idx, 0});
            load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
            load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {0, 0}));
            load(L_i, subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 2));
            // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
            zero(dQ_i_T);
            mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T,  dQ_i_T);
            swap_layout_and_transpose(dQ_i, dQ_i_T);
            atomic_pk_add_bf16_with_warpid<2>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4 + 1, 0}, warpid);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // 10. S_ij = Q_i K_j^T * scale
            // 11. P_ij = exp(S_ij - L_i)
            // 13. dP_ij = dO_i @ V_j^T
            // 14. dS_ij = P_ij o (dP_ij - delta_i)
            load(delta_i, subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 2));
            load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {0, 0}));
            zero(P_ij);
            mma_ABt(P_ij, Q_i, K_j, P_ij);
            mul(P_ij, P_ij, scale_factor);
            sub_row(P_ij, P_ij, L_i);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {0, 0}));
            load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {0, 0}));
            exp(P_ij, P_ij);
            copy(P_ij_bf16, P_ij);
            zero(dP_ij);
            mma_ABt(dP_ij, dO_i, V_j, dP_ij);
            sub_row(dP_ij, dP_ij, delta_i);
            mul(dP_ij, dP_ij, scale_factor);
            mul(dP_ij, dP_ij, P_ij);
            copy(dP_ij_bf16, dP_ij);
            swap_layout_and_transpose(dP_ij_bf16_accum_row, dP_ij_bf16);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // 12. dV_j += P_ij^T @ dO_i
            // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
            auto attn_i_smem_subtile = subtile_inplace<WARP_SIZE_KV, DOT_SLICE_QO>(attn_i_smem, {warpid, 0});
            store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
            load(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}));
            P_ij_bf16_col = swap_layout_inplace<col_l, mfma_32x32x16>(P_ij_bf16);
            mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
            dP_ij_bf16_col = swap_layout_inplace<col_l, mfma_32x32x16>(dP_ij_bf16);
            mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
            // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            load(dP_ij_bf16_col_T, attn_i_smem);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);
        }
        // dot slice 3
        {
            G::load<1, false>(dO_i_smem[toc][1], g.dOg, {batch_idx, next_q_seq_idx * 2 + 1, next_q_head_idx, 0});
            load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
            load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {1, 0}));
            load(L_i, subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 3));
            zero(dQ_i_T);
            mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T,  dQ_i_T);
            swap_layout_and_transpose(dQ_i, dQ_i_T);
            atomic_pk_add_bf16_with_warpid<2>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4 + 2, 0}, warpid);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // 10. S_ij = Q_i K_j^T * scale
            // 11. P_ij = exp(S_ij - L_i)
            // 13. dP_ij = dO_i @ V_j^T
            // 14. dS_ij = P_ij o (dP_ij - delta_i)
            load(delta_i, subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 3));
            load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {1, 0}));
            zero(P_ij);
            mma_ABt(P_ij, Q_i, K_j, P_ij);
            mul(P_ij, P_ij, scale_factor);
            sub_row(P_ij, P_ij, L_i);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {1, 0}));
            load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {1, 0}));
            exp(P_ij, P_ij);
            copy(P_ij_bf16, P_ij);
            zero(dP_ij);
            mma_ABt(dP_ij, dO_i, V_j, dP_ij);
            sub_row(dP_ij, dP_ij, delta_i);
            mul(dP_ij, dP_ij, scale_factor);
            mul(dP_ij, dP_ij, P_ij);
            copy(dP_ij_bf16, dP_ij);
            swap_layout_and_transpose(dP_ij_bf16_accum_row, dP_ij_bf16);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            auto attn_i_smem_subtile = subtile_inplace<WARP_SIZE_KV, DOT_SLICE_QO>(attn_i_smem, {warpid, 0});
            store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
            load(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}));
            // 12. dV_j += P_ij^T @ dO_i
            // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
            P_ij_bf16_col = swap_layout_inplace<col_l, mfma_32x32x16>(P_ij_bf16);
            mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
            dP_ij_bf16_col = swap_layout_inplace<col_l, mfma_32x32x16>(dP_ij_bf16);
            mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
            // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
            __builtin_amdgcn_s_waitcnt(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            load(dP_ij_bf16_col_T, attn_i_smem);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);
        }
        tic ^= 1; toc ^= 1;
    }

    // 9. for 1 <= i <= T_r (1024 / 32 = 32)  
    for (int i = 1; i < num_steps - 1; ++i, tic ^= 1, toc ^= 1) {
        const int last_q_head_idx = (i - 1) / num_steps_per_head + first_q_head;
        const int last_q_seq_idx = (i - 1) % num_steps_per_head;

        const int q_head_idx = i / num_steps_per_head + first_q_head;
        const int q_seq_idx = i % num_steps_per_head;

        const int next_q_head_idx = (i + 1) / num_steps_per_head + first_q_head;
        const int next_q_seq_idx = (i + 1) % num_steps_per_head;

        // dot slice 0
        {
            load(L_smem[toc], g.L_vec, {batch_idx, next_q_head_idx, 0, next_q_seq_idx});
            G::load<1, false>(Q_i_smem[toc][0], g.Q, {batch_idx, next_q_seq_idx * 2, next_q_head_idx, 0});
            load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
            load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
            load(L_i, subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 0));
            zero(dQ_i_T);
            mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T,  dQ_i_T);
            swap_layout_and_transpose(dQ_i, dQ_i_T);
            atomic_pk_add_bf16_with_warpid<2>(g.dQg, dQ_i, {batch_idx, last_q_head_idx, last_q_seq_idx * 4 + 3, 0}, warpid);
            __builtin_amdgcn_s_waitcnt(0xc07f);
            // __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // 10. S_ij = Q_i K_j^T * scale
            // 11. P_ij = exp(S_ij - L_i)
            // 13. dP_ij = dO_i @ V_j^T
            // 14. dS_ij = P_ij o (dP_ij - delta_i)
            load(delta_i, subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 0));
            load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
            zero(P_ij);
            mma_ABt(P_ij, Q_i, K_j, P_ij);
            mul(P_ij, P_ij, scale_factor);
            sub_row(P_ij, P_ij, L_i);
            __builtin_amdgcn_s_waitcnt(0xc07f);
            // __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
            load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
            exp(P_ij, P_ij);
            copy(P_ij_bf16, P_ij);
            zero(dP_ij);
            mma_ABt(dP_ij, dO_i, V_j, dP_ij);
            sub_row(dP_ij, dP_ij, delta_i);
            mul(dP_ij, dP_ij, scale_factor);
            __builtin_amdgcn_s_waitcnt(0xc07f);
            // __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // 12. dV_j += P_ij^T @ dO_i
            // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
            load(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}));
            mul(dP_ij, dP_ij, P_ij);
            copy(dP_ij_bf16, dP_ij);
            swap_layout_and_transpose(dP_ij_bf16_accum_row, dP_ij_bf16);
            auto attn_i_smem_subtile = subtile_inplace<WARP_SIZE_KV, DOT_SLICE_QO>(attn_i_smem, {warpid, 0});
            store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
            P_ij_bf16_col = swap_layout_inplace<col_l, mfma_32x32x16>(P_ij_bf16);
            mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
            dP_ij_bf16_col = swap_layout_inplace<col_l, mfma_32x32x16>(dP_ij_bf16);
            mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
            __builtin_amdgcn_s_waitcnt(0xc07f);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            load(dP_ij_bf16_col_T, attn_i_smem);
            __builtin_amdgcn_s_waitcnt(0xc07f);
            // __builtin_amdgcn_s_barrier();
            // __builtin_amdgcn_sched_barrier(0);
            // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
        }

        // dot slice 1
        {
            load(delta_smem[toc], g.delta_vec, {batch_idx, next_q_head_idx, 0, next_q_seq_idx});
            G::load<1, false>(dO_i_smem[toc][0], g.dOg, {batch_idx, next_q_seq_idx * 2, next_q_head_idx, 0});
            load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
            load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {1, 0}));
            load(L_i, subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 1));
            zero(dQ_i_T);
            mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T,  dQ_i_T);
            swap_layout_and_transpose(dQ_i, dQ_i_T);
            atomic_pk_add_bf16_with_warpid<2>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4, 0}, warpid);
            __builtin_amdgcn_s_waitcnt(0xc07f);
            // __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // 10. S_ij = Q_i K_j^T * scale
            // 11. P_ij = exp(S_ij - L_i)
            // 13. dP_ij = dO_i @ V_j^T
            // 14. dS_ij = P_ij o (dP_ij - delta_i)
            load(delta_i, subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 1));
            load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {1, 0}));
            zero(P_ij);
            mma_ABt(P_ij, Q_i, K_j, P_ij);
            mul(P_ij, P_ij, scale_factor);
            sub_row(P_ij, P_ij, L_i);
            __builtin_amdgcn_s_waitcnt(0xc07f);
            // __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {1, 0}));
            load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {1, 0}));
            exp(P_ij, P_ij);
            copy(P_ij_bf16, P_ij);
            zero(dP_ij);
            mma_ABt(dP_ij, dO_i, V_j, dP_ij);
            sub_row(dP_ij, dP_ij, delta_i);
            mul(dP_ij, dP_ij, scale_factor);
            __builtin_amdgcn_s_waitcnt(0xc07f);
            // __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            load(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}));
            mul(dP_ij, dP_ij, P_ij);
            copy(dP_ij_bf16, dP_ij);
            swap_layout_and_transpose(dP_ij_bf16_accum_row, dP_ij_bf16);
            auto attn_i_smem_subtile = subtile_inplace<WARP_SIZE_KV, DOT_SLICE_QO>(attn_i_smem, {warpid, 0});
            store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
            // 12. dV_j += P_ij^T @ dO_i
            // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
            P_ij_bf16_col = swap_layout_inplace<col_l, mfma_32x32x16>(P_ij_bf16);
            mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
            dP_ij_bf16_col = swap_layout_inplace<col_l, mfma_32x32x16>(dP_ij_bf16);
            mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
            __builtin_amdgcn_s_waitcnt(0xc07f);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
            load(dP_ij_bf16_col_T, attn_i_smem);
            __builtin_amdgcn_s_waitcnt(0xc07f);
            // __builtin_amdgcn_s_barrier();
            // __builtin_amdgcn_sched_barrier(0);
        }

        // dot slice 2
        {
            G::load<1, false>(Q_i_smem[toc][1], g.Q, {batch_idx, next_q_seq_idx * 2 + 1, next_q_head_idx, 0});
            load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
            load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {0, 0}));
            load(L_i, subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 2));
            zero(dQ_i_T);
            mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T,  dQ_i_T);
            swap_layout_and_transpose(dQ_i, dQ_i_T);
            atomic_pk_add_bf16_with_warpid<2>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4 + 1, 0}, warpid);
            __builtin_amdgcn_s_waitcnt(0xc07f);
            // __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // 10. S_ij = Q_i K_j^T * scale
            // 11. P_ij = exp(S_ij - L_i)
            // 13. dP_ij = dO_i @ V_j^T
            // 14. dS_ij = P_ij o (dP_ij - delta_i)
            load(delta_i, subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 2));
            load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {0, 0}));
            zero(P_ij);
            mma_ABt(P_ij, Q_i, K_j, P_ij);
            mul(P_ij, P_ij, scale_factor);
            sub_row(P_ij, P_ij, L_i);
            __builtin_amdgcn_s_waitcnt(0xc07f);
            // __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {0, 0}));
            load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {0, 0}));
            exp(P_ij, P_ij);
            copy(P_ij_bf16, P_ij);
            zero(dP_ij);
            mma_ABt(dP_ij, dO_i, V_j, dP_ij);
            sub_row(dP_ij, dP_ij, delta_i);
            mul(dP_ij, dP_ij, scale_factor);
            __builtin_amdgcn_s_waitcnt(0xc07f);
            // __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            load(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}));
            mul(dP_ij, dP_ij, P_ij);
            copy(dP_ij_bf16, dP_ij);
            swap_layout_and_transpose(dP_ij_bf16_accum_row, dP_ij_bf16);
            auto attn_i_smem_subtile = subtile_inplace<WARP_SIZE_KV, DOT_SLICE_QO>(attn_i_smem, {warpid, 0});
            store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
            // 12. dV_j += P_ij^T @ dO_i            
            // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
            P_ij_bf16_col = swap_layout_inplace<col_l, mfma_32x32x16>(P_ij_bf16);
            mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
            dP_ij_bf16_col = swap_layout_inplace<col_l, mfma_32x32x16>(dP_ij_bf16);
            mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
            __builtin_amdgcn_s_waitcnt(0xc07f);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
            load(dP_ij_bf16_col_T, attn_i_smem);
            __builtin_amdgcn_s_waitcnt(0xc07f);
            // __builtin_amdgcn_s_barrier();
            // __builtin_amdgcn_sched_barrier(0);
        }

        // dot slice 3
        {
            G::load<1, false>(dO_i_smem[toc][1], g.dOg, {batch_idx, next_q_seq_idx * 2 + 1, next_q_head_idx, 0});
            load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
            load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {1, 0}));
            load(L_i, subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 3));
            zero(dQ_i_T);
            mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T,  dQ_i_T);
            swap_layout_and_transpose(dQ_i, dQ_i_T);
            atomic_pk_add_bf16_with_warpid<2>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4 + 2, 0}, warpid);
            __builtin_amdgcn_s_waitcnt(0xc07f);
            // __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // 10. S_ij = Q_i K_j^T * scale
            // 11. P_ij = exp(S_ij - L_i)
            // 13. dP_ij = dO_i @ V_j^T
            // 14. dS_ij = P_ij o (dP_ij - delta_i)
            load(delta_i, subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 3));
            load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {1, 0}));
            zero(P_ij);
            mma_ABt(P_ij, Q_i, K_j, P_ij);
            mul(P_ij, P_ij, scale_factor);
            sub_row(P_ij, P_ij, L_i);
            __builtin_amdgcn_s_waitcnt(0xc07f);
            // __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {1, 0}));
            load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {1, 0}));
            exp(P_ij, P_ij);
            copy(P_ij_bf16, P_ij);
            zero(dP_ij);
            mma_ABt(dP_ij, dO_i, V_j, dP_ij);
            sub_row(dP_ij, dP_ij, delta_i);
            mul(dP_ij, dP_ij, scale_factor);
            __builtin_amdgcn_s_waitcnt(0xc07f);
            // __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            load(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}));
            mul(dP_ij, dP_ij, P_ij);
            copy(dP_ij_bf16, dP_ij);
            swap_layout_and_transpose(dP_ij_bf16_accum_row, dP_ij_bf16);
            auto attn_i_smem_subtile = subtile_inplace<WARP_SIZE_KV, DOT_SLICE_QO>(attn_i_smem, {warpid, 0});
            store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
            // 12. dV_j += P_ij^T @ dO_i
            // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
            P_ij_bf16_col = swap_layout_inplace<col_l, mfma_32x32x16>(P_ij_bf16);
            mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
            dP_ij_bf16_col = swap_layout_inplace<col_l, mfma_32x32x16>(dP_ij_bf16);
            mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
            __builtin_amdgcn_s_waitcnt(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
            load(dP_ij_bf16_col_T, attn_i_smem);
            __builtin_amdgcn_s_waitcnt(0xc07f);
            // __builtin_amdgcn_s_barrier();
            // __builtin_amdgcn_sched_barrier(0);
        }
    }

    const int last_q_head_idx = (num_steps - 2) / num_steps_per_head + first_q_head;
    const int last_q_seq_idx = (num_steps - 2) % num_steps_per_head;

    const int q_head_idx = (num_steps - 1) / num_steps_per_head + first_q_head;
    const int q_seq_idx = (num_steps - 1) % num_steps_per_head;

    // Sequence Epilogue
    // dot slice 0
    {
        load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
        load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
        load(L_i, subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 0));
        zero(dQ_i_T);
        mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T,  dQ_i_T);
        swap_layout_and_transpose(dQ_i, dQ_i_T);
        atomic_pk_add_bf16_with_warpid<2>(g.dQg, dQ_i, {batch_idx, last_q_head_idx, last_q_seq_idx * 4 + 3, 0}, warpid);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // 10. S_ij = Q_i K_j^T * scale
        // 11. P_ij = exp(S_ij - L_i)
        // 13. dP_ij = dO_i @ V_j^T
        // 14. dS_ij = P_ij o (dP_ij - delta_i)
        load(delta_i, subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 0));
        load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
        zero(P_ij);
        mma_ABt(P_ij, Q_i, K_j, P_ij);
        mul(P_ij, P_ij, scale_factor);
        sub_row(P_ij, P_ij, L_i);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
        load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
        exp(P_ij, P_ij);
        copy(P_ij_bf16, P_ij);
        zero(dP_ij);
        mma_ABt(dP_ij, dO_i, V_j, dP_ij);
        sub_row(dP_ij, dP_ij, delta_i);
        mul(dP_ij, dP_ij, scale_factor);
        mul(dP_ij, dP_ij, P_ij);
        copy(dP_ij_bf16, dP_ij);
        swap_layout_and_transpose(dP_ij_bf16_accum_row, dP_ij_bf16);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // 12. dV_j += P_ij^T @ dO_i
        // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
        auto attn_i_smem_subtile = subtile_inplace<WARP_SIZE_KV, DOT_SLICE_QO>(attn_i_smem, {warpid, 0});
        store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
        load(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}));
        P_ij_bf16_col = swap_layout_inplace<col_l, mfma_32x32x16>(P_ij_bf16);
        mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        dP_ij_bf16_col = swap_layout_inplace<col_l, mfma_32x32x16>(dP_ij_bf16);
        mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
        load(dP_ij_bf16_col_T, attn_i_smem);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    }
    // dot slice 1
    {
        load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
        load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {1, 0}));
        load(L_i, subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 1));
        zero(dQ_i_T);
        mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T,  dQ_i_T);
        swap_layout_and_transpose(dQ_i, dQ_i_T);
        atomic_pk_add_bf16_with_warpid<2>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4, 0}, warpid);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // 10. S_ij = Q_i K_j^T * scale
        // 11. P_ij = exp(S_ij - L_i)
        // 13. dP_ij = dO_i @ V_j^T
        // 14. dS_ij = P_ij o (dP_ij - delta_i)
        load(delta_i, subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 1));
        load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {1, 0}));
        zero(P_ij);
        mma_ABt(P_ij, Q_i, K_j, P_ij);
        mul(P_ij, P_ij, scale_factor);
        sub_row(P_ij, P_ij, L_i);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {1, 0}));
        load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {1, 0}));
        exp(P_ij, P_ij);
        copy(P_ij_bf16, P_ij);
        zero(dP_ij);
        mma_ABt(dP_ij, dO_i, V_j, dP_ij);
        sub_row(dP_ij, dP_ij, delta_i);
        mul(dP_ij, dP_ij, scale_factor);
        mul(dP_ij, dP_ij, P_ij);
        copy(dP_ij_bf16, dP_ij);
        swap_layout_and_transpose(dP_ij_bf16_accum_row, dP_ij_bf16);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
        
        // 12. dV_j += P_ij^T @ dO_i
        // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
        auto attn_i_smem_subtile = subtile_inplace<WARP_SIZE_KV, DOT_SLICE_QO>(attn_i_smem, {warpid, 0});
        store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
        load(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}));
        P_ij_bf16_col = swap_layout_inplace<col_l, mfma_32x32x16>(P_ij_bf16);
        mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        dP_ij_bf16_col = swap_layout_inplace<col_l, mfma_32x32x16>(dP_ij_bf16);
        mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
        load(dP_ij_bf16_col_T, attn_i_smem);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    }
    // dot slice 2
    {
        load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
        load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {0, 0}));
        load(L_i, subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 2));
        zero(dQ_i_T);
        mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T,  dQ_i_T);
        swap_layout_and_transpose(dQ_i, dQ_i_T);
        atomic_pk_add_bf16_with_warpid<2>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4 + 1, 0}, warpid);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // 10. S_ij = Q_i K_j^T * scale
        // 11. P_ij = exp(S_ij - L_i)
        // 13. dP_ij = dO_i @ V_j^T
        // 14. dS_ij = P_ij o (dP_ij - delta_i)
        load(delta_i, subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 2));
        load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {0, 0}));
        zero(P_ij);
        mma_ABt(P_ij, Q_i, K_j, P_ij);
        mul(P_ij, P_ij, scale_factor);
        sub_row(P_ij, P_ij, L_i);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {0, 0}));
        load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {0, 0}));
        exp(P_ij, P_ij);
        copy(P_ij_bf16, P_ij);
        zero(dP_ij);
        mma_ABt(dP_ij, dO_i, V_j, dP_ij);
        sub_row(dP_ij, dP_ij, delta_i);
        mul(dP_ij, dP_ij, scale_factor);
        mul(dP_ij, dP_ij, P_ij);
        copy(dP_ij_bf16, dP_ij);
        swap_layout_and_transpose(dP_ij_bf16_accum_row, dP_ij_bf16);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // 12. dV_j += P_ij^T @ dO_i
        // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
        auto attn_i_smem_subtile = subtile_inplace<WARP_SIZE_KV, DOT_SLICE_QO>(attn_i_smem, {warpid, 0});
        store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
        load(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}));
        P_ij_bf16_col = swap_layout_inplace<col_l, mfma_32x32x16>(P_ij_bf16);
        mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        dP_ij_bf16_col = swap_layout_inplace<col_l, mfma_32x32x16>(dP_ij_bf16);
        mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
        load(dP_ij_bf16_col_T, attn_i_smem);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    }
    // dot slice 3
    {
        load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
        load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {1, 0}));
        load(L_i, subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 3));
        zero(dQ_i_T);
        mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T,  dQ_i_T);
        swap_layout_and_transpose(dQ_i, dQ_i_T);
        atomic_pk_add_bf16_with_warpid<2>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4 + 2, 0}, warpid);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // 10. S_ij = Q_i K_j^T * scale
        // 11. P_ij = exp(S_ij - L_i)
        // 13. dP_ij = dO_i @ V_j^T
        // 14. dS_ij = P_ij o (dP_ij - delta_i)
        load(delta_i, subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 3));
        load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {1, 0}));
        zero(P_ij);
        mma_ABt(P_ij, Q_i, K_j, P_ij);
        mul(P_ij, P_ij, scale_factor);
        sub_row(P_ij, P_ij, L_i);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {1, 0}));
        load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {1, 0}));
        exp(P_ij, P_ij);
        copy(P_ij_bf16, P_ij);
        zero(dP_ij);
        mma_ABt(dP_ij, dO_i, V_j, dP_ij);
        sub_row(dP_ij, dP_ij, delta_i);
        mul(dP_ij, dP_ij, scale_factor);
        mul(dP_ij, dP_ij, P_ij);
        copy(dP_ij_bf16, dP_ij);
        swap_layout_and_transpose(dP_ij_bf16_accum_row, dP_ij_bf16);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
        
        // 12. dV_j += P_ij^T @ dO_i
        // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
        auto attn_i_smem_subtile = subtile_inplace<WARP_SIZE_KV, DOT_SLICE_QO>(attn_i_smem, {warpid, 0});
        store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
        load(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}));
        P_ij_bf16_col = swap_layout_inplace<col_l, mfma_32x32x16>(P_ij_bf16);
        mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        dP_ij_bf16_col = swap_layout_inplace<col_l, mfma_32x32x16>(dP_ij_bf16);
        mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
        load(dP_ij_bf16_col_T, attn_i_smem);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        zero(dQ_i_T);
        mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T,  dQ_i_T);
        swap_layout_and_transpose(dQ_i, dQ_i_T);
        atomic_pk_add_bf16_with_warpid<2>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4 + 3, 0}, warpid);
    }

    // 18. Write dK_j and dV_j back to HBM (using KV head index)
    kv_tile<D, float, accum_row_l, mfma_32x32x16> dK_j, dV_j;
    swap_layout_and_transpose(dK_j, dK_j_T);
    swap_layout_and_transpose(dV_j, dV_j_T);
    store<1>(g.dKg, dK_j, {batch_idx, j, kv_head_idx, 0});
    store<1>(g.dVg, dV_j, {batch_idx, j, kv_head_idx, 0});
}
#endif

template<int D>
void dispatch_bwd_combined(attn_bwd_combined_globals<D> g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)attend_bwd_combined_ker<D>, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    attend_bwd_combined_ker<D><<<g.grid(), g.block(), mem_size>>>(g);
    hipDeviceSynchronize();
}

template<int D> struct attn_dq_shuffle_globals { 
  gl<bf16, -1, -1, -1, -1> dQg_in, dQg_out;
  dim3 grid() { return dim3(ATTN_B, ATTN_H_Q, ATTN_N / (DOT_SLICE_QO * NUM_WARPS)); }
  dim3 block() { return dim3(NUM_THREADS); }
  size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

template<int D> __launch_bounds__(NUM_THREADS, 1)
__global__ void attend_dq_shuffle_ker(const attn_dq_shuffle_globals<D> g) {
    
    const int batch_idx = blockIdx.x;
    const int q_head_idx = blockIdx.y; // Using Q head index for dQ shuffle
    const int seq_idx = blockIdx.z;

    const int warpid = kittens::warpid();

    qo_tile<D, bf16, accum_row_l> dQg;

    load_shuffled<2>(dQg, g.dQg_in, {batch_idx, q_head_idx, seq_idx * NUM_WARPS + warpid, 0});
    store<1>(g.dQg_out, dQg, {batch_idx, seq_idx * NUM_WARPS + warpid, q_head_idx, 0});
}

template<int D>
void dispatch_dq_shuffle(attn_dq_shuffle_globals<D> g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)attend_dq_shuffle_ker<D>, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    attend_dq_shuffle_ker<D><<<g.grid(), g.block(), mem_size>>>(g);
    hipDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernel_bkwd, m) {
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
      &attn_bwd_combined_globals<ATTN_D>::dOg, 
      &attn_bwd_combined_globals<ATTN_D>::dQg,
      &attn_bwd_combined_globals<ATTN_D>::dKg,
      &attn_bwd_combined_globals<ATTN_D>::dVg,
      &attn_bwd_combined_globals<ATTN_D>::L_vec, 
      &attn_bwd_combined_globals<ATTN_D>::delta_vec
  );

  py::bind_function<dispatch_dq_shuffle<ATTN_D>>(m, "dispatch_dq_shuffle", 
      &attn_dq_shuffle_globals<ATTN_D>::dQg_in,
      &attn_dq_shuffle_globals<ATTN_D>::dQg_out
  );
}


