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

using G = kittens::group<NUM_WARPS>;

using namespace kittens;

template<int D> struct attn_bwd_combined_globals { 
  gl<bf16, -1, -1, -1, -1> Q, K, V;
  gl<bf16, -1, -1, -1, -1> dOg, dQg, dKg, dVg;
  gl<float, -1, -1, -1, -1> L_vec, delta_vec;
  dim3 grid() { return dim3((ATTN_N / BLOCK_SIZE_KV), ATTN_H_KV, ATTN_B); }
  dim3 block() { return dim3(NUM_THREADS); }
  size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};


template<int D> __launch_bounds__(NUM_THREADS, 1)
__global__ __attribute__((amdgpu_num_vgpr(30))) void attend_bwd_combined_ker(const attn_bwd_combined_globals<D> g) {

  const int seq_idx = blockIdx.x;
  const int kv_head_idx = blockIdx.y; // This is the KV head index
  const int batch_idx = blockIdx.z;
  const int first_q_head = kv_head_idx * GROUP_SIZE;

  const int warpid = kittens::warpid();
  const int j = seq_idx * NUM_WARPS + warpid;

  const int num_steps_per_head = ATTN_N / STEP_QO;
  const int num_steps = num_steps_per_head * GROUP_SIZE;

  constexpr float L_SCALE_FACTOR = 1.44269504089f;
  constexpr float P_SCALE_FACTOR = (D == 128) ? 0.08838834764f*1.44269504089f : 0.125f*1.44269504089f;
  constexpr float dP_SCALE_FACTOR = (D == 128) ? 0.08838834764f : 0.125f;

  // Shared tiles
  extern __shared__ alignment_dummy __shm[];
  shared_allocator al((int*)&__shm[0]);

  st_bf<BLOCK_SIZE_KV, D, st_16x16_s> (&K_j_smem) = al.allocate<st_bf<BLOCK_SIZE_KV, D, st_16x16_s>>();
  st_bf<SLICE_QO, D, st_16x32_s> (&Q_i_smem)[2][2] = al.allocate<st_bf<SLICE_QO, D, st_16x32_s>, 2, 2>();
  st_bf<SLICE_QO, D, st_16x32_s> (&dO_i_smem)[2][2] = al.allocate<st_bf<SLICE_QO, D, st_16x32_s>, 2, 2>();
  st_bf<BLOCK_SIZE_KV, DOT_SLICE_QO, st_16x16_swizzled_s> (&attn_i_smem) = al.allocate<st_bf<BLOCK_SIZE_KV, DOT_SLICE_QO, st_16x16_swizzled_s>>();
  sv_fl<STEP_QO> (&L_smem)[2] = al.allocate<sv_fl<STEP_QO>, 2>();
  sv_fl<STEP_QO> (&delta_smem)[2] = al.allocate<sv_fl<STEP_QO>, 2>();

  // Register tiles
  using Q_ranges = ducks::rt::split_many_t<ducks::rt::type_list<ducks::rt::range<368, 383>>, 4>; // 16 registers - a[112:127]
  using dO_ranges = ducks::rt::split_many_t<ducks::rt::type_list<ducks::rt::range<78, 93>>, 4>; // 16 registers - v[72:87]
  using dO_col_ranges = ducks::rt::split_many_t<ducks::rt::type_list<ducks::rt::range<94, 109>>, 4>; // 16 registers - v[88:103]
  using K_ranges = ducks::rt::split_many_t<ducks::rt::type_list<ducks::rt::range<256, 303>, ducks::rt::range<62, 77>>, 4>; // 64 registers - a[0:47] & v[56:71]
  using V_ranges = ducks::rt::split_many_t<ducks::rt::type_list<ducks::rt::range<304, 367>>, 4>; // 64 registers - a[48:111]
  using P_ranges = ducks::rt::split_many_t<ducks::rt::type_list<ducks::rt::range<46, 61>>, 4>; // 16 registers - v[40:55]
  using dP_ranges = ducks::rt::split_many_t<ducks::rt::type_list<ducks::rt::range<62, 77>>, 4>; // 16 registers - v[56:71]
  using P_bf16_ranges = ducks::rt::split_many_t<ducks::rt::type_list<ducks::rt::range<118, 125>>, 2>; // 8 registers - v[116:123]
  using dP_bf16_ranges = ducks::rt::split_many_t<ducks::rt::type_list<ducks::rt::range<62, 69>>, 2>; // 8 registers - v[56:63]
  using P_bf16_col_ranges = ducks::rt::split_many_t<ducks::rt::type_list<ducks::rt::range<118, 125>>, 4>; // 8 registers
  using dP_bf16_col_ranges = ducks::rt::split_many_t<ducks::rt::type_list<ducks::rt::range<62, 69>>, 4>; // 8 registers
  using dS_ranges = ducks::rt::split_many_t<ducks::rt::type_list<ducks::rt::range<30, 61>>, 4>; // 32 registers - v[24:55]
  using dQ_ranges = ducks::rt::split_many_t<ducks::rt::type_list<ducks::rt::range<110, 117>>, 4>; // 8 registers - v[108:115]
  ducks::rt::clobber<Q_ranges>();
  ducks::rt::clobber<dO_ranges>();
  ducks::rt::clobber<dO_col_ranges>();
  ducks::rt::clobber<K_ranges>();
  ducks::rt::clobber<V_ranges>();
  ducks::rt::clobber<P_ranges>();
  ducks::rt::clobber<dP_ranges>();
  ducks::rt::clobber<P_bf16_ranges>();
  ducks::rt::clobber<dP_bf16_ranges>();
  ducks::rt::clobber<dS_ranges>();
  ducks::rt::clobber<dQ_ranges>();


  using dV_ranges = ducks::rt::split_many_t<ducks::rt::type_list<ducks::rt::range<128, 255>>, 16>; // 128 registers v[128:255]
  using dK_ranges = ducks::rt::split_many_t<ducks::rt::type_list<ducks::rt::range<384, 511>>, 16>; // 128 registers a[128:255]
  ducks::rt::clobber<dV_ranges>();
  ducks::rt::clobber<dK_ranges>();

  rt<bf16, DOT_SLICE_QO, D, row_l, rt_16x32_s, Q_ranges> Q_i; // 16 registers
  rt<bf16, DOT_SLICE_QO, D, row_l, rt_16x32_s, dO_ranges> dO_i; // 16 registers
  rt<bf16, DOT_SLICE_QO, D, col_l, rt_16x32_s, Q_ranges> Q_i_col; // 16 registers
  rt<bf16, DOT_SLICE_QO, D, col_l, rt_16x32_s, dO_col_ranges> dO_i_col; // 16 registers
  rt<bf16, WARP_SIZE_KV, D, row_l, rt_16x32_s, K_ranges> K_j; // 64 registers
  rt<bf16, WARP_SIZE_KV, D, row_l, rt_16x32_s, V_ranges> V_j; // 64 registers
  constexpr int L_i = 126;
  constexpr int delta_i = 127;

  rt<float, DOT_SLICE_QO, WARP_SIZE_KV, col_l, rt_16x16_s, P_ranges> P_ij; // 16 registers
  rt<float, DOT_SLICE_QO, WARP_SIZE_KV, col_l, rt_16x16_s, dP_ranges> dP_ij; // 16 registers
  rt<bf16, DOT_SLICE_QO, WARP_SIZE_KV, col_l, rt_16x16_s, P_bf16_ranges> P_ij_bf16; // 8 registers
  rt<bf16, DOT_SLICE_QO, WARP_SIZE_KV, col_l, rt_16x16_s, dP_bf16_ranges> dP_ij_bf16; // 8 registers
  rt<bf16, WARP_SIZE_KV, DOT_SLICE_QO, row_l, rt_16x16_s, ducks::rt::transpose_2d<dP_bf16_ranges, 1, 4>> dP_ij_bf16_accum_row; // 8 registers

  rt<bf16, DOT_SLICE_QO, WARP_SIZE_KV, col_l, rt_16x32_s, P_bf16_col_ranges> P_ij_bf16_col; // 8 registers
  rt<bf16, DOT_SLICE_QO, WARP_SIZE_KV, col_l, rt_16x32_s, dP_bf16_col_ranges> dP_ij_bf16_col; // 8 registers

  rt<bf16, 256, 32, col_l, rt_32x16_4_s, K_ranges> K_j_col; // 64 registers // for dq
  rt<bf16, 256, 16, col_l, rt_32x16_4_s, dS_ranges> dP_ij_bf16_col_T; // 32 registers // for dq

  rt<float, D, WARP_SIZE_KV, col_l, rt_32x32_s, dK_ranges> dK_j_T; // 128 registers
  rt<float, D, WARP_SIZE_KV, col_l, rt_32x32_s, dV_ranges> dV_j_T; // 128 registers
  rt<float, 32, 16, col_l, rt_16x16_s, dQ_ranges> dQ_i_T; // 8 registers // for dq
  rt<float, 16, 32, row_l, rt_16x16_s, ducks::rt::transpose_2d<dQ_ranges, 2, 1>> dQ_i; // 8 registers // for dq

  // This is used for both dK_j_T and dV_j_T
  rt<float, WARP_SIZE_KV, D, row_l, rt_32x32_s, ducks::rt::transpose_2d<dV_ranges, 4, 2>> dV_j;

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
      load<L_i>(subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 0));
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();

      // 10. S_ij = Q_i K_j^T * scale
      // 11. P_ij = exp(S_ij - L_i)
      // 13. dP_ij = dO_i @ V_j^T
      // 14. dS_ij = P_ij o (dP_ij - delta_i)
      load<delta_i>(subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 0));
      load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
      mul<L_i, L_i>(L_SCALE_FACTOR);
      mma_ABt(P_ij, Q_i, K_j);
      mul(P_ij, P_ij, P_SCALE_FACTOR);
      sub_row<L_i>(P_ij, P_ij);
      exp2(P_ij, P_ij);
      copy(P_ij_bf16, P_ij);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();

      load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
      load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
      mma_ABt(dP_ij, dO_i, V_j);
      sub_row<delta_i>(dP_ij, dP_ij);
      mul(dP_ij, dP_ij, P_ij);
      copy(dP_ij_bf16, dP_ij);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();

      // 12. dV_j += P_ij^T @ dO_i
      // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
      auto attn_i_smem_subtile = subtile_inplace<WARP_SIZE_KV, DOT_SLICE_QO>(attn_i_smem, {warpid, 0});
      store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
      swap_layout_inplace(P_ij_bf16_col, P_ij_bf16);
      mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col);
      swap_layout_inplace(dP_ij_bf16_col, dP_ij_bf16);
      mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
    
      load(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}));
      load(dP_ij_bf16_col_T, attn_i_smem);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
    }

    // dot slice 1
    {
      // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
      mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T);

      load(delta_smem[toc], g.delta_vec, {batch_idx, next_q_head_idx, 0, next_q_seq_idx});
      G::load<1, false>(dO_i_smem[toc][0], g.dOg, {batch_idx, next_q_seq_idx * 2, next_q_head_idx, 0});
      load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
      load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {1, 0}));
      load<L_i>(subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 1));
      mul(dQ_i, dQ_i, dP_SCALE_FACTOR);
      atomic_pk_add_bf16_with_warpid<2>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4, 0}, warpid);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();

      // 10. S_ij = Q_i K_j^T * scale
      // 11. P_ij = exp(S_ij - L_i)
      // 13. dP_ij = dO_i @ V_j^T
      // 14. dS_ij = P_ij o (dP_ij - delta_i)
      load<delta_i>(subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 1));
      load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {1, 0}));
      mul<L_i, L_i>(L_SCALE_FACTOR);
      mma_ABt(P_ij, Q_i, K_j);
      mul(P_ij, P_ij, P_SCALE_FACTOR);
      sub_row<L_i>(P_ij, P_ij);
      exp2(P_ij, P_ij);
      copy(P_ij_bf16, P_ij);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();

      load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {1, 0}));
      load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {1, 0}));
      mma_ABt(dP_ij, dO_i, V_j);
      sub_row<delta_i>(dP_ij, dP_ij);
      mul(dP_ij, dP_ij, P_ij);
      copy(dP_ij_bf16, dP_ij);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();

      // 12. dV_j += P_ij^T @ dO_i
      // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
      auto attn_i_smem_subtile = subtile_inplace<WARP_SIZE_KV, DOT_SLICE_QO>(attn_i_smem, {warpid, 0});
      store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
      swap_layout_inplace(P_ij_bf16_col, P_ij_bf16);
      mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
      swap_layout_inplace(dP_ij_bf16_col, dP_ij_bf16);
      mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();

      load(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}));
      load(dP_ij_bf16_col_T, attn_i_smem);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
    }

    // dot slice 2
    {
      // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
      mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T);

      G::load<1, false>(Q_i_smem[toc][1], g.Q, {batch_idx, next_q_seq_idx * 2 + 1, next_q_head_idx, 0});
      load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
      load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {0, 0}));
      load<L_i>(subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 2));
      mul(dQ_i, dQ_i, dP_SCALE_FACTOR);
      atomic_pk_add_bf16_with_warpid<2>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4 + 1, 0}, warpid);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();

      // 10. S_ij = Q_i K_j^T * scale
      // 11. P_ij = exp(S_ij - L_i)
      // 13. dP_ij = dO_i @ V_j^T
      // 14. dS_ij = P_ij o (dP_ij - delta_i)
      load<delta_i>(subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 2));
      load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {0, 0}));
      mul<L_i, L_i>(L_SCALE_FACTOR);
      mma_ABt(P_ij, Q_i, K_j);
      mul(P_ij, P_ij, P_SCALE_FACTOR);
      sub_row<L_i>(P_ij, P_ij);
      exp2(P_ij, P_ij);
      copy(P_ij_bf16, P_ij);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();

      load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {0, 0}));
      load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {0, 0}));
      mma_ABt(dP_ij, dO_i, V_j);
      sub_row<delta_i>(dP_ij, dP_ij);
      mul(dP_ij, dP_ij, P_ij);
      copy(dP_ij_bf16, dP_ij);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();

      // 12. dV_j += P_ij^T @ dO_i
      // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
      auto attn_i_smem_subtile = subtile_inplace<WARP_SIZE_KV, DOT_SLICE_QO>(attn_i_smem, {warpid, 0});
      store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
      swap_layout_inplace(P_ij_bf16_col, P_ij_bf16);
      mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
      swap_layout_inplace(dP_ij_bf16_col, dP_ij_bf16);
      mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();

      load(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}));
      load(dP_ij_bf16_col_T, attn_i_smem);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
    }

    // dot slice 3
    {
      // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
      mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T);

      G::load<1, false>(dO_i_smem[toc][1], g.dOg, {batch_idx, next_q_seq_idx * 2 + 1, next_q_head_idx, 0});
      load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
      load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {1, 0}));
      load<L_i>(subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 3));
      mul(dQ_i, dQ_i, dP_SCALE_FACTOR);
      atomic_pk_add_bf16_with_warpid<2>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4 + 2, 0}, warpid);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();

      // 10. S_ij = Q_i K_j^T * scale
      // 11. P_ij = exp(S_ij - L_i)
      // 13. dP_ij = dO_i @ V_j^T
      // 14. dS_ij = P_ij o (dP_ij - delta_i)
      load<delta_i>(subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 3));
      load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {1, 0}));
      mul<L_i, L_i>(L_SCALE_FACTOR);
      mma_ABt(P_ij, Q_i, K_j);
      mul(P_ij, P_ij, P_SCALE_FACTOR);
      sub_row<L_i>(P_ij, P_ij);
      exp2(P_ij, P_ij);
      copy(P_ij_bf16, P_ij);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();

      load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {1, 0}));
      load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {1, 0}));
      mma_ABt(dP_ij, dO_i, V_j);
      sub_row<delta_i>(dP_ij, dP_ij);
      mul(dP_ij, dP_ij, P_ij);
      copy(dP_ij_bf16, dP_ij);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();

      auto attn_i_smem_subtile = subtile_inplace<WARP_SIZE_KV, DOT_SLICE_QO>(attn_i_smem, {warpid, 0});
      store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
      // 12. dV_j += P_ij^T @ dO_i
      // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
      swap_layout_inplace(P_ij_bf16_col, P_ij_bf16);
      mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
      swap_layout_inplace(dP_ij_bf16_col, dP_ij_bf16);
      mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();

      load(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}));
      load(dP_ij_bf16_col_T, attn_i_smem);
      __builtin_amdgcn_s_waitcnt(0);
      __builtin_amdgcn_s_barrier();
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

      load<L_i>(subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 0));
      G::load<1, false>(Q_i_smem[toc][0], g.Q, {batch_idx, next_q_seq_idx * 2, next_q_head_idx, 0});
      load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
      load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
      load(L_smem[toc], g.L_vec, {batch_idx, next_q_head_idx, 0, next_q_seq_idx});
      mul(dQ_i, dQ_i, dP_SCALE_FACTOR);
      atomic_pk_add_bf16_with_warpid<2>(g.dQg, dQ_i, {batch_idx, last_q_head_idx, last_q_seq_idx * 4 + 3, 0}, warpid);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();

      // 10. S_ij = Q_i K_j^T * scale
      // 11. P_ij = exp(S_ij - L_i)
      // 13. dP_ij = dO_i @ V_j^T
      // 14. dS_ij = P_ij o (dP_ij - delta_i)
      load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
      load<delta_i>(subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 0));
      mul<L_i, L_i>(L_SCALE_FACTOR);
      mma_ABt(P_ij, Q_i, K_j);
      mul(P_ij, P_ij, P_SCALE_FACTOR);
      sub_row<L_i>(P_ij, P_ij);
      exp2(P_ij, P_ij);
      copy(P_ij_bf16, P_ij);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();

      load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
      load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
      mma_ABt(dP_ij, dO_i, V_j);
      sub_row<delta_i>(dP_ij, dP_ij);
      mul(dP_ij, dP_ij, P_ij);
      copy(dP_ij_bf16, dP_ij);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();

      // 12. dV_j += P_ij^T @ dO_i
      // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
      auto attn_i_smem_subtile = subtile_inplace<WARP_SIZE_KV, DOT_SLICE_QO>(attn_i_smem, {warpid, 0});
      store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
      swap_layout_inplace(P_ij_bf16_col, P_ij_bf16);
      mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T); 
      swap_layout_inplace(dP_ij_bf16_col, dP_ij_bf16);
      mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
    
      load(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}));
      load(dP_ij_bf16_col_T, attn_i_smem);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
    }

    // dot slice 1
    {
      // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
      mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T);
      __builtin_amdgcn_sched_barrier(0);

      load(delta_smem[toc], g.delta_vec, {batch_idx, next_q_head_idx, 0, next_q_seq_idx});
      G::load<1, false>(dO_i_smem[toc][0], g.dOg, {batch_idx, next_q_seq_idx * 2, next_q_head_idx, 0});
      load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
      load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {1, 0}));
      load<L_i>(subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 1));
      mul(dQ_i, dQ_i, dP_SCALE_FACTOR);
      atomic_pk_add_bf16_with_warpid<2>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4, 0}, warpid);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();

      // 10. S_ij = Q_i K_j^T * scale
      // 11. P_ij = exp(S_ij - L_i)
      // 13. dP_ij = dO_i @ V_j^T
      // 14. dS_ij = P_ij o (dP_ij - delta_i)
      load<delta_i>(subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 1));
      load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {1, 0}));
      mul<L_i, L_i>(L_SCALE_FACTOR);
      mma_ABt(P_ij, Q_i, K_j);
      mul(P_ij, P_ij, P_SCALE_FACTOR);
      sub_row<L_i>(P_ij, P_ij);
      exp2(P_ij, P_ij);
      copy(P_ij_bf16, P_ij);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();

      load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {1, 0}));
      load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {1, 0}));
      mma_ABt(dP_ij, dO_i, V_j);
      sub_row<delta_i>(dP_ij, dP_ij);
      mul(dP_ij, dP_ij, P_ij);
      copy(dP_ij_bf16, dP_ij);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();

      // 12. dV_j += P_ij^T @ dO_i
      // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
      auto attn_i_smem_subtile = subtile_inplace<WARP_SIZE_KV, DOT_SLICE_QO>(attn_i_smem, {warpid, 0});
      store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
      swap_layout_inplace(P_ij_bf16_col, P_ij_bf16);
      mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
      swap_layout_inplace(dP_ij_bf16_col, dP_ij_bf16);
      mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();

      load(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}));
      load(dP_ij_bf16_col_T, attn_i_smem);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
    }

    // dot slice 2
    {
      // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
      mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T);

      G::load<1, false>(Q_i_smem[toc][1], g.Q, {batch_idx, next_q_seq_idx * 2 + 1, next_q_head_idx, 0});
      load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
      load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {0, 0}));
      load<L_i>(subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 2));
      mul(dQ_i, dQ_i, dP_SCALE_FACTOR);
      atomic_pk_add_bf16_with_warpid<2>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4 + 1, 0}, warpid);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();

      // 10. S_ij = Q_i K_j^T * scale
      // 11. P_ij = exp(S_ij - L_i)
      // 13. dP_ij = dO_i @ V_j^T
      // 14. dS_ij = P_ij o (dP_ij - delta_i)
      load<delta_i>(subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 2));
      load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {0, 0}));
      mul<L_i, L_i>(L_SCALE_FACTOR);
      mma_ABt(P_ij, Q_i, K_j);
      mul(P_ij, P_ij, P_SCALE_FACTOR);
      sub_row<L_i>(P_ij, P_ij);
      exp2(P_ij, P_ij);
      copy(P_ij_bf16, P_ij);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();

      load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {0, 0}));
      load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {0, 0}));
      mma_ABt(dP_ij, dO_i, V_j);
      sub_row<delta_i>(dP_ij, dP_ij);
      mul(dP_ij, dP_ij, P_ij);
      copy(dP_ij_bf16, dP_ij);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();

      // 12. dV_j += P_ij^T @ dO_i
      // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
      auto attn_i_smem_subtile = subtile_inplace<WARP_SIZE_KV, DOT_SLICE_QO>(attn_i_smem, {warpid, 0});
      store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
      swap_layout_inplace(P_ij_bf16_col, P_ij_bf16);
      mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
      swap_layout_inplace(dP_ij_bf16_col, dP_ij_bf16);
      mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();

      load(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}));
      load(dP_ij_bf16_col_T, attn_i_smem);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
    }

    // dot slice 3
    {
      // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
      mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T);

      G::load<1, false>(dO_i_smem[toc][1], g.dOg, {batch_idx, next_q_seq_idx * 2 + 1, next_q_head_idx, 0});
      load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
      load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {1, 0}));
      load<L_i>(subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 3));
      mul(dQ_i, dQ_i, dP_SCALE_FACTOR);
      atomic_pk_add_bf16_with_warpid<2>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4 + 2, 0}, warpid);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();

      // 10. S_ij = Q_i K_j^T * scale
      // 11. P_ij = exp(S_ij - L_i)
      // 13. dP_ij = dO_i @ V_j^T
      // 14. dS_ij = P_ij o (dP_ij - delta_i)
      load<delta_i>(subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 3));
      load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {1, 0}));
      mul<L_i, L_i>(L_SCALE_FACTOR);
      mma_ABt(P_ij, Q_i, K_j);
      mul(P_ij, P_ij, P_SCALE_FACTOR);
      sub_row<L_i>(P_ij, P_ij);
      exp2(P_ij, P_ij);
      copy(P_ij_bf16, P_ij);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();

      load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {1, 0}));
      load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {1, 0}));
      mma_ABt(dP_ij, dO_i, V_j);
      sub_row<delta_i>(dP_ij, dP_ij);
      mul(dP_ij, dP_ij, P_ij);
      copy(dP_ij_bf16, dP_ij);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();

      auto attn_i_smem_subtile = subtile_inplace<WARP_SIZE_KV, DOT_SLICE_QO>(attn_i_smem, {warpid, 0});
      store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
      // 12. dV_j += P_ij^T @ dO_i
      // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
      swap_layout_inplace(P_ij_bf16_col, P_ij_bf16);
      mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
      swap_layout_inplace(dP_ij_bf16_col, dP_ij_bf16);
      mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();

      load(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}));
      load(dP_ij_bf16_col_T, attn_i_smem);
      __builtin_amdgcn_s_waitcnt(0);
      __builtin_amdgcn_s_barrier();
    }
  }

  const int last_q_head_idx = (num_steps - 2) / num_steps_per_head + first_q_head;
  const int last_q_seq_idx = (num_steps - 2) % num_steps_per_head;

  const int q_head_idx = (num_steps - 1) / num_steps_per_head + first_q_head;
  const int q_seq_idx = (num_steps - 1) % num_steps_per_head;
  // Epilogue
  {

    // dot slice 0
    {
      // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
      mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T);

      load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
      load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
      load<L_i>(subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 0));
      mul(dQ_i, dQ_i, dP_SCALE_FACTOR);
      atomic_pk_add_bf16_with_warpid<2>(g.dQg, dQ_i, {batch_idx, last_q_head_idx, last_q_seq_idx * 4 + 3, 0}, warpid);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();

      // 10. S_ij = Q_i K_j^T * scale
      // 11. P_ij = exp(S_ij - L_i)
      // 13. dP_ij = dO_i @ V_j^T
      // 14. dS_ij = P_ij o (dP_ij - delta_i)
      load<delta_i>(subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 0));
      load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
      mul<L_i, L_i>(L_SCALE_FACTOR);
      mma_ABt(P_ij, Q_i, K_j);
      mul(P_ij, P_ij, P_SCALE_FACTOR);
      sub_row<L_i>(P_ij, P_ij);
      exp2(P_ij, P_ij);
      copy(P_ij_bf16, P_ij);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();

      load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
      load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
      mma_ABt(dP_ij, dO_i, V_j);
      sub_row<delta_i>(dP_ij, dP_ij);
      mul(dP_ij, dP_ij, P_ij);
      copy(dP_ij_bf16, dP_ij);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();

      // 12. dV_j += P_ij^T @ dO_i
      // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
      auto attn_i_smem_subtile = subtile_inplace<WARP_SIZE_KV, DOT_SLICE_QO>(attn_i_smem, {warpid, 0});
      store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
      swap_layout_inplace(P_ij_bf16_col, P_ij_bf16);
      mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T); 
      swap_layout_inplace(dP_ij_bf16_col, dP_ij_bf16);
      mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
    
      load(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}));
      load(dP_ij_bf16_col_T, attn_i_smem);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
    }

    // dot slice 1
    {
      // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
      mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T);

      load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
      load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {1, 0}));
      load<L_i>(subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 1));
      mul(dQ_i, dQ_i, dP_SCALE_FACTOR);
      atomic_pk_add_bf16_with_warpid<2>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4, 0}, warpid);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();

      // 10. S_ij = Q_i K_j^T * scale
      // 11. P_ij = exp(S_ij - L_i)
      // 13. dP_ij = dO_i @ V_j^T
      // 14. dS_ij = P_ij o (dP_ij - delta_i)
      load<delta_i>(subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 1));
      load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {1, 0}));
      mul<L_i, L_i>(L_SCALE_FACTOR);
      mma_ABt(P_ij, Q_i, K_j);
      mul(P_ij, P_ij, P_SCALE_FACTOR);
      sub_row<L_i>(P_ij, P_ij);
      exp2(P_ij, P_ij);
      copy(P_ij_bf16, P_ij);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();

      load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {1, 0}));
      load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {1, 0}));
      mma_ABt(dP_ij, dO_i, V_j);
      sub_row<delta_i>(dP_ij, dP_ij);
      mul(dP_ij, dP_ij, P_ij);
      copy(dP_ij_bf16, dP_ij);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();

      // 12. dV_j += P_ij^T @ dO_i
      // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
      auto attn_i_smem_subtile = subtile_inplace<WARP_SIZE_KV, DOT_SLICE_QO>(attn_i_smem, {warpid, 0});
      store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
      swap_layout_inplace(P_ij_bf16_col, P_ij_bf16);
      mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
      swap_layout_inplace(dP_ij_bf16_col, dP_ij_bf16);
      mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();

      load(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}));
      load(dP_ij_bf16_col_T, attn_i_smem);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
    }

    // dot slice 2
    {
      // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
      mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T);

      load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
      load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {0, 0}));
      load<L_i>(subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 2));
      mul(dQ_i, dQ_i, dP_SCALE_FACTOR);
      atomic_pk_add_bf16_with_warpid<2>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4 + 1, 0}, warpid);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();

      // 10. S_ij = Q_i K_j^T * scale
      // 11. P_ij = exp(S_ij - L_i)
      // 13. dP_ij = dO_i @ V_j^T
      // 14. dS_ij = P_ij o (dP_ij - delta_i)
      load<delta_i>(subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 2));
      load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {0, 0}));
      mul<L_i, L_i>(L_SCALE_FACTOR);
      mma_ABt(P_ij, Q_i, K_j);
      mul(P_ij, P_ij, P_SCALE_FACTOR);
      sub_row<L_i>(P_ij, P_ij);
      exp2(P_ij, P_ij);
      copy(P_ij_bf16, P_ij);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();

      load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {0, 0}));
      load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {0, 0}));
      mma_ABt(dP_ij, dO_i, V_j);
      sub_row<delta_i>(dP_ij, dP_ij);
      mul(dP_ij, dP_ij, P_ij);
      copy(dP_ij_bf16, dP_ij);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();

      // 12. dV_j += P_ij^T @ dO_i
      // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
      auto attn_i_smem_subtile = subtile_inplace<WARP_SIZE_KV, DOT_SLICE_QO>(attn_i_smem, {warpid, 0});
      store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
      swap_layout_inplace(P_ij_bf16_col, P_ij_bf16);
      mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
      swap_layout_inplace(dP_ij_bf16_col, dP_ij_bf16);
      mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();

      load(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}));
      load(dP_ij_bf16_col_T, attn_i_smem);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();
    }

    // dot slice 3
    {
      // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
      mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T);

      load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
      load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {1, 0}));
      load<L_i>(subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 3));
      mul(dQ_i, dQ_i, dP_SCALE_FACTOR);
      atomic_pk_add_bf16_with_warpid<2>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4 + 2, 0}, warpid);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();

      // 10. S_ij = Q_i K_j^T * scale
      // 11. P_ij = exp(S_ij - L_i)
      // 13. dP_ij = dO_i @ V_j^T
      // 14. dS_ij = P_ij o (dP_ij - delta_i)
      load<delta_i>(subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 3));
      load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {1, 0}));
      mul<L_i, L_i>(L_SCALE_FACTOR);
      mma_ABt(P_ij, Q_i, K_j);
      mul(P_ij, P_ij, P_SCALE_FACTOR);
      sub_row<L_i>(P_ij, P_ij);
      exp2(P_ij, P_ij);
      copy(P_ij_bf16, P_ij);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();

      load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {1, 0}));
      load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {1, 0}));
      mma_ABt(dP_ij, dO_i, V_j);
      sub_row<delta_i>(dP_ij, dP_ij);
      mul(dP_ij, dP_ij, P_ij);
      copy(dP_ij_bf16, dP_ij);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();

      auto attn_i_smem_subtile = subtile_inplace<WARP_SIZE_KV, DOT_SLICE_QO>(attn_i_smem, {warpid, 0});
      store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
      // 12. dV_j += P_ij^T @ dO_i
      // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
      swap_layout_inplace(P_ij_bf16_col, P_ij_bf16);
      mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
      swap_layout_inplace(dP_ij_bf16_col, dP_ij_bf16);
      mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
      // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();

      load(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warpid}));
      load(dP_ij_bf16_col_T, attn_i_smem);
      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_barrier();

      mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T);
    }
  }

  store<1>(g.dVg, dV_j, {batch_idx, 0, kv_head_idx, 0}, {0, j, 0, 0});
  __builtin_amdgcn_s_waitcnt(0);
  __builtin_amdgcn_s_barrier();

  // We first copy dV_j_T from accumulator GPRs to vector GPRs and then perform the store
  accvgpr_read(dV_j_T, dK_j_T);
  mul(dV_j, dV_j, dP_SCALE_FACTOR);
  store<1>(g.dKg, dV_j, {batch_idx, 0, kv_head_idx, 0}, {0, j, 0, 0});

  mul(dQ_i, dQ_i, dP_SCALE_FACTOR);
  atomic_pk_add_bf16_with_warpid<2>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4 + 3, 0}, warpid);
}

template<int D>
void dispatch_bwd_combined(attn_bwd_combined_globals<D> g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)attend_bwd_combined_ker<D>, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    attend_bwd_combined_ker<D><<<g.grid(), g.block(), mem_size>>>(g);
    hipDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernel_bkwd, m) {
  m.doc() = "tk_kernel python module";

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
}


