#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include "utils.cpp"

using namespace kittens;

#define NUM_WARPS 4
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)
#define ASSEMBLY_MODE

using namespace kittens;

using G = kittens::group<NUM_WARPS>;

struct micro_globals {
  gl<bf16, -1, -1, -1, -1> Q;
  gl<bf16, -1, -1, -1, -1> dO;
  gl<bf16, -1, -1, -1, -1> K;
  gl<bf16, -1, -1, -1, -1> V;
  gl<float, -1, -1, -1, -1> L;
  gl<float, -1, -1, -1, -1> delta;
  gl<bf16, -1, -1, -1, -1> dK;
  gl<bf16, -1, -1, -1, -1> dV;
  gl<bf16, -1, -1, -1, -1> dQ;
  dim3 grid() {return dim3(1); }
  dim3 block() {return dim3(NUM_THREADS);}
  size_t dynamic_shared_memory() {return MAX_SHARED_MEMORY;}
};

#ifdef ASSEMBLY_MODE
__device__ __forceinline__ void process(const micro_globals g) {
  extern __shared__ alignment_dummy __shm[];
  shared_allocator al((int*)&__shm[0]);

  st_bf<256, 128, ducks::st_layout::accumulator, ducks::st_matrix::mfma_16x16x32> (&K_j_smem) = al.allocate<st_bf<256, 128, ducks::st_layout::accumulator, ducks::st_matrix::mfma_16x16x32>>();
  st_bf<16, 128, ducks::st_layout::classical, ducks::st_matrix::mfma_16x16x32> (&Q_i_smem) = al.allocate<st_bf<16, 128, ducks::st_layout::classical, ducks::st_matrix::mfma_16x16x32>>();
  st_bf<16, 128, ducks::st_layout::classical, ducks::st_matrix::mfma_16x16x32> (&dO_i_smem) = al.allocate<st_bf<16, 128, ducks::st_layout::classical, ducks::st_matrix::mfma_16x16x32>>();

  // We parameterize this using mfma_32x32x16 because we want the base tile for it to be 32x16. Not that it uses that intrinsic.
  st_bf<256, 16, ducks::st_layout::classical, ducks::st_matrix::mfma_32x32x16> (&attn_i_smem) = al.allocate<st_bf<256, 16, ducks::st_layout::classical, ducks::st_matrix::mfma_32x32x16>>();
  sv_fl<16> (&L_smem) = al.allocate<sv_fl<16>>();
  sv_fl<16> (&delta_smem) = al.allocate<sv_fl<16>>();

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

  rt_asm<bf16, 16, 128, row_l, mfma_16x16x32, Q_ranges> Q_i; // 16 registers
  rt_asm<bf16, 16, 128, row_l, mfma_16x16x32, dO_ranges> dO_i; // 16 registers
  rt_asm<bf16, 16, 128, col_l, mfma_32x32x16, Q_ranges> Q_i_col; // 16 registers
  rt_asm<bf16, 16, 128, col_l, mfma_32x32x16, dO_ranges> dO_i_col; // 16 registers
  rt_asm<bf16, 64, 128, row_l, mfma_16x16x32, K_ranges> K_j; // 64 registers
  rt_asm<bf16, 64, 128, row_l, mfma_16x16x32, V_ranges> V_j; // 64 registers
  rt<float, 16, 64, accum_col_l, mfma_16x16x32>::col_vec L_i, delta_i;

  rt_asm<float, 16, 64, accum_col_l, mfma_16x16x32, P_ranges> P_ij; // 16 registers
  rt_asm<float, 16, 64, accum_col_l, mfma_16x16x32, dP_ranges> dP_ij; // 16 registers
  rt_asm<bf16, 16, 64, accum_col_l, mfma_16x16x32, P_bf16_ranges> P_ij_bf16; // 8 registers
  rt_asm<bf16, 16, 64, accum_col_l, mfma_16x16x32, dP_bf16_ranges> dP_ij_bf16; // 8 registers
  rt_asm<bf16, 64, 16, accum_row_l, mfma_16x16x32, ducks::rt_asm::transpose_2d<dP_bf16_ranges, 1, 4>> dP_ij_bf16_accum_row; // 8 registers

  rt_asm<bf16, 16, 64, col_l, mfma_32x32x16, P_bf16_col_ranges> P_ij_bf16_col; // 8 registers
  rt_asm<bf16, 16, 64, col_l, mfma_32x32x16, dP_bf16_col_ranges> dP_ij_bf16_col; // 8 registers

  rt_asm<bf16, 256, 32, col_l, mfma_16x16x32, K_ranges> K_j_col; // 64 registers // for dq
  rt_asm<bf16, 256, 16, col_l, mfma_16x16x32, dS_ranges> dP_ij_bf16_col_T; // 32 registers // for dq

  rt_asm<float, 128, 64, accum_col_l, mfma_32x32x16, dK_ranges> dK_j_T; // 128 registers
  rt_asm<float, 128, 64, accum_col_l, mfma_32x32x16, dV_ranges> dV_j_T; // 128 registers
  rt_asm<float, 32, 16, accum_col_l, mfma_16x16x32, dQ_ranges> dQ_i_T; // 8 registers // for dq
  rt_asm<float, 16, 32, accum_row_l, mfma_16x16x32, ducks::rt_asm::transpose_2d<dQ_ranges, 2, 1>> dQ_i; // 8 registers // for dq

  // This is used for both dK_j_T and dV_j_T
  rt_asm<float, 64, 128, accum_row_l, mfma_32x32x16, ducks::rt_asm::transpose_2d<dV_ranges, 4, 2>> dV_j;

  const int warp_id = kittens::warpid();
  const float scale_factor = 1.0f / sqrt(128);

  // Each thread writes Q, dO, K, L, delta to shared memory
  G::load(Q_i_smem, g.Q, {0, 0, 0, 0});
  G::load(dO_i_smem, g.dO, {0, 0, 0, 0});
  G::load(K_j_smem, g.K, {0, 0, 0, 0});
  load(L_smem, g.L, {0, 0, 0, 0});
  load(delta_smem, g.delta, {0, 0, 0, 0});

  // Load V_j from HBM to registers
  load(V_j, g.V, {0, 0, 0, 0}, {0, 0, warp_id, 0});
  __builtin_amdgcn_s_waitcnt(0);
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_sched_barrier(0);

  load(Q_i, Q_i_smem);
  load(K_j, subtile_inplace<64, 128>(K_j_smem, {warp_id, 0}));
  load(L_i, L_smem);
  __builtin_amdgcn_s_waitcnt(0);
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_sched_barrier(0);

  load(dO_i, dO_i_smem);
  load(delta_i, delta_smem);
  mma_ABt(P_ij, Q_i, K_j);
  mul(P_ij, P_ij, scale_factor);
  sub_row(P_ij, P_ij, L_i);
  __builtin_amdgcn_s_waitcnt(0);
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_sched_barrier(0);

  exp(P_ij, P_ij);
  copy(P_ij_bf16, P_ij);
  mma_ABt(dP_ij, dO_i, V_j);
  sub_row(dP_ij, dP_ij, delta_i);
  mul(dP_ij, dP_ij, scale_factor);
  mul(dP_ij, dP_ij, P_ij);
  copy(dP_ij_bf16, dP_ij);
  load(Q_i_col, Q_i_smem);
  load(dO_i_col, dO_i_smem);
  __builtin_amdgcn_s_waitcnt(0);
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_sched_barrier(0);

  auto attn_i_smem_subtile = subtile_inplace<64, 16>(attn_i_smem, {warp_id, 0});
  store(attn_i_smem_subtile, dP_ij_bf16_accum_row); // to check
  swap_layout_inplace(P_ij_bf16_col, P_ij_bf16);
  swap_layout_inplace(dP_ij_bf16_col, dP_ij_bf16);
  mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col); 
  mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col);
  __builtin_amdgcn_s_waitcnt(0);
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_sched_barrier(0);


  load(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warp_id})); // to check
  load(dP_ij_bf16_col_T, attn_i_smem); // to check
  __builtin_amdgcn_s_waitcnt(0);
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_sched_barrier(0);
  
  mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T); // to check
  atomic_pk_add_bf16_with_warpid<2>(g.dQ, dQ_i, {0, 0, 0, 0}, warp_id); // to check
  store(g.dV, dV_j, {0, 0, 0, 0}, {0, 0, warp_id, 0});
  __builtin_amdgcn_s_waitcnt(0);
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_sched_barrier(0);

  // We first copy dV_j_T from accumulator GPRs to vector GPRs and then perform the store
  accvgpr_read(dV_j_T, dK_j_T);
  store(g.dK, dV_j, {0, 0, 0, 0}, {0, 0, warp_id, 0});
}
#else
__device__ __forceinline__ void process(const micro_globals g) {
  extern __shared__ alignment_dummy __shm[];
  shared_allocator al((int*)&__shm[0]);

  st_bf<256, 128, ducks::st_layout::accumulator, ducks::st_matrix::mfma_16x16x32> (&K_j_smem) = al.allocate<st_bf<256, 128, ducks::st_layout::accumulator, ducks::st_matrix::mfma_16x16x32>>();
  st_bf<16, 128, ducks::st_layout::classical, ducks::st_matrix::mfma_16x16x32> (&Q_i_smem) = al.allocate<st_bf<16, 128, ducks::st_layout::classical, ducks::st_matrix::mfma_16x16x32>>();
  st_bf<16, 128, ducks::st_layout::classical, ducks::st_matrix::mfma_16x16x32> (&dO_i_smem) = al.allocate<st_bf<16, 128, ducks::st_layout::classical, ducks::st_matrix::mfma_16x16x32>>();

  // We parameterize this using mfma_32x32x16 because we want the base tile for it to be 32x16. Not that it uses that intrinsic.
  st_bf<256, 16, ducks::st_layout::classical, ducks::st_matrix::mfma_32x32x16> (&attn_i_smem) = al.allocate<st_bf<256, 16, ducks::st_layout::classical, ducks::st_matrix::mfma_32x32x16>>();
  sv_fl<16> (&L_smem) = al.allocate<sv_fl<16>>();
  sv_fl<16> (&delta_smem) = al.allocate<sv_fl<16>>();

  rt<bf16, 16, 128, row_l, mfma_16x16x32> Q_i;
  rt<bf16, 16, 128, row_l, mfma_16x16x32> dO_i;
  rt<bf16, 16, 128, col_l, mfma_32x32x16> Q_i_col;
  rt<bf16, 16, 128, col_l, mfma_32x32x16> dO_i_col;
  rt<bf16, 64, 128, row_l, mfma_16x16x32> K_j;
  rt<bf16, 64, 128, row_l, mfma_16x16x32> V_j;
  rt<float, 16, 64, accum_col_l, mfma_16x16x32>::col_vec L_i, delta_i;

  rt<float, 16, 64, accum_col_l, mfma_16x16x32> P_ij;
  rt<float, 16, 64, accum_col_l, mfma_16x16x32> dP_ij;
  rt<bf16, 16, 64, accum_col_l, mfma_16x16x32> P_ij_bf16;
  rt<bf16, 16, 64, accum_col_l, mfma_16x16x32> dP_ij_bf16;
  rt<bf16, 64, 16, accum_row_l, mfma_16x16x32> dP_ij_bf16_accum_row;

  rt<bf16, 16, 64, col_l, mfma_32x32x16> P_ij_bf16_col;
  rt<bf16, 16, 64, col_l, mfma_32x32x16> dP_ij_bf16_col;

  rt<bf16, 256, 32, col_l, mfma_16x16x32> K_j_col; // for dq
  rt<bf16, 256, 16, col_l, mfma_16x16x32> dP_ij_bf16_col_T; // for dq

  rt<float, 128, 64, accum_col_l, mfma_32x32x16> dK_j_T;
  rt<float, 128, 64, accum_col_l, mfma_32x32x16> dV_j_T;
  rt<float, 32, 16, accum_col_l, mfma_16x16x32> dQ_i_T; // for dq
  rt<float, 16, 32, accum_row_l, mfma_16x16x32> dQ_i; // for dq

  zero(dK_j_T);
  zero(dV_j_T);

  const int warp_id = kittens::warpid();
  const float scale_factor = 1.0f / sqrt(128);

  // Each thread writes Q, dO, K, L, delta to shared memory
  G::load(Q_i_smem, g.Q, {0, 0, 0, 0});
  G::load(dO_i_smem, g.dO, {0, 0, 0, 0});
  G::load(K_j_smem, g.K, {0, 0, 0, 0});
  load(L_smem, g.L, {0, 0, 0, 0});
  load(delta_smem, g.delta, {0, 0, 0, 0});

  // Load V_j from HBM to registers
  load(V_j, g.V, {0, 0, warp_id, 0});
  __builtin_amdgcn_s_waitcnt(0);
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_sched_barrier(0);

  load(Q_i, Q_i_smem);
  load(K_j, subtile_inplace<64, 128>(K_j_smem, {warp_id, 0}));
  load(L_i, L_smem);
  __builtin_amdgcn_s_waitcnt(0);
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_sched_barrier(0);

  load(dO_i, dO_i_smem);
  load(delta_i, delta_smem);
  zero(P_ij);
  mma_ABt(P_ij, Q_i, K_j, P_ij);
  mul(P_ij, P_ij, scale_factor);
  sub_row(P_ij, P_ij, L_i);
  __builtin_amdgcn_s_waitcnt(0);
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_sched_barrier(0);

  load(Q_i_col, Q_i_smem);
  load(dO_i_col, dO_i_smem);
  exp(P_ij, P_ij);
  copy(P_ij_bf16, P_ij);
  zero(dP_ij);
  mma_ABt(dP_ij, dO_i, V_j, dP_ij);
  sub_row(dP_ij, dP_ij, delta_i);
  mul(dP_ij, dP_ij, scale_factor);
  mul(dP_ij, dP_ij, P_ij);
  copy(dP_ij_bf16, dP_ij);
  swap_layout_and_transpose(dP_ij_bf16_accum_row, dP_ij_bf16);
  __builtin_amdgcn_s_waitcnt(0);
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_sched_barrier(0);

  auto attn_i_smem_subtile = subtile_inplace<64, 16>(attn_i_smem, {warp_id, 0});
  store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
  P_ij_bf16_col = swap_layout_inplace<col_l, mfma_32x32x16>(P_ij_bf16);
  dP_ij_bf16_col = swap_layout_inplace<col_l, mfma_32x32x16>(dP_ij_bf16);
  mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
  mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
  __builtin_amdgcn_s_waitcnt(0);
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_sched_barrier(0);

  load(K_j_col, subtile_inplace<256, 32>(K_j_smem, {0, warp_id}));
  load(dP_ij_bf16_col_T, attn_i_smem);
  __builtin_amdgcn_s_waitcnt(0);
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_sched_barrier(0);

  zero(dQ_i_T);
  mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T, dQ_i_T);
  swap_layout_and_transpose(dQ_i, dQ_i_T);
  atomic_pk_add_bf16_with_warpid<2>(g.dQ, dQ_i, {0, 0, 0, 0}, warp_id);

  rt<float, 64, 128, accum_row_l, mfma_32x32x16> dK_j;
  rt<float, 64, 128, accum_row_l, mfma_32x32x16> dV_j;
  swap_layout_and_transpose(dK_j, dK_j_T);
  swap_layout_and_transpose(dV_j, dV_j_T);

  store(g.dK, dK_j, {0, 0, warp_id, 0});
  store(g.dV, dV_j, {0, 0, warp_id, 0});
}
#endif

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const micro_globals g) {
  process(g);
}

void dispatch_micro(micro_globals g) {
  unsigned long mem_size = g.dynamic_shared_memory();
  hipFuncSetAttribute((void*)micro_tk, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
  micro_tk<<<g.grid(), g.block(), mem_size>>>(g);
  hipDeviceSynchronize();
}

struct attn_dq_shuffle_globals { 
  gl<bf16, -1, -1, -1, -1> dQ_in, dQ_out;
  dim3 grid() { return dim3(1); }
  dim3 block() { return dim3(64); }
  size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

__global__ __launch_bounds__(64, 1)
void attend_dq_shuffle_ker(const attn_dq_shuffle_globals g) {

  rt<bf16, 16, 128, accum_row_l, mfma_16x16x32> dQ;

  load_shuffled<2>(dQ, g.dQ_in, {0, 0, 0, 0});
  store(g.dQ_out, dQ, {0, 0, 0, 0});
}

void dispatch_dq_shuffle(attn_dq_shuffle_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)attend_dq_shuffle_ker, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    attend_dq_shuffle_ker<<<g.grid(), g.block(), mem_size>>>(g);
    hipDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernel, m) {
  m.doc() = "tk_kernel python module";
  py::bind_function<dispatch_micro>(m, "dispatch_micro", 
    &micro_globals::Q, 
    &micro_globals::dO, 
    &micro_globals::K, 
    &micro_globals::V, 
    &micro_globals::L, 
    &micro_globals::delta, 
    &micro_globals::dK, 
    &micro_globals::dV,
    &micro_globals::dQ
  );

  py::bind_function<dispatch_dq_shuffle>(m, "dispatch_dq_shuffle", 
    &attn_dq_shuffle_globals::dQ_in,
    &attn_dq_shuffle_globals::dQ_out
  );
}


