#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

constexpr int ATTN_B = 16; // batch size
constexpr int ATTN_H_Q = 64; // number of query heads
constexpr int ATTN_H_KV = 8; // number of key/value heads (for GQA)
constexpr int GROUP_SIZE = ATTN_H_Q / ATTN_H_KV; // queries per KV head group
constexpr int ATTN_N = 1024; // sequence length
constexpr int ATTN_D = 128; // dimension
constexpr int STEP_QO = 64; // block size for QO
constexpr int BLOCK_SIZE_KV = 256; // block size for KV
constexpr int SLICE_QO = 32;
constexpr int DOT_SLICE_QO = 16;
constexpr int WARP_SIZE_KV = 32; // warp size for KV

#define NUM_WARPS 8
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using G = kittens::group<NUM_WARPS>;

using namespace kittens;

template<int D, typename T=bf16, typename L=row_l, typename S=rt_16x32_s> using qo_tile = rt<T, DOT_SLICE_QO, D, L, S>;
template<int D, typename T=bf16, typename L=row_l, typename S=rt_16x32_s> using kv_tile = rt<T, WARP_SIZE_KV, D, L, S>;
template<int D, typename T=bf16, typename L=row_l, typename S=rt_16x32_s> using qo_tile_T_dq = rt<T, 16, 16, L, S>;
template<int D, typename T=bf16, typename L=row_l, typename S=rt_16x32_s> using qo_tile_dq = rt<T, 16, 16, L, S>;
template<int D, typename T=bf16, typename L=row_l, typename S=rt_16x32_s> using kv_tile_T = rt<T, D, WARP_SIZE_KV, L, S>;
template<int D, typename T=float, typename L=col_l, typename S=rt_16x16_s> using attn_tile = rt<T, DOT_SLICE_QO, WARP_SIZE_KV, L, S>;
template<int D, typename T=bf16, typename L=col_l, typename S=rt_16x16_s> using attn_tile_T = rt<T, WARP_SIZE_KV, DOT_SLICE_QO, L, S>;

template<int D, typename T=bf16, typename L=col_l, typename S=rt_32x16_s> using attn_tile_T_dq = rt<T, 256, 16, L, S>;
template<int D, typename T=bf16, typename L=row_l, typename S=rt_16x32_s> using kv_tile_dq = rt<T, 256, 16, L, S>;

template<int D> struct attn_bwd_combined_globals { 
    gl<bf16, -1, -1, -1, -1> Q, K, V;
    gl<bf16, -1, -1, -1, -1> dOg, dQg, dKg, dVg;
    gl<float, -1, -1, -1, -1> L_vec, delta_vec;
    hipStream_t stream;
    dim3 grid() { return dim3((ATTN_N / BLOCK_SIZE_KV), ATTN_H_KV, ATTN_B); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

template<int axis, ducks::rt::row_layout RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void atomic_pk_add_bf16_with_warpid(const GL &dst, const RT &src, const COORD &idx, int warpid) { 
    using T = base_types::packing<typename RT::dtype>::unpacked_type;
    using T2 = base_types::packing<typename RT::dtype>::packed_type;
    using U = typename GL::dtype;
    using U2 = base_types::packing<U>::packed_type;

    static_assert(std::is_same_v<U, bf16>, "atomic_pk_add_bf16 is only supported for bf16");

    U *dst_ptr = (U*)&dst[(idx.template unit_coord<axis, 3>())];
    const int row_stride = dst.template stride<axis>();
    int laneid = kittens::laneid();

    const uint32_t buffer_size = row_stride * RT::rows * sizeof(U); 
    std::uintptr_t as_int = reinterpret_cast<std::uintptr_t>(dst_ptr);
    std::uint64_t  as_u64 = static_cast<std::uint64_t>(as_int);
    buffer_resource br = make_buffer_resource(as_u64, buffer_size, 0x00020000);

    int lane_offset = laneid * 2 + warpid * 256;

    const U2 val_0 = base_types::convertor<U2, T2>::convert(src.tiles[0][0].data[0]);
    const U2 val_1 = base_types::convertor<U2, T2>::convert(src.tiles[0][0].data[1]);

    uint32_t byte_offset_0 = static_cast<uint32_t>((lane_offset) * sizeof(U));
    uint32_t byte_offset_1 = static_cast<uint32_t>((lane_offset + 128) * sizeof(U));

    uint32_t val_0_bits = *reinterpret_cast<const uint32_t*>(&val_0);
    uint32_t val_1_bits = *reinterpret_cast<const uint32_t*>(&val_1);

    asm volatile(
        "buffer_atomic_pk_add_bf16 %0, %1, %2, 0 offen\n"
        :
        : "v"(val_0_bits), "v"(byte_offset_0),      // %0, %1
            "s"(*(i32x4*)&br)                         // %8
        : "memory"
    );

    asm volatile(
        "buffer_atomic_pk_add_bf16 %0, %1, %2, 0 offen\n"
        :
        : "v"(val_1_bits), "v"(byte_offset_1),      // %2, %3
            "s"(*(i32x4*)&br)                         // %8
        : "memory"
    );
}

/*------------------------------------------------------------------------------------------------*/

template<int D> __launch_bounds__(NUM_THREADS, 1)
__global__ void attend_bwd_combined_ker(const attn_bwd_combined_globals<D> g) {
    
    const int seq_idx = blockIdx.x;
    const int kv_head_idx = blockIdx.y; // This is the KV head index
    const int batch_idx = blockIdx.z;
    const int first_q_head = kv_head_idx * GROUP_SIZE;

    const int warpid = kittens::warpid();
    const int j = seq_idx * NUM_WARPS + warpid;
    const int stagger = warpid / 4;

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
    kv_tile<D, bf16, row_l, rt_16x32_s> K_j, V_j;
    kv_tile_dq<D, bf16, col_l, rt_32x16_4_s> K_j_col; // for dq
    qo_tile_T_dq<D, float, col_l, rt_16x16_s> dQ_i_T; // for dq
    kv_tile_T<D, float, col_l, rt_32x32_s> dK_j_T, dV_j_T;
    qo_tile<D, bf16, row_l, rt_16x32_s> Q_i, dO_i;
    qo_tile<D, bf16, col_l, rt_16x32_s> Q_i_col, dO_i_col;
    qo_tile_dq<D, float, row_l, rt_16x16_s> dQ_i;
    attn_tile<D, float, col_l, rt_16x16_s>::col_vec L_i, delta_i;

    attn_tile<D, float, col_l, rt_16x16_s> P_ij;
    attn_tile<D, bf16, col_l, rt_16x16_s> P_ij_bf16;
    attn_tile<D, float, col_l, rt_16x16_s> dP_ij;
    attn_tile<D, bf16, col_l, rt_16x16_s> dP_ij_bf16;
    attn_tile_T<D, bf16, row_l, rt_16x16_s> dP_ij_bf16_accum_row;

    attn_tile<D, bf16, col_l, rt_16x32_s> P_ij_bf16_col;
    attn_tile<D, bf16, col_l, rt_16x32_s> dP_ij_bf16_col;
    attn_tile_T_dq<D, bf16, col_l, rt_32x16_4_s> dP_ij_bf16_col_T; // for dq

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

    if (stagger) {
        __builtin_amdgcn_s_barrier();
    }

    // 9. for 1 <= i <= T_r (1024 / 32 = 32)  
    for (int i = 0; i < num_steps - 1; ++i, tic ^= 1, toc ^= 1) {
        const int q_head_idx = i / num_steps_per_head + first_q_head;
        const int q_seq_idx = i % num_steps_per_head;

        const int next_q_head_idx = (i + 1) / num_steps_per_head + first_q_head;
        const int next_q_seq_idx = (i + 1) % num_steps_per_head;

        // dot slice 0
        {
            load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
            load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
            load(L_i, subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 0));
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // 10. S_ij = Q_i K_j^T * scale
            // 11. P_ij = exp2(S_ij - L_i)
            __builtin_amdgcn_s_setprio(1);
            zero(P_ij);
            mul(L_i, L_i, L_SCALE_FACTOR);
            mma_ABt(P_ij, Q_i, K_j, P_ij);
            mul(P_ij, P_ij, P_SCALE_FACTOR);
            sub_row(P_ij, P_ij, L_i);
            exp2(P_ij, P_ij);
            copy(P_ij_bf16, P_ij);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // 13. dP_ij = dO_i @ V_j^T
            // 14. dS_ij = P_ij o (dP_ij - delta_i)
            load(delta_i, subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 0));
            load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            __builtin_amdgcn_s_setprio(1);
            zero(dP_ij);
            mma_ABt(dP_ij, dO_i, V_j, dP_ij);
            sub_row(dP_ij, dP_ij, delta_i);
            mul(dP_ij, dP_ij, P_ij);
            copy(dP_ij_bf16, dP_ij);
            transpose(dP_ij_bf16_accum_row, dP_ij_bf16);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            load(L_smem[toc], g.L_vec, {batch_idx, next_q_head_idx, 0, next_q_seq_idx});
            G::load<1, false>(Q_i_smem[toc][0], g.Q, {batch_idx, next_q_seq_idx * 2, next_q_head_idx, 0});
            auto attn_i_smem_subtile = subtile_inplace<WARP_SIZE_KV, DOT_SLICE_QO>(attn_i_smem, {warpid, 0});
            store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
            load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
            load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // 12. dV_j += P_ij^T @ dO_i
            // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
            __builtin_amdgcn_s_setprio(1);
            P_ij_bf16_col = swap_layout_inplace<col_l, rt_16x32_s>(P_ij_bf16);
            mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
            dP_ij_bf16_col = swap_layout_inplace<col_l, rt_16x32_s>(dP_ij_bf16);
            mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
            load(dP_ij_bf16_col_T, attn_i_smem);
            load(K_j_col, subtile_inplace<256, 16>(K_j_smem, {0, warpid}));
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            __builtin_amdgcn_s_setprio(1);
            zero(dQ_i_T);
            mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T,  dQ_i_T);
            transpose(dQ_i, dQ_i_T);
            mul(dQ_i, dQ_i, dP_SCALE_FACTOR);
            atomic_pk_add_bf16_with_warpid<2>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4, 0}, warpid);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

        }

        // dot slice 1
        {
            load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
            load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {1, 0}));
            load(L_i, subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 1));
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // 10. S_ij = Q_i K_j^T * scale
            // 11. P_ij = exp2(S_ij - L_i)
            __builtin_amdgcn_s_setprio(1);
            zero(P_ij);
            mul(L_i, L_i, L_SCALE_FACTOR);
            mma_ABt(P_ij, Q_i, K_j, P_ij);
            mul(P_ij, P_ij, P_SCALE_FACTOR);
            sub_row(P_ij, P_ij, L_i);
            exp2(P_ij, P_ij);
            copy(P_ij_bf16, P_ij);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // 13. dP_ij = dO_i @ V_j^T
            // 14. dS_ij = P_ij o (dP_ij - delta_i)
            load(delta_i, subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 1));
            load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {1, 0}));
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            __builtin_amdgcn_s_setprio(1);
            zero(dP_ij);
            mma_ABt(dP_ij, dO_i, V_j, dP_ij);
            sub_row(dP_ij, dP_ij, delta_i);
            mul(dP_ij, dP_ij, P_ij);
            copy(dP_ij_bf16, dP_ij);
            transpose(dP_ij_bf16_accum_row, dP_ij_bf16);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            load(delta_smem[toc], g.delta_vec, {batch_idx, next_q_head_idx, 0, next_q_seq_idx});
            G::load<1, false>(dO_i_smem[toc][0], g.dOg, {batch_idx, next_q_seq_idx * 2, next_q_head_idx, 0});
            auto attn_i_smem_subtile = subtile_inplace<WARP_SIZE_KV, DOT_SLICE_QO>(attn_i_smem, {warpid, 0});
            store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
            load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {1, 0}));
            load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {1, 0}));
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // 12. dV_j += P_ij^T @ dO_i
            // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
            __builtin_amdgcn_s_setprio(1);
            P_ij_bf16_col = swap_layout_inplace<col_l, rt_16x32_s>(P_ij_bf16);
            mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
            dP_ij_bf16_col = swap_layout_inplace<col_l, rt_16x32_s>(dP_ij_bf16);
            mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
            load(dP_ij_bf16_col_T, attn_i_smem);
            load(K_j_col, subtile_inplace<256, 16>(K_j_smem, {0, warpid}));
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            __builtin_amdgcn_s_setprio(1);
            zero(dQ_i_T);
            mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T,  dQ_i_T);
            transpose(dQ_i, dQ_i_T);
            mul(dQ_i, dQ_i, dP_SCALE_FACTOR);
            atomic_pk_add_bf16_with_warpid<2>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4 + 1, 0}, warpid);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);
        }

        // dot slice 2
        {
            load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
            load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {0, 0}));
            load(L_i, subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 2));
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // 10. S_ij = Q_i K_j^T * scale
            // 11. P_ij = exp2(S_ij - L_i)
            __builtin_amdgcn_s_setprio(1);
            zero(P_ij);
            mul(L_i, L_i, L_SCALE_FACTOR);
            mma_ABt(P_ij, Q_i, K_j, P_ij);
            mul(P_ij, P_ij, P_SCALE_FACTOR);
            sub_row(P_ij, P_ij, L_i);
            exp2(P_ij, P_ij);
            copy(P_ij_bf16, P_ij);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // 13. dP_ij = dO_i @ V_j^T
            // 14. dS_ij = P_ij o (dP_ij - delta_i)
            load(delta_i, subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 2));
            load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {0, 0}));
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            __builtin_amdgcn_s_setprio(1);
            zero(dP_ij);
            mma_ABt(dP_ij, dO_i, V_j, dP_ij);
            sub_row(dP_ij, dP_ij, delta_i);
            mul(dP_ij, dP_ij, P_ij);
            copy(dP_ij_bf16, dP_ij);
            transpose(dP_ij_bf16_accum_row, dP_ij_bf16);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            G::load<1, false>(Q_i_smem[toc][1], g.Q, {batch_idx, next_q_seq_idx * 2 + 1, next_q_head_idx, 0});
            auto attn_i_smem_subtile = subtile_inplace<WARP_SIZE_KV, DOT_SLICE_QO>(attn_i_smem, {warpid, 0});
            store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
            load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {0, 0}));
            load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {0, 0}));
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // 12. dV_j += P_ij^T @ dO_i
            // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
            __builtin_amdgcn_s_setprio(1);
            P_ij_bf16_col = swap_layout_inplace<col_l, rt_16x32_s>(P_ij_bf16);
            mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
            dP_ij_bf16_col = swap_layout_inplace<col_l, rt_16x32_s>(dP_ij_bf16);
            mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
            load(dP_ij_bf16_col_T, attn_i_smem);
            load(K_j_col, subtile_inplace<256, 16>(K_j_smem, {0, warpid}));
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            __builtin_amdgcn_s_setprio(1);
            zero(dQ_i_T);
            mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T,  dQ_i_T);
            transpose(dQ_i, dQ_i_T);
            mul(dQ_i, dQ_i, dP_SCALE_FACTOR);
            atomic_pk_add_bf16_with_warpid<2>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4 + 2, 0}, warpid);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);
        }

        // dot slice 3
        {
            load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
            load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {1, 0}));
            load(L_i, subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 3));
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // 10. S_ij = Q_i K_j^T * scale
            // 11. P_ij = exp2(S_ij - L_i)
            __builtin_amdgcn_s_setprio(1);
            zero(P_ij);
            mul(L_i, L_i, L_SCALE_FACTOR);
            mma_ABt(P_ij, Q_i, K_j, P_ij);
            mul(P_ij, P_ij, P_SCALE_FACTOR);
            sub_row(P_ij, P_ij, L_i);
            exp2(P_ij, P_ij);
            copy(P_ij_bf16, P_ij);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // 13. dP_ij = dO_i @ V_j^T
            // 14. dS_ij = P_ij o (dP_ij - delta_i)
            load(delta_i, subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 3));
            load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {1, 0}));
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            __builtin_amdgcn_s_setprio(1);
            zero(dP_ij);
            mma_ABt(dP_ij, dO_i, V_j, dP_ij);
            sub_row(dP_ij, dP_ij, delta_i);
            mul(dP_ij, dP_ij, P_ij);
            copy(dP_ij_bf16, dP_ij);
            transpose(dP_ij_bf16_accum_row, dP_ij_bf16);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            G::load<1, false>(dO_i_smem[toc][1], g.dOg, {batch_idx, next_q_seq_idx * 2 + 1, next_q_head_idx, 0});
            auto attn_i_smem_subtile = subtile_inplace<WARP_SIZE_KV, DOT_SLICE_QO>(attn_i_smem, {warpid, 0});
            store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
            load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {1, 0}));
            load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {1, 0}));
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // 12. dV_j += P_ij^T @ dO_i
            // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
            __builtin_amdgcn_s_setprio(1);
            P_ij_bf16_col = swap_layout_inplace<col_l, rt_16x32_s>(P_ij_bf16);
            mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
            dP_ij_bf16_col = swap_layout_inplace<col_l, rt_16x32_s>(dP_ij_bf16);
            mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
            load(dP_ij_bf16_col_T, attn_i_smem);
            load(K_j_col, subtile_inplace<256, 16>(K_j_smem, {0, warpid}));
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            __builtin_amdgcn_s_setprio(1);
            zero(dQ_i_T);
            mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T,  dQ_i_T);
            transpose(dQ_i, dQ_i_T);
            mul(dQ_i, dQ_i, dP_SCALE_FACTOR);
            atomic_pk_add_bf16_with_warpid<2>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4 + 3, 0}, warpid);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);
        }
    }

    const int q_head_idx = (num_steps - 1) / num_steps_per_head + first_q_head;
    const int q_seq_idx = (num_steps - 1) % num_steps_per_head;

    // Sequence Epilogue
    // dot slice 0
    {
        load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
        load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
        load(L_i, subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 0));
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // 10. S_ij = Q_i K_j^T * scale
        // 11. P_ij = exp2(S_ij - L_i)
        __builtin_amdgcn_s_setprio(1);
        zero(P_ij);
        mul(L_i, L_i, L_SCALE_FACTOR);
        mma_ABt(P_ij, Q_i, K_j, P_ij);
        mul(P_ij, P_ij, P_SCALE_FACTOR);
        sub_row(P_ij, P_ij, L_i);
        exp2(P_ij, P_ij);
        copy(P_ij_bf16, P_ij);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // 13. dP_ij = dO_i @ V_j^T
        // 14. dS_ij = P_ij o (dP_ij - delta_i)
        load(delta_i, subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 0));
        load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_s_setprio(1);
        zero(dP_ij);
        mma_ABt(dP_ij, dO_i, V_j, dP_ij);
        sub_row(dP_ij, dP_ij, delta_i);
        mul(dP_ij, dP_ij, P_ij);
        copy(dP_ij_bf16, dP_ij);
        transpose(dP_ij_bf16_accum_row, dP_ij_bf16);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        auto attn_i_smem_subtile = subtile_inplace<WARP_SIZE_KV, DOT_SLICE_QO>(attn_i_smem, {warpid, 0});
        store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
        load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {0, 0}));
        load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {0, 0}));
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // 12. dV_j += P_ij^T @ dO_i
        // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
        __builtin_amdgcn_s_setprio(1);
        P_ij_bf16_col = swap_layout_inplace<col_l, rt_16x32_s>(P_ij_bf16);
        mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        dP_ij_bf16_col = swap_layout_inplace<col_l, rt_16x32_s>(dP_ij_bf16);
        mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
        load(dP_ij_bf16_col_T, attn_i_smem);
        load(K_j_col, subtile_inplace<256, 16>(K_j_smem, {0, warpid}));
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_s_setprio(1);
        zero(dQ_i_T);
        mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T,  dQ_i_T);
        transpose(dQ_i, dQ_i_T);
        mul(dQ_i, dQ_i, dP_SCALE_FACTOR);
        atomic_pk_add_bf16_with_warpid<2>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4, 0}, warpid);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

    }

    // dot slice 1
    {
        load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
        load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {1, 0}));
        load(L_i, subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 1));
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // 10. S_ij = Q_i K_j^T * scale
        // 11. P_ij = exp2(S_ij - L_i)
        __builtin_amdgcn_s_setprio(1);
        zero(P_ij);
        mul(L_i, L_i, L_SCALE_FACTOR);
        mma_ABt(P_ij, Q_i, K_j, P_ij);
        mul(P_ij, P_ij, P_SCALE_FACTOR);
        sub_row(P_ij, P_ij, L_i);
        exp2(P_ij, P_ij);
        copy(P_ij_bf16, P_ij);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // 13. dP_ij = dO_i @ V_j^T
        // 14. dS_ij = P_ij o (dP_ij - delta_i)
        load(delta_i, subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 1));
        load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {1, 0}));
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_s_setprio(1);
        zero(dP_ij);
        mma_ABt(dP_ij, dO_i, V_j, dP_ij);
        sub_row(dP_ij, dP_ij, delta_i);
        mul(dP_ij, dP_ij, P_ij);
        copy(dP_ij_bf16, dP_ij);
        transpose(dP_ij_bf16_accum_row, dP_ij_bf16);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        auto attn_i_smem_subtile = subtile_inplace<WARP_SIZE_KV, DOT_SLICE_QO>(attn_i_smem, {warpid, 0});
        store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
        load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][0], {1, 0}));
        load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][0], {1, 0}));
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // 12. dV_j += P_ij^T @ dO_i
        // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
        __builtin_amdgcn_s_setprio(1);
        P_ij_bf16_col = swap_layout_inplace<col_l, rt_16x32_s>(P_ij_bf16);
        mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        dP_ij_bf16_col = swap_layout_inplace<col_l, rt_16x32_s>(dP_ij_bf16);
        mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
        load(dP_ij_bf16_col_T, attn_i_smem);
        load(K_j_col, subtile_inplace<256, 16>(K_j_smem, {0, warpid}));
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_s_setprio(1);
        zero(dQ_i_T);
        mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T,  dQ_i_T);
        transpose(dQ_i, dQ_i_T);
        mul(dQ_i, dQ_i, dP_SCALE_FACTOR);
        atomic_pk_add_bf16_with_warpid<2>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4 + 1, 0}, warpid);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    }

    // dot slice 2
    {
        load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
        load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {0, 0}));
        load(L_i, subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 2));
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // 10. S_ij = Q_i K_j^T * scale
        // 11. P_ij = exp2(S_ij - L_i)
        __builtin_amdgcn_s_setprio(1);
        zero(P_ij);
        mul(L_i, L_i, L_SCALE_FACTOR);
        mma_ABt(P_ij, Q_i, K_j, P_ij);
        mul(P_ij, P_ij, P_SCALE_FACTOR);
        sub_row(P_ij, P_ij, L_i);
        exp2(P_ij, P_ij);
        copy(P_ij_bf16, P_ij);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // 13. dP_ij = dO_i @ V_j^T
        // 14. dS_ij = P_ij o (dP_ij - delta_i)
        load(delta_i, subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 2));
        load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {0, 0}));
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_s_setprio(1);
        zero(dP_ij);
        mma_ABt(dP_ij, dO_i, V_j, dP_ij);
        sub_row(dP_ij, dP_ij, delta_i);
        mul(dP_ij, dP_ij, P_ij);
        copy(dP_ij_bf16, dP_ij);
        transpose(dP_ij_bf16_accum_row, dP_ij_bf16);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        auto attn_i_smem_subtile = subtile_inplace<WARP_SIZE_KV, DOT_SLICE_QO>(attn_i_smem, {warpid, 0});
        store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
        load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {0, 0}));
        load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {0, 0}));
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // 12. dV_j += P_ij^T @ dO_i
        // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
        __builtin_amdgcn_s_setprio(1);
        P_ij_bf16_col = swap_layout_inplace<col_l, rt_16x32_s>(P_ij_bf16);
        mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        dP_ij_bf16_col = swap_layout_inplace<col_l, rt_16x32_s>(dP_ij_bf16);
        mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
        load(dP_ij_bf16_col_T, attn_i_smem);
        load(K_j_col, subtile_inplace<256, 16>(K_j_smem, {0, warpid}));
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_s_setprio(1);
        zero(dQ_i_T);
        mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T,  dQ_i_T);
        transpose(dQ_i, dQ_i_T);
        mul(dQ_i, dQ_i, dP_SCALE_FACTOR);
        atomic_pk_add_bf16_with_warpid<2>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4 + 2, 0}, warpid);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    }

    // dot slice 3
    {
        load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
        load(Q_i, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {1, 0}));
        load(L_i, subvec_inplace<DOT_SLICE_QO>(L_smem[tic], 3));
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // 10. S_ij = Q_i K_j^T * scale
        // 11. P_ij = exp2(S_ij - L_i)
        __builtin_amdgcn_s_setprio(1);
        zero(P_ij);
        mul(L_i, L_i, L_SCALE_FACTOR);
        mma_ABt(P_ij, Q_i, K_j, P_ij);
        mul(P_ij, P_ij, P_SCALE_FACTOR);
        sub_row(P_ij, P_ij, L_i);
        exp2(P_ij, P_ij);
        copy(P_ij_bf16, P_ij);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // 13. dP_ij = dO_i @ V_j^T
        // 14. dS_ij = P_ij o (dP_ij - delta_i)
        load(delta_i, subvec_inplace<DOT_SLICE_QO>(delta_smem[tic], 3));
        load(dO_i, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {1, 0}));
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_s_setprio(1);
        zero(dP_ij);
        mma_ABt(dP_ij, dO_i, V_j, dP_ij);
        sub_row(dP_ij, dP_ij, delta_i);
        mul(dP_ij, dP_ij, P_ij);
        copy(dP_ij_bf16, dP_ij);
        transpose(dP_ij_bf16_accum_row, dP_ij_bf16);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        auto attn_i_smem_subtile = subtile_inplace<WARP_SIZE_KV, DOT_SLICE_QO>(attn_i_smem, {warpid, 0});
        store(attn_i_smem_subtile, dP_ij_bf16_accum_row);
        load(dO_i_col, subtile_inplace<DOT_SLICE_QO, D>(dO_i_smem[tic][1], {1, 0}));
        load(Q_i_col, subtile_inplace<DOT_SLICE_QO, D>(Q_i_smem[tic][1], {1, 0}));
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // 12. dV_j += P_ij^T @ dO_i
        // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
        __builtin_amdgcn_s_setprio(1);
        P_ij_bf16_col = swap_layout_inplace<col_l, rt_16x32_s>(P_ij_bf16);
        mma_AtB(dV_j_T, dO_i_col, P_ij_bf16_col, dV_j_T);
        dP_ij_bf16_col = swap_layout_inplace<col_l, rt_16x32_s>(dP_ij_bf16);
        mma_AtB(dK_j_T, Q_i_col, dP_ij_bf16_col, dK_j_T);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
        load(dP_ij_bf16_col_T, attn_i_smem);
        load(K_j_col, subtile_inplace<256, 16>(K_j_smem, {0, warpid}));
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_s_setprio(1);
        zero(dQ_i_T);
        mma_AtB(dQ_i_T, K_j_col, dP_ij_bf16_col_T,  dQ_i_T);
        transpose(dQ_i, dQ_i_T);
        mul(dQ_i, dQ_i, dP_SCALE_FACTOR);
        atomic_pk_add_bf16_with_warpid<2>(g.dQg, dQ_i, {batch_idx, q_head_idx, q_seq_idx * 4 + 3, 0}, warpid);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    }

    if (!stagger) {
        __builtin_amdgcn_s_barrier();
    }

    // 18. Write dK_j and dV_j back to HBM (using KV head index)
    kv_tile<D, float, row_l, rt_32x32_s> dK_j, dV_j;
    transpose(dK_j, dK_j_T);
    transpose(dV_j, dV_j_T);
    store<1>(g.dVg, dV_j, {batch_idx, j, kv_head_idx, 0});
    mul(dK_j, dK_j, dP_SCALE_FACTOR);
    store<1>(g.dKg, dK_j, {batch_idx, j, kv_head_idx, 0});
}

template<int D>
void dispatch_bwd_combined(attn_bwd_combined_globals<D> g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)attend_bwd_combined_ker<D>, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    attend_bwd_combined_ker<D><<<g.grid(), g.block(), mem_size, g.stream>>>(g);
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