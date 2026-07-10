#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

#ifndef ATTN_B
constexpr int ATTN_B = 16; // batch size
#endif

#ifndef ATTN_H
constexpr int ATTN_H = 64; // number of heads
#endif

#ifndef ATTN_H_KV
constexpr int ATTN_H_KV = 8; // number of heads for key and value
#endif

constexpr int GROUP_SIZE = ATTN_H / ATTN_H_KV; // queries per KV head group

#ifndef ATTN_N
constexpr int ATTN_N = 1024; // sequence length
#endif

#ifndef ATTN_D
constexpr int ATTN_D = 64; // dimension
#endif

constexpr int Q_BLOCK_SIZE = 32; // q block size
constexpr int KV_BLOCK_SIZE = 64; // kv block size

#define NUM_WARPS 8
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

#define MFMA_MASK 0x08
#define VALU_MASK 0x02
#define EXP_MASK  0x400

using namespace kittens;
using _gl_QKVO = gl<bf16, -1, -1, -1, -1>;

using G = kittens::group<NUM_WARPS>;

#define SCHED_BARRIER(mask, cnt, group) __builtin_amdgcn_sched_group_barrier(mask, cnt, group)

template<int Pairs, int VALU_CNT, int Group>
__device__ __forceinline__ void sched_barrier_pairs() {
    SCHED_BARRIER(MFMA_MASK, 1, Group);
    SCHED_BARRIER(VALU_MASK, VALU_CNT, Group);
    if constexpr (Pairs > 1) sched_barrier_pairs<Pairs - 1, VALU_CNT, Group>();
}

template<int Pairs, int EXP_CNT, int Group>
__device__ __forceinline__ void sched_barrier_exp_pairs() {
    SCHED_BARRIER(MFMA_MASK, 1, Group);
    SCHED_BARRIER(EXP_MASK, EXP_CNT, Group);
    if constexpr (Pairs > 1) sched_barrier_exp_pairs<Pairs - 1, EXP_CNT, Group>();
}

template<typename T, ducks::rt_layout::all layout, ducks::rt_shape::all shape>
__device__ inline void exp2(rt_base<T, layout, shape> &dst, const rt_base<T, layout, shape> &src) {
    static_assert(std::is_same_v<shape, rt_32x32_s>, "Only 32x32 tiles are supported");

    #pragma unroll
    for(int k = 0; k < dst.packed_per_thread; k++) {
        dst.data[k] = base_ops::exp2::op(src.data[k]);
    }

}

constexpr float RESCALE_THRESHOLD = 8.0f;  // https://github.com/Dao-AILab/flash-attention/blob/add175637c5d54b74bc25372e49ce282d6f236fc/flash_attn/cute/flash_fwd_sm100.py#L1229

template<typename RV>
__device__ __forceinline__ int rv_all_below(const RV& prev, const RV& cur, float T) {
    int ok = 1;
    #pragma unroll
    for (int o = 0; o < RV::outer_dim; ++o) {
        #pragma unroll
        for (int i = 0; i < RV::inner_dim; ++i) {
            ok &= (float(cur.data[o][i]) - float(prev.data[o][i]) <= T);
        }
    }
    return ok;  // 1 or 0, per lane
}
// Wave-uniform: true iff EVERY lane says "all below"
__device__ __forceinline__ int wave_all_ok(int lane_ok) {
    return __all(lane_ok);  // 1 fast vote; no ds_bpermute ladder
}

template<int D, typename T=bf16, typename L=row_l, typename S=rt_32x16_s> using qo_tile = rt<T, Q_BLOCK_SIZE, D, L, S>;
template<int D, typename T=bf16, typename L=col_l, typename S=rt_16x32_s> using qo_tile_transposed = rt<T, D, Q_BLOCK_SIZE, L, S>;
template<int D, typename T=bf16, typename L=row_l, typename S=rt_32x16_s> using kv_tile = rt<T, KV_BLOCK_SIZE, D, L, S>;
template<int D, typename T=bf16, typename L=col_l, typename S=rt_16x32_s> using kv_tile_transposed = rt<T, D, KV_BLOCK_SIZE, L, S>;
template<int D, typename T=float, typename L=col_l, typename S=rt_16x32_4_s> using attn_tile = rt<T, KV_BLOCK_SIZE, Q_BLOCK_SIZE, L, S>;

template<int D> struct attn_globals { 
    _gl_QKVO Qg, Kg, Vg, Og; 
    gl<float, -1, -1, -1, -1> L_vec;
    hipStream_t stream;
    dim3 grid() { return dim3(ATTN_H, ((ATTN_N / Q_BLOCK_SIZE + NUM_WARPS - 1) / NUM_WARPS), ATTN_B); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

template<int D> __launch_bounds__(NUM_THREADS, 2)
__global__ void attend_ker(const attn_globals<D> g) {

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<KV_BLOCK_SIZE, ATTN_D, st_32x32_s> (&k_smem)[2] = al.allocate<st_bf<KV_BLOCK_SIZE, ATTN_D, st_32x32_s>, 2>();
    st_bf<KV_BLOCK_SIZE, ATTN_D, st_8x32_s> (&v_smem)[2] = al.allocate<st_bf<KV_BLOCK_SIZE, ATTN_D, st_8x32_s>, 2>();
    
    const int head_idx = (blockIdx.x % ATTN_H_KV) * GROUP_SIZE + (blockIdx.x / ATTN_H_KV);
    const int batch_idx = blockIdx.z;
    const int head_idx_kv = head_idx / GROUP_SIZE;
    const int block_tile_idx = blockIdx.y;
    const int tile_idx = block_tile_idx * NUM_WARPS + warpid();
    const int stagger = warpid() / 4;

    /********** Readfirstlane hoisting **********/
    // Create base buffer resources once
    const bf16* k_base = (bf16*)&g.Kg[{batch_idx, 0, head_idx_kv, 0}];
    const bf16* v_base = (bf16*)&g.Vg[{batch_idx, 0, head_idx_kv, 0}];
    const int k_row_stride = g.Kg.template stride<1>() * sizeof(bf16);
    const int v_row_stride = g.Vg.template stride<1>() * sizeof(bf16);
    i32x4 k_srsrc_base = make_srsrc(k_base, k_row_stride * ATTN_N, k_row_stride);
    i32x4 v_srsrc_base = make_srsrc(v_base, v_row_stride * ATTN_N, v_row_stride);

    const int wid = warpid() % NUM_WARPS;
    constexpr int elem_per_warp = (16 / sizeof(bf16)) * kittens::WARP_THREADS;
    uint32_t k_lds_base_0 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(&k_smem[0].data[0]) + wid * elem_per_warp * sizeof(bf16)
    ));
    uint32_t v_lds_base_0 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(&v_smem[0].data[0]) + wid * elem_per_warp * sizeof(bf16)
    ));
    uint32_t k_lds_base_1 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(&k_smem[1].data[0]) + wid * elem_per_warp * sizeof(bf16)
    ));
    uint32_t v_lds_base_1 = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(&v_smem[1].data[0]) + wid * elem_per_warp * sizeof(bf16)
    ));
    /*****************************************/


    const int num_tiles = ATTN_N / KV_BLOCK_SIZE;

    constexpr float TEMPERATURE_SCALE = (D == 128) ? 0.08838834764f*1.44269504089f : 0.125f*1.44269504089f;

    // Initialize all of the register tiles.
    qo_tile<D, bf16> q_reg; // Q and K are both row layout, as we use mma_ABt.
    qo_tile_transposed<D, bf16> q_reg_transposed;
    kv_tile<D, bf16> k_reg;
    kv_tile_transposed<D, bf16> k_reg_transposed;

    kv_tile<D, bf16, col_l, rt_16x32_4_s> v_reg;
    qo_tile_transposed<D, float, col_l, rt_32x32_s> o_reg; // Output tile.
    attn_tile<D, float, col_l, rt_32x32_s> att_block[2]; // attention tile, in float.
    attn_tile<D, bf16, col_l, rt_32x32_s> att_block_bf16;
    attn_tile<D, bf16, col_l, rt_16x32_4_s> att_block_bf16_in;
    typename attn_tile<D, float, col_l, rt_32x32_s>::row_vec max_vec, norm_vec, max_vec_prev, scale_vec;

    zero(o_reg);
    zero(norm_vec);
    ones(scale_vec);

    using T = typename st_bf<KV_BLOCK_SIZE, ATTN_D, st_32x32_s>::dtype;
    constexpr int bytes_per_thread = st_32x32_s::template bytes_per_thread<T>();
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_THREADS;
    constexpr int memcpy_per_tile = KV_BLOCK_SIZE * ATTN_D * sizeof(T) / bytes_per_memcpy;

    uint32_t swizzled_offsets_V[memcpy_per_tile];
    uint32_t swizzled_offsets_K[memcpy_per_tile];
    G::prefill_swizzled_offsets<1, false>(k_smem[0], g.Kg, swizzled_offsets_K);
    G::prefill_swizzled_offsets<1, false>(v_smem[0], g.Vg, swizzled_offsets_V);

    G::load<1, false>(k_smem[0], g.Kg, {batch_idx, 0, head_idx_kv, 0}, swizzled_offsets_K, k_srsrc_base, k_base, k_lds_base_0);
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();

    qo_tile<D, float> q_reg_fl;
    load<1, qo_tile<D, float>, _gl_QKVO>(q_reg_fl, g.Qg, {batch_idx, tile_idx, head_idx, 0});
    mul(q_reg_fl, q_reg_fl, TEMPERATURE_SCALE);  // Use sqrtf for clarity
    copy(q_reg, q_reg_fl);
    transpose(q_reg_transposed, q_reg);

    // All warps then collaboratively load in the first slice of V (V0) and the second slice of K (K1) into shared memory
    G::load<1, false>(k_smem[1], g.Kg, {batch_idx, 1, head_idx_kv, 0}, swizzled_offsets_K, k_srsrc_base, k_base, k_lds_base_1);
    // All warps then load in the first slice of K (K0)
    G::load<1, false>(v_smem[0], g.Vg, {batch_idx, 0, head_idx_kv, 0}, swizzled_offsets_V, v_srsrc_base, v_base, v_lds_base_0);
    load(k_reg, k_smem[0]);
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt lgkmcnt(0)");
    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();

    // Each warp performs QK0
    zero(att_block[0]);
    transpose(k_reg_transposed, k_reg);
    mma_AtB(att_block[0], k_reg_transposed, q_reg_transposed, att_block[0]);

    // Each warp performs a partial softmax of QK0 (i.e. some of the online softmax up until but not including the second exponential scaling of the attention block likely)
    col_max(max_vec, att_block[0]);
    copy(max_vec_prev, max_vec);
    sub_col(att_block[0], att_block[0], max_vec);
    exp2(att_block[0].tiles[0][0], att_block[0].tiles[0][0]);
    __builtin_amdgcn_sched_barrier(0);

    if (stagger) {
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
    }

    __builtin_amdgcn_sched_barrier(0);
    // All warps then load in the second slice of K (K1)
    load(k_reg, k_smem[1]);
    // All warps then collaboratively load in the third slice of K (K2) into shared memory
    G::load<1, false>(k_smem[0], g.Kg, {batch_idx, 2, head_idx_kv, 0}, swizzled_offsets_K, k_srsrc_base, k_base, k_lds_base_0);
    // All warps then collaboratively load in the second slice of V (V1) into shared memory 
    G::load<1, false>(v_smem[1], g.Vg, {batch_idx, 1, head_idx_kv, 0}, swizzled_offsets_V, v_srsrc_base, v_base, v_lds_base_1);
    asm volatile("s_waitcnt lgkmcnt(0)");
    asm volatile("s_waitcnt vmcnt(2)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();

    int pending_scale = 0;


    // hot loop
    #pragma unroll 2
    for (int j = 3; j < num_tiles - 1; j += 2) {
        // Cluster 0:
        //      QK1
        zero(att_block[1]);
        transpose(k_reg_transposed, k_reg);
        if (pending_scale) {
            mul(norm_vec, norm_vec, scale_vec);
        }
        mma_AtB(att_block[1], k_reg_transposed, q_reg_transposed, att_block[1]);
        copy(subtile_inplace<32>(att_block_bf16, 0), subtile_inplace<32>(att_block[0], 0));
        //      Finish softmax for QK0
        exp2(att_block[0].tiles[1][0], att_block[0].tiles[1][0]);
        col_sum(norm_vec, att_block[0], norm_vec);
        // copy(att_block_bf16, att_block[0]);
        copy(subtile_inplace<32>(att_block_bf16, 1), subtile_inplace<32>(att_block[0], 1));
        att_block_bf16_in = *reinterpret_cast<attn_tile<D, bf16, col_l, rt_16x32_4_s>*>(&att_block_bf16);
        sched_barrier_exp_pairs<6, 3, 1>();
        sched_barrier_pairs<10, 5, 1>();
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 1:
        //      Load K3 into shared 
        G::load<1, false>(k_smem[1], g.Kg, {batch_idx, j, head_idx_kv, 0}, swizzled_offsets_K, k_srsrc_base, k_base, k_lds_base_1);
        //      Load V0 into registers
        load(v_reg, v_smem[0]);
        asm volatile("s_waitcnt lgkmcnt(0)");
        asm volatile("s_waitcnt vmcnt(2)");
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 2:
        //      A0V0
        __builtin_amdgcn_s_setprio(1);
        mma_AtB(o_reg, subtile_inplace<16>(v_reg, 0), subtile_inplace<16>(att_block_bf16_in, 0), o_reg);
        //      Partial softmax for QK1
        col_max(max_vec, att_block[1], max_vec_prev);
        sched_barrier_pairs<4, 5, 2>();
        int lane_ok = rv_all_below(max_vec_prev, max_vec, RESCALE_THRESHOLD);
        int all_ok  = wave_all_ok(lane_ok);
        if (__builtin_expect(all_ok, 1)) {
            copy(max_vec, max_vec_prev);
            pending_scale = 0;
        } else {
            sub(scale_vec, max_vec_prev, max_vec);   // diff = prev - new
            exp2(scale_vec, scale_vec);
            mul_col(o_reg, o_reg, scale_vec);
            copy(max_vec_prev, max_vec);
            pending_scale = 1;
        }
        mma_AtB(o_reg, subtile_inplace<16>(v_reg, 1), subtile_inplace<16>(att_block_bf16_in, 1), o_reg);
        mma_AtB(o_reg, subtile_inplace<16>(v_reg, 2), subtile_inplace<16>(att_block_bf16_in, 2), o_reg);
        mma_AtB(o_reg, subtile_inplace<16>(v_reg, 3), subtile_inplace<16>(att_block_bf16_in, 3), o_reg);
        sub_col(att_block[1], att_block[1], max_vec);
        exp2(att_block[1].tiles[0][0], att_block[1].tiles[0][0]);
        sched_barrier_pairs<6, 5, 2>();
        sched_barrier_exp_pairs<6, 3, 2>();
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 3:
        //      Load V2 into shared
        G::load<1, false>(v_smem[0], g.Vg, {batch_idx, j - 1, head_idx_kv, 0}, swizzled_offsets_V, v_srsrc_base, v_base, v_lds_base_0);
        //      Load K2 into registers
        load(k_reg, k_smem[0]);
        asm volatile("s_waitcnt lgkmcnt(0)");
        asm volatile("s_waitcnt vmcnt(2)");
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 4:
        //      QK2
        zero(att_block[0]);
        transpose(k_reg_transposed, k_reg);
        if (pending_scale) {
            mul(norm_vec, norm_vec, scale_vec);
        }
        mma_AtB(att_block[0], k_reg_transposed, q_reg_transposed, att_block[0]);
        //      Finish softmax for QK1
        exp2(att_block[1].tiles[1][0], att_block[1].tiles[1][0]);
        col_sum(norm_vec, att_block[1], norm_vec);
        copy(att_block_bf16, att_block[1]);
        att_block_bf16_in = *reinterpret_cast<attn_tile<D, bf16, col_l, rt_16x32_4_s>*>(&att_block_bf16);
        sched_barrier_exp_pairs<6, 3, 3>();
        sched_barrier_pairs<10, 5, 3>();
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 5:
        //      Load K4 into shared
        G::load<1, false>(k_smem[0], g.Kg, {batch_idx, j + 1, head_idx_kv, 0}, swizzled_offsets_K, k_srsrc_base, k_base, k_lds_base_0);
        //      Load V1 into registers
        load(v_reg, v_smem[1]);
        asm volatile("s_waitcnt lgkmcnt(0)");
        asm volatile("s_waitcnt vmcnt(2)");
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 6:
        //      A1V1
        __builtin_amdgcn_s_setprio(1);
        mma_AtB(o_reg, subtile_inplace<16>(v_reg, 0), subtile_inplace<16>(att_block_bf16_in, 0), o_reg);
        //      Partial softmax for QK2
        col_max(max_vec, att_block[0], max_vec_prev);
        sched_barrier_pairs<4, 5, 4>();
        lane_ok = rv_all_below(max_vec_prev, max_vec, RESCALE_THRESHOLD);
        all_ok  = wave_all_ok(lane_ok);
        if (__builtin_expect(all_ok, 1)) {
            copy(max_vec, max_vec_prev);
            pending_scale = 0;
        } else {
            sub(scale_vec, max_vec_prev, max_vec);   // diff = prev - new
            exp2(scale_vec, scale_vec);
            mul_col(o_reg, o_reg, scale_vec);
            copy(max_vec_prev, max_vec);
            pending_scale = 1;
        }
        mma_AtB(o_reg, subtile_inplace<16>(v_reg, 1), subtile_inplace<16>(att_block_bf16_in, 1), o_reg);
        mma_AtB(o_reg, subtile_inplace<16>(v_reg, 2), subtile_inplace<16>(att_block_bf16_in, 2), o_reg);
        mma_AtB(o_reg, subtile_inplace<16>(v_reg, 3), subtile_inplace<16>(att_block_bf16_in, 3), o_reg);
        sub_col(att_block[0], att_block[0], max_vec);
        exp2(att_block[0].tiles[0][0], att_block[0].tiles[0][0]);
        sched_barrier_pairs<6, 5, 4>();
        sched_barrier_exp_pairs<6, 3, 4>();
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 7:
        //      Load V3 into shared
        G::load<1, false>(v_smem[1], g.Vg, {batch_idx, j, head_idx_kv, 0}, swizzled_offsets_V, v_srsrc_base, v_base, v_lds_base_1);
        //      Load K3 into registers
        load(k_reg, k_smem[1]);
        asm volatile("s_waitcnt lgkmcnt(0)");
        asm volatile("s_waitcnt vmcnt(2)");
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    }

    // Epilogue
    // Cluster 0:
    //      QK3
    zero(att_block[1]);
    transpose(k_reg_transposed, k_reg);
    mma_AtB(att_block[1], k_reg_transposed, q_reg_transposed, att_block[1]);
    //      Finish softmax for QK2
    exp2(att_block[0].tiles[1][0], att_block[0].tiles[1][0]);
    mul(norm_vec, norm_vec, scale_vec);
    col_sum(norm_vec, att_block[0], norm_vec);
    copy(att_block_bf16, att_block[0]);
    att_block_bf16_in = *reinterpret_cast<attn_tile<D, bf16, col_l, rt_16x32_4_s>*>(&att_block_bf16);
    sched_barrier_exp_pairs<6, 3, 5>();
    sched_barrier_pairs<10, 5, 5>();
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 1:
    //      Load K5 into shared
    G::load<1, false>(k_smem[1], g.Kg, {batch_idx, num_tiles - 1, head_idx_kv, 0}, swizzled_offsets_K, k_srsrc_base, k_base, k_lds_base_1);
    //      Load V2 into registers
    load(v_reg, v_smem[0]);
    asm volatile("s_waitcnt lgkmcnt(0)");
    asm volatile("s_waitcnt vmcnt(2)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 2:
    //      A2V2
    __builtin_amdgcn_s_setprio(1);
    mma_AtB(o_reg, v_reg, att_block_bf16_in, o_reg);
    //      Partial softmax for QK3
    col_max(max_vec, att_block[1], max_vec_prev);
    sub(scale_vec, max_vec_prev, max_vec);
    copy(max_vec_prev, max_vec);
    exp2(scale_vec, scale_vec);
    sub_col(att_block[1], att_block[1], max_vec);
    exp2(att_block[1].tiles[0][0], att_block[1].tiles[0][0]);
    sched_barrier_pairs<10, 5, 6>();
    sched_barrier_exp_pairs<6, 3, 6>();
    __builtin_amdgcn_sched_barrier(0);
    mul_col(o_reg, o_reg, scale_vec);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 3:
    //      Load V4 into shared
    G::load<1, false>(v_smem[0], g.Vg, {batch_idx, num_tiles - 2, head_idx_kv, 0}, swizzled_offsets_V, v_srsrc_base, v_base, v_lds_base_0);
    //      Load K4 into registers
    load(k_reg, k_smem[0]);
    asm volatile("s_waitcnt lgkmcnt(0)");
    asm volatile("s_waitcnt vmcnt(2)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 4:
    //      QK4
    zero(att_block[0]);
    transpose(k_reg_transposed, k_reg);
    mma_AtB(att_block[0], k_reg_transposed, q_reg_transposed, att_block[0]);
    //      Finish softmax for QK3
    exp2(att_block[1].tiles[1][0], att_block[1].tiles[1][0]);
    mul(norm_vec, norm_vec, scale_vec);
    col_sum(norm_vec, att_block[1], norm_vec);
    copy(att_block_bf16, att_block[1]);
    att_block_bf16_in = *reinterpret_cast<attn_tile<D, bf16, col_l, rt_16x32_4_s>*>(&att_block_bf16);
    sched_barrier_exp_pairs<6, 3, 7>();
    sched_barrier_pairs<10, 5, 7>();
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 5:
    //      Load V3 into registers
    load(v_reg, v_smem[1]);
    asm volatile("s_waitcnt lgkmcnt(0)");
    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 6:
    //      A3V3
    __builtin_amdgcn_s_setprio(1);
    mma_AtB(o_reg, v_reg, att_block_bf16_in, o_reg);
    //      Partial softmax for QK4
    col_max(max_vec, att_block[0], max_vec_prev);
    sub(scale_vec, max_vec_prev, max_vec);
    copy(max_vec_prev, max_vec);
    exp2(scale_vec, scale_vec);
    sub_col(att_block[0], att_block[0], max_vec);
    exp2(att_block[0].tiles[0][0], att_block[0].tiles[0][0]);
    sched_barrier_pairs<10, 5, 8>();
    sched_barrier_exp_pairs<6, 3, 8>();
    __builtin_amdgcn_sched_barrier(0);
    mul_col(o_reg, o_reg, scale_vec);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 7:
    //      Load V5 into shared
    G::load<1, false>(v_smem[1], g.Vg, {batch_idx, num_tiles - 1, head_idx_kv, 0}, swizzled_offsets_V, v_srsrc_base, v_base, v_lds_base_1);
    //      Load K5 into registers
    load(k_reg, k_smem[1]);
    asm volatile("s_waitcnt lgkmcnt(0)");
    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 8:
    //      QK5
    zero(att_block[1]);
    transpose(k_reg_transposed, k_reg);
    mma_AtB(att_block[1], k_reg_transposed, q_reg_transposed, att_block[1]);
    //      Finish softmax for QK4
    exp2(att_block[0].tiles[1][0], att_block[0].tiles[1][0]);
    mul(norm_vec, norm_vec, scale_vec);
    col_sum(norm_vec, att_block[0], norm_vec);
    copy(att_block_bf16, att_block[0]);
    att_block_bf16_in = *reinterpret_cast<attn_tile<D, bf16, col_l, rt_16x32_4_s>*>(&att_block_bf16);
    sched_barrier_exp_pairs<6, 3, 9>();
    sched_barrier_pairs<10, 5, 9>();
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 9:
    //      Load V4 into registers
    load(v_reg, v_smem[0]);
    asm volatile("s_waitcnt lgkmcnt(0)");
    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 10:
    //      A4V4
    mma_AtB(o_reg, v_reg, att_block_bf16_in, o_reg);
    //      Full softmax for QK5
    col_max(max_vec, att_block[1], max_vec_prev);
    sub(scale_vec, max_vec_prev, max_vec);
    copy(max_vec_prev, max_vec);
    exp2(scale_vec, scale_vec);
    sub_col(att_block[1], att_block[1], max_vec);
    exp2(att_block[1].tiles[0][0], att_block[1].tiles[0][0]);
    sched_barrier_pairs<10, 5, 10>();
    sched_barrier_exp_pairs<6, 3, 10>();
    __builtin_amdgcn_sched_barrier(0);

    exp2(att_block[1].tiles[1][0], att_block[1].tiles[1][0]);
    mul(norm_vec, norm_vec, scale_vec);
    col_sum(norm_vec, att_block[1], norm_vec);
    copy(att_block_bf16, att_block[1]);
    att_block_bf16_in = *reinterpret_cast<attn_tile<D, bf16, col_l, rt_16x32_4_s>*>(&att_block_bf16);
    __builtin_amdgcn_sched_barrier(0);
    mul_col(o_reg, o_reg, scale_vec);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 11:
    //      Load V5 into registers
    load(v_reg, v_smem[1]);
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 12:
    //      A5V5
    mma_AtB(o_reg, v_reg, att_block_bf16_in, o_reg);
    div_col(o_reg, o_reg, norm_vec);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Conclusion
    if (!stagger) {
        __builtin_amdgcn_s_barrier();
    }

    qo_tile<D, float, row_l, rt_32x32_s> o_reg_transposed;
    transpose(o_reg_transposed, o_reg);
    store<1>(g.Og, o_reg_transposed, {batch_idx, tile_idx, head_idx, 0});

    // multiply by ln(2)
    mul(max_vec, max_vec, 0.69314718056f);
    log(norm_vec, norm_vec);
    add(norm_vec, norm_vec, max_vec);
    store(g.L_vec, norm_vec, {batch_idx, head_idx, 0, tile_idx});
}

template<int D>
void dispatch_micro(attn_globals<D> g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)attend_ker<D>, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    attend_ker<D><<<g.grid(), g.block(), mem_size, g.stream>>>(g);
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module";
    py::bind_function<dispatch_micro<ATTN_D>>(m, "dispatch_micro", 
        &attn_globals<ATTN_D>::Qg, 
        &attn_globals<ATTN_D>::Kg, 
        &attn_globals<ATTN_D>::Vg, 
        &attn_globals<ATTN_D>::Og,
        &attn_globals<ATTN_D>::L_vec
    );
}