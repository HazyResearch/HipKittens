#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include "utils.cpp"

constexpr int ATTN_B = 16; // batch size
constexpr int ATTN_H = 64; // number of heads
constexpr int ATTN_H_KV = 8; // number of heads for key and value
constexpr int ATTN_N = 8192; // sequence length
constexpr int ATTN_D = 128; // dimension
constexpr int Q_BLOCK_SIZE = 32; // q block size
constexpr int KV_BLOCK_SIZE = 64; // kv block size

#define NUM_WARPS 8
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using namespace kittens;
using _gl_QKVO = gl<bf16, -1, -1, -1, -1>;

template<int D, typename T=bf16, typename L=row_l> using qo_tile = rt<T, Q_BLOCK_SIZE, D, L>;
template<int D, typename T=bf16, typename L=col_l> using qo_tile_transposed = rt<T, D, Q_BLOCK_SIZE, L>;
template<int D, typename T=bf16, typename L=row_l> using kv_tile = rt<T, KV_BLOCK_SIZE, D, L>;
template<int D, typename T=bf16, typename L=col_l> using kv_tile_transposed = rt<T, D, KV_BLOCK_SIZE, L>;
template<int D, typename T=float, typename L=accum_l> using attn_tile = rt<T, KV_BLOCK_SIZE, Q_BLOCK_SIZE, L>;

template<int D> struct attn_globals { 
    _gl_QKVO Qg, Kg, Vg, Og; 
    dim3 grid() { return dim3(ATTN_H, ((ATTN_N / Q_BLOCK_SIZE + NUM_WARPS - 1) / NUM_WARPS), ATTN_B); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

template<int D> __launch_bounds__(NUM_THREADS, 2)
__global__ void attend_ker(const attn_globals<D> g) {

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<KV_BLOCK_SIZE, ATTN_D> (&k_smem)[2] = al.allocate<st_bf<KV_BLOCK_SIZE, ATTN_D>, 2>();
    st_bf<KV_BLOCK_SIZE, ATTN_D> (&v_smem)[2] = al.allocate<st_bf<KV_BLOCK_SIZE, ATTN_D>, 2>();
    
    const int head_idx = (blockIdx.x % 8) * 8 + (blockIdx.x / 8);
    const int batch_idx = blockIdx.z;
    const int GROUP_SIZE = ATTN_H / ATTN_H_KV;
    const int head_idx_kv = head_idx / GROUP_SIZE;
    const int block_tile_idx = blockIdx.y;
    const int tile_idx = block_tile_idx * NUM_WARPS + warpid();
    const int stagger = warpid() / 4;

    const int num_tiles = ATTN_N / KV_BLOCK_SIZE;

    constexpr float TEMPERATURE_SCALE = (D == 128) ? 0.08838834764f*1.44269504089f : 0.125f*1.44269504089f;

    // Initialize all of the register tiles.
    qo_tile<D, bf16> q_reg; // Q and K are both row layout, as we use mma_ABt.
    qo_tile_transposed<D, bf16> q_reg_transposed;
    kv_tile<D, bf16> k_reg;
    kv_tile_transposed<D, bf16> k_reg_transposed;

    kv_tile<D, bf16, col_l> v_reg;
    qo_tile_transposed<D, float, accum_l> o_reg; // Output tile.
    attn_tile<D, float, accum_l> att_block[2]; // attention tile, in float.
    attn_tile<D, bf16, accum_l> att_block_bf16;
    attn_tile<D, bf16, col_l> att_block_col_bf16;
    typename attn_tile<D, float, accum_l>::row_vec max_vec, norm_vec, max_vec_prev;

    using T = typename st_bf<KV_BLOCK_SIZE, ATTN_D>::dtype;
    constexpr int bytes_per_thread = 16;
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_THREADS;
    constexpr int memcpy_per_tile = KV_BLOCK_SIZE * ATTN_D * sizeof(T) / bytes_per_memcpy;

    uint32_t swizzled_offsets_V[memcpy_per_tile];
    uint32_t swizzled_offsets_K[memcpy_per_tile];
    prefill_swizzled_offsets<1, false, rt_bf<KV_BLOCK_SIZE, ATTN_D>, st_bf<KV_BLOCK_SIZE, ATTN_D>, _gl_QKVO, coord<st_bf<KV_BLOCK_SIZE, ATTN_D>>, NUM_THREADS>(g.Kg, {batch_idx, 0, head_idx_kv, 0}, k_smem[0], swizzled_offsets_K);
    prefill_swizzled_offsets<1, false, rt_bf<KV_BLOCK_SIZE, ATTN_D, ducks::rt_layout::accumulator>, st_bf<KV_BLOCK_SIZE, ATTN_D>, _gl_QKVO, coord<st_bf<KV_BLOCK_SIZE, ATTN_D>>, NUM_THREADS>(g.Vg, {batch_idx, 0, head_idx_kv, 0}, v_smem[0], swizzled_offsets_V);

    load_global_to_shared_direct_with_swizzled_offsets<1, false, st_bf<KV_BLOCK_SIZE, ATTN_D>, _gl_QKVO, coord<st_bf<KV_BLOCK_SIZE,ATTN_D>>, NUM_THREADS>(
        g.Kg, {batch_idx, 0, head_idx_kv, 0}, k_smem[0], swizzled_offsets_K);

    // Pre-scale Q by temperature
    qo_tile<D, float> q_reg_fl;
    load<1, qo_tile<D, float>, _gl_QKVO>(q_reg_fl, g.Qg, {batch_idx, tile_idx, head_idx, 0});
    mul(q_reg_fl, q_reg_fl, TEMPERATURE_SCALE);  // Use sqrtf for clarity
    copy(q_reg, q_reg_fl);
    swap_layout_and_transpose(q_reg_transposed, q_reg);

    zero(o_reg);
    zero(norm_vec);
    neg_infty(max_vec_prev);

    // All warps then collaboratively load in the first slice of V (V0) and the second slice of K (K1) into shared memory
    load_global_to_shared_direct_with_swizzled_offsets<1, false, st_bf<KV_BLOCK_SIZE, ATTN_D>, _gl_QKVO, coord<st_bf<KV_BLOCK_SIZE,ATTN_D>>, NUM_THREADS>(
        g.Kg, {batch_idx, 1, head_idx_kv, 0}, k_smem[1], swizzled_offsets_K);
    // All warps then load in the first slice of K (K0)
    load_global_to_shared_direct_with_swizzled_offsets<1, false, st_bf<KV_BLOCK_SIZE, ATTN_D>, _gl_QKVO, coord<st_bf<KV_BLOCK_SIZE,ATTN_D>>, NUM_THREADS>(
        g.Vg, {batch_idx, 0, head_idx_kv, 0}, v_smem[0], swizzled_offsets_V);
    load_lds_reg_row(k_reg, k_smem[0]);
    swap_layout_and_transpose(k_reg_transposed, k_reg);
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();

    // Each warp performs QK0
    asm volatile("s_waitcnt lgkmcnt(0)");
    zero(att_block[0]);
    mma_AtB(att_block[0], k_reg_transposed, q_reg_transposed, att_block[0]);

    // Each warp performs a partial softmax of QK0 (i.e. some of the online softmax up until but not including the second exponential scaling of the attention block likely)
    // col_max(max_vec, att_block[0], max_vec);
    col_max(max_vec, att_block[0]);
    sub_col(att_block[0], att_block[0], max_vec);
    exp2(att_block[0], att_block[0]);

    if (stagger) {
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
    }

    // All warps then load in the second slice of K (K1)
    load_lds_reg_row(k_reg, k_smem[1]);
    // All warps then collaboratively load in the third slice of K (K2) into shared memory
    load_global_to_shared_direct_with_swizzled_offsets<1, false, st_bf<KV_BLOCK_SIZE, ATTN_D>, _gl_QKVO, coord<st_bf<KV_BLOCK_SIZE,ATTN_D>>, NUM_THREADS>(
        g.Kg, {batch_idx, 2, head_idx_kv, 0}, k_smem[0], swizzled_offsets_K);
    swap_layout_and_transpose(k_reg_transposed, k_reg);
    // All warps then collaboratively load in the second slice of V (V1) into shared memory 
    load_global_to_shared_direct_with_swizzled_offsets<1, false, st_bf<KV_BLOCK_SIZE, ATTN_D>, _gl_QKVO, coord<st_bf<KV_BLOCK_SIZE,ATTN_D>>, NUM_THREADS>(
        g.Vg, {batch_idx, 1, head_idx_kv, 0}, v_smem[1], swizzled_offsets_V);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();

    

    // hot loop
    #pragma unroll (1)
    for (int j = 3; j < num_tiles - 1; j += 2) {
        // Cluster 0:
        //      QK1
        asm volatile("s_waitcnt lgkmcnt(0)");
        zero(att_block[1]);
        mma_AtB(att_block[1], k_reg_transposed, q_reg_transposed, att_block[1]);
        //      Finish softmax for QK0
        sub(max_vec_prev, max_vec_prev, max_vec); 
        exp2(max_vec_prev, max_vec_prev);  
        mul(norm_vec, norm_vec, max_vec_prev);
        col_sum(norm_vec, att_block[0], norm_vec);
        copy(att_block_bf16, att_block[0]);
        att_block_col_bf16 = *(attn_tile<D, bf16, col_l>*)(&att_block_bf16);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 1:
        //      Load K3 into shared 
        load_global_to_shared_direct_with_swizzled_offsets<1, false, st_bf<KV_BLOCK_SIZE, ATTN_D>, _gl_QKVO, coord<st_bf<KV_BLOCK_SIZE,ATTN_D>>, NUM_THREADS>(
            g.Kg, {batch_idx, j, head_idx_kv, 0}, k_smem[1], swizzled_offsets_K);
        //      Load V0 into registers
        load_lds_reg_col_new(v_reg, v_smem[0]);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 2:
        //      A0V0
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mul_col(o_reg, o_reg, max_vec_prev);
        mma_AtB(o_reg, v_reg, att_block_col_bf16, o_reg);
        //      Partial softmax for QK1
        copy(max_vec_prev, max_vec);
        col_max(max_vec, att_block[1], max_vec);
        sub_col(att_block[1], att_block[1], max_vec);
        exp2(att_block[1], att_block[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 3:
        //      Load K2 into registers
        load_lds_reg_row(k_reg, k_smem[0]);
        //      Load V2 into shared
        load_global_to_shared_direct_with_swizzled_offsets<1, false, st_bf<KV_BLOCK_SIZE, ATTN_D>, _gl_QKVO, coord<st_bf<KV_BLOCK_SIZE,ATTN_D>>, NUM_THREADS>(
            g.Vg, {batch_idx, j - 1, head_idx_kv, 0}, v_smem[0], swizzled_offsets_V);
        swap_layout_and_transpose(k_reg_transposed, k_reg);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 4:
        //      QK2
        // asm volatile("s_waitcnt lgkmcnt(0)");
        zero(att_block[0]);
        mma_AtB(att_block[0], k_reg_transposed, q_reg_transposed, att_block[0]);
        //      Finish softmax for QK1
        sub(max_vec_prev, max_vec_prev, max_vec); 
        exp2(max_vec_prev, max_vec_prev);  
        mul(norm_vec, norm_vec, max_vec_prev);
        col_sum(norm_vec, att_block[1], norm_vec);
        copy(att_block_bf16, att_block[1]);
        att_block_col_bf16 = *(attn_tile<D, bf16, col_l>*)(&att_block_bf16);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 5:
        //      Load K4 into shared
        load_global_to_shared_direct_with_swizzled_offsets<1, false, st_bf<KV_BLOCK_SIZE, ATTN_D>, _gl_QKVO, coord<st_bf<KV_BLOCK_SIZE,ATTN_D>>, NUM_THREADS>(
            g.Kg, {batch_idx, j + 1, head_idx_kv, 0}, k_smem[0], swizzled_offsets_K);
        //      Load V1 into registers
        load_lds_reg_col_new(v_reg, v_smem[1]);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 6:
        //      A1V1
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mul_col(o_reg, o_reg, max_vec_prev);
        mma_AtB(o_reg, v_reg, att_block_col_bf16, o_reg);
        //      Partial softmax for QK2
        copy(max_vec_prev, max_vec);
        col_max(max_vec, att_block[0], max_vec);
        sub_col(att_block[0], att_block[0], max_vec);
        exp2(att_block[0], att_block[0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 7:
        //      Load K3 into registers
        load_lds_reg_row(k_reg, k_smem[1]);
        //      Load V3 into shared
        load_global_to_shared_direct_with_swizzled_offsets<1, false, st_bf<KV_BLOCK_SIZE, ATTN_D>, _gl_QKVO, coord<st_bf<KV_BLOCK_SIZE,ATTN_D>>, NUM_THREADS>(
            g.Vg, {batch_idx, j, head_idx_kv, 0}, v_smem[1], swizzled_offsets_V);
        swap_layout_and_transpose(k_reg_transposed, k_reg);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    }

    // Epilogue
    // Cluster 0:
    //      QK3
    asm volatile("s_waitcnt lgkmcnt(0)");
    zero(att_block[1]);
    swap_layout_and_transpose(k_reg_transposed, k_reg);
    mma_AtB(att_block[1], k_reg_transposed, q_reg_transposed, att_block[1]);
    //      Finish softmax for QK2
    sub(max_vec_prev, max_vec_prev, max_vec); 
    exp2(max_vec_prev, max_vec_prev);  
    mul(norm_vec, norm_vec, max_vec_prev);
    col_sum(norm_vec, att_block[0], norm_vec);
    copy(att_block_bf16, att_block[0]);
    att_block_col_bf16 = *(attn_tile<D, bf16, col_l>*)(&att_block_bf16);

    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 1:
    //      Load K5 into shared
    load_global_to_shared_direct_with_swizzled_offsets<1, false, st_bf<KV_BLOCK_SIZE, ATTN_D>, _gl_QKVO, coord<st_bf<KV_BLOCK_SIZE,ATTN_D>>, NUM_THREADS>(
        g.Kg, {batch_idx, num_tiles - 1, head_idx_kv, 0}, k_smem[1], swizzled_offsets_K);
    //      Load V2 into registers
    load_lds_reg_col_new(v_reg, v_smem[0]);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 2:
    //      A2V2
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_setprio(1);
    mul_col(o_reg, o_reg, max_vec_prev);
    mma_AtB(o_reg, v_reg, att_block_col_bf16, o_reg);
    //      Partial softmax for QK3
    copy(max_vec_prev, max_vec);
    col_max(max_vec, att_block[1], max_vec);
    sub_col(att_block[1], att_block[1], max_vec);
    exp2(att_block[1], att_block[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 3:
    //      Load K4 into registers
    load_lds_reg_row(k_reg, k_smem[0]);
    //      Load V4 into shared
    load_global_to_shared_direct_with_swizzled_offsets<1, false, st_bf<KV_BLOCK_SIZE, ATTN_D>, _gl_QKVO, coord<st_bf<KV_BLOCK_SIZE,ATTN_D>>, NUM_THREADS>(
        g.Vg, {batch_idx, num_tiles - 2, head_idx_kv, 0}, v_smem[0], swizzled_offsets_V);
    swap_layout_and_transpose(k_reg_transposed, k_reg);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 4:
    //      QK4
    asm volatile("s_waitcnt lgkmcnt(0)");
    zero(att_block[0]);
    mma_AtB(att_block[0], k_reg_transposed, q_reg_transposed, att_block[0]);
    //      Finish softmax for QK3
    sub(max_vec_prev, max_vec_prev, max_vec); 
    exp2(max_vec_prev, max_vec_prev);  
    mul(norm_vec, norm_vec, max_vec_prev);
    col_sum(norm_vec, att_block[1], norm_vec);
    copy(att_block_bf16, att_block[1]);
    att_block_col_bf16 = *(attn_tile<D, bf16, col_l>*)(&att_block_bf16);

    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 5:
    //      Load V3 into registers
    load_lds_reg_col_new(v_reg, v_smem[1]);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 6:
    //      A3V3
    asm volatile("s_waitcnt lgkmcnt(0)");
    mul_col(o_reg, o_reg, max_vec_prev);
    mma_AtB(o_reg, v_reg, att_block_col_bf16, o_reg);
    //      Partial softmax for QK4
    copy(max_vec_prev, max_vec);
    col_max(max_vec, att_block[0], max_vec);
    sub_col(att_block[0], att_block[0], max_vec);
    exp2(att_block[0], att_block[0]);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 7:
    //      Load V5 into shared
    load_global_to_shared_direct_with_swizzled_offsets<1, false, st_bf<KV_BLOCK_SIZE, ATTN_D>, _gl_QKVO, coord<st_bf<KV_BLOCK_SIZE,ATTN_D>>, NUM_THREADS>(
        g.Vg, {batch_idx, num_tiles - 1, head_idx_kv, 0}, v_smem[1], swizzled_offsets_V);
    //      Load K5 into registers
    load_lds_reg_row(k_reg, k_smem[1]);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 8:
    //      QK5
    asm volatile("s_waitcnt lgkmcnt(0)");
    zero(att_block[1]);
    swap_layout_and_transpose(k_reg_transposed, k_reg);
    mma_AtB(att_block[1], k_reg_transposed, q_reg_transposed, att_block[1]);
    //      Finish softmax for QK4
    sub(max_vec_prev, max_vec_prev, max_vec); 
    exp2(max_vec_prev, max_vec_prev); 
    mul(norm_vec, norm_vec, max_vec_prev);
    col_sum(norm_vec, att_block[0], norm_vec);
    copy(att_block_bf16, att_block[0]);
    att_block_col_bf16 = *(attn_tile<D, bf16, col_l>*)(&att_block_bf16);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 9:
    //      Load V4 into registers
    load_lds_reg_col_new(v_reg, v_smem[0]);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 10:
    //      A4V4
    asm volatile("s_waitcnt lgkmcnt(0)");
    mul_col(o_reg, o_reg, max_vec_prev);
    mma_AtB(o_reg, v_reg, att_block_col_bf16, o_reg);
    //      Full softmax for QK5
    copy(max_vec_prev, max_vec);
    col_max(max_vec, att_block[1], max_vec);
    sub_col(att_block[1], att_block[1], max_vec);
    exp2(att_block[1], att_block[1]);
    sub(max_vec_prev, max_vec_prev, max_vec);
    exp2(max_vec_prev, max_vec_prev);  
    mul(norm_vec, norm_vec, max_vec_prev);
    col_sum(norm_vec, att_block[1], norm_vec);
    copy(att_block_bf16, att_block[1]);
    att_block_col_bf16 = *(attn_tile<D, bf16, col_l>*)(&att_block_bf16);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 11:
    //      Load V5 into registers
    load_lds_reg_col_new(v_reg, v_smem[1]);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 12:
    //      A5V5
    asm volatile("s_waitcnt lgkmcnt(0)");
    mul_col(o_reg, o_reg, max_vec_prev);
    mma_AtB(o_reg, v_reg, att_block_col_bf16, o_reg);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Conclusion
    if (!stagger) {
        __builtin_amdgcn_s_barrier();
    }

    div_col(o_reg, o_reg, norm_vec);
    qo_tile<D, float, accum_l> o_reg_transposed;
    swap_layout_and_transpose(o_reg_transposed, o_reg);
    store_transposed(g.Og, o_reg_transposed, {batch_idx, tile_idx, head_idx, 0});
}

template<int D>
void dispatch_micro(attn_globals<D> g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)attend_ker<D>, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    attend_ker<D><<<g.grid(), g.block(), mem_size>>>(g);
    hipDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module";
    py::bind_function<dispatch_micro<ATTN_D>>(m, "dispatch_micro", &attn_globals<ATTN_D>::Qg, &attn_globals<ATTN_D>::Kg, &attn_globals<ATTN_D>::Vg, &attn_globals<ATTN_D>::Og);
}