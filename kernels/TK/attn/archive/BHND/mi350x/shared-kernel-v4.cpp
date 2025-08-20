#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include "utils.cpp"

constexpr int ATTN_B = 16; // batch size
constexpr int ATTN_H = 16; // number of heads
constexpr int ATTN_N = 4096; // sequence length
constexpr int ATTN_D = 64; // dimension
constexpr int N_STEP = 256;
constexpr int BLOCK_SIZE = 32; // block size

#define NUM_WARPS 16
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using namespace kittens;
using _gl_QKVO = gl<bf16, -1, -1, -1, -1>;

/*
Flash Attention: (B = H = 1)
Input: Q (B, H, N, D), K (B, H, N, D), V (B, H, N, D)
Output: O (B, H, N, D)

Each warp is responsible for a 32x64 tile of the output. i.e. Given an output of (1024, 64), there are 1024*64/32/64 = 32 tiles.

1. Set block sizes Bc = Br = 32
2. Initialize O = (o) (B, H, N, D), L = (0) (B, H, N), m = (-inf) (B, H, N) in HBM
3. Divide Q into N/32 = 1024/32 = 32 tiles of 32x64 each. Divide K, V into N/32 = 1024/32 = 32 tiles of 32x64 each.
4. Divide O into 32 tiles of 32x64 each. Divide L into 32 vectors of 32 elements each, divide m into 32 vectors of 32 elements each.

5. Given i = blockIdx.x, load Q_i from global to registers. Set O_i = 0, l_i = 0, m_i = -inf.
6. For 1 <= j <= 32 do
7.     Load K_j, V_j from global to registers (32x64)
8.     Compute S_ij = Q_i @ K_j.T (32x32)
9.     Compute m'_ij = row_max(S_ij) (32x1)
10.            p'_ij = exp(S_ij - m'_ij) (32x32)
11.            l'_ij = row_sum(p'_ij) (32x1)
12.    Compute m_i_new = max(m_i, m'_ij) (32x1)
13.            l_i_new = exp(m_i - m_i_new) * l_i + exp(m'_ij - m_i_new) * l'_ij (32x1)
14.    O_i = diag(l_i_new)^-1 @ (diag(l_i) @ exp(m_i - m_i_new) * O_i + exp(m'_ij - m_i_new) * P'_ij @ V_j) (32x64)
15.    l_i = l_i_new, m_i = m_i_new
16. Store O_i back to global memory.
*/

template<int D, typename T=bf16, typename L=row_l> using qkvo_tile = rt<T, BLOCK_SIZE, D, L>;
template<int D, typename T=bf16, typename L=col_l> using qkvo_tile_transposed = rt<T, D, BLOCK_SIZE, L>;
template<int D, typename T=float, typename L=accum_l> using attn_tile = rt<T, BLOCK_SIZE, BLOCK_SIZE, L>;

template<int D> struct attn_globals { 
    _gl_QKVO Qg, Kg, Vg, Og; 
    dim3 grid() { return dim3(ATTN_B, ATTN_H, ((ATTN_N / BLOCK_SIZE + NUM_WARPS - 1) / NUM_WARPS)); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

template<int D> __launch_bounds__(NUM_THREADS, 2)
__global__ void attend_ker(const attn_globals<D> g) {

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<N_STEP, ATTN_D> (&k_smem)[2] = al.allocate<st_bf<N_STEP, ATTN_D>, 2>();
    st_bf<N_STEP, ATTN_D> (&v_smem)[2] = al.allocate<st_bf<N_STEP, ATTN_D>, 2>();
    
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int block_tile_idx = blockIdx.z;
    const int tile_idx = block_tile_idx * NUM_WARPS + warpid();
    const int stagger = warpid() / 8;

    const int num_tiles = ATTN_N / N_STEP;
    const int num_sub_tiles = N_STEP / BLOCK_SIZE;

    constexpr float TEMPERATURE_SCALE = (D == 128) ? 0.08838834764f*1.44269504089f : 0.125f*1.44269504089f;

    // Initialize all of the register tiles.
    qkvo_tile<D, bf16> q_reg; // Q and K are both row layout, as we use mma_ABt.
    qkvo_tile_transposed<D, bf16> q_reg_transposed;
    qkvo_tile<D, bf16> k_reg;
    qkvo_tile_transposed<D, bf16> k_reg_transposed;

    qkvo_tile<D, bf16, col_l> v_reg;
    qkvo_tile_transposed<D, float, accum_l> o_reg; // Output tile.
    attn_tile<D, float, accum_l> att_block; // attention tile, in float.
    attn_tile<D, bf16, accum_l> att_block_bf16;
    attn_tile<D, bf16, col_l> att_block_col_bf16;
    typename attn_tile<D, float, accum_l>::row_vec max_vec, norm_vec, max_vec_prev;

    int tic = 0, toc = 1;
    using T = typename st_bf<N_STEP, ATTN_D>::dtype;
    constexpr int bytes_per_thread = 16;
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_THREADS;
    constexpr int memcpy_per_tile = N_STEP * ATTN_D * sizeof(T) / bytes_per_memcpy;

    load_global_to_shared_direct<2, false, st_bf<N_STEP, ATTN_D>, _gl_QKVO, coord<st_bf<N_STEP,ATTN_D>>, NUM_THREADS>(
        g.Kg, {batch_idx, head_idx, 0, 0}, k_smem[tic]);
    load_global_to_shared_direct<2, false, st_bf<N_STEP, ATTN_D>, _gl_QKVO, coord<st_bf<N_STEP,ATTN_D>>, NUM_THREADS>(
        g.Vg, {batch_idx, head_idx, 0, 0}, v_smem[tic]);
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();

    uint32_t swizzled_offsets_V[memcpy_per_tile];
    uint32_t swizzled_offsets_K[memcpy_per_tile];
    prefill_swizzled_offsets<2, false, st_bf<N_STEP, ATTN_D>, _gl_QKVO, coord<st_bf<N_STEP, ATTN_D>>, NUM_THREADS>(g.Kg, {batch_idx, head_idx, 0, 0}, k_smem[tic], swizzled_offsets_K);
    prefill_swizzled_offsets<2, false, st_bf<N_STEP, ATTN_D>, _gl_QKVO, coord<st_bf<N_STEP, ATTN_D>>, NUM_THREADS>(g.Vg, {batch_idx, head_idx, 0, 0}, v_smem[tic], swizzled_offsets_V);

    // Pre-scale Q by temperature
    qkvo_tile<D, float> q_reg_fl;
    load(q_reg_fl, g.Qg, {batch_idx, head_idx, tile_idx, 0});
    mul(q_reg_fl, q_reg_fl, TEMPERATURE_SCALE);  // Use sqrtf for clarity
    copy(q_reg, q_reg_fl);
    swap_layout_and_transpose(q_reg_transposed, q_reg);

    zero(o_reg);
    zero(norm_vec);
    neg_infty(max_vec);

    if (stagger) {
        __builtin_amdgcn_s_barrier();
    }


    for (int j = 0; j < num_tiles - 1; j++, tic^=1, toc^=1) {


        load_global_to_shared_direct_with_swizzled_offsets<2, false, st_bf<N_STEP, ATTN_D>, _gl_QKVO, coord<st_bf<N_STEP, ATTN_D>>, NUM_THREADS>(
            g.Kg, {batch_idx, head_idx, j + 1, 0}, k_smem[toc], swizzled_offsets_K);
        load_global_to_shared_direct_with_swizzled_offsets<2, false, st_bf<N_STEP, ATTN_D>, _gl_QKVO, coord<st_bf<N_STEP, ATTN_D>>, NUM_THREADS>(
            g.Vg, {batch_idx, head_idx, j + 1, 0}, v_smem[toc], swizzled_offsets_V);

        for (int i = 0; i < num_sub_tiles; i++) {
            // Cluster 0: Shared to register for current K and V, and MMA for Q @ K.T
            load_lds_reg(k_reg, subtile_inplace<BLOCK_SIZE, ATTN_D>(k_smem[tic], {i, 0}));
            load_lds_reg_col(v_reg, subtile_inplace<BLOCK_SIZE, ATTN_D>(v_smem[tic], {i, 0}));
            __builtin_amdgcn_sched_barrier(0);
            // zero
            zero(att_block);
            // Store previous max values
            copy(max_vec_prev, max_vec);
            // A = Q @ K.T (now temperature is fully applied)
            // asm volatile("s_waitcnt lgkmcnt(8)\n");
            swap_layout_and_transpose(k_reg_transposed, k_reg);
            mma_AtB(att_block, k_reg_transposed, q_reg_transposed, att_block);
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // Cluster 1: Softmax and normalization
            // Update max in-place and compute correction
            col_max(max_vec, att_block, max_vec);  // max_vec = max(max_vec, row_max(att_block))
            sub(max_vec_prev, max_vec_prev, max_vec);  // max_vec_prev = old_max - new_max
            exp2(max_vec_prev, max_vec_prev);  // max_vec_prev = exp2(old_max - new_max)
            // Apply max normalization to attention scores
            sub_col(att_block, att_block, max_vec);
            exp2(att_block, att_block);
            // Update running normalization
            mul(norm_vec, norm_vec, max_vec_prev);
            col_sum(norm_vec, att_block, norm_vec);
            copy(att_block_bf16, att_block);  // float → bf16, same layout
            att_block_col_bf16 = swap_layout_inplace<col_l>(att_block_bf16);
            // Update running output
            mul_col(o_reg, o_reg, max_vec_prev);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // Next Cluster. . .
            mma_AtB(o_reg, v_reg, att_block_col_bf16, o_reg);
        }
    }

    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Epilogue
    for (int i = 0; i < num_sub_tiles; i++) {
        // Cluster 0: Shared to register for current K and V, and MMA for Q @ K.T
        load_lds_reg(k_reg, subtile_inplace<BLOCK_SIZE, ATTN_D>(k_smem[tic], {i, 0}));
        load_lds_reg_col(v_reg, subtile_inplace<BLOCK_SIZE, ATTN_D>(v_smem[tic], {i, 0}));
        __builtin_amdgcn_sched_barrier(0);
        // zero
        zero(att_block);
        // Store previous max values
        copy(max_vec_prev, max_vec);
        // A = Q @ K.T (now temperature is fully applied)
        // asm volatile("s_waitcnt lgkmcnt(8)\n");
        swap_layout_and_transpose(k_reg_transposed, k_reg);
        mma_AtB(att_block, k_reg_transposed, q_reg_transposed, att_block);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 1: Softmax and normalization
        // Update max in-place and compute correction
        col_max(max_vec, att_block, max_vec);  // max_vec = max(max_vec, row_max(att_block))
        sub(max_vec_prev, max_vec_prev, max_vec);  // max_vec_prev = old_max - new_max
        exp2(max_vec_prev, max_vec_prev);  // max_vec_prev = exp2(old_max - new_max)
        // Apply max normalization to attention scores
        sub_col(att_block, att_block, max_vec);
        exp2(att_block, att_block);
        // Update running normalization
        mul(norm_vec, norm_vec, max_vec_prev);
        col_sum(norm_vec, att_block, norm_vec);
        copy(att_block_bf16, att_block);  // float → bf16, same layout
        att_block_col_bf16 = swap_layout_inplace<col_l>(att_block_bf16);
        // Update running output
        mul_col(o_reg, o_reg, max_vec_prev);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Next Cluster. . .
        mma_AtB(o_reg, v_reg, att_block_col_bf16, o_reg);
    }

    if (!stagger) {
        __builtin_amdgcn_s_barrier();
    }

    // 16. O_i = diag(l_i)^-1 @ O_i
    div_col(o_reg, o_reg, norm_vec);

    // 17. Store O_i back to global memory.
    qkvo_tile<D, float, accum_l> o_reg_transposed = *(qkvo_tile<D, float, accum_l>*)&o_reg;
    store_transposed(g.Og, o_reg_transposed, {batch_idx, head_idx, tile_idx, 0});
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