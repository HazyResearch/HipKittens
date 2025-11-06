#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

constexpr int ATTN_B = 16; // batch size
constexpr int ATTN_H = 16; // number of heads
constexpr int ATTN_N = 1024; // sequence length
constexpr int ATTN_D = 64; // dimension
constexpr int BLOCK_SIZE = 16; // block size

#define NUM_WARPS 1
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using namespace kittens;

/*
Flash Attention: (B = H = 1)
Input: Q (B, H, N, D), K (B, H, N, D), V (B, H, N, D)
Output: O (B, H, N, D)

Each warp is responsible for a 16x64 tile of the output. i.e. Given an output of (1024, 64), there are 1024*64/16/64 = 64 tiles.

1. Set block sizes Bc = Br = 16
2. Initialize O = (o) (B, H, N, D), L = (0) (B, H, N), m = (-inf) (B, H, N) in HBM
3. Divide Q into N/16 = 1024/16 = 64 tiles of 16x64 each. Divide K, V into N/16 = 1024/16 = 64 tiles of 16x64 each.
4. Divide O into 64 tiles of 16x64 each. Divide L into 64 vectors of 16 elements each, divide m into 64 vectors of 16 elements each.

5. Given i = blockIdx.x, load Q_i from global to registers. Set O_i = 0, l_i = 0, m_i = -inf.
6. For 1 <= j <= 64 do
7.     Load K_j, V_j from global to registers (16x64)
8.     Compute S_ij = Q_i @ K_j.T (16x16)
9.     Compute m'_ij = row_max(S_ij) (16x1)
10.            p'_ij = exp(S_ij - m'_ij) (16x16)
11.            l'_ij = row_sum(p'_ij) (16x1)
12.    Compute m_i_new = max(m_i, m'_ij) (16x1)
13.            l_i_new = exp(m_i - m_i_new) * l_i + exp(m'_ij - m_i_new) * l'_ij (16x1)
14.    O_i = diag(l_i_new)^-1 @ (diag(l_i) @ exp(m_i - m_i_new) * O_i + exp(m'_ij - m_i_new) * P'_ij @ V_j) (16x64)
15.    l_i = l_i_new, m_i = m_i_new
16. Store O_i back to global memory.
*/

template<int D, typename T=bf16, typename L=row_l> using qkvo_tile = rt<T, BLOCK_SIZE, D, L>;
template<int D, typename T=float, typename L=row_l> using attn_tile = rt<T, BLOCK_SIZE, BLOCK_SIZE, L>;

template<int D> using global_layout = gl<bf16, -1, -1, -1, D>; // B, N, H, specified at runtime, D known at compile time for this kernel
template<int D> struct attn_globals { 
    gl<bf16, -1, -1, -1, D> Qg, Kg, Vg, Og; 
    dim3 grid() { return dim3(ATTN_B, ATTN_H, ATTN_N / BLOCK_SIZE); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 0; }
};

template<int D> __launch_bounds__(NUM_THREADS, 0)
__global__ void attend_ker(const attn_globals<D> g) {
    
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tile_idx = blockIdx.z;

    const float scale_factor = 1.0f / sqrt(D);

    // Initialize all of the register tiles.
    qkvo_tile<D, bf16> q_reg, k_reg; // Q and K are both row layout, as we use mma_ABt.
    qkvo_tile<D, bf16, col_l> v_reg; // V is column layout, as we use mma_AB.
    qkvo_tile<D, float, col_l> o_reg; // Output tile.
    qkvo_tile<D, float, col_l> o_reg_next; // attention tile, in float, for the mma_AB.
    attn_tile<D, float, col_l> att_block; // attention tile, in float. (We want to use float wherever possible.)
    attn_tile<D, bf16, col_l> att_block_bf16; // bf16 attention tile for the second mma_AB. We cast right before that op.
    attn_tile<D, bf16> att_block_mma; // bf16 attention tile in row layout for the second mma_AB.
    typename attn_tile<D, float, col_l>::col_vec max_vec_last, max_vec, max_vec_new, norm_vec_last, norm_vec, norm_vec_new; // these are column vectors for the online softmax.

    // 5. Given i = blockIdx.x, load Q_i from global to registers. Set O_i = 0, l_i = 0, m_i = -inf.
    zero(o_reg);
    zero(norm_vec_last);
    zero(norm_vec);
    zero(norm_vec_new);
    neg_infty(max_vec_last);
    neg_infty(max_vec);
    neg_infty(max_vec_new);
    load(q_reg, g.Qg, {batch_idx, head_idx, tile_idx, 0});

    // 6. For 1 <= j <= 64 do
    for (int j = 0; j < 64; j++) {
        // zero out the accumulators
        zero(att_block);
        zero(o_reg_next);

        // 7.     Load K_j, V_j from global to registers (16x64)
        load(k_reg, g.Kg, {batch_idx, head_idx, j, 0});
        load(v_reg, g.Vg, {batch_idx, head_idx, j, 0});

        // 8.     Compute S_ij = Q_i @ K_j.T (16x16)
        mma_ABt(att_block, q_reg, k_reg, att_block);
        mul(att_block, att_block, scale_factor);

        // 9.     Compute m'_ij = row_max(S_ij) (16x1)
        row_max(max_vec, att_block);

        // 10.            p'_ij = exp(S_ij - m'_ij) (16x16)
        sub_row(att_block, att_block, max_vec);
        exp(att_block, att_block);

        // 11.            l'_ij = row_sum(p'_ij) (16x1)
        row_sum(norm_vec, att_block);

        // 12.    Compute m_i_new = max(m_i, m'_ij) (16x1)
        max(max_vec_new, max_vec_last, max_vec);

        // 13.            l_i_new = exp(m_i - m_i_new) * l_i + exp(m'_ij - m_i_new) * l'_ij (16x1)
        sub(max_vec_last, max_vec_last, max_vec_new);
        exp(max_vec_last, max_vec_last);

        sub(max_vec, max_vec, max_vec_new);
        exp(max_vec, max_vec);

        mul(norm_vec_last, max_vec_last, norm_vec_last);
        mul(norm_vec, max_vec, norm_vec);
        add(norm_vec_new, norm_vec_last, norm_vec);

        // 14.    O_i = exp(m_i - m_i_new) @ O_i + exp(m'_ij - m_i_new) * P'_ij @ V_j (16x64)
        mul_row(o_reg, o_reg, max_vec_last);
        copy(att_block_bf16, att_block);
        swap_layout(att_block_mma, att_block_bf16);
        mma_AB(o_reg_next, att_block_mma, v_reg, o_reg_next);
        mul_row(o_reg_next, o_reg_next, max_vec);
        add(o_reg, o_reg, o_reg_next);

        // 15.    l_i = l_i_new, m_i = m_i_new
        copy(max_vec_last, max_vec_new);
        copy(norm_vec_last, norm_vec_new);
    }

    // 16. O_i = diag(l_i)^-1 @ O_i
    div_row(o_reg, o_reg, norm_vec_last);

    // 17. Store O_i back to global memory.
    store(g.Og, o_reg, {batch_idx, head_idx, tile_idx, 0});

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
    py::bind_kernel<attend_ker<ATTN_D>>(m, "attend_ker", &attn_globals<ATTN_D>::Qg, &attn_globals<ATTN_D>::Kg, &attn_globals<ATTN_D>::Vg, &attn_globals<ATTN_D>::Og); 
    py::bind_function<dispatch_micro<ATTN_D>>(m, "dispatch_micro", &attn_globals<ATTN_D>::Qg, &attn_globals<ATTN_D>::Kg, &attn_globals<ATTN_D>::Vg, &attn_globals<ATTN_D>::Og);
}



