#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include "../utils.cpp"
using namespace kittens;

constexpr int ATTN_B = 16; // batch size
constexpr int ATTN_H = 16; // number of heads
constexpr int ATTN_N = 4096; // sequence length
constexpr int ATTN_D = 64; // dimension
constexpr int N_STEP = 128;
constexpr int SUB_N_STEP = 32;
constexpr int BLOCK_SIZE = 32; // block size

#define NUM_WARPS 8
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

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
template<int D, typename T=float, typename L=row_l> using attn_tile = rt<T, BLOCK_SIZE, BLOCK_SIZE, L>;

template<int D> using global_layout = gl<bf16, -1, -1, -1, D>; // B, N, H, specified at runtime, D known at compile time for this kernel
template<int D> struct attn_globals { 
    _gl_QKVO Qg, Kg, Vg, Og; 
    dim3 grid() { return dim3(ATTN_B, ATTN_H, ((ATTN_N / BLOCK_SIZE + NUM_WARPS - 1) / NUM_WARPS)); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY - 32768; }
};

template<int D> __launch_bounds__(NUM_THREADS, 0)
__global__ void attend_ker(const attn_globals<D> g) {
    
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<N_STEP, ATTN_D> (&k_smem)[2] = al.allocate<st_bf<N_STEP, ATTN_D>, 2>();
    st_bf<N_STEP, ATTN_D> (&v_smem)[2] = al.allocate<st_bf<N_STEP, ATTN_D>, 2>();
    
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int block_tile_idx = blockIdx.z;
    const int tile_idx = block_tile_idx * NUM_WARPS + warpid();

    constexpr float TEMPERATURE_SCALE = (D == 128) ? 0.08838834764f*1.44269504089f : 0.125f*1.44269504089f;

    // Initialize all of the register tiles.
    qkvo_tile<D, bf16> q_reg; // Q and K are both row layout, as we use mma_ABt.
    qkvo_tile<D, bf16> k_reg;
    qkvo_tile_transposed<D, bf16> q_reg_transposed;
    qkvo_tile_transposed<D, bf16> k_reg_transposed;
    qkvo_tile<D, bf16, col_l> v_reg;
    // qkvo_tile<D, float, accum_l> o_reg; // Output tile.
    qkvo_tile_transposed<D, float, accum_l> o_reg; // Output tile.
    attn_tile<D, float, accum_l> att_block; // attention tile, in float.
    attn_tile<D, bf16, accum_l> att_block_bf16;
    attn_tile<D, bf16, col_l> att_block_col_bf16;
    typename attn_tile<D, float, accum_l>::row_vec max_vec, norm_vec, max_vec_prev;

    // Pre-scale Q by temperature
    qkvo_tile<D, float> q_reg_fl;
    load(q_reg_fl, g.Qg, {batch_idx, head_idx, tile_idx, 0});
    mul(q_reg_fl, q_reg_fl, TEMPERATURE_SCALE);  // Use sqrtf for clarity
    copy(q_reg, q_reg_fl);
    q_reg_transposed = swap_layout_and_transpose_inplace(q_reg);

    int tic = 0, toc = 1;
    load_global_to_shared_direct<2, false, st_bf<N_STEP, ATTN_D>, _gl_QKVO, coord<st_bf<N_STEP,ATTN_D>>, NUM_THREADS>(
        g.Kg, {batch_idx, head_idx, 0, 0}, k_smem[tic]);
    load_global_to_shared_direct<2, false, st_bf<N_STEP, ATTN_D>, _gl_QKVO, coord<st_bf<N_STEP,ATTN_D>>, NUM_THREADS>(
        g.Vg, {batch_idx, head_idx, 0, 0}, v_smem[tic]);

    // Initialize output and running statistics
    zero(o_reg);
    zero(norm_vec);
    neg_infty(max_vec);

    int num_tiles = ATTN_N / N_STEP;
    int num_sub_tiles = N_STEP / SUB_N_STEP;

    // Iterate over chunks of keys and values.
    for (int j = 0; j < num_tiles - 1; j++, tic^=1, toc^=1) {

        __builtin_amdgcn_s_barrier();
        // load the k and v tiles from global to shared memory
        load_global_to_shared_direct<2, false, st_bf<N_STEP, ATTN_D>, _gl_QKVO, coord<st_bf<N_STEP,ATTN_D>>, NUM_THREADS>(
            g.Kg, {batch_idx, head_idx, j + 1, 0}, k_smem[toc]);
        load_global_to_shared_direct<2, false, st_bf<N_STEP, ATTN_D>, _gl_QKVO, coord<st_bf<N_STEP,ATTN_D>>, NUM_THREADS>(
            g.Vg, {batch_idx, head_idx, j + 1, 0}, v_smem[toc]);

        // Process sub-tiles
        for (int i = 0; i < num_sub_tiles; i++) {

            // load the k and v tiles from shared memory to registers
            load_lds_reg(k_reg, subtile_inplace<SUB_N_STEP, ATTN_D>(k_smem[tic], {i, 0}));
            k_reg_transposed = swap_layout_and_transpose_inplace(k_reg);

            // zero
            zero(att_block);
            copy(max_vec_prev, max_vec);

            // A = Q @ K.T
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_AtB(att_block, k_reg_transposed, q_reg_transposed, att_block);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            
            // Update max in-place and compute correction
            col_max(max_vec, att_block, max_vec);  
            sub(max_vec_prev, max_vec_prev, max_vec); 
            exp2(max_vec_prev, max_vec_prev); 
            
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

            load_lds_reg(v_reg, subtile_inplace<SUB_N_STEP, ATTN_D>(v_smem[tic], {i, 0}));

            // O += A @ V
            mma_AtB(o_reg, v_reg, att_block_col_bf16, o_reg);
        }
    }

    // Epilogue - process final tile
    for (int i = 0; i < num_sub_tiles; i++) {

        // load the k and v tiles from shared memory to registers
        qkvo_tile<D, bf16> k_reg;
        load_lds_reg(k_reg, subtile_inplace<SUB_N_STEP, ATTN_D>(k_smem[tic], {i, 0}));
        k_reg_transposed = swap_layout_and_transpose_inplace(k_reg);

        // zero
        zero(att_block);

        // Store previous max values
        copy(max_vec_prev, max_vec);

        // A = Q @ K.T
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_AtB(att_block, k_reg_transposed, q_reg_transposed, att_block);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        // Update max in-place and compute correction
        col_max(max_vec, att_block, max_vec);  
        sub(max_vec_prev, max_vec_prev, max_vec); 
        exp2(max_vec_prev, max_vec_prev); 
        
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

        load_lds_reg(v_reg, subtile_inplace<SUB_N_STEP, ATTN_D>(v_smem[tic], {i, 0}));

        mma_AtB(o_reg, v_reg, att_block_col_bf16, o_reg);
    }

    // Final normalization
    div_col(o_reg, o_reg, norm_vec);

    // Store O_i back to global memory.
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



