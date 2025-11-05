#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include "utils.cpp"
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
    qkvo_tile<D, bf16> q_reg, k_reg; // Q and K are both row layout, as we use mma_ABt.
    qkvo_tile<D, bf16, col_l> v_reg; // V is column layout, as we use mma_AB.
    qkvo_tile<D, float, accum_l> o_reg; // Output tile.
    attn_tile<D, float, accum_l> att_block; // attention tile, in float. (We want to use float wherever possible.)
    attn_tile<D, float> att_block_row;
    attn_tile<D, bf16> att_block_row_bf16;
    typename attn_tile<D, float, accum_l>::col_vec max_vec_last_scaled, max_vec_scaled, max_vec, norm_vec; // these are column vectors for the online softmax.

    int tic = 0, toc = 1;
    load_global_to_shared_direct<2, false, st_bf<N_STEP, ATTN_D>, _gl_QKVO, coord<st_bf<N_STEP,ATTN_D>>, NUM_THREADS>(
        g.Kg, {batch_idx, head_idx, 0, 0}, k_smem[tic]);
    load_global_to_shared_direct<2, false, st_bf<N_STEP, ATTN_D>, _gl_QKVO, coord<st_bf<N_STEP,ATTN_D>>, NUM_THREADS>(
        g.Vg, {batch_idx, head_idx, 0, 0}, v_smem[tic]);

    // 5. Given i = blockIdx.x, load Q_i from global to registers. Set O_i = 0, l_i = 0, m_i = -inf.
    zero(o_reg);
    zero(norm_vec);
    neg_infty(max_vec);
    load(q_reg, g.Qg, {batch_idx, head_idx, tile_idx, 0});

    int num_tiles = ATTN_N / N_STEP;
    int num_sub_tiles = N_STEP / SUB_N_STEP;

    // 6. For 1 <= j <= 64 do
    for (int j = 0; j < num_tiles - 1; j++, tic^=1, toc^=1) {

        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();
        // load the k and v tiles from global to shared memory
        load_global_to_shared_direct<2, false, st_bf<N_STEP, ATTN_D>, _gl_QKVO, coord<st_bf<N_STEP,ATTN_D>>, NUM_THREADS>(
            g.Kg, {batch_idx, head_idx, j + 1, 0}, k_smem[toc]);
        load_global_to_shared_direct<2, false, st_bf<N_STEP, ATTN_D>, _gl_QKVO, coord<st_bf<N_STEP,ATTN_D>>, NUM_THREADS>(
            g.Vg, {batch_idx, head_idx, j + 1, 0}, v_smem[toc]);

        for (int i = 0; i < num_sub_tiles; i++) {

            // load the k and v tiles from shared memory to registers
            load_lds_reg(k_reg, subtile_inplace<SUB_N_STEP, ATTN_D>(k_smem[tic], {i, 0}));
            load_lds_reg(v_reg, subtile_inplace<SUB_N_STEP, ATTN_D>(v_smem[tic], {i, 0}));

            // A = Q @ K.T
            zero(att_block);
            mma_ABt(att_block, q_reg, k_reg, att_block);
            mul(max_vec_last_scaled, max_vec, TEMPERATURE_SCALE);

            // softmax
            row_max(max_vec, att_block, max_vec);
            mul(max_vec_scaled, max_vec, TEMPERATURE_SCALE);
            mul(att_block, att_block, TEMPERATURE_SCALE);
            sub_row(att_block, att_block, max_vec_scaled);
            exp2(att_block, att_block);
            sub(max_vec_last_scaled, max_vec_last_scaled, max_vec_scaled);
            exp2(max_vec_last_scaled, max_vec_last_scaled);
            mul(norm_vec, norm_vec, max_vec_last_scaled);
            row_sum(norm_vec, att_block, norm_vec);
            mul_row(o_reg, o_reg, max_vec_last_scaled);
            att_block_row = swap_layout_inplace<row_l>(att_block); 
            copy(att_block_row_bf16, att_block_row);

            // O += A @ V
            mma_AB(o_reg, att_block_row_bf16, v_reg, o_reg);
        }
    }

    // Epilogue
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();

    for (int i = 0; i < num_sub_tiles; i++) {

        // load the k and v tiles from shared memory to registers
        load_lds_reg(k_reg, subtile_inplace<SUB_N_STEP, ATTN_D>(k_smem[tic], {i, 0}));
        load_lds_reg(v_reg, subtile_inplace<SUB_N_STEP, ATTN_D>(v_smem[tic], {i, 0}));

        // A = Q @ K.T
        zero(att_block);
        mma_ABt(att_block, q_reg, k_reg, att_block);
        mul(max_vec_last_scaled, max_vec, TEMPERATURE_SCALE);

        // softmax
        row_max(max_vec, att_block, max_vec);
        mul(max_vec_scaled, max_vec, TEMPERATURE_SCALE);
        mul(att_block, att_block, TEMPERATURE_SCALE);
        sub_row(att_block, att_block, max_vec_scaled);
        exp2(att_block, att_block);
        sub(max_vec_last_scaled, max_vec_last_scaled, max_vec_scaled);
        exp2(max_vec_last_scaled, max_vec_last_scaled);
        mul(norm_vec, norm_vec, max_vec_last_scaled);
        row_sum(norm_vec, att_block, norm_vec);
        mul_row(o_reg, o_reg, max_vec_last_scaled);
        att_block_row = swap_layout_inplace<row_l>(att_block); 
        copy(att_block_row_bf16, att_block_row);

        // O += A @ V
        mma_AB(o_reg, att_block_row_bf16, v_reg, o_reg);
    }

    // 16. O_i = diag(l_i)^-1 @ O_i
    div_row(o_reg, o_reg, norm_vec);

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
    py::bind_function<dispatch_micro<ATTN_D>>(m, "dispatch_micro", &attn_globals<ATTN_D>::Qg, &attn_globals<ATTN_D>::Kg, &attn_globals<ATTN_D>::Vg, &attn_globals<ATTN_D>::Og);
}



