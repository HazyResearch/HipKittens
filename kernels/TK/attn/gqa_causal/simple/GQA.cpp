#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

constexpr int ATTN_B = 1; // batch size
constexpr int ATTN_H = 1; // number of heads
constexpr int ATTN_N = 32; // sequence length
constexpr int ATTN_D = 64; // dimension
constexpr int N_STEP = 32;
constexpr int BLOCK_SIZE = 32; // block size

#define NUM_WARPS 1
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using namespace kittens;
using _gl_QKVO = gl<bf16, -1, -1, -1, -1>;

template<int D, typename T=bf16, typename L=row_l> using qkvo_tile = rt<T, BLOCK_SIZE, D, L>;
template<int D, typename T=bf16, typename L=col_l> using qkvo_tile_transposed = rt<T, D, BLOCK_SIZE, L>;
template<int D, typename T=float, typename L=accum_col_l> using attn_tile = rt<T, BLOCK_SIZE, BLOCK_SIZE, L>;

template<int D> struct attn_globals { 
    _gl_QKVO Qg, Kg, Vg, Og; 
    dim3 grid() { return dim3(ATTN_H, ((ATTN_N / BLOCK_SIZE + NUM_WARPS - 1) / NUM_WARPS), ATTN_B); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

template<ducks::rt::accumulator_col_layout RT>
__device__ inline void mask_causal_accum_col(
    RT &dst,
    const typename base_types::packing<typename RT::dtype>::unpacked_type &neg_inf
) {
    const int lane = laneid();
    const int col  = lane & 31;   // lane % 32 → column
    const int ro   = lane >> 5;   // lane / 32 → which 4-row block (0 or 1)

    #pragma unroll
    for (int i = 0; i < dst.height; ++i) {
        #pragma unroll
        for (int j = 0; j < dst.width; ++j) {
            #pragma unroll
            for (int ii = 0; ii < 4; ++ii) {
                const int rbase = ii*8 + ro*4; // rows rbase+0..+3 at this col
                auto &d0 = dst.tiles[i][j].data[ii*2];
                auto &d1 = dst.tiles[i][j].data[ii*2 + 1];

                // Keep if row ≤ col, mask if row > col
                if (rbase + 0 > col) d0.x = neg_inf;
                if (rbase + 1 > col) d0.y = neg_inf;
                if (rbase + 2 > col) d1.x = neg_inf;
                if (rbase + 3 > col) d1.y = neg_inf;
            }
        }
    }
}

template<int D> __launch_bounds__(NUM_THREADS, 2)
__global__ void attend_ker(const attn_globals<D> g) {

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<N_STEP, ATTN_D> (&k_smem) = al.allocate<st_bf<N_STEP, ATTN_D>>();
    
    st_bf<N_STEP, ATTN_D, ducks::st_layout::col> (&v_smem) = al.allocate<st_bf<N_STEP, ATTN_D, ducks::st_layout::col>>();
    
    // const int head_idx = (blockIdx.x % 8) * 8 + (blockIdx.x / 8);
    const int head_idx = blockIdx.x;
    const int batch_idx = blockIdx.z;
    const int head_idx_kv = head_idx;
    const int block_tile_idx = blockIdx.y;
    const int tile_idx = block_tile_idx * NUM_WARPS + warpid();
    const int stagger = warpid() / 4;

    constexpr float TEMPERATURE_SCALE = (D == 128) ? 0.08838834764f*1.44269504089f : 0.125f*1.44269504089f;

    // Initialize all of the register tiles.
    qkvo_tile<D, bf16> q_reg; // Q and K are both row layout, as we use mma_ABt.
    qkvo_tile_transposed<D, bf16> q_reg_transposed;
    qkvo_tile<D, bf16> k_reg;
    qkvo_tile_transposed<D, bf16> k_reg_transposed;

    qkvo_tile<D, bf16, col_l> v_reg;
    qkvo_tile_transposed<D, float, accum_col_l> o_reg; // Output tile.
    attn_tile<D, float, accum_col_l> att_block; // attention tile, in float.
    attn_tile<D, bf16, accum_col_l> att_block_bf16;
    attn_tile<D, bf16, col_l> att_block_col_bf16;
    typename attn_tile<D, float, accum_col_l>::row_vec max_vec, norm_vec, max_vec_prev;

    load<1, false>(k_smem, g.Kg, {batch_idx, 0, head_idx_kv, 0});
    load<1, false>(v_smem, g.Vg, {batch_idx, 0, head_idx_kv, 0});
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();

    // Pre-scale Q by temperature
    qkvo_tile<D, float> q_reg_fl;
    load<1, qkvo_tile<D, float>, _gl_QKVO>(q_reg_fl, g.Qg, {batch_idx, tile_idx, head_idx, 0});
    mul(q_reg_fl, q_reg_fl, TEMPERATURE_SCALE);  // Use sqrtf for clarity
    copy(q_reg, q_reg_fl);
    swap_layout_and_transpose(q_reg_transposed, q_reg);

    zero(o_reg);
    zero(norm_vec);
    neg_infty(max_vec);

    int num_sub_tiles;
    int causal = true; 
    int num_tiles;
    int q_end_pos;
    if (causal) {
        q_end_pos = (tile_idx + 1) * BLOCK_SIZE;
        num_tiles = (q_end_pos + N_STEP - 1) / N_STEP;
        num_tiles = min(num_tiles, ATTN_N / N_STEP);
    } else {
        q_end_pos = ATTN_N;
        num_tiles = ATTN_N / N_STEP;
    } 

    for (int j = 0; j < num_tiles; j++) {

        int current_num_sub_tiles;
        if (causal && j == num_tiles - 1) {
            int tile_start = j * N_STEP;
            int positions_in_tile = q_end_pos - tile_start;
            current_num_sub_tiles = (positions_in_tile + BLOCK_SIZE - 1) / BLOCK_SIZE;
        } else {
            current_num_sub_tiles = N_STEP / BLOCK_SIZE;
        }

        load<1, false>(k_smem, g.Kg, {batch_idx, j, head_idx_kv, 0});
        load<1, false>(v_smem, g.Vg, {batch_idx, j, head_idx_kv, 0});

        for (int i = 0; i < current_num_sub_tiles; i++) {
            // Cluster 0: Shared to register for current K and V, and MMA for Q @ K.T
            load(k_reg, subtile_inplace<BLOCK_SIZE, ATTN_D>(k_smem, {i, 0}));
            load(v_reg, subtile_inplace<BLOCK_SIZE, ATTN_D>(v_smem, {i, 0}));
            __builtin_amdgcn_sched_barrier(0);
            zero(att_block);
            copy(max_vec_prev, max_vec);
            // A = Q @ K.T (now temperature is fully applied)
            swap_layout_and_transpose(k_reg_transposed, k_reg);
            mma_AtB(att_block, k_reg_transposed, q_reg_transposed, att_block);
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);
            
            // Masking
            if (causal) {
                const int q_pos = tile_idx;                                  // query tile index (columns)
                const int k_pos = j * (N_STEP / BLOCK_SIZE) + i;             // key tile index (rows)
            
                if (k_pos > q_pos) {
                    neg_infty(att_block);                                    // whole future tile
                } else if (k_pos == q_pos) {
                    mask_causal_accum_col(att_block, kittens::base_types::constants<float>::neg_infty());
                }
            }

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
            mma_AtB(o_reg, v_reg, att_block_col_bf16, o_reg);
        }
    }
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);
    div_col(o_reg, o_reg, norm_vec);
    qkvo_tile<D, float, accum_row_l> o_reg_transposed;
    swap_layout_and_transpose(o_reg_transposed, o_reg);
    store<1>(g.Og, o_reg_transposed, {batch_idx, tile_idx, head_idx, 0});
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
    py::bind_function<dispatch_micro<ATTN_D>>(m, "dispatch_micro", 
        &attn_globals<ATTN_D>::Qg, 
        &attn_globals<ATTN_D>::Kg, 
        &attn_globals<ATTN_D>::Vg, 
        &attn_globals<ATTN_D>::Og
    );
}