#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include "../utils.cpp"

constexpr int ATTN_B = 16; // batch size
constexpr int ATTN_H = 64; // number of heads
constexpr int ATTN_H_KV = 8; // number of heads for key and value
constexpr int ATTN_N = 16384; // sequence length
constexpr int ATTN_D = 128; // dimension
constexpr int N_STEP = 128;
constexpr int BLOCK_SIZE = 32; // block size

#define NUM_WARPS 8
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using namespace kittens;
using _gl_QKVO = gl<bf16, -1, -1, -1, -1>;


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
    const int GROUP_SIZE = ATTN_H / ATTN_H_KV;
    const int head_idx_kv = blockIdx.y / GROUP_SIZE;
    const int block_tile_idx = blockIdx.z;
    const int tile_idx = block_tile_idx * NUM_WARPS + warpid();
    const int stagger = warpid() / 4;

    const int num_tiles = ATTN_N / N_STEP;
    const int num_sub_tiles = N_STEP / BLOCK_SIZE;

    constexpr float TEMPERATURE_SCALE = (D == 128) ? 0.08838834764f*1.44269504089f : 0.125f*1.44269504089f;

    // Initialize all of the register tiles.
    qkvo_tile<D, bf16> q_reg; // Q and K are both row layout, as we use mma_ABt.
    qkvo_tile_transposed<D, bf16> q_reg_transposed;
    qkvo_tile<D, bf16> k_reg, k_reg_2;
    qkvo_tile_transposed<D, bf16> k_reg_transposed, k_reg_transposed_2;

    qkvo_tile<D, bf16, col_l> v_reg, v_reg_2;
    qkvo_tile_transposed<D, float, accum_l> o_reg; // Output tile.
    attn_tile<D, float, accum_l> att_block; // attention tile, in float.
    attn_tile<D, bf16, accum_l> att_block_bf16;
    attn_tile<D, bf16, col_l> att_block_col_bf16;
    typename attn_tile<D, float, accum_l>::row_vec max_vec, norm_vec, max_vec_prev;

    int tic = 0, toc = 1;

    load_global_to_shared_direct<1, false, st_bf<N_STEP, ATTN_D>, _gl_QKVO, coord<st_bf<N_STEP,ATTN_D>>, NUM_THREADS>(
        g.Kg, {batch_idx, 0, head_idx_kv, 0}, k_smem[tic]);
    load_global_to_shared_direct<1, false, st_bf<N_STEP, ATTN_D>, _gl_QKVO, coord<st_bf<N_STEP,ATTN_D>>, NUM_THREADS>(
        g.Vg, {batch_idx, 0, head_idx_kv, 0}, v_smem[tic]);
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

    if (stagger) {
        __builtin_amdgcn_s_barrier();
    }


    for (int j = 0; j < num_tiles - 1; j++, tic^=1, toc^=1) {

        // Cluster: (load)
        load_global_to_shared_direct<1, false, st_bf<N_STEP, ATTN_D>, _gl_QKVO, coord<st_bf<N_STEP, ATTN_D>>, NUM_THREADS>(g.Kg, {batch_idx, j + 1, head_idx_kv, 0}, k_smem[toc]);
        load_lds_reg(k_reg, subtile_inplace<BLOCK_SIZE, ATTN_D>(k_smem[tic], {0, 0}));
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster: (compute)
        __builtin_amdgcn_s_setprio(1);
        zero(att_block);
        copy(max_vec_prev, max_vec); // Store previous max values
        swap_layout_and_transpose(k_reg_transposed, k_reg);
        mma_AtB(att_block, k_reg_transposed, q_reg_transposed, att_block); // A = Q @ K.T

        col_max(max_vec, att_block, max_vec);  // Update max in-place and compute correction
        sub_col(att_block, att_block, max_vec);
        exp2(att_block, att_block);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster: (load)
        load_global_to_shared_direct<1, false, st_bf<N_STEP, ATTN_D>, _gl_QKVO, coord<st_bf<N_STEP, ATTN_D>>, NUM_THREADS>(
            g.Vg, {batch_idx, j + 1, head_idx_kv, 0}, v_smem[toc]);
        load_lds_reg_col(v_reg, subtile_inplace<BLOCK_SIZE, ATTN_D>(v_smem[tic], {0, 0}));

        load_lds_reg(k_reg, subtile_inplace<BLOCK_SIZE, ATTN_D>(k_smem[tic], {1, 0}));
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);


        // Cluster: (compute)
        sub(max_vec_prev, max_vec_prev, max_vec); 
        exp2(max_vec_prev, max_vec_prev);  
        mul(norm_vec, norm_vec, max_vec_prev);
        col_sum(norm_vec, att_block, norm_vec);
        copy(att_block_bf16, att_block);  
        att_block_col_bf16 = swap_layout_inplace<col_l>(att_block_bf16); 

        zero(att_block); 
        swap_layout_and_transpose(k_reg_transposed, k_reg);
        mma_AtB(att_block, k_reg_transposed, q_reg_transposed, att_block); // A = Q @ K.T
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);


        // Cluster: (compute) -- Finish Tile 0 and Start Tile 1
        mul_col(o_reg, o_reg, max_vec_prev);   // Update running output
        mma_AtB(o_reg, v_reg, att_block_col_bf16, o_reg);

        copy(max_vec_prev, max_vec); // Store previous max values
        col_max(max_vec, att_block, max_vec);  // Update max in-place and compute correction
        sub_col(att_block, att_block, max_vec);
        exp2(att_block, att_block); 
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();


        // Cluster: (load)
        load_lds_reg_col(v_reg, subtile_inplace<BLOCK_SIZE, ATTN_D>(v_smem[tic], {1, 0}));
        load_lds_reg(k_reg, subtile_inplace<BLOCK_SIZE, ATTN_D>(k_smem[tic], {2, 0}));
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);


        // Cluster: (compute)
        sub(max_vec_prev, max_vec_prev, max_vec); 
        exp2(max_vec_prev, max_vec_prev);  
        mul(norm_vec, norm_vec, max_vec_prev);
        col_sum(norm_vec, att_block, norm_vec);
        copy(att_block_bf16, att_block);  
        att_block_col_bf16 = swap_layout_inplace<col_l>(att_block_bf16); 

        zero(att_block); 
        swap_layout_and_transpose(k_reg_transposed, k_reg);
        mma_AtB(att_block, k_reg_transposed, q_reg_transposed, att_block); // A = Q @ K.T
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        
        // Cluster: (compute) -- Finish Tile 0 and Start Tile 1
        mul_col(o_reg, o_reg, max_vec_prev);   // Update running output
        mma_AtB(o_reg, v_reg, att_block_col_bf16, o_reg);

        copy(max_vec_prev, max_vec); // Store previous max values
        col_max(max_vec, att_block, max_vec);  // Update max in-place and compute correction
        sub_col(att_block, att_block, max_vec);
        exp2(att_block, att_block); 
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();

        // Cluster: (load)
        load_lds_reg_col(v_reg, subtile_inplace<BLOCK_SIZE, ATTN_D>(v_smem[tic], {2, 0}));
        load_lds_reg(k_reg, subtile_inplace<BLOCK_SIZE, ATTN_D>(k_smem[tic], {3, 0}));
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);


        // Cluster: (compute)
        sub(max_vec_prev, max_vec_prev, max_vec); 
        exp2(max_vec_prev, max_vec_prev);  
        mul(norm_vec, norm_vec, max_vec_prev);
        col_sum(norm_vec, att_block, norm_vec);
        copy(att_block_bf16, att_block);  
        att_block_col_bf16 = swap_layout_inplace<col_l>(att_block_bf16); 

        zero(att_block); 
        swap_layout_and_transpose(k_reg_transposed, k_reg);
        mma_AtB(att_block, k_reg_transposed, q_reg_transposed, att_block); // A = Q @ K.T
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        
        // Cluster: (compute) -- Finish Tile 0 and Start Tile 1
        mul_col(o_reg, o_reg, max_vec_prev);   // Update running output
        mma_AtB(o_reg, v_reg, att_block_col_bf16, o_reg);

        copy(max_vec_prev, max_vec); // Store previous max values
        col_max(max_vec, att_block, max_vec);  // Update max in-place and compute correction
        sub_col(att_block, att_block, max_vec);
        exp2(att_block, att_block); 
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();


        // Cluster: (load)
        load_lds_reg_col(v_reg, subtile_inplace<BLOCK_SIZE, ATTN_D>(v_smem[tic], {3, 0}));
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);


        // Cluster: (compute)
        sub(max_vec_prev, max_vec_prev, max_vec); 
        exp2(max_vec_prev, max_vec_prev);  
        mul(norm_vec, norm_vec, max_vec_prev);
        col_sum(norm_vec, att_block, norm_vec);
        copy(att_block_bf16, att_block);  
        att_block_col_bf16 = swap_layout_inplace<col_l>(att_block_bf16); 

        mul_col(o_reg, o_reg, max_vec_prev);   // Update running output
        mma_AtB(o_reg, v_reg, att_block_col_bf16, o_reg);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

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
    qkvo_tile<D, float, accum_l> o_reg_transposed;
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


