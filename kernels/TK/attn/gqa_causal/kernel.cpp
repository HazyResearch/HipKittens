#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

constexpr int ATTN_B = 16; // batch size
constexpr int ATTN_H = 64; // number of heads
constexpr int ATTN_H_KV = 8; // number of heads for key and value
constexpr int ATTN_N = 256; // sequence length
constexpr int ATTN_D = 128; // dimension
constexpr int Q_BLOCK_SIZE = 32; // q block size
constexpr int KV_BLOCK_SIZE = 64; // kv block size

#define NUM_WARPS 8
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using namespace kittens;
using _gl_QKVO = gl<bf16, -1, -1, -1, -1>;

using G = kittens::group<NUM_WARPS>;

template<int D, typename T=bf16, typename L=row_l> using qo_tile = rt<T, Q_BLOCK_SIZE, D, L>;
template<int D, typename T=bf16, typename L=col_l> using qo_tile_transposed = rt<T, D, Q_BLOCK_SIZE, L>;
template<int D, typename T=bf16, typename L=row_l> using kv_tile = rt<T, KV_BLOCK_SIZE, D, L>;
template<int D, typename T=bf16, typename L=col_l> using kv_tile_transposed = rt<T, D, KV_BLOCK_SIZE, L>;
template<int D, typename T=float, typename L=accum_col_l> using attn_tile = rt<T, KV_BLOCK_SIZE, Q_BLOCK_SIZE, L>;


/**********************************************************/


template<ducks::rt::accumulator_col_layout RT>  // RT = rt<float, KV_BLOCK_SIZE, Q_BLOCK_SIZE, accum_col_l>
__device__ inline void mask_causal_kv_q_accum_col(
    RT &dst,
    int q_abs,
    int k_abs,
    const typename base_types::packing<typename RT::dtype>::unpacked_type &neg_inf
) {
    if (k_abs > q_abs + (Q_BLOCK_SIZE - 1)) { neg_infty(dst); return; } // whole future tile
    if (k_abs + KV_BLOCK_SIZE - 1 <= q_abs) { return; }            // whole past tile

    const int lane = laneid();
    const int col  = lane & 31;   // query column
    const int ro   = lane >> 5;   // row-block selector

    #pragma unroll
    for (int i = 0; i < dst.height; ++i) {      // i steps rows by 32
        const int base32 = i * 32;
        #pragma unroll
        for (int j = 0; j < dst.width; ++j) {
            #pragma unroll
            for (int ii = 0; ii < 4; ++ii) {
                const int rbase = base32 + ii*8 + ro*4;  // row within KV tile

                auto &d0 = dst.tiles[i][j].data[ii*2];
                auto &d1 = dst.tiles[i][j].data[ii*2 + 1];

                if (k_abs + (rbase + 0) > q_abs + col) d0.x = neg_inf;
                if (k_abs + (rbase + 1) > q_abs + col) d0.y = neg_inf;
                if (k_abs + (rbase + 2) > q_abs + col) d1.x = neg_inf;
                if (k_abs + (rbase + 3) > q_abs + col) d1.y = neg_inf;
            }
        }
    }
}

template<typename AT>
__device__ inline void mask_kv_tile(AT &A, int q_tile, int k_tile) {
    const int q_abs = q_tile * Q_BLOCK_SIZE;
    const int k_abs = k_tile * KV_BLOCK_SIZE;
    mask_causal_kv_q_accum_col(A, q_abs, k_abs,
        kittens::base_types::constants<float>::neg_infty());
}


/**********************************************************/


template<int D> struct attn_globals { 
    _gl_QKVO Qg, Kg, Vg, Og; 
    gl<float, -1, -1, -1, -1> L_vec;
    dim3 grid() { return dim3(ATTN_H, ((ATTN_N / Q_BLOCK_SIZE + NUM_WARPS - 1) / NUM_WARPS), ATTN_B); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

template<int D> __launch_bounds__(NUM_THREADS, 2)
__global__ void attend_ker(const attn_globals<D> g) {

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<KV_BLOCK_SIZE, ATTN_D, ducks::st_layout::row> (&k_smem)[2] = al.allocate<st_bf<KV_BLOCK_SIZE, ATTN_D, ducks::st_layout::row>, 2>();
    st_bf<KV_BLOCK_SIZE, ATTN_D, ducks::st_layout::accumulator_col> (&v_smem)[2] = al.allocate<st_bf<KV_BLOCK_SIZE, ATTN_D, ducks::st_layout::accumulator_col>, 2>();
    
    const int head_idx = (blockIdx.x % 8) * 8 + (blockIdx.x / 8);
    const int batch_idx = blockIdx.z;
    const int GROUP_SIZE = ATTN_H / ATTN_H_KV;
    const int head_idx_kv = head_idx / GROUP_SIZE;
    const int block_tile_idx = blockIdx.y;
    const int tile_idx = block_tile_idx * NUM_WARPS + warpid();
    const int stagger = warpid() / 4;

    int k_idx_buf0 = -1, k_idx_buf1 = -1;
    int k_curr_idx = -1; 

    int causal = true;
    int num_tiles;
    if (causal) {
        int q_end_pos = (tile_idx + 1) * Q_BLOCK_SIZE;
        num_tiles = (q_end_pos + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;
        num_tiles = min(num_tiles, ATTN_N / KV_BLOCK_SIZE);
    } else {
        num_tiles = ATTN_N / KV_BLOCK_SIZE;
    }

    // Find max_num_tiles across all warps in the block using shared memory
    __shared__ int block_max_tiles[NUM_WARPS];
    if (laneid() == 0) {
        block_max_tiles[warpid()] = num_tiles;
    }
    __syncthreads();
    int max_num_tiles = block_max_tiles[0];
    #pragma unroll
    for (int i = 1; i < NUM_WARPS; i++) {
        max_num_tiles = max(max_num_tiles, block_max_tiles[i]);
    }
    __syncthreads();

    constexpr float TEMPERATURE_SCALE = (D == 128) ? 0.08838834764f*1.44269504089f : 0.125f*1.44269504089f;

    // Initialize all of the register tiles.
    qo_tile<D, bf16> q_reg;
    qo_tile_transposed<D, bf16> q_reg_transposed;
    kv_tile<D, bf16> k_reg;
    kv_tile_transposed<D, bf16> k_reg_transposed;
    kv_tile<D, bf16, accum_col_l> v_reg;
    qo_tile_transposed<D, float, accum_col_l> o_reg;
    attn_tile<D, float, accum_col_l> att_block[2];
    attn_tile<D, bf16, accum_col_l> att_block_bf16;
    typename attn_tile<D, float, accum_col_l>::row_vec max_vec, norm_vec, max_vec_prev;

    G::load<1, false>(k_smem[0], g.Kg, {batch_idx, 0, head_idx_kv, 0}); 
    k_idx_buf0 = 0;
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();

    qo_tile<D, float> q_reg_fl;
    load<1, qo_tile<D, float>, _gl_QKVO>(q_reg_fl, g.Qg, {batch_idx, tile_idx, head_idx, 0});
    mul(q_reg_fl, q_reg_fl, TEMPERATURE_SCALE);
    copy(q_reg, q_reg_fl);
    swap_layout_and_transpose(q_reg_transposed, q_reg);

    zero(o_reg);
    zero(norm_vec);
    neg_infty(max_vec_prev);

    // All warps collaboratively load K1 and V0
    G::load<1, false>(k_smem[1], g.Kg, {batch_idx, 1, head_idx_kv, 0});
    k_idx_buf1 = 1; 
    G::load<1, false>(v_smem[0], g.Vg, {batch_idx, 0, head_idx_kv, 0});
    load(k_reg, k_smem[0]);
    k_curr_idx = k_idx_buf0; 
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();

    // Process first tile always
    zero(att_block[0]);
    swap_layout_and_transpose(k_reg_transposed, k_reg);
    mma_AtB(att_block[0], k_reg_transposed, q_reg_transposed, att_block[0]);
    if (causal) mask_kv_tile(att_block[0], tile_idx, k_curr_idx);
    col_max(max_vec, att_block[0]);
    sub_col(att_block[0], att_block[0], max_vec);
    exp2(att_block[0], att_block[0]);

    if (stagger) {
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
    }

    // Load K1 and prepare for next iteration
    load(k_reg, k_smem[1]);
    k_curr_idx = k_idx_buf1; 
    G::load<1, false>(k_smem[0], g.Kg, {batch_idx, 2, head_idx_kv, 0});
    k_idx_buf0 = 2; 
    G::load<1, false>(v_smem[1], g.Vg, {batch_idx, 1, head_idx_kv, 0});
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    

    // Early return path for small max_num_tiles
    if (max_num_tiles <= 4) {
        if (num_tiles > 0) {
            col_sum(norm_vec, att_block[0]);
            copy(att_block_bf16, att_block[0]);
            load(v_reg, v_smem[0]);
            __builtin_amdgcn_s_waitcnt(0);
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_barrier();
            mma_AtB(o_reg, v_reg, att_block_bf16, o_reg);
            copy(max_vec_prev, max_vec);
        }
        
        if (num_tiles > 1) {
            zero(att_block[1]);
            swap_layout_and_transpose(k_reg_transposed, k_reg);
            mma_AtB(att_block[1], k_reg_transposed, q_reg_transposed, att_block[1]);
            if (causal) mask_kv_tile(att_block[1], tile_idx, k_curr_idx);
            
            col_max(max_vec, att_block[1]);
            sub(max_vec_prev, max_vec_prev, max_vec);
            exp2(max_vec_prev, max_vec_prev);
            mul(norm_vec, norm_vec, max_vec_prev);
            mul_col(o_reg, o_reg, max_vec_prev);
            
            sub_col(att_block[1], att_block[1], max_vec);
            exp2(att_block[1], att_block[1]);
            col_sum(norm_vec, att_block[1], norm_vec);
            copy(att_block_bf16, att_block[1]);
            
            load<1, false>(v_smem[0], g.Vg, {batch_idx, 2, head_idx_kv, 0});
            load(v_reg, v_smem[1]);
            __builtin_amdgcn_s_waitcnt(0);
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_barrier();
            mma_AtB(o_reg, v_reg, att_block_bf16, o_reg);
            copy(max_vec_prev, max_vec);
        }
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        
        if (num_tiles > 2 && max_num_tiles > 2) {
            load<1, false>(k_smem[1], g.Kg, {batch_idx, 3, head_idx_kv, 0});
            load(k_reg, k_smem[0]);
            __builtin_amdgcn_s_waitcnt(0);
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_barrier();
            k_curr_idx = k_idx_buf0;
            
            zero(att_block[0]);
            swap_layout_and_transpose(k_reg_transposed, k_reg);
            mma_AtB(att_block[0], k_reg_transposed, q_reg_transposed, att_block[0]);
            if (causal) mask_kv_tile(att_block[0], tile_idx, k_curr_idx);
            
            col_max(max_vec, att_block[0]);
            sub(max_vec_prev, max_vec_prev, max_vec);
            exp2(max_vec_prev, max_vec_prev);
            mul(norm_vec, norm_vec, max_vec_prev);
            mul_col(o_reg, o_reg, max_vec_prev);
            
            sub_col(att_block[0], att_block[0], max_vec);
            exp2(att_block[0], att_block[0]);
            col_sum(norm_vec, att_block[0], norm_vec);
            copy(att_block_bf16, att_block[0]);
            
            load<1, false>(v_smem[1], g.Vg, {batch_idx, 3, head_idx_kv, 0});
            load(v_reg, v_smem[0]);
            __builtin_amdgcn_s_waitcnt(0);
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_barrier();
            mma_AtB(o_reg, v_reg, att_block_bf16, o_reg);
            copy(max_vec_prev, max_vec);
        }
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();

        if (num_tiles > 3 && max_num_tiles > 3) {
            load(k_reg, k_smem[1]);
            __builtin_amdgcn_s_waitcnt(0);
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_barrier();
            k_curr_idx = 3;

            zero(att_block[1]);
            swap_layout_and_transpose(k_reg_transposed, k_reg);
            mma_AtB(att_block[1], k_reg_transposed, q_reg_transposed, att_block[1]);
            if (causal) mask_kv_tile(att_block[1], tile_idx, k_curr_idx);

            col_max(max_vec, att_block[1]);
            sub(max_vec_prev, max_vec_prev, max_vec);
            exp2(max_vec_prev, max_vec_prev);
            mul(norm_vec, norm_vec, max_vec_prev);
            mul_col(o_reg, o_reg, max_vec_prev);

            sub_col(att_block[1], att_block[1], max_vec);
            exp2(att_block[1], att_block[1]);
            col_sum(norm_vec, att_block[1], norm_vec);
            copy(att_block_bf16, att_block[1]);

            load(v_reg, v_smem[1]);
            __builtin_amdgcn_s_waitcnt(0);
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_barrier();
            mma_AtB(o_reg, v_reg, att_block_bf16, o_reg);
            copy(max_vec_prev, max_vec);
        }
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        
        if (num_tiles > 0) {
            div_col(o_reg, o_reg, norm_vec);
            qo_tile<D, float, accum_row_l> o_reg_transposed;
            swap_layout_and_transpose(o_reg_transposed, o_reg);
            store<1>(g.Og, o_reg_transposed, {batch_idx, tile_idx, head_idx, 0});
            
            mul(max_vec_prev, max_vec_prev, 0.69314718056f);
            log(norm_vec, norm_vec);
            add(norm_vec, norm_vec, max_vec_prev);
            store(g.L_vec, norm_vec, {batch_idx, head_idx, 0, tile_idx});
        }
        
        return;
    }

    // hot loop
    // #pragma unroll  // for some reason unroll makes it slower
    for (int j = 3; j <= max_num_tiles + 1; j += 2) {
        // Break early if we've gone past available tiles
        if (j - 3 >= max_num_tiles) break;
        
        int tile_complete_first = j - 3;
        int tile_complete_second = j - 2;
        int tile_start_first = j - 2;
        int tile_start_second = j - 1;
        
        bool should_complete_first = (tile_complete_first < num_tiles);
        bool should_compute_first = (tile_start_first < num_tiles);
        bool should_complete_second = (tile_complete_second < num_tiles);
        bool should_compute_second = (tile_start_second < num_tiles);
        
        // Cluster 0: QK1
        if (should_compute_first) {
            zero(att_block[1]);
            swap_layout_and_transpose(k_reg_transposed, k_reg);
            mma_AtB(att_block[1], k_reg_transposed, q_reg_transposed, att_block[1]);
            if (causal) mask_kv_tile(att_block[1], tile_idx, k_curr_idx);
        }
        
        if (should_complete_first) {
            sub(max_vec_prev, max_vec_prev, max_vec); 
            exp2(max_vec_prev, max_vec_prev);  
            mul(norm_vec, norm_vec, max_vec_prev);
            col_sum(norm_vec, att_block[0], norm_vec);
            copy(att_block_bf16, att_block[0]);
        }
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    
        // Cluster 1: ALL warps must participate in collective loads
        // Load K(j) if in bounds, otherwise repeat a safe index
        int k_load_idx = (j < max_num_tiles) ? j : (max_num_tiles - 1);
        G::load<1, false>(k_smem[1], g.Kg, {batch_idx, k_load_idx, head_idx_kv, 0});
        k_idx_buf1 = j;
        load(v_reg, v_smem[0]);
        asm volatile("s_waitcnt lgkmcnt(0)");
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    
        // Cluster 2: A0V0
        if (should_complete_first) {
            __builtin_amdgcn_s_setprio(1);
            mul_col(o_reg, o_reg, max_vec_prev);
            mma_AtB(o_reg, v_reg, att_block_bf16, o_reg);
            __builtin_amdgcn_s_setprio(0);
        }
        
        if (should_compute_first) {
            copy(max_vec_prev, max_vec);
            col_max(max_vec, att_block[1], max_vec);
            sub_col(att_block[1], att_block[1], max_vec);
            exp2(att_block[1], att_block[1]);
        }
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    
        // Cluster 3: Loads
        int v_load_idx = ((j - 1) < max_num_tiles) ? (j - 1) : (max_num_tiles - 1);
        G::load<1, false>(v_smem[0], g.Vg, {batch_idx, v_load_idx, head_idx_kv, 0});
        load(k_reg, k_smem[0]);
        k_curr_idx = k_idx_buf0;
        asm volatile("s_waitcnt lgkmcnt(0)");
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    
        // Cluster 4: QK2
        if (should_compute_second) {
            zero(att_block[0]);
            swap_layout_and_transpose(k_reg_transposed, k_reg);
            mma_AtB(att_block[0], k_reg_transposed, q_reg_transposed, att_block[0]);
            if (causal) mask_kv_tile(att_block[0], tile_idx, k_curr_idx);
        }
        
        if (should_complete_second) {
            sub(max_vec_prev, max_vec_prev, max_vec); 
            exp2(max_vec_prev, max_vec_prev);  
            mul(norm_vec, norm_vec, max_vec_prev);
            col_sum(norm_vec, att_block[1], norm_vec);
            copy(att_block_bf16, att_block[1]);
        }
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    
        // Cluster 5: Loads
        int k_load_idx2 = ((j + 1) < max_num_tiles) ? (j + 1) : (max_num_tiles - 1);
        G::load<1, false>(k_smem[0], g.Kg, {batch_idx, k_load_idx2, head_idx_kv, 0});
        k_idx_buf0 = j + 1;
        load(v_reg, v_smem[1]);
        asm volatile("s_waitcnt lgkmcnt(0)");
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    
        // Cluster 6: A1V1
        if (should_complete_second) {
            __builtin_amdgcn_s_setprio(1);
            mul_col(o_reg, o_reg, max_vec_prev);
            mma_AtB(o_reg, v_reg, att_block_bf16, o_reg);
            __builtin_amdgcn_s_setprio(0);
        }
        
        if (should_compute_second) {
            copy(max_vec_prev, max_vec);
            col_max(max_vec, att_block[0], max_vec);
            sub_col(att_block[0], att_block[0], max_vec);
            exp2(att_block[0], att_block[0]);
        }
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    
        // Cluster 7: Loads
        int v_load_idx2 = (j < max_num_tiles) ? j : (max_num_tiles - 1);
        G::load<1, false>(v_smem[1], g.Vg, {batch_idx, v_load_idx2, head_idx_kv, 0});
        load(k_reg, k_smem[1]);
        k_curr_idx = k_idx_buf1;
        asm volatile("s_waitcnt lgkmcnt(0)");
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    }
    
    // Epilogue: finalize output
    if (num_tiles > 0) {
        div_col(o_reg, o_reg, norm_vec);
        
        if (!stagger) {
            __builtin_amdgcn_s_barrier();
        }
        
        qo_tile<D, float, accum_row_l> o_reg_transposed;
        swap_layout_and_transpose(o_reg_transposed, o_reg);
        store<1>(g.Og, o_reg_transposed, {batch_idx, tile_idx, head_idx, 0});
        
        mul(max_vec, max_vec, 0.69314718056f);
        log(norm_vec, norm_vec);
        add(norm_vec, norm_vec, max_vec);
        store(g.L_vec, norm_vec, {batch_idx, head_idx, 0, tile_idx});
    }
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
        &attn_globals<ATTN_D>::Og,
        &attn_globals<ATTN_D>::L_vec
    );
}

