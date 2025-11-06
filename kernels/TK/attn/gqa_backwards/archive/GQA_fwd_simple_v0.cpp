#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

constexpr int ATTN_B = 1; // batch size
constexpr int ATTN_H = 1; // number of heads
constexpr int ATTN_H_KV = 1; // number of heads for key and value
constexpr int ATTN_N = 1024; // sequence length
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
    st_bf<KV_BLOCK_SIZE, ATTN_D, ducks::st_layout::row> (&k_smem) = al.allocate<st_bf<KV_BLOCK_SIZE, ATTN_D, ducks::st_layout::row>>();
    st_bf<KV_BLOCK_SIZE, ATTN_D, ducks::st_layout::accumulator_col> (&v_smem) = al.allocate<st_bf<KV_BLOCK_SIZE, ATTN_D, ducks::st_layout::accumulator_col>>();
    
    // const int head_idx = (blockIdx.x % 8) * 8 + (blockIdx.x / 8);
    const int head_idx = blockIdx.x;
    const int batch_idx = blockIdx.z;
    const int GROUP_SIZE = ATTN_H / ATTN_H_KV;
    const int head_idx_kv = head_idx / GROUP_SIZE;
    const int block_tile_idx = blockIdx.y;
    const int tile_idx = block_tile_idx * NUM_WARPS + warpid();
    const int stagger = warpid() / 4;

    const int num_tiles = ATTN_N / KV_BLOCK_SIZE;

    // constexpr float TEMPERATURE_SCALE = (D == 128) ? 0.08838834764f*1.44269504089f : 0.125f*1.44269504089f;
    constexpr float TEMPERATURE_SCALE = (D == 128) ? 0.08838834764f : 0.125f;

    // Initialize all of the register tiles.
    qo_tile<D, bf16> q_reg; // Q and K are both row layout, as we use mma_ABt.
    qo_tile_transposed<D, bf16> q_reg_transposed;
    kv_tile<D, bf16> k_reg;
    kv_tile_transposed<D, bf16> k_reg_transposed;

    kv_tile<D, bf16, accum_col_l> v_reg;
    qo_tile_transposed<D, float, accum_col_l> o_reg; // Output tile.
    attn_tile<D, float, accum_col_l> att_block; // attention tile, in float.
    attn_tile<D, bf16, accum_col_l> att_block_bf16;
    typename attn_tile<D, float, accum_col_l>::row_vec max_vec, norm_vec, max_vec_prev;


    // qo_tile<D, float> q_reg_fl;
    // load<1, qo_tile<D, float>, _gl_QKVO>(q_reg_fl, g.Qg, {batch_idx, tile_idx, head_idx, 0});
    // mul(q_reg_fl, q_reg_fl, TEMPERATURE_SCALE);  // Use sqrtf for clarity
    // copy(q_reg, q_reg_fl);
    // swap_layout_and_transpose(q_reg_transposed, q_reg);
    load<1>(q_reg, g.Qg, {batch_idx, tile_idx, head_idx, 0});
    swap_layout_and_transpose(q_reg_transposed, q_reg);

    zero(o_reg);
    zero(norm_vec);
    neg_infty(max_vec);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    for (int j = 0; j < num_tiles; j++) {

        G::load<1, false>(k_smem, g.Kg, {batch_idx, j, head_idx_kv, 0});
        G::load<1, false>(v_smem, g.Vg, {batch_idx, j, head_idx_kv, 0});
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        load(k_reg, k_smem);
        load(v_reg, v_smem);
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // QK
        // QK = torch.matmul(q_block.float(), k_block.float().transpose(-2, -1)) * scale
        zero(att_block);
        swap_layout_and_transpose(k_reg_transposed, k_reg);
        mma_AtB(att_block, k_reg_transposed, q_reg_transposed, att_block);
        // mul(att_block, att_block, TEMPERATURE_SCALE);

        // max_vec_prev = max_vec.clone()
        // block_max = torch.max(QK, dim=-1, keepdim=True)[0]
        // max_vec_new = torch.maximum(max_vec, block_max)
        // Softmax with correct online algorithm
        copy(max_vec_prev, max_vec);  // Store old max
        typename attn_tile<D, float, accum_col_l>::row_vec block_max;
        neg_infty(block_max);
        col_max(block_max, att_block, block_max);
        max(max_vec, block_max, max_vec);  // Update running max

        // alpha = torch.exp(max_vec_prev - max_vec_new)
        // Compute correction factor: exp(old_max - new_max)
        typename attn_tile<D, float, accum_col_l>::row_vec alpha;
        sub(alpha, max_vec_prev, max_vec);
        exp(alpha, alpha);

        // beta = torch.exp(block_max - max_vec_new)
        typename attn_tile<D, float, accum_col_l>::row_vec beta;
        sub(beta, block_max, max_vec);
        exp(beta, beta);

        // Apply corrections to previous contributions
        mul(norm_vec, norm_vec, alpha);
        mul_col(o_reg, o_reg, alpha);

        // Compute exp(QK - block_max) * beta
        sub_col(att_block, att_block, block_max);
        exp(att_block, att_block);
        mul_col(att_block, att_block, beta);

        // Update normalization: norm_vec += sum(att_block)
        col_sum(norm_vec, att_block, norm_vec);

        // Convert to bf16 for matrix multiplication
        copy(att_block_bf16, att_block);

        // Update output: o_reg += att_block @ v_block
        mma_AtB(o_reg, v_reg, att_block_bf16, o_reg);
    }

    div_col(o_reg, o_reg, norm_vec);

    qo_tile<D, float, accum_row_l> o_reg_transposed;
    swap_layout_and_transpose(o_reg_transposed, o_reg);
    store<1>(g.Og, o_reg_transposed, {batch_idx, tile_idx, head_idx, 0});

    // multiply by ln(2)
    // mul(max_vec, max_vec, 0.69314718056f);
    log(norm_vec, norm_vec);
    add(norm_vec, norm_vec, max_vec);
    store(g.L_vec, norm_vec, {batch_idx, head_idx, 0, tile_idx});
}

template<int D>
void dispatch_fwd(attn_globals<D> g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)attend_ker<D>, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    attend_ker<D><<<g.grid(), g.block(), mem_size>>>(g);
    hipDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernel_fwd, m) {
    m.doc() = "tk_kernel python module";
    py::bind_function<dispatch_fwd<ATTN_D>>(m, "dispatch_fwd", 
        &attn_globals<ATTN_D>::Qg, 
        &attn_globals<ATTN_D>::Kg, 
        &attn_globals<ATTN_D>::Vg, 
        &attn_globals<ATTN_D>::Og,
        &attn_globals<ATTN_D>::L_vec
    );
}
