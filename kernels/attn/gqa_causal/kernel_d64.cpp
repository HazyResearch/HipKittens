/**
 * @file kernel_d64.cpp
 * @brief CDNA3 Flash Attention Forward (Causal, D=64) using MFMA 16x16x16
 *
 * FlashAttention-2 with causal masking and online softmax (base-2 exp).
 * D=64 variant: half the MFMA calls, lower register pressure.
 *
 * Optimization: register-buffer pipeline overlaps global memory loads
 * with MFMA compute via load_global_to_register_buffer / store_register_buffer_to_shared.
 */

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

#ifndef ATTN_B
#define ATTN_B 4
#endif
#ifndef ATTN_H
#define ATTN_H 32
#endif
#ifndef ATTN_H_KV
#define ATTN_H_KV 8
#endif
#ifndef ATTN_N
#define ATTN_N 1024
#endif

constexpr int D          = 64;
constexpr int Q_ROWS     = 16;
constexpr int KV_BLOCK   = 64;
constexpr int NUM_WARPS  = 4;
constexpr int NUM_THREADS = kittens::WARP_THREADS * NUM_WARPS;

constexpr int GROUP_SIZE = ATTN_H / ATTN_H_KV;
constexpr float TEMPERATURE = 0.125f * 1.44269504089f; // 1/sqrt(64) * log2(e)

// Register buffer: 2 float4s per thread for st_bf<64, 64>
constexpr int BUF_ELEMS = 2;

using gl_bf16 = gl<bf16, -1, -1, -1, -1>;
using gl_fl32 = gl<float, -1, -1, -1, -1>;

using G = kittens::group<NUM_WARPS>;

struct attn_globals {
    gl_bf16 Q;
    gl_bf16 K;
    gl_bf16 V;
    gl_bf16 O;
    gl_fl32 L;
    int N_seq;
    hipStream_t stream;

    dim3 grid()  { return dim3(ATTN_H, (N_seq + Q_ROWS * NUM_WARPS - 1) / (Q_ROWS * NUM_WARPS), ATTN_B); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 8192; } // 8KB for single st_bf<64,64>
};

__global__ __launch_bounds__(NUM_THREADS, 2)
void flash_attention_causal_d64_fwd(const attn_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    st_bf<KV_BLOCK, D> (&kv_smem) = al.allocate<st_bf<KV_BLOCK, D>>();

    const int warp_id  = kittens::warpid();
    const int head_idx = blockIdx.x;
    const int batch_idx = blockIdx.z;
    const int head_idx_kv = head_idx / GROUP_SIZE;
    const int q_block_idx = blockIdx.y;
    const int q_row_start = q_block_idx * (Q_ROWS * NUM_WARPS) + warp_id * Q_ROWS;

    const int N_seq = g.N_seq;
    const bool warp_active = (q_row_start < N_seq);

    const int q_row_end = q_row_start + Q_ROWS - 1;

    // ========== Persistent registers (live across all iterations) ==========
    rt_bf<Q_ROWS, D> q_reg;
    rt_fl<Q_ROWS, D, ducks::rt_layout::col> o_reg;
    zero(o_reg);

    typename rt_fl<Q_ROWS, D, ducks::rt_layout::col>::col_vec max_vec;
    typename rt_fl<Q_ROWS, D, ducks::rt_layout::col>::col_vec norm_vec;
    typename rt_fl<Q_ROWS, D, ducks::rt_layout::col>::col_vec max_vec_prev;
    neg_infty(max_vec);
    zero(norm_vec);

    // Register buffer for pipelining global→shared loads (8 VGPRs)
    float4 kv_buf[BUF_ELEMS];

    // ========== Load Q once, pre-scale by temperature ==========
    if (warp_active) {
        load<2>(q_reg, g.Q, {batch_idx, head_idx, q_row_start / Q_ROWS, 0});
        const bf16 temp_bf16 = __float2bfloat16(TEMPERATURE);
        const bf16_2 temp_packed = {temp_bf16, temp_bf16};
        mul(q_reg, q_reg, temp_packed);
    }

    // For causal: use the maximum possible KV blocks across all warps in the block
    // All warps must iterate the same number of times for register-buffer loads
    const int block_q_row_end = (q_block_idx + 1) * (Q_ROWS * NUM_WARPS) - 1;
    const int max_kv_block = (min(block_q_row_end, N_seq - 1) / KV_BLOCK) + 1;
    const int kv_blocks = min(max_kv_block, (N_seq + KV_BLOCK - 1) / KV_BLOCK);

    // ========== Prologue: issue first K load to register buffer ==========
    load_global_to_register_buffer<2, false, NUM_THREADS>(kv_buf, BUF_ELEMS, g.K,
        {batch_idx, head_idx_kv, 0, 0}, kv_smem);

    // ========== Main KV loop (register-buffer pipeline) ==========
    for (int kv = 0; kv < kv_blocks; kv++) {
        const int kv_start = kv * KV_BLOCK;

        // P tile must survive from K phase to V phase
        rt_bf<Q_ROWS, KV_BLOCK> p_reg;

        // Only compute if this warp is active AND KV block is relevant for causal
        const bool warp_computes = warp_active && (kv_start <= q_row_end);

        // ---- K phase: register buffer → shared → registers ----
        store_register_buffer_to_shared<NUM_THREADS>(kv_smem, kv_buf);
        __syncthreads();

        if (warp_computes) {
            // Load K from shared to registers (scoped for register reuse)
            rt_bf<KV_BLOCK, D> k_reg;
            #pragma unroll
            for (int sub = 0; sub < KV_BLOCK / Q_ROWS; sub++) {
                load(
                    subtile_inplace<Q_ROWS>(k_reg, sub),
                    subtile_inplace<Q_ROWS, D>(kv_smem, {sub, 0})
                );
            }

            // Issue V[kv] global → register buffer (OVERLAPS with QK compute below)
            load_global_to_register_buffer<2, false, NUM_THREADS>(kv_buf, BUF_ELEMS, g.V,
                {batch_idx, head_idx_kv, kv, 0}, kv_smem);

            // ---------- QK: S[16, 64] = Q[16, 64] @ K[64, 64]^T ----------
            rt_fl<Q_ROWS, KV_BLOCK, ducks::rt_layout::col> s_reg;
            zero(s_reg);

            __builtin_amdgcn_s_setprio(1);
            mma_ABt(s_reg, q_reg, k_reg, s_reg);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_sched_barrier(0);

            // Apply causal mask
            const bool needs_causal_mask = (kv_start + KV_BLOCK - 1 > q_row_start);
            if (needs_causal_mask) {
                const int mask_offset = kv_start - q_row_start;
                tril(s_reg, s_reg, mask_offset, -__builtin_inff());
            }

            // Handle boundary: mask out positions beyond N_seq
            if (kv_start + KV_BLOCK > N_seq) {
                const int valid_cols = N_seq - kv_start;
                right_fill(s_reg, s_reg, valid_cols, -__builtin_inff());
            }

            // ---------- Online softmax (base-2 exp) ----------
            copy(max_vec_prev, max_vec);
            row_max(max_vec, s_reg, max_vec_prev);

            typename rt_fl<Q_ROWS, D, ducks::rt_layout::col>::col_vec scale_vec;
            sub(scale_vec, max_vec_prev, max_vec);
            exp2(scale_vec, scale_vec);

            mul_row(o_reg, o_reg, scale_vec);
            mul(norm_vec, norm_vec, scale_vec);

            sub_row(s_reg, s_reg, max_vec);
            exp2(s_reg, s_reg);
            row_sum(norm_vec, s_reg, norm_vec);

            // ---------- Convert S to bf16 row-major for PV multiply ----------
            rt_bf<Q_ROWS, KV_BLOCK, ducks::rt_layout::col> p_col;
            copy(p_col, s_reg);
            swap_layout(p_reg, p_col);
        } else {
            // Inactive/skipped warps still issue V load (all threads must participate)
            load_global_to_register_buffer<2, false, NUM_THREADS>(kv_buf, BUF_ELEMS, g.V,
                {batch_idx, head_idx_kv, kv, 0}, kv_smem);
        }

        // ---- V phase: register buffer → shared → registers ----
        store_register_buffer_to_shared<NUM_THREADS>(kv_smem, kv_buf);
        __syncthreads();

        if (warp_computes) {
            // Load V from shared to registers (col-major for mma_AB's B operand)
            rt_bf<KV_BLOCK, D, ducks::rt_layout::col> v_reg;
            #pragma unroll
            for (int sub = 0; sub < KV_BLOCK / Q_ROWS; sub++) {
                load(
                    subtile_inplace<Q_ROWS>(v_reg, sub),
                    subtile_inplace<Q_ROWS, D>(kv_smem, {sub, 0})
                );
            }

            // ---------- PV: O[16, 64] += P[16, 64] @ V[64, 64] ----------
            __builtin_amdgcn_s_setprio(1);
            mma_AB(o_reg, p_reg, v_reg, o_reg);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_sched_barrier(0);
        }

        // Issue K[kv+1] global → register buffer (OVERLAPS with PV compute above)
        if (kv + 1 < kv_blocks) {
            load_global_to_register_buffer<2, false, NUM_THREADS>(kv_buf, BUF_ELEMS, g.K,
                {batch_idx, head_idx_kv, kv + 1, 0}, kv_smem);
        }

        __syncthreads();
    }

    if (!warp_active) return;

    div_row(o_reg, o_reg, norm_vec);

    rt_bf<Q_ROWS, D, ducks::rt_layout::col> o_bf_col;
    copy(o_bf_col, o_reg);
    rt_bf<Q_ROWS, D> o_bf_row;
    swap_layout(o_bf_row, o_bf_col);
    store<2>(g.O, o_bf_row, {batch_idx, head_idx, q_row_start / Q_ROWS, 0});

    {
        const int lane = kittens::laneid();
        const int row_in_group = 4 * (lane / 16);
        constexpr float LN2 = 0.693147180559945f;

        float m0 = max_vec.data[0][0].x;
        float m1 = max_vec.data[0][0].y;
        float m2 = max_vec.data[0][1].x;
        float m3 = max_vec.data[0][1].y;
        float n0 = norm_vec.data[0][0].x;
        float n1 = norm_vec.data[0][0].y;
        float n2 = norm_vec.data[0][1].x;
        float n3 = norm_vec.data[0][1].y;

        float lse0 = m0 * LN2 + __logf(n0 + 1e-10f);
        float lse1 = m1 * LN2 + __logf(n1 + 1e-10f);
        float lse2 = m2 * LN2 + __logf(n2 + 1e-10f);
        float lse3 = m3 * LN2 + __logf(n3 + 1e-10f);

        if (lane % 16 == 0) {
            int base_row = q_row_start + row_in_group;
            if (base_row + 0 < N_seq) g.L[{batch_idx, head_idx, base_row + 0, 0}] = lse0;
            if (base_row + 1 < N_seq) g.L[{batch_idx, head_idx, base_row + 1, 0}] = lse1;
            if (base_row + 2 < N_seq) g.L[{batch_idx, head_idx, base_row + 2, 0}] = lse2;
            if (base_row + 3 < N_seq) g.L[{batch_idx, head_idx, base_row + 3, 0}] = lse3;
        }
    }
}

void dispatch_attn_causal_d64(attn_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)flash_attention_causal_d64_fwd, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    flash_attention_causal_d64_fwd<<<g.grid(), g.block(), mem_size, g.stream>>>(g);
}

PYBIND11_MODULE(attn_causal_fwd_d64, m) {
    m.doc() = "CDNA3 Flash Attention Forward (Causal, D=64)";
    py::bind_function<dispatch_attn_causal_d64>(m, "attention",
        &attn_globals::Q, &attn_globals::K, &attn_globals::V,
        &attn_globals::O, &attn_globals::L, &attn_globals::N_seq);
}
