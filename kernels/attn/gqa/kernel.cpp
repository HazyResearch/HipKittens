/**
 * @file kernel.cpp
 * @brief CDNA3 Flash Attention Forward (Non-Causal, D=128) using MFMA 16x16x16
 *
 * FlashAttention-2 online softmax with base-2 exp for numerical stability.
 * Uses mfma_f32_16x16x16 (bf16 inputs, f32 accumulator).
 *
 * Tile config:
 *   Q_ROWS = 16 per warp, KV_BLOCK = 64, NUM_WARPS = 4
 *   Total Q rows per block = 64
 *   Shared memory: 16KB (single st_bf<64,128>, K and V alternate)
 *
 * Optimization: register-buffer pipeline overlaps global memory loads
 * with MFMA compute via load_global_to_register_buffer / store_register_buffer_to_shared.
 */

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

// Compile-time shape parameters (set via -D flags or defaults)
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

// Kernel configuration
constexpr int D          = 128;   // Head dimension
constexpr int Q_ROWS     = 16;    // Q rows per warp (one MFMA row tile)
constexpr int KV_BLOCK   = 64;    // KV rows processed per iteration
constexpr int NUM_WARPS  = 4;     // Warps per block
constexpr int NUM_THREADS = kittens::WARP_THREADS * NUM_WARPS; // 256

// GQA group size
constexpr int GROUP_SIZE = ATTN_H / ATTN_H_KV;

// Temperature constant: 1/sqrt(D) * log2(e) for base-2 softmax
constexpr float TEMPERATURE = 0.0883883476f * 1.44269504089f; // 1/sqrt(128) * log2(e)

// Register buffer: 4 float4s per thread for st_bf<64, 128>
// (64*128*sizeof(bf16)) / (256*sizeof(float4)) = 16384/4096 = 4
constexpr int BUF_ELEMS = 4;

// Global layout: [B, H, N, D] — axis=2 is the sequence dimension
using gl_bf16 = gl<bf16, -1, -1, -1, -1>;
using gl_fl32 = gl<float, -1, -1, -1, -1>;

using G = kittens::group<NUM_WARPS>;

struct attn_globals {
    gl_bf16 Q;      // [B, H_Q, N, D]
    gl_bf16 K;      // [B, H_KV, N, D]
    gl_bf16 V;      // [B, H_KV, N, D]
    gl_bf16 O;      // [B, H_Q, N, D]
    gl_fl32 L;      // [B, H_Q, N, 1] — LSE (log-sum-exp)
    int N_seq;      // Sequence length
    hipStream_t stream;

    dim3 grid()  { return dim3(ATTN_H, (N_seq + Q_ROWS * NUM_WARPS - 1) / (Q_ROWS * NUM_WARPS), ATTN_B); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 16384; } // 16KB for single st_bf<64,128>
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void flash_attention_fwd(const attn_globals g) {
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

    // ========== Persistent registers (live across all iterations) ==========
    rt_bf<Q_ROWS, D> q_reg;
    rt_fl<Q_ROWS, D, ducks::rt_layout::col> o_reg;
    zero(o_reg);

    typename rt_fl<Q_ROWS, D, ducks::rt_layout::col>::col_vec max_vec;
    typename rt_fl<Q_ROWS, D, ducks::rt_layout::col>::col_vec norm_vec;
    typename rt_fl<Q_ROWS, D, ducks::rt_layout::col>::col_vec max_vec_prev;
    neg_infty(max_vec);
    zero(norm_vec);

    // Register buffer for pipelining global→shared loads (16 VGPRs)
    float4 kv_buf[BUF_ELEMS];

    // ========== Load Q once, pre-scale by temperature ==========
    if (warp_active) {
        load<2>(q_reg, g.Q, {batch_idx, head_idx, q_row_start / Q_ROWS, 0});
        const bf16 temp_bf16 = __float2bfloat16(TEMPERATURE);
        const bf16_2 temp_packed = {temp_bf16, temp_bf16};
        mul(q_reg, q_reg, temp_packed);
    }

    // ========== Prologue: issue first K load to register buffer ==========
    const int kv_blocks = (N_seq + KV_BLOCK - 1) / KV_BLOCK;
    load_global_to_register_buffer<2, false, NUM_THREADS>(kv_buf, BUF_ELEMS, g.K,
        {batch_idx, head_idx_kv, 0, 0}, kv_smem);

    // ========== Main KV loop (register-buffer pipeline) ==========
    for (int kv = 0; kv < kv_blocks; kv++) {
        // P tile must survive from K phase to V phase
        rt_bf<Q_ROWS, KV_BLOCK> p_reg;

        // ---- K phase: register buffer → shared → registers ----
        store_register_buffer_to_shared<NUM_THREADS>(kv_smem, kv_buf);
        __syncthreads();

        if (warp_active) {
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
            // Safe: k_reg loads from shared are complete, kv_buf can be reused
            load_global_to_register_buffer<2, false, NUM_THREADS>(kv_buf, BUF_ELEMS, g.V,
                {batch_idx, head_idx_kv, kv, 0}, kv_smem);

            // ---------- QK: S[16, 64] = Q[16, 128] @ K[64, 128]^T ----------
            rt_fl<Q_ROWS, KV_BLOCK, ducks::rt_layout::col> s_reg;
            zero(s_reg);

            __builtin_amdgcn_s_setprio(1);
            mma_ABt(s_reg, q_reg, k_reg, s_reg);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_sched_barrier(0);

            // Handle boundary: mask out positions beyond N_seq
            const int kv_start = kv * KV_BLOCK;
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
            // Inactive warps still issue V load (all threads must participate)
            load_global_to_register_buffer<2, false, NUM_THREADS>(kv_buf, BUF_ELEMS, g.V,
                {batch_idx, head_idx_kv, kv, 0}, kv_smem);
        }
        // V loads completing in background during QK+softmax above

        // ---- V phase: register buffer → shared → registers ----
        store_register_buffer_to_shared<NUM_THREADS>(kv_smem, kv_buf);
        __syncthreads();

        if (warp_active) {
            // Load V from shared to registers (col-major for mma_AB's B operand)
            rt_bf<KV_BLOCK, D, ducks::rt_layout::col> v_reg;
            #pragma unroll
            for (int sub = 0; sub < KV_BLOCK / Q_ROWS; sub++) {
                load(
                    subtile_inplace<Q_ROWS>(v_reg, sub),
                    subtile_inplace<Q_ROWS, D>(kv_smem, {sub, 0})
                );
            }

            // ---------- PV: O[16, 128] += P[16, 64] @ V[64, 128] ----------
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

    // ========== Final normalization: O /= norm_vec ==========
    div_row(o_reg, o_reg, norm_vec);

    // ========== Convert O to bf16 and store ==========
    rt_bf<Q_ROWS, D, ducks::rt_layout::col> o_bf_col;
    copy(o_bf_col, o_reg);

    rt_bf<Q_ROWS, D> o_bf_row;
    swap_layout(o_bf_row, o_bf_col);

    store<2>(g.O, o_bf_row, {batch_idx, head_idx, q_row_start / Q_ROWS, 0});

    // ========== Store LSE = max_vec * ln(2) + log(norm_vec) ==========
    {
        const int lane = kittens::laneid();
        constexpr float LN2 = 0.693147180559945f;
        const int row_in_group = 4 * (lane / 16);

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

void dispatch_attn(attn_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)flash_attention_fwd, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    flash_attention_fwd<<<g.grid(), g.block(), mem_size, g.stream>>>(g);
}

PYBIND11_MODULE(attn_fwd, m) {
    m.doc() = "CDNA3 Flash Attention Forward (Non-Causal, D=128)";
    py::bind_function<dispatch_attn>(m, "attention",
        &attn_globals::Q, &attn_globals::K, &attn_globals::V,
        &attn_globals::O, &attn_globals::L, &attn_globals::N_seq);
}
