// moe_swiglu.cpp -- GEMM1 + fused SwiGLU kernels (bf16 out, and the MXFP8 re-quant variant) + their
// host dispatches. #included by moe_gemm.cpp after moe_compute.cuh; not a standalone translation unit.

// SwiGLU variant globals: same inputs, but the N dim packs gate/up interleaved
// (col 2j = gate, col 2j+1 = up); output is bf16 [M, N/2].
struct moe_swiglu_globals {
    const uint8_t  *a;
    const uint8_t  *b;
    scale_a_gl      scale_a;
    scale_b_gl      scale_b;
    const int32_t  *expt_data;
    kittens::bf16  *c;          // output bf16 [M, N/2]
    const float    *bias;       // optional per-expert bias [E, N] added to acc pre-SwiGLU (nullptr=off)
    const float    *gammas;     // optional per-token scale [M] applied post-SwiGLU (nullptr=off)
    int M, N, K, E;
    float alpha;               // SwiGLU gate gain: s = silu(alpha*gate)
    float limit;               // clamp magnitude (<=0 disables): gate<=limit, up in [-limit,limit]
    int   add_residual;        // 0: out = s*up ; 1: out = s*(up+1)   (aiter ADD_RESIDUAL)
};

// SwiGLU + MXFP8 (FP8 e4m3 + per-32 e8m0) output variant: fuses the GEMM1->GEMM2 activation
// re-quant. Output FP8 [M, N/2] + e8m0 scale [M, (N/2)/32] (one e8m0 per 32-col group; since
// OUT_BLOCK_N==32 that is one scale per row per N-tile).
struct moe_swiglu_mxfp8_globals {
    const uint8_t  *a;
    const uint8_t  *b;
    scale_a_gl      scale_a;
    scale_b_gl      scale_b;
    const int32_t  *expt_data;
    uint8_t        *c;          // output FP8 e4m3 [M, N/2]
    uint8_t        *c_scale;    // output e8m0     [M, (N/2)/32]
    const float    *bias;       // optional [E, N]
    const float    *gammas;     // optional [M]
    int M, N, K, E;
    float alpha, limit;
    int   add_residual;
};

// SwiGLU pointwise (matches aiter `_swiglu`): s = silu(alpha*gate); out = add_res ? s*(up+1) : s*up.
__device__ __forceinline__ float moe_swiglu_apply(
    float gate, float up, float alpha, float limit, int add_residual) {
    if (limit > 0.f) { gate = fminf(gate, limit); up = fmaxf(-limit, fminf(up, limit)); }
    const float s = gate / (1.0f + __expf(-alpha * gate));
    return add_residual ? (s * up + s) : (s * up);
}

// ======================== Kernel: grouped GEMM + fused SwiGLU, bf16 output ========================
// Epilogue stages the [BLOCK_M, BLOCK_N] tile into LDS, then computes (matches aiter `_swiglu`):
//   gate = acc[m,2j] (even col), up = acc[m,2j+1] (odd col)        [interleaved pairs]
//   if limit>0: gate = min(gate, limit); up = clip(up, [-limit,limit])
//   s   = silu(alpha*gate) = gate / (1 + exp(-alpha*gate))
//   out = add_residual ? s*(up+1) : s*up
// OUT_BLOCK_N = BLOCK_N/2, output bf16 [M, N/2].
__global__ MOE_LAUNCH_ATTRS
void moe_gemm_swiglu_kernel(const moe_swiglu_globals g) {
    acc_rt C_accum[MTILES][NSUB_M][NSUB_N];
    int row_base[MTILES]; bool tile_valid[MTILES];
    int warp_row, warp_col, group_expert;
    int group_m, tile_n;
    moe_tile_ids(group_m, tile_n);
    if (!moe_compute_tile(g.a, g.b, g.scale_a, g.scale_b, g.expt_data, g.N, g.K,
                          group_m, tile_n, C_accum, row_base, tile_valid, warp_row, warp_col, group_expert))
        return;

    // f32 epilogue staging buffer [BLOCK_M, BLOCK_N] in dynamic shared memory (sized by the dispatcher).
    extern __shared__ uint8_t raw_lds[];
    float *const ep = reinterpret_cast<float *>(raw_lds);

    constexpr int OUT_BLOCK_N = BLOCK_N / 2;
    const int yN = g.N / 2;
    const int n_base = tile_n * BLOCK_N;            // input feature base (gate+up)
    const int out_col_base = tile_n * OUT_BLOCK_N;
    const int L = kittens::laneid();
    const int t_col = L % 16;
    const int t_half = L / 16;
    const int m_w = warp_row * WARP_M;
    const int n_w = warp_col * WARP_N;

    // One m-tile at a time: stage its accumulator into ep, then cooperative SwiGLU. The shared ep
    // buffer is reused per tile; tile_valid[] is uniform across the WG so barriers stay balanced.
    #pragma unroll
    for (int r = 0; r < MTILES; r++) {
        if (!tile_valid[r]) continue;
        kittens::sync::sync();   // prior ep reads (or K-loop LDS reads) done before we overwrite ep
        #pragma unroll
        for (int si = 0; si < NSUB_M; si++)
            #pragma unroll
            for (int sj = 0; sj < NSUB_N; sj++) {
                const int n = n_w + sj * 16 + t_col;
                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    const int m = m_w + si * 16 + t_half * 8 + k * 2;
                    ep[m * BLOCK_N + n]       = C_AT(r, si, sj, k).x;
                    ep[(m + 1) * BLOCK_N + n] = C_AT(r, si, sj, k).y;
                }
            }
        kittens::sync::sync();   // ep fully written before SwiGLU reads

        for (int idx = threadIdx.x; idx < BLOCK_M * OUT_BLOCK_N; idx += blockDim.x) {
            const int m = idx / OUT_BLOCK_N;
            const int j = idx % OUT_BLOCK_N;
            float gate = ep[m * BLOCK_N + 2 * j];
            float up   = ep[m * BLOCK_N + 2 * j + 1];
            if (g.bias) {                                   // bias added to acc pre-SwiGLU
                gate += g.bias[(size_t)group_expert * g.N + n_base + 2 * j];
                up   += g.bias[(size_t)group_expert * g.N + n_base + 2 * j + 1];
            }
            float out = moe_swiglu_apply(gate, up, g.alpha, g.limit, g.add_residual);
            if (g.gammas) out *= g.gammas[row_base[r] + m]; // per-token scale post-SwiGLU
            g.c[(size_t)(row_base[r] + m) * yN + (out_col_base + j)] =
                kittens::base_types::convertor<kittens::bf16, float>::convert(out);
        }
    }
}

// ======================== Kernel: grouped GEMM + SwiGLU + MXFP8 re-quant (FP8 e4m3 + e8m0) ====
// One thread per row computes its OUT_BLOCK_N SwiGLU outputs, then per 32-col group finds the amax,
// derives the e8m0 scale (matches aiter `_mxfp8_quant_op`), and writes FP8 + the e8m0 byte.
// NGROUP = OUT_BLOCK_N/32 MXFP8 groups per (row, N-tile).
__global__ MOE_LAUNCH_ATTRS
void moe_gemm_swiglu_mxfp8_kernel(const moe_swiglu_mxfp8_globals g) {
    acc_rt C_accum[MTILES][NSUB_M][NSUB_N];
    int row_base[MTILES]; bool tile_valid[MTILES];
    int warp_row, warp_col, group_expert;
    int group_m, tile_n;
    moe_tile_ids(group_m, tile_n);
    if (!moe_compute_tile(g.a, g.b, g.scale_a, g.scale_b, g.expt_data, g.N, g.K,
                          group_m, tile_n, C_accum, row_base, tile_valid, warp_row, warp_col, group_expert))
        return;

    extern __shared__ uint8_t raw_lds[];
    float *const ep = reinterpret_cast<float *>(raw_lds);
    const int L = kittens::laneid();
    const int t_col = L % 16, t_half = L / 16;
    const int m_w = warp_row * WARP_M, n_w = warp_col * WARP_N;

    constexpr int OUT_BLOCK_N = BLOCK_N / 2;
    constexpr int NGROUP      = OUT_BLOCK_N / 32;        // MXFP8 groups per (row, N-tile)
    // OUT_BLOCK_N (=BLOCK_N/2) must be a whole number of 32-col MXFP8 groups, else the tail cols of
    // the tile are never quantized/written. Requires BLOCK_N % 64 == 0.
    static_assert(OUT_BLOCK_N % 32 == 0, "mxfp8 re-quant needs BLOCK_N a multiple of 64 (OUT_BLOCK_N % 32 == 0)");
    const int yN = g.N / 2;
    const int n_groups = yN / 32;
    const int n_base = tile_n * BLOCK_N;
    const int out_col_base = tile_n * OUT_BLOCK_N;
    const int group_base = tile_n * NGROUP;          // first MXFP8 group index for this tile

    // One m-tile at a time: stage acc into ep, then one-thread-per-row SwiGLU + MXFP8 re-quant.
    #pragma unroll
    for (int r = 0; r < MTILES; r++) {
        if (!tile_valid[r]) continue;
        kittens::sync::sync();
        #pragma unroll
        for (int si = 0; si < NSUB_M; si++)
            #pragma unroll
            for (int sj = 0; sj < NSUB_N; sj++) {
                const int n = n_w + sj * 16 + t_col;
                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    const int m = m_w + si * 16 + t_half * 8 + k * 2;
                    ep[m * BLOCK_N + n]       = C_AT(r, si, sj, k).x;
                    ep[(m + 1) * BLOCK_N + n] = C_AT(r, si, sj, k).y;
                }
            }
        kittens::sync::sync();

        const int m = threadIdx.x;
        if (m < BLOCK_M) {
            float vals[OUT_BLOCK_N];
            #pragma unroll
            for (int j = 0; j < OUT_BLOCK_N; j++) {
                float gate = ep[m * BLOCK_N + 2 * j];
                float up   = ep[m * BLOCK_N + 2 * j + 1];
                if (g.bias) {
                    gate += g.bias[(size_t)group_expert * g.N + n_base + 2 * j];
                    up   += g.bias[(size_t)group_expert * g.N + n_base + 2 * j + 1];
                }
                float out = moe_swiglu_apply(gate, up, g.alpha, g.limit, g.add_residual);
                if (g.gammas) out *= g.gammas[row_base[r] + m];
                vals[j] = out;
            }
            // Per 32-col MXFP8 group: amax -> e8m0 (aiter _mxfp8_quant_op) -> FP8 quant.
            #pragma unroll
            for (int gp = 0; gp < NGROUP; gp++) {
                float amax = 0.0f;
                #pragma unroll
                for (int jj = 0; jj < 32; jj++) amax = fmaxf(amax, fabsf(vals[gp * 32 + jj]));
                uint32_t ai; __builtin_memcpy(&ai, &amax, 4);
                ai = (ai + 0x200000u) & 0xFF800000u;
                float amax_p2; __builtin_memcpy(&amax_p2, &ai, 4);
                // All-zero group: log2f(0) = -inf -> qscale = inf -> 0*inf = NaN. Emit a zero
                // block (e8m0 scale 0, FP8 zeros) instead of quantizing.
                if (amax_p2 == 0.0f) {
                    g.c_scale[(size_t)(row_base[r] + m) * n_groups + group_base + gp] = 0;
                    #pragma unroll
                    for (int jj = 0; jj < 32; jj++)
                        g.c[(size_t)(row_base[r] + m) * yN + (out_col_base + gp * 32 + jj)] = 0;
                    continue;
                }
                // amax_p2 is an exact power of two (mantissa zeroed above), so read its unbiased
                // exponent straight from the bits. This is exact and avoids log2f rounding a
                // boundary power-of-two down by an ULP (which would halve qscale and saturate FP8).
                float su = (float)((int)((ai >> 23) & 0xFFu) - 127 - 8);
                su = fmaxf(-127.0f, fminf(127.0f, su));
                const uint8_t e8m0 = (uint8_t)((int)su + 127);
                const float qscale = exp2f(-su);
                g.c_scale[(size_t)(row_base[r] + m) * n_groups + group_base + gp] = e8m0;
                #pragma unroll
                for (int jj = 0; jj < 32; jj++) {
                    const int j = gp * 32 + jj;
                    kittens::fp8e4m3 q = kittens::base_types::convertor<kittens::fp8e4m3, float>::convert(vals[j] * qscale);
                    g.c[(size_t)(row_base[r] + m) * yN + (out_col_base + j)] = *reinterpret_cast<uint8_t *>(&q);
                }
            }
        }
    }
}

void dispatch_swiglu(const moe_swiglu_globals &g, const moe_launch_config &launch) {
    moe_validate_dims(g.M, g.N, g.K, g.E);
    moe_check_mtiles_contract(g.expt_data, g.M / BLOCK_M);
    // f32 epilogue staging buffer ep[BLOCK_M, BLOCK_N] (dynamic LDS; the GEMM core uses static tiles).
    size_t mem_size = (size_t)BLOCK_M * BLOCK_N * sizeof(float);
    hipError_t err = hipFuncSetAttribute(
        (void *)moe_gemm_swiglu_kernel,
        hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    if (err != hipSuccess) {
        fprintf(stderr, "moe_gemm swiglu dispatch: hipFuncSetAttribute failed: %s\n", hipGetErrorString(err));
        moe_check(false, "moe_gemm swiglu dispatch: hipFuncSetAttribute failed");
    }
    moe_gemm_swiglu_kernel<<<launch.grid, launch.block, mem_size, launch.stream>>>(g);
}

void dispatch_swiglu_mxfp8(const moe_swiglu_mxfp8_globals &g, const moe_launch_config &launch) {
    moe_validate_dims(g.M, g.N, g.K, g.E);
    moe_check_mtiles_contract(g.expt_data, g.M / BLOCK_M);
    // f32 epilogue staging buffer ep[BLOCK_M, BLOCK_N] (dynamic LDS; the GEMM core uses static tiles).
    size_t mem_size = (size_t)BLOCK_M * BLOCK_N * sizeof(float);
    hipError_t err = hipFuncSetAttribute(
        (void *)moe_gemm_swiglu_mxfp8_kernel,
        hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    if (err != hipSuccess) {
        fprintf(stderr, "moe_gemm swiglu_mxfp8 dispatch: hipFuncSetAttribute failed: %s\n", hipGetErrorString(err));
        moe_check(false, "moe_gemm swiglu_mxfp8 dispatch: hipFuncSetAttribute failed");
    }
    moe_gemm_swiglu_mxfp8_kernel<<<launch.grid, launch.block, mem_size, launch.stream>>>(g);
}
