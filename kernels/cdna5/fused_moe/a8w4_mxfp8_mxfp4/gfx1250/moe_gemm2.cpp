// moe_gemm2.cpp -- GEMM2 (down-projection, bf16 output) kernel + host dispatch. #included by
// moe_gemm.cpp after moe_compute.cuh; not a standalone translation unit.

// GEMM2 (down-projection) globals: a plain grouped MXFP8 x MXFP4 GEMM with bf16 output and an
// optional residual add. The activation A is the MXFP8 intermediate produced by the GEMM1
// swiglu_mxfp8 kernel (same FP8 e4m3 + e8m0 format), so the shared moe_compute_tile core is reused
// unchanged. B is the per-expert down-projection weight [E, N=d_model, K=d_ff]. No SwiGLU.
struct moe_gemm2_globals {
    const uint8_t  *a;          // MXFP8 intermediate [M, K=d_ff]
    const uint8_t  *b;          // MXFP4 down weight  [E, N=d_model, K/2]
    scale_a_gl      scale_a;    // [M, K/32] as [M, K/K_STEP] packed u32
    scale_b_gl      scale_b;    // [E, N, K/32] as [E, N, K/K_STEP] packed u32
    const int32_t  *expt_data;  // [grid_m]
    kittens::bf16  *c;          // output bf16 [M, N=d_model]
    const kittens::bf16 *residual;  // optional [M, N] added post-GEMM (nullptr = off)
    int M, N, K, E;
};

// ======================== Kernel: GEMM2 down-projection, bf16 output (+ optional residual) =======
// Plain grouped MXFP8 x MXFP4 GEMM (no SwiGLU): A = MXFP8 intermediate [M, d_ff], B = down weight
// [E, d_model, d_ff], C = bf16 [M, d_model]. Reuses moe_compute_tile; the only difference vs the
// f32 kernel is a bf16 (optionally residual-added) epilogue. Same MOE_MTILES / MOE_XCD levers apply.
__global__ MOE_LAUNCH_ATTRS
void moe_gemm2_kernel(const moe_gemm2_globals g, out_gl_bf16 c_gl, out_gl_bf16 res_gl) {
    acc_rt C_accum[MTILES][NSUB_M][NSUB_N];
    int row_base[MTILES]; bool tile_valid[MTILES];
    int warp_row, warp_col, group_expert;
    int group_m, tile_n;
    moe_tile_ids(group_m, tile_n);
    if (!moe_compute_tile(g.a, g.b, g.scale_a, g.scale_b, g.expt_data, g.N, g.K,
                          group_m, tile_n, C_accum, row_base, tile_valid, warp_row, warp_col, group_expert))
        return;

    // Store each 16x16 col-layout accumulator subtile to bf16 global C via kittens::store (which
    // converts f32->bf16). Optional residual: load it into a matching col-layout rt (g2r, bf16->f32)
    // and add on-register before the store. Coord is in 16-row/16-col tile units.
    const int n_t0 = (tile_n * BLOCK_N + warp_col * WARP_N) / 16;
    #pragma unroll
    for (int r = 0; r < MTILES; r++) {
        if (!tile_valid[r]) continue;
        const int m_t0 = (row_base[r] + warp_row * WARP_M) / 16;
        #pragma unroll
        for (int si = 0; si < NSUB_M; si++)
            #pragma unroll
            for (int sj = 0; sj < NSUB_N; sj++) {
                if (g.residual) {
                    acc_rt res;
                    kittens::load(res, res_gl, {0, 0, m_t0 + si, n_t0 + sj});
                    kittens::add(C_accum[r][si][sj], C_accum[r][si][sj], res);
                }
                kittens::store(c_gl, C_accum[r][si][sj], {0, 0, m_t0 + si, n_t0 + sj});
            }
    }
}

void dispatch_gemm2(const moe_gemm2_globals &g, const moe_launch_config &launch) {
    moe_validate_dims(g.M, g.N, g.K, g.E);
    moe_check_mtiles_contract(g.expt_data, g.M / BLOCK_M);
    size_t mem_size = launch.dynamic_shared_memory();
    hipError_t err = hipFuncSetAttribute(
        (void *)moe_gemm2_kernel,
        hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    if (err != hipSuccess) {
        fprintf(stderr, "moe_gemm2 dispatch: hipFuncSetAttribute failed: %s\n", hipGetErrorString(err));
        moe_check(false, "moe_gemm2 dispatch: hipFuncSetAttribute failed");
    }
    moe_gemm2_kernel<<<launch.grid, launch.block, mem_size, launch.stream>>>(
        g, make_out_gl_bf16(g.c, g.M, g.N),
        make_out_gl_bf16(const_cast<kittens::bf16 *>(g.residual), g.M, g.N));
}
