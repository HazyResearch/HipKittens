// moe_gemm1.cpp -- GEMM1 (plain grouped GEMM, f32 output) kernel + host dispatch. #included by
// moe_gemm.cpp after moe_compute.cuh; not a standalone translation unit.

struct moe_globals {
    const uint8_t  *a;         // activation, FP8 E4M3 [M,K]      (M = grid_m*BLOCK_M, grouped)
    const uint8_t  *b;         // weight,    packed FP4 [E,N,K/2]
    scale_a_gl      scale_a;   // activation E8M0 [M,K/32] as [M,K/K_STEP] packed u32 (MXFP8)
    scale_b_gl      scale_b;   // weight     E8M0 [E,N,K/32] as [E,N,K/K_STEP] packed u32 (MXFP4)
    const int32_t  *expt_data; // [grid_m] expert id per m-tile, -1 = padding (skip)
    float          *c;         // output f32 [M,N]
    int M, N, K, E;
};

// ======================== Kernel: grouped GEMM, f32 output ========================

__global__ MOE_LAUNCH_ATTRS
void moe_gemm_kernel(const moe_globals g, out_gl_f32 c_gl) {
    acc_rt C_accum[MTILES][NSUB_M][NSUB_N];
    int row_base[MTILES]; bool tile_valid[MTILES];
    int warp_row, warp_col, group_expert;
    int group_m, tile_n0;
    moe_tile_ids(group_m, tile_n0);
    // PERSISTENCE: each workgroup computes PERSIST_NI consecutive tile_n's (NI=1 => the original single
    // tile). All tile_n of a group_m share the same expert (expt_data is per m-tile), so a padding m-tile
    // is padding for every tile_n -> return once.
    const int grid_n = g.N / BLOCK_N;
    #pragma unroll
    for (int nl = 0; nl < PERSIST_NI; nl++) {
        const int tile_n = tile_n0 * PERSIST_NI + nl;
        if (tile_n >= grid_n) break;
        if (!moe_compute_tile(g.a, g.b, g.scale_a, g.scale_b, g.expt_data, g.N, g.K,
                              group_m, tile_n, C_accum, row_base, tile_valid, warp_row, warp_col, group_expert))
            return;   // padding m-tile (group_m-wide) -> nothing to write for any tile_n

        // Store each 16x16 col-layout accumulator subtile to global C via kittens::store. Coord is in
        // 16-row/16-col tile units; row_base[r]+warp_row*WARP_M and the N origin are multiples of 16.
        const int n_t0 = (tile_n * BLOCK_N + warp_col * WARP_N) / 16;
        #pragma unroll
        for (int r = 0; r < MTILES; r++) {
            if (!tile_valid[r]) continue;           // padding m-tile -> don't write
            const int m_t0 = (row_base[r] + warp_row * WARP_M) / 16;
            #pragma unroll
            for (int si = 0; si < NSUB_M; si++)
                #pragma unroll
                for (int sj = 0; sj < NSUB_N; sj++)
                    kittens::store(c_gl, C_accum[r][si][sj], {0, 0, m_t0 + si, n_t0 + sj});
        }
    }
}

void dispatch(const moe_globals &g, const moe_launch_config &launch) {
    moe_validate_dims(g.M, g.N, g.K, g.E);
    moe_check_mtiles_contract(g.expt_data, g.M / BLOCK_M);
    size_t mem_size = launch.dynamic_shared_memory();
    hipError_t err = hipFuncSetAttribute(
        (void *)moe_gemm_kernel,
        hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    if (err != hipSuccess) {
        fprintf(stderr, "moe_gemm dispatch: hipFuncSetAttribute failed: %s\n", hipGetErrorString(err));
        moe_check(false, "moe_gemm dispatch: hipFuncSetAttribute failed");
    }
    moe_gemm_kernel<<<launch.grid, launch.block, mem_size, launch.stream>>>(g, make_out_gl_f32(g.c, g.M, g.N));
}
