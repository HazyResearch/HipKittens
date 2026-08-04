#include "mma_scaled.cuh"

#ifdef TEST_WARP_REGISTER_TILE_MMA_SCALED

using namespace kittens;

using GG_test = group<8>;

// Test 1: Scale pipeline (G::load into st<fp8e8m0> + pack_scales)
__global__ __launch_bounds__(512, 1)
void scale_pipeline_kernel(
    const gl<fp8e8m0, -1, 1, 16, 64> scale_A_gl,
    uint32_t *__restrict__ out) {
    using ST_Scale = st<fp8e8m0, 16, 64, st_16x64_s>;
    __shared__ ST_Scale scale_smem;

    GG_test::load(scale_smem, scale_A_gl, {0, 0, 0, 0});
    asm volatile("s_waitcnt vmcnt(0)");
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();

    int lid  = laneid();
    int wid  = warpid();
    if (wid >= 2) return;

    int row_offset = wid * 64;
    fp8e8m0_4 packed = pack_scales(scale_smem.data, row_offset);

    out[wid * 64 + lid] = (uint32_t)packed;
}

void run_scale_pipeline_test(test_data &results) {
    test_info info;
    info.label = "scale_pipeline";

    constexpr int M       = 256;
    constexpr int K       = 128;
    constexpr int scale_K = K / 32;
    constexpr int k_iters = K / 128;

    std::vector<uint8_t> raw_scales(M * scale_K);
    std::mt19937 rng(42);
    for (auto &v : raw_scales) v = rng() % 255;

    std::vector<uint32_t> iter_scales(k_iters * M);
    for (int ki = 0; ki < k_iters; ki++) {
        int kb_base = ki * 4;
        for (int row = 0; row < M; row++) {
            uint32_t p = 0;
            for (int j = 0; j < 4; j++)
                p |= (uint32_t)raw_scales[row * scale_K + kb_base + j] << (j * 8);
            iter_scales[ki * M + row] = p;
        }
    }

    uint32_t *d_sa, *d_out;
    hipMalloc(&d_sa, k_iters * M * sizeof(uint32_t));
    hipMalloc(&d_out, 2 * 64 * sizeof(uint32_t));
    hipMemcpy(d_sa, iter_scales.data(), k_iters * M * sizeof(uint32_t), hipMemcpyHostToDevice);
    hipMemset(d_out, 0, 2 * 64 * sizeof(uint32_t));

    gl<fp8e8m0, -1, 1, 16, 64> SA_gl(reinterpret_cast<fp8e8m0 *>(d_sa), 1, nullptr, nullptr, nullptr);
    scale_pipeline_kernel<<<1, 512>>>(SA_gl, d_out);
    hipDeviceSynchronize();

    std::vector<uint32_t> gpu_out(2 * 64);
    hipMemcpy(gpu_out.data(), d_out, 2 * 64 * sizeof(uint32_t), hipMemcpyDeviceToHost);

    int fail_count = 0;
    for (int wid = 0; wid < 2; wid++) {
        int row_offset = wid * 64;
        for (int lid = 0; lid < 64; lid++) {
            int r16   = lid % 16;
            int k_sub = lid / 16;

            uint32_t w0 = iter_scales[row_offset + 0 * 16 + r16];
            uint32_t w1 = iter_scales[row_offset + 1 * 16 + r16];
            uint32_t w2 = iter_scales[row_offset + 2 * 16 + r16];
            uint32_t w3 = iter_scales[row_offset + 3 * 16 + r16];

            uint8_t b0 = (w0 >> (k_sub * 8)) & 0xFF;
            uint8_t b1 = (w1 >> (k_sub * 8)) & 0xFF;
            uint8_t b2 = (w2 >> (k_sub * 8)) & 0xFF;
            uint8_t b3 = (w3 >> (k_sub * 8)) & 0xFF;

            uint32_t expected = b0 | (b1 << 8) | ((uint32_t)b2 << 16) | ((uint32_t)b3 << 24);
            uint32_t actual   = gpu_out[wid * 64 + lid];

            if (actual != expected) {
                if (fail_count < 4)
                    printf("  FAIL wid=%d lid=%d: got 0x%08X expected 0x%08X\n",
                           wid, lid, actual, expected);
                fail_count++;
            }
        }
    }

    info.result = (fail_count == 0) ? test_result::PASSED : test_result::FAILED;
    if (fail_count > 0)
        printf("  scale_pipeline: %d/128 failures\n", fail_count);

    hipFree(d_sa); hipFree(d_out);
    results.push_back(info);
}


// Test 2: End-to-end mini GEMM (G::load → S→R → mma_ABt_scaled → store)
constexpr int GEMM_M       = 256;
constexpr int GEMM_N       = 256;
constexpr int GEMM_K       = 256;
constexpr int GEMM_BLOCK_K = 128;
constexpr int GEMM_HALF    = 128;
constexpr int GEMM_REG_M   = 64;
constexpr int GEMM_REG_N   = 32;

using GG = group<8>;

__global__ __launch_bounds__(512, 1)
void mini_gemm_kernel(
    const gl<fp8e4m3, 1, 1, GEMM_M, GEMM_K> A,
    const gl<fp8e4m3, 1, 1, GEMM_N, GEMM_K> B,
    const gl<float,   1, 1, GEMM_M, GEMM_N> C,
    const gl<fp8e8m0, -1, 1, 16, 64> scale_A_gl,
    const gl<fp8e8m0, -1, 1, 16, 64> scale_B_gl) {
    using ST_A = st_fp8e4m3<GEMM_HALF, GEMM_BLOCK_K, st_16x128_s>;
    using ST_B = st_fp8e4m3<GEMM_HALF, GEMM_BLOCK_K, st_16x128_s>;
    using ST_Scale = st<fp8e8m0, 16, 64, st_16x64_s>;

    __shared__ ST_A As[2];
    __shared__ ST_B Bs[2];
    __shared__ ST_Scale scale_A_smem, scale_B_smem;

    rt_fp8e4m3<GEMM_REG_M, GEMM_BLOCK_K> a;
    rt_fp8e4m3<GEMM_REG_N, GEMM_BLOCK_K> b0, b1;
    rt_fl<GEMM_REG_M, GEMM_REG_N, col_l, rt_16x16_s> cA, cB, cC, cD;

    zero(cA); zero(cB); zero(cC); zero(cD);

    int warp_m = warpid() / 4;
    int warp_n = warpid() % 4;

    constexpr int bpt      = ST_A::underlying_subtile_bytes_per_thread;
    constexpr int bpm      = bpt * 8 * WARP_THREADS;
    constexpr int copies_A = GEMM_HALF * GEMM_BLOCK_K / bpm;
    constexpr int copies_B = GEMM_HALF * GEMM_BLOCK_K / bpm;
    uint32_t sw_A[copies_A], sw_B[copies_B];
    GG::prefill_swizzled_offsets(As[0], A, sw_A);
    GG::prefill_swizzled_offsets(Bs[0], B, sw_B);

    int a_row_h0  = warp_m * GEMM_REG_M;
    int a_row_h1  = GEMM_HALF + warp_m * GEMM_REG_M;
    int b_row_h0  = warp_n * GEMM_REG_N;
    int b_row_h1  = GEMM_HALF + warp_n * GEMM_REG_N;

    for (int k = 0; k < GEMM_K / GEMM_BLOCK_K; k++) {
        GG::load(As[0], A, {0, 0, 0, k}, sw_A);
        GG::load(As[1], A, {0, 0, 1, k}, sw_A);
        GG::load(Bs[0], B, {0, 0, 0, k}, sw_B);
        GG::load(Bs[1], B, {0, 0, 1, k}, sw_B);
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();

        GG::load(scale_A_smem, scale_A_gl, {k, 0, 0, 0});
        GG::load(scale_B_smem, scale_B_gl, {k, 0, 0, 0});
        asm volatile("s_waitcnt vmcnt(0)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        fp8e8m0_4 sa_h0 = pack_scales(scale_A_smem.data, a_row_h0);
        fp8e8m0_4 sa_h1 = pack_scales(scale_A_smem.data, a_row_h1);
        fp8e8m0_4 sb_h0 = pack_scales(scale_B_smem.data, b_row_h0);
        fp8e8m0_4 sb_h1 = pack_scales(scale_B_smem.data, b_row_h1);

        auto as0 = subtile_inplace<GEMM_REG_M, GEMM_BLOCK_K>(As[0], {warp_m, 0});
        auto as1 = subtile_inplace<GEMM_REG_M, GEMM_BLOCK_K>(As[1], {warp_m, 0});
        auto bs0 = subtile_inplace<GEMM_REG_N, GEMM_BLOCK_K>(Bs[0], {warp_n, 0});
        auto bs1 = subtile_inplace<GEMM_REG_N, GEMM_BLOCK_K>(Bs[1], {warp_n, 0});

        load(a, as0); load(b0, bs0); load(b1, bs1);
        asm volatile("s_waitcnt lgkmcnt(0)");

        mma_ABt_scaled(cA, a, b0, cA, &sa_h0, &sb_h0);
        mma_ABt_scaled(cB, a, b1, cB, &sa_h0, &sb_h1);

        load(a, as1);
        asm volatile("s_waitcnt lgkmcnt(0)");

        mma_ABt_scaled(cC, a, b0, cC, &sa_h1, &sb_h0);
        mma_ABt_scaled(cD, a, b1, cD, &sa_h1, &sb_h1);

        __builtin_amdgcn_s_barrier();
    }

    store(C, cA, {0, 0, warp_m,          warp_n});
    store(C, cB, {0, 0, warp_m,          4 + warp_n});
    store(C, cC, {0, 0, 2 + warp_m,      warp_n});
    store(C, cD, {0, 0, 2 + warp_m,      4 + warp_n});
}

static uint8_t compute_e8m0(const float *vals, int count) {
    float mx = 0.0f;
    for (int i = 0; i < count; i++) mx = std::max(mx, std::abs(vals[i]));
    if (mx == 0.0f) return 0;
    return (uint8_t)std::clamp((int)std::floor(std::log2(mx)) + 127, 0, 254);
}

void run_mini_gemm_test(test_data &results) {
    test_info info;
    info.label = "mini_gemm_256x256x256";

    constexpr int M       = GEMM_M;
    constexpr int N       = GEMM_N;
    constexpr int K       = GEMM_K;
    constexpr int scale_K = K / 32;
    constexpr int k_iters = K / GEMM_BLOCK_K;

    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.0f, 0.5f);

    std::vector<float> a_f(M * K), b_f(N * K);
    for (auto &v : a_f) v = dist(rng);
    for (auto &v : b_f) v = dist(rng);

    std::vector<fp8e4m3> a_q(M * K), b_q(N * K);
    std::vector<uint8_t> sa_raw(M * scale_K), sb_raw(N * scale_K);

    for (int row = 0; row < M; row++)
        for (int kb = 0; kb < scale_K; kb++) {
            float blk[32];
            for (int i = 0; i < 32; i++) blk[i] = a_f[row * K + kb * 32 + i];
            uint8_t s = compute_e8m0(blk, 32);
            sa_raw[row * scale_K + kb] = s;
            float inv = std::ldexp(1.0f, 127 - (int)s);
            for (int i = 0; i < 32; i++) a_q[row * K + kb * 32 + i] = fp8e4m3(blk[i] * inv);
        }
    for (int row = 0; row < N; row++)
        for (int kb = 0; kb < scale_K; kb++) {
            float blk[32];
            for (int i = 0; i < 32; i++) blk[i] = b_f[row * K + kb * 32 + i];
            uint8_t s = compute_e8m0(blk, 32);
            sb_raw[row * scale_K + kb] = s;
            float inv = std::ldexp(1.0f, 127 - (int)s);
            for (int i = 0; i < 32; i++) b_q[row * K + kb * 32 + i] = fp8e4m3(blk[i] * inv);
        }

    std::vector<float> c_ref(M * N, 0.0f);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int kb = 0; kb < scale_K; kb++) {
                float sa = std::ldexp(1.0f, (int)sa_raw[i * scale_K + kb] - 127);
                float sb = std::ldexp(1.0f, (int)sb_raw[j * scale_K + kb] - 127);
                for (int ki = 0; ki < 32; ki++)
                    acc += (float)a_q[i * K + kb * 32 + ki] * sa
                         * (float)b_q[j * K + kb * 32 + ki] * sb;
            }
            c_ref[i * N + j] = acc;
        }

    std::vector<uint32_t> sa_iter(k_iters * M), sb_iter(k_iters * N);
    for (int ki = 0; ki < k_iters; ki++) {
        int kb_base = ki * 4;
        for (int row = 0; row < M; row++) {
            uint32_t p = 0;
            for (int j = 0; j < 4; j++)
                p |= (uint32_t)sa_raw[row * scale_K + kb_base + j] << (j * 8);
            sa_iter[ki * M + row] = p;
        }
        for (int row = 0; row < N; row++) {
            uint32_t p = 0;
            for (int j = 0; j < 4; j++)
                p |= (uint32_t)sb_raw[row * scale_K + kb_base + j] << (j * 8);
            sb_iter[ki * N + row] = p;
        }
    }

    fp8e4m3 *d_a, *d_b; float *d_c; uint32_t *d_sa, *d_sb;
    hipMalloc(&d_a, M * K);
    hipMalloc(&d_b, N * K);
    hipMalloc(&d_c, M * N * sizeof(float));
    hipMalloc(&d_sa, k_iters * M * sizeof(uint32_t));
    hipMalloc(&d_sb, k_iters * N * sizeof(uint32_t));
    hipMemcpy(d_a, a_q.data(), M * K, hipMemcpyHostToDevice);
    hipMemcpy(d_b, b_q.data(), N * K, hipMemcpyHostToDevice);
    hipMemcpy(d_sa, sa_iter.data(), k_iters * M * sizeof(uint32_t), hipMemcpyHostToDevice);
    hipMemcpy(d_sb, sb_iter.data(), k_iters * N * sizeof(uint32_t), hipMemcpyHostToDevice);
    hipMemset(d_c, 0, M * N * sizeof(float));

    gl<fp8e4m3, 1, 1, M, K> A_gl(d_a, nullptr, nullptr, nullptr, nullptr);
    gl<fp8e4m3, 1, 1, N, K> B_gl(d_b, nullptr, nullptr, nullptr, nullptr);
    gl<float,   1, 1, M, N> C_gl(d_c, nullptr, nullptr, nullptr, nullptr);
    gl<fp8e8m0, -1, 1, 16, 64> SA_gl(reinterpret_cast<fp8e8m0 *>(d_sa), k_iters, nullptr, nullptr, nullptr);
    gl<fp8e8m0, -1, 1, 16, 64> SB_gl(reinterpret_cast<fp8e8m0 *>(d_sb), k_iters, nullptr, nullptr, nullptr);

    mini_gemm_kernel<<<1, 512>>>(A_gl, B_gl, C_gl, SA_gl, SB_gl);
    hipDeviceSynchronize();

    std::vector<float> c_gpu(M * N);
    hipMemcpy(c_gpu.data(), d_c, M * N * sizeof(float), hipMemcpyDeviceToHost);

    float c_max = 0;
    for (int i = 0; i < M * N; i++) c_max = std::max(c_max, std::abs(c_ref[i]));
    float atol = c_max * 0.001f;

    int fail_count = 0;
    float max_err = 0;
    for (int i = 0; i < M * N; i++) {
        float err = std::abs(c_gpu[i] - c_ref[i]);
        max_err = std::max(max_err, err);
        if (err > atol) fail_count++;
    }

    info.result = (fail_count == 0) ? test_result::PASSED : test_result::FAILED;
    if (fail_count > 0)
        printf("  mini_gemm: FAILED %d/%d (max_err=%.4f, atol=%.4f)\n",
               fail_count, M * N, max_err, atol);

    hipFree(d_a); hipFree(d_b); hipFree(d_c); hipFree(d_sa); hipFree(d_sb);
    results.push_back(info);
}
void warp::reg::tile::mma_scaled::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/register/tile/mma_scaled tests! -----\n" << std::endl;

    run_scale_pipeline_test(results);
    run_mini_gemm_test(results);
}
#endif
