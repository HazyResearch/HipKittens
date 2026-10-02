#include "mma_scaled_art.cuh"

#ifdef TEST_WARP_REGISTER_TILE_MMA_SCALED_ART

using namespace kittens;

// Scaled fp8 GEMM on art (pinned-register) tiles: load and mma_ABt_scaled with all eight
// combinations of VGPR/AGPR operands and accumulators, against the rt mma_scaled CPU reference.
constexpr int ART_M = 128, ART_N = 128, ART_K = 256, ART_BLOCK_K = 128;
constexpr int ART_WAVE_TILE = 64;   // each of the 4 waves computes a 64x64 output tile

template <int BASE, int N>
using fp8_ranges = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<BASE, BASE + 8 * N - 1>>, 8>;
template <int BASE, int N>
using acc_ranges = ducks::art::split_many_t<ducks::art::type_list<ducks::art::range<BASE, BASE + 4 * N - 1>>, 4>;

using G4 = group<4>;

template <bool A_AGPR, bool B_AGPR, bool C_AGPR>
__global__ __launch_bounds__(256, 1) __attribute__((amdgpu_num_vgpr(128)))
void art_mini_gemm_kernel(
    const gl<fp8e4m3, 1, 1, ART_M, ART_K> A,
    const gl<fp8e4m3, 1, 1, ART_N, ART_K> B,
    float *__restrict__ C,
    const gl<fp8e8m0, -1, 1, 16, 64> scale_A_gl,
    const gl<fp8e8m0, -1, 1, 16, 64> scale_B_gl) {
    using ART_A = art<fp8e4m3, ART_WAVE_TILE, ART_BLOCK_K, ducks::rt_layout::row, rt_16x128_s,
                      fp8_ranges<A_AGPR ? 320 : 128, 4>>;
    using ART_B = art<fp8e4m3, ART_WAVE_TILE, ART_BLOCK_K, ducks::rt_layout::row, rt_16x128_s,
                      fp8_ranges<B_AGPR ? 352 : 160, 4>>;
    using ART_C = art<float, ART_WAVE_TILE, ART_WAVE_TILE, ducks::rt_layout::col, rt_16x16_s,
                      acc_ranges<C_AGPR ? 256 : 192, 16>>;
    using ST = st_fp8e4m3<ART_M, ART_BLOCK_K, st_16x128_s>;
    using ST_Scale = st<fp8e8m0, 16, 64, st_16x64_s>;
    __shared__ __attribute__((aligned(2048))) ST As, Bs;
    __shared__ ST_Scale scale_A_smem, scale_B_smem;

    // The pinned registers are only written by inline asm; clobbering them makes the kernel allocate them.
    ducks::art::clobber<typename ART_A::register_ranges>();
    ducks::art::clobber<typename ART_B::register_ranges>();
    ducks::art::clobber<typename ART_C::register_ranges>();
    ART_A a;
    ART_B b;
    ART_C c;
    zero(c);

    const int warp_m = warpid() / 2, warp_n = warpid() % 2;
    constexpr int copies = ART_M * ART_BLOCK_K / (ST::underlying_subtile_bytes_per_thread * 4 * WARP_THREADS);
    uint32_t sw_A[copies], sw_B[copies];
    G4::prefill_swizzled_offsets(As, A, sw_A);
    G4::prefill_swizzled_offsets(Bs, B, sw_B);

    for (int k = 0; k < ART_K / ART_BLOCK_K; k++) {
        G4::load(As, A, {0, 0, 0, k}, sw_A);
        G4::load(Bs, B, {0, 0, 0, k}, sw_B);
        G4::load(scale_A_smem, scale_A_gl, {k, 0, 0, 0});
        G4::load(scale_B_smem, scale_B_gl, {k, 0, 0, 0});
        asm volatile("s_waitcnt vmcnt(0)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        const uint32_t sa = static_cast<uint32_t>(pack_scales(scale_A_smem.data, warp_m * ART_WAVE_TILE));
        const uint32_t sb = static_cast<uint32_t>(pack_scales(scale_B_smem.data, warp_n * ART_WAVE_TILE));
        auto as = subtile_inplace<ART_WAVE_TILE, ART_BLOCK_K>(As, {warp_m, 0});
        auto bs = subtile_inplace<ART_WAVE_TILE, ART_BLOCK_K>(Bs, {warp_n, 0});
        if (k == 0) {
            load(a, as);
            load(b, bs);
        } else {
            const auto aa = get_address(a, as), ba = get_address(b, bs);
            [&]<int... Ns>(std::integer_sequence<int, Ns...>) {
                ([&] {
                    load<Ns, 0>(a, as, aa);
                    load<Ns, 0, 0>(b, bs, ba[0]);
                    load<Ns, 0, 1>(b, bs, ba[1]);
                }(), ...);
            }(std::make_integer_sequence<int, 4>{});
        }
        asm volatile("s_waitcnt lgkmcnt(0)");
        mma_ABt_scaled(c, a, b, c, sa, sb);
        __builtin_amdgcn_s_barrier();
    }

    // col layout accumulator: lane l of subtile (n, m) holds column m*16 + l%16, rows n*16 + 4*(l/16) .. +3
    const int lane = laneid();
    [&]<int... Is>(std::integer_sequence<int, Is...>) {
        ([&]<int I>() {
            constexpr int n = I / 4, m = I % 4;
            using range = ducks::art::get_nth_range_t<typename ART_C::register_ranges, I>;
            const float v[4] = {macros::v_mov_b32_p2up<range::lo + 0, float>(), macros::v_mov_b32_p2up<range::lo + 1, float>(),
                                macros::v_mov_b32_p2up<range::lo + 2, float>(), macros::v_mov_b32_p2up<range::lo + 3, float>()};
            const int row = warp_m * ART_WAVE_TILE + n * 16 + 4 * (lane / 16);
            const int col = warp_n * ART_WAVE_TILE + m * 16 + lane % 16;
            for (int r = 0; r < 4; r++) C[(row + r) * ART_N + col] = v[r];
        }.template operator()<Is>(), ...);
    }(std::make_integer_sequence<int, 16>{});
}

static uint8_t art_compute_e8m0(const float *vals, int count) {
    float mx = 0.0f;
    for (int i = 0; i < count; i++) mx = std::max(mx, std::abs(vals[i]));
    if (mx == 0.0f) return 0;
    return (uint8_t)std::clamp((int)std::floor(std::log2(mx)) + 127, 0, 254);
}

template <bool A_AGPR, bool B_AGPR, bool C_AGPR>
void run_art_mini_gemm_test(test_data &results) {
    test_info info;
    info.label = std::string("art_mini_gemm_128x128x256_A") + (A_AGPR ? "a" : "v") +
                 "_B" + (B_AGPR ? "a" : "v") + "_C" + (C_AGPR ? "a" : "v");

    constexpr int M = ART_M, N = ART_N, K = ART_K, scale_K = K / 32, k_iters = K / ART_BLOCK_K;
    constexpr int SCALE_ROWS = 256;   // the 16x64 E8M0 scale tile holds one packed word for each of 256 rows

    std::mt19937 rng(321);
    std::normal_distribution<float> dist(0.0f, 0.5f);
    std::vector<float> a_f(M * K), b_f(N * K);
    for (auto &v : a_f) v = dist(rng);
    for (auto &v : b_f) v = dist(rng);

    auto quantize = [&](const std::vector<float> &f, int rows, std::vector<fp8e4m3> &q, std::vector<uint8_t> &s) {
        for (int row = 0; row < rows; row++)
            for (int kb = 0; kb < scale_K; kb++) {
                const float *blk = &f[row * K + kb * 32];
                const uint8_t e = art_compute_e8m0(blk, 32);
                s[row * scale_K + kb] = e;
                const float inv = std::ldexp(1.0f, 127 - (int)e);
                for (int i = 0; i < 32; i++) q[row * K + kb * 32 + i] = fp8e4m3(blk[i] * inv);
            }
    };
    std::vector<fp8e4m3> a_q(M * K), b_q(N * K);
    std::vector<uint8_t> sa_raw(M * scale_K), sb_raw(N * scale_K);
    quantize(a_f, M, a_q, sa_raw);
    quantize(b_f, N, b_q, sb_raw);

    std::vector<float> c_ref(M * N, 0.0f);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int kb = 0; kb < scale_K; kb++) {
                const float sa = std::ldexp(1.0f, (int)sa_raw[i * scale_K + kb] - 127);
                const float sb = std::ldexp(1.0f, (int)sb_raw[j * scale_K + kb] - 127);
                for (int ki = 0; ki < 32; ki++)
                    acc += (float)a_q[i * K + kb * 32 + ki] * sa * (float)b_q[j * K + kb * 32 + ki] * sb;
            }
            c_ref[i * N + j] = acc;
        }

    auto pack_iter = [&](const std::vector<uint8_t> &raw, int rows, std::vector<uint32_t> &out) {
        for (int ki = 0; ki < k_iters; ki++)
            for (int row = 0; row < rows; row++) {
                uint32_t p = 0;
                for (int j = 0; j < 4; j++) p |= (uint32_t)raw[row * scale_K + ki * 4 + j] << (j * 8);
                out[ki * SCALE_ROWS + row] = p;
            }
    };
    std::vector<uint32_t> sa_iter(k_iters * SCALE_ROWS, 0), sb_iter(k_iters * SCALE_ROWS, 0);
    pack_iter(sa_raw, M, sa_iter);
    pack_iter(sb_raw, N, sb_iter);

    fp8e4m3 *d_a, *d_b; float *d_c; uint32_t *d_sa, *d_sb;
    hipMalloc(&d_a, M * K);
    hipMalloc(&d_b, N * K);
    hipMalloc(&d_c, M * N * sizeof(float));
    hipMalloc(&d_sa, sa_iter.size() * sizeof(uint32_t));
    hipMalloc(&d_sb, sb_iter.size() * sizeof(uint32_t));
    hipMemcpy(d_a, a_q.data(), M * K, hipMemcpyHostToDevice);
    hipMemcpy(d_b, b_q.data(), N * K, hipMemcpyHostToDevice);
    hipMemcpy(d_sa, sa_iter.data(), sa_iter.size() * sizeof(uint32_t), hipMemcpyHostToDevice);
    hipMemcpy(d_sb, sb_iter.data(), sb_iter.size() * sizeof(uint32_t), hipMemcpyHostToDevice);
    hipMemset(d_c, 0, M * N * sizeof(float));

    gl<fp8e4m3, 1, 1, M, K> A_gl(d_a, nullptr, nullptr, nullptr, nullptr);
    gl<fp8e4m3, 1, 1, N, K> B_gl(d_b, nullptr, nullptr, nullptr, nullptr);
    gl<fp8e8m0, -1, 1, 16, 64> SA_gl(reinterpret_cast<fp8e8m0 *>(d_sa), k_iters, nullptr, nullptr, nullptr);
    gl<fp8e8m0, -1, 1, 16, 64> SB_gl(reinterpret_cast<fp8e8m0 *>(d_sb), k_iters, nullptr, nullptr, nullptr);

    art_mini_gemm_kernel<A_AGPR, B_AGPR, C_AGPR><<<1, 256>>>(A_gl, B_gl, d_c, SA_gl, SB_gl);
    const hipError_t launch_err = hipGetLastError();
    const hipError_t sync_err = hipDeviceSynchronize();

    std::vector<float> c_gpu(M * N);
    const hipError_t copy_err = hipMemcpy(c_gpu.data(), d_c, M * N * sizeof(float), hipMemcpyDeviceToHost);

    float c_max = 0;
    for (float v : c_ref) c_max = std::max(c_max, std::abs(v));
    const float atol = c_max * 0.001f;
    int fail_count = 0;
    float max_err = 0;
    for (int i = 0; i < M * N; i++) {
        const float err = std::abs(c_gpu[i] - c_ref[i]);
        max_err = std::max(max_err, err);
        if (!std::isfinite(c_gpu[i]) || err > atol) fail_count++;
    }
    const bool runtime_ok = launch_err == hipSuccess && sync_err == hipSuccess && copy_err == hipSuccess;
    info.result = (runtime_ok && fail_count == 0) ? test_result::PASSED : test_result::FAILED;
    if (!runtime_ok)
        printf("  %s: launch=%s, sync=%s, copy=%s\n", info.label.c_str(), hipGetErrorString(launch_err),
               hipGetErrorString(sync_err), hipGetErrorString(copy_err));
    if (fail_count > 0)
        printf("  %s: FAILED %d/%d (max_err=%.4f, atol=%.4f)\n", info.label.c_str(), fail_count, M * N, max_err, atol);

    hipFree(d_a); hipFree(d_b); hipFree(d_c); hipFree(d_sa); hipFree(d_sb);
    results.push_back(info);
}

void warp::reg::tile::mma_scaled_art::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/register/tile/mma_scaled_art tests! -----\n" << std::endl;
    run_art_mini_gemm_test<false, false, false>(results);
    run_art_mini_gemm_test<false, false, true>(results);
    run_art_mini_gemm_test<false, true, false>(results);
    run_art_mini_gemm_test<false, true, true>(results);
    run_art_mini_gemm_test<true, false, false>(results);
    run_art_mini_gemm_test<true, false, true>(results);
    run_art_mini_gemm_test<true, true, false>(results);
    run_art_mini_gemm_test<true, true, true>(results);
}
#endif
