/**
 * @file harness.h
 * @brief Standalone host harness for the gfx1201 bf16 GEMM kernels.
 *
 * Each kernel `.cpp` defines `gemm_globals` and `dispatch(gemm_globals)` and
 * includes this header last. Built with `-DHARNESS_MAIN` it adds a `main()`:
 *
 *     ./<kernel> M N K iters verify
 *
 * which fills A and B with uniform(-1, 1) bf16, times `iters` launches after one
 * warm-up, and checks the result against an fp32 CPU reference:
 * verify=1 checks every output, verify=2 checks NSAMPLE (default 512) random
 * outputs as full-K dot products (use it for large shapes), verify=0 skips it.
 *
 * Without HARNESS_MAIN this header is empty, so the kernel file can be compiled
 * into other code.
 */

#pragma once

#ifdef HARNESS_MAIN

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static inline void hip_check(hipError_t e, const char* what, int line) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "HIP error at line %d (%s): %s\n",
                     line, what, hipGetErrorString(e));
        std::exit(1);
    }
}
#define HIP_OK(call) hip_check((call), #call, __LINE__)

static inline float bf16_to_float(__hip_bfloat16 v) { return static_cast<float>(v); }
static inline __hip_bfloat16 float_to_bf16(float v) { return __hip_bfloat16(v); }

static void cpu_gemm_abt_ref(const std::vector<float>& A,
                             const std::vector<float>& B,
                             std::vector<float>& C,
                             int M, int N, int K)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k) {
                acc += A[i * K + k] * B[j * K + k]; // B^T -> B[j,k] = B[j*K+k]
            }
            C[i * N + j] = acc;
        }
    }
}

int main(int argc, char** argv)
{
    int M = (argc > 1) ? std::atoi(argv[1]) : 256;
    int N = (argc > 2) ? std::atoi(argv[2]) : 256;
    int K = (argc > 3) ? std::atoi(argv[3]) : 256;
    int n_iters = (argc > 4) ? std::atoi(argv[4]) : 1;
    int verify  = (argc > 5) ? std::atoi(argv[5]) : 1;

    std::printf("bf16 GEMM  M=%d N=%d K=%d  iters=%d verify=%d\n",
                M, N, K, n_iters, verify);

    // ---- host fp32 reference + bf16 buffers ----
    std::vector<float> A_h(M * K), B_h(N * K), C_ref(M * N);
    std::vector<__hip_bfloat16> A_bf(M * K), B_bf(N * K), C_bf(M * N, __hip_bfloat16(0.f));

    // Random operands. On this card bf16 throughput is data dependent at the
    // power limit, so low-entropy fills (zeros, constants) time faster than real
    // data and are not representative.
    std::mt19937 rng(0xC0FFEEu);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto& x : A_h) x = dist(rng);
    for (auto& x : B_h) x = dist(rng);
    for (size_t i = 0; i < A_h.size(); ++i) A_bf[i] = float_to_bf16(A_h[i]);
    for (size_t i = 0; i < B_h.size(); ++i) B_bf[i] = float_to_bf16(B_h[i]);

    // ---- device buffers ----
    __hip_bfloat16 *A_d = nullptr, *B_d = nullptr, *C_d = nullptr;
    HIP_OK(hipMalloc(&A_d, A_bf.size() * sizeof(__hip_bfloat16)));
    HIP_OK(hipMalloc(&B_d, B_bf.size() * sizeof(__hip_bfloat16)));
    HIP_OK(hipMalloc(&C_d, C_bf.size() * sizeof(__hip_bfloat16)));
    HIP_OK(hipMemcpy(A_d, A_bf.data(), A_bf.size() * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(B_d, B_bf.data(), B_bf.size() * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
    HIP_OK(hipMemset(C_d, 0, C_bf.size() * sizeof(__hip_bfloat16)));

    // ---- build kittens globals ----
    using namespace kittens;
    gl_bf A_gl(reinterpret_cast<__hip_bfloat16*>(A_d),
               size_t(1), size_t(1), size_t(M), size_t(K));
    gl_bf B_gl(reinterpret_cast<__hip_bfloat16*>(B_d),
               size_t(1), size_t(1), size_t(N), size_t(K));
    gl_bf C_gl(reinterpret_cast<__hip_bfloat16*>(C_d),
               size_t(1), size_t(1), size_t(M), size_t(N));
    gemm_globals g{A_gl, B_gl, C_gl, /*stream=*/ 0};

    // ---- warmup + timed run ----
    dispatch(g);
    HIP_OK(hipDeviceSynchronize());

    hipEvent_t t0, t1;
    HIP_OK(hipEventCreate(&t0));
    HIP_OK(hipEventCreate(&t1));
    HIP_OK(hipEventRecord(t0));
    for (int i = 0; i < n_iters; ++i) dispatch(g);
    HIP_OK(hipEventRecord(t1));
    HIP_OK(hipEventSynchronize(t1));
    float ms_total = 0.f;
    HIP_OK(hipEventElapsedTime(&ms_total, t0, t1));
    double ms_per = static_cast<double>(ms_total) / n_iters;
    double gflops = 2.0 * M * N * K / 1.0e9;
    std::printf("  %.3f ms/iter  %.1f GFLOP/s\n",
                ms_per, gflops / (ms_per * 1.0e-3));

    if (!verify) { std::puts("  skipped verification"); return 0; }

    // ---- bring C back and compare ----
    HIP_OK(hipMemcpy(C_bf.data(), C_d, C_bf.size() * sizeof(__hip_bfloat16),
                     hipMemcpyDeviceToHost));

    // verify == 2: sampled. A kernel that drops k-tiles is wrong at every output,
    // so a few hundred exact full-K dot products catch it without the hours a
    // full CPU reference takes at large shapes.
    if (verify == 2) {
        const int NS = getenv("NSAMPLE") ? atoi(getenv("NSAMPLE")) : 512;
        std::mt19937 rng(12345);
        double max_abs = 0.0, mean_abs = 0.0;
        int n_bad = 0;
        for (int t = 0; t < NS; ++t) {
            const int i = rng() % M, j = rng() % N;
            // float, like cpu_gemm_abt_ref: the tolerance below is calibrated to
            // fp32 accumulation drift over K terms.
            float acc = 0.f;
            for (int k = 0; k < K; ++k)
                acc += A_h[(size_t)i * K + k] * B_h[(size_t)j * K + k];
            const float got = bf16_to_float(C_bf[(size_t)i * N + j]);
            const double e = std::fabs((double)got - (double)acc);
            max_abs = std::max(max_abs, e);
            mean_abs += e;
            if (e > 0.5 + 0.01 * std::fabs(acc)) {
                if (n_bad < 6)
                    std::printf("    bad: row=%d col=%d got=%.4f ref=%.4f\n", i, j, got, acc);
                ++n_bad;
            }
        }
        std::printf("  SAMPLED max_abs_err=%.4f  mean_abs_err=%.4f  bad=%d/%d\n",
                    max_abs, mean_abs / NS, n_bad, NS);
        return n_bad ? 1 : 0;
    }

    cpu_gemm_abt_ref(A_h, B_h, C_ref, M, N, K);
    double max_abs = 0.0, mean_abs = 0.0;
    int n_bad = 0;
    for (int i = 0; i < M * N; ++i) {
        float got = bf16_to_float(C_bf[i]);
        float ref = C_ref[i];
        double e   = std::fabs(static_cast<double>(got) - static_cast<double>(ref));
        max_abs = std::max(max_abs, e);
        mean_abs += e;
        if (e > 0.5 + 0.01 * std::fabs(ref)) {
            if (n_bad < 6)
                std::printf("    bad: row=%d col=%d got=%.4f ref=%.4f\n",
                            i / N, i % N, got, ref);
            ++n_bad;
        }
    }
    mean_abs /= (M * N);
    std::printf("  max_abs_err=%.4f  mean_abs_err=%.4f  bad=%d/%d\n",
                max_abs, mean_abs, n_bad, M * N);
    return (max_abs < 1.0 || n_bad < 10) ? 0 : 1;
}

#endif // HARNESS_MAIN
