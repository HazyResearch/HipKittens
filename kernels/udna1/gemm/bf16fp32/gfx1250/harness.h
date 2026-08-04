/**
 * @file harness.h
 * @brief Standalone host harness for the gfx1250 GEMM ladder.
 *
 * Every rung defines `dispatch(gemm_globals)` and includes this at the end of the file, which
 * gives it a `main()`. Each rung is therefore its own executable, taking `M N K iters verify`.
 *
 * The harness fills A and B with uniform random values, times `iters` dispatches with HIP
 * events, and at `verify=1` checks the result against a CPU fp32 reference in the same
 * invocation that did the timing, so a throughput number is never reported for a run that was
 * not also checked.
 *
 * Compile with `-DHARNESS_MAIN` to enable the `main` body; a rung does not pull it in otherwise.
 */

#pragma once

#ifdef HARNESS_MAIN

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

static inline void hip_check(hipError_t e, const char* what, int line) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "HIP error at line %d (%s): %s\n",
                     line, what, hipGetErrorString(e));
        std::exit(1);
    }
}
#define HIP_OK(call) hip_check((call), #call, __LINE__)

/* `dev_elem` follows `GFX1250_ELEM`: bf16 by default, `half` for the fp16 build. Both host
 * conversions round to nearest even, so the input quantization is unbiased either way and only
 * the device-side epilogue rounding differs between the two formats. */
using dev_elem = gfx1250_gemm::elem_t;

static inline float elem_to_float(__hip_bfloat16 v) { return __bfloat162float(v); }
static inline float elem_to_float(__half v)         { return __half2float(v); }
static inline __hip_bfloat16 float_to_elem(float v, __hip_bfloat16*) { return __float2bfloat16(v); }
static inline __half        float_to_elem(float v, __half*)          { return __float2half(v); }

/* Every rung writes C column-major, `C[c*M + r]`, so the reference is read through the index map
 * below. A layout mistake false-passes at M == N, where a transposed output and a consistently
 * transposed reference agree element for element, so check layout changes at a non-square shape:
 * 4096x8192x2048. */
#ifdef GEMM_C_COL_MAJOR
#error "C is always column-major on this ladder; GEMM_C_COL_MAJOR is not a knob."
#endif

// Map a row-major reference index to the device buffer index.
static inline size_t hk_c_index(int i, int M, int N) {
    const int r = i / N, c = i % N;
    return (size_t)c * (size_t)M + (size_t)r;
}

// Per-format numerics for the tolerance below.
//   eps_in  = half-ULP relative error of the round-to-nearest input quantization
//   eps_out = worst-case relative error of the fp32 -> elem epilogue conversion
template<typename T> struct elem_numerics;
template<> struct elem_numerics<__hip_bfloat16> {
    static constexpr double eps_in  = 1.0 / 512.0;      // 2^-9  (7 stored mantissa bits)
    static constexpr double eps_out = 1.0 / 512.0;      // 2^-9  round-to-nearest-even
    static constexpr const char* name = "bf16";
};
template<> struct elem_numerics<__half> {
    static constexpr double eps_in  = 1.0 / 4096.0;     // 2^-12 (10 stored mantissa bits)
    static constexpr double eps_out = 1.0 / 4096.0;     // 2^-12 __float2half is round-to-nearest
    static constexpr const char* name = "fp16";
};

/* CPU fp32 reference for C = A . Bt. Only the outer i loop is parallel: each C[i,j] accumulates
 * over k in the same order whatever the thread count, so the reference is bitwise reproducible. */
static void cpu_gemm_abt_ref(const std::vector<float>& A,
                             const std::vector<float>& B,
                             std::vector<float>& C,
                             int M, int N, int K)
{
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
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

// Bytes of host RAM available (MemAvailable from /proc/meminfo), 0 if unknown.
static size_t host_mem_available_bytes() {
    FILE* f = std::fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char key[64]; unsigned long kb = 0; size_t avail = 0;
    while (std::fscanf(f, "%63s %lu kB\n", key, &kb) == 2) {
        if (std::strcmp(key, "MemAvailable:") == 0) { avail = size_t(kb) * 1024; break; }
    }
    std::fclose(f);
    return avail;
}

int main(int argc, char** argv)
{
    int M = (argc > 1) ? std::atoi(argv[1]) : 256;
    int N = (argc > 2) ? std::atoi(argv[2]) : 256;
    int K = (argc > 3) ? std::atoi(argv[3]) : 256;
    int n_iters = (argc > 4) ? std::atoi(argv[4]) : 1;
    int verify  = (argc > 5) ? std::atoi(argv[5]) : 1;

    std::printf("  C layout: COLUMN-major (TN contract)\n");
    std::printf("gemm (%s->fp32->%s)  M=%d N=%d K=%d  iters=%d verify=%d\n",
                elem_numerics<dev_elem>::name, elem_numerics<dev_elem>::name,
                M, N, K, n_iters, verify);

    // Fail cleanly rather than invoking the OOM killer. Verifying needs three fp32 host arrays
    // (A_h, B_h, C_ref) and three narrow ones (A_bf, B_bf, C_bf).
    if (verify) {
        const size_t elems = size_t(M) * K + size_t(N) * K + size_t(M) * N;
        const size_t need  = elems * 4 + elems * 2;
        const size_t avail = host_mem_available_bytes();
        if (avail && avail < need + need / 2) {
            std::fprintf(stderr, "harness: insufficient host RAM to verify at %dx%dx%d "
                         "(need ~%.2f GB + 50%% margin, MemAvailable %.2f GB). Rerun with "
                         "verify=0 or a smaller size.\n", M, N, K, need / 1e9, avail / 1e9);
            std::exit(2);
        }
    }

    // ---- host fp32 reference inputs, plus their narrowed copies for the device ----
    std::vector<float> A_h(M * K), B_h(N * K), C_ref(M * N);
    std::vector<dev_elem> A_bf(M * K), B_bf(N * K),
                          C_bf(M * N, float_to_elem(0.f, (dev_elem*)nullptr));

    /* Inputs are U(-1,1). The output range follows from that: |C| grows as sqrt(K)/3, which is
     * about 181 at K=8192 -- comfortably inside both storage formats. */
    const float in_scale = 1.0f;
    std::mt19937 rng(0xC0FFEEu);
    std::uniform_real_distribution<float> dist(-in_scale, in_scale);
    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k)
            A_h[i * K + k] = dist(rng);
    for (int j = 0; j < N; ++j)
        for (int k = 0; k < K; ++k)
            B_h[j * K + k] = dist(rng);
    for (size_t i = 0; i < A_h.size(); ++i) A_bf[i] = float_to_elem(A_h[i], (dev_elem*)nullptr);
    for (size_t i = 0; i < B_h.size(); ++i) B_bf[i] = float_to_elem(B_h[i], (dev_elem*)nullptr);

    // ---- device buffers ----
    dev_elem *A_d = nullptr, *B_d = nullptr, *C_d = nullptr;
    HIP_OK(hipMalloc(&A_d, A_bf.size() * sizeof(dev_elem)));
    HIP_OK(hipMalloc(&B_d, B_bf.size() * sizeof(dev_elem)));
    HIP_OK(hipMalloc(&C_d, C_bf.size() * sizeof(dev_elem)));
    HIP_OK(hipMemcpy(A_d, A_bf.data(), A_bf.size() * sizeof(dev_elem), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(B_d, B_bf.data(), B_bf.size() * sizeof(dev_elem), hipMemcpyHostToDevice));
    HIP_OK(hipMemset(C_d, 0, C_bf.size() * sizeof(dev_elem)));

    // ---- build kittens globals ----
    using namespace kittens;
    gl_e A_gl(A_d, size_t(1), size_t(1), size_t(M), size_t(K));
    gl_e B_gl(B_d, size_t(1), size_t(1), size_t(N), size_t(K));
    gl_c C_gl(C_d, size_t(1), size_t(1), size_t(M), size_t(N));
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

    // ---- bring C back, compute the reference, compare ----
    HIP_OK(hipMemcpy(C_bf.data(), C_d, C_bf.size() * sizeof(dev_elem),
                     hipMemcpyDeviceToHost));
    cpu_gemm_abt_ref(A_h, B_h, C_ref, M, N, K);

    /* Tolerance is derived, not fitted. Input quantization accumulates as a random walk over the
     * reduction, kappa * sqrt(K) * sigma_ab * eps_in * sqrt(2), where sigma_ab = 1/3 is the
     * standard deviation of a*b for a,b ~ U(-1,1) and kappa = 6 covers the tail of M*N samples;
     * output conversion adds |ref| * eps_out. `tol_abs_fixed` is the gate that sets the exit
     * status and scales with sqrt(K) for the same reason, anchored to 0.5 at K=8192. */
    constexpr double kappa    = 6.0;
    constexpr double sigma_ab = 1.0 / 3.0;
    const double tol_floor = kappa * std::sqrt((double)K) * sigma_ab
                           * elem_numerics<dev_elem>::eps_in * std::sqrt(2.0);
    const double tol_rel   = elem_numerics<dev_elem>::eps_out;
    const double tol_abs_fixed = 0.5 * std::sqrt((double)K / 8192.0);

    double max_abs = 0.0, mean_abs = 0.0;
    // mean(|got|-|ref|) is a rounding-mode probe: round-to-nearest-even is unbiased and reads
    // near zero, while truncation toward zero always drops magnitude and reads negative.
    double mag_bias = 0.0;
    int n_bad = 0, n_bad_scaled = 0;
    // A value the storage format cannot hold shows up as a non-finite `got`. fp16 overflows at
    // 65504; bf16 has fp32's exponent range.
    int n_nonfinite = 0;
    double max_ref_abs = 0.0;
    for (int i = 0; i < M * N; ++i) {
        const float got = elem_to_float(C_bf[hk_c_index(i, M, N)]);
        const float ref = C_ref[i];
        if (!std::isfinite(got)) ++n_nonfinite;
        max_ref_abs = std::max(max_ref_abs, (double)std::fabs(ref));
        double e   = std::fabs(static_cast<double>(got) - static_cast<double>(ref));
        max_abs = std::max(max_abs, e);
        mean_abs += e;
        mag_bias += std::fabs(static_cast<double>(got)) - std::fabs(static_cast<double>(ref));
        if (e > tol_abs_fixed + 0.01 * std::fabs(ref)) ++n_bad;
        if (e > tol_floor + tol_rel * std::fabs(ref)) ++n_bad_scaled;
    }
    mean_abs /= (M * N);
    mag_bias /= (M * N);
    std::printf("  max_abs_err=%.4f  mean_abs_err=%.4f  bad=%d/%d  (fixed gate %.4f+0.01*|ref|)\n",
                max_abs, mean_abs, n_bad, M * N, tol_abs_fixed);
    std::printf("  mag_bias=%.6f (mean |got|-|ref|; ~0 = unbiased rounding)\n", mag_bias);
    std::printf("  scaled_tol=%.4f+%.5f*|ref| (%s, K=%d)  bad_scaled=%d/%d\n",
                tol_floor, tol_rel, elem_numerics<dev_elem>::name, K,
                n_bad_scaled, M * N);
    std::printf("  in_scale=%.3g  max|ref|=%.1f  nonfinite_out=%d/%d\n",
                (double)in_scale, max_ref_abs, n_nonfinite, M * N);

    // Strict: the exit status is bad == 0, so no element may exceed the fixed per-element
    // tolerance. Anything gating on this harness gates on that.
    return (n_bad == 0) ? 0 : 1;
}

#endif // HARNESS_MAIN
