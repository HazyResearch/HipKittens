// int8 GEMM on gfx1201 through the HipKittens int8 MMA primitive.
//
// C[i][j] = sum_k A[i][k] * B[j][k]   -- A row-major MxK, B row-major NxK, int8 in,
// int32 out.
//
// The kernel and its tuning parameters live in gemm_i8_kernel.h (`gemm_i8_config`). This
// file picks two configurations, dispatches between them per shape, and is the
// benchmark/verification harness:
//
//   ./gemm_i8 M N K [iters] [verify]

#include "gemm_i8_kernel.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

// The two shipped configurations. Everything not named takes gemm_i8_config's default.
//                                    BM   BN   BK  MULTIBUF FRAG_PIPELINE
using cfg_main = gemm_i8_config<256, 256, 32, true,    0>;   // 4-slot LDS ring
using cfg_alt  = gemm_i8_config<192, 256, 64, false,   1>;   // double buffer, split barrier

// Neither configuration has a partial-tile path, so a shape must divide its tile.
template <typename C>
static bool tiles(int M, int N, int K) { return !(M % C::BM) && !(N % C::BN) && !(K % C::BK); }

template <typename C>
static void launch(const int8_t* A, const int8_t* B, int* Cout, int M, int N, int K, hipStream_t s) {
    static const bool attr_set = [] {   // once per configuration, not per launch
        return hipFuncSetAttribute(reinterpret_cast<const void*>(gemm_i8<C>),
                                   hipFuncAttributeMaxDynamicSharedMemorySize, C::LDS_TOTAL) == hipSuccess;
    }();
    (void)attr_set;
    gemm_i8<C><<<dim3(N / C::BN, M / C::BM), dim3(C::NTHREAD + C::PROD_THREADS),
                 C::LDS_TOTAL, s>>>(A, B, Cout, M, N, K);
}

/**
 * Launch the fastest configuration that tiles M x N x K: 192x256 where M has a factor of
 * 3 (~3% faster where it applies), otherwise 256x256. Returns the chosen tile's name, or
 * nullptr if neither tiles the shape.
 */
static const char* dispatch(const int8_t* A, const int8_t* B, int* Cout,
                            int M, int N, int K, hipStream_t s = 0) {
    if (tiles<cfg_alt>(M, N, K))  { launch<cfg_alt>(A, B, Cout, M, N, K, s);  return "192x256"; }
    if (tiles<cfg_main>(M, N, K)) { launch<cfg_main>(A, B, Cout, M, N, K, s); return "256x256"; }
    return nullptr;
}

int main(int argc, char** argv)
{
    const int M = argc > 1 ? atoi(argv[1]) : 4096;
    const int N = argc > 2 ? atoi(argv[2]) : 4096;
    const int K = argc > 3 ? atoi(argv[3]) : 4096;
    const int iters = argc > 4 ? atoi(argv[4]) : 100;
    const int verify = argc > 5 ? atoi(argv[5]) : 1;

    if (!tiles<cfg_alt>(M, N, K) && !tiles<cfg_main>(M, N, K)) {
        printf("shape %dx%dx%d does not tile: need M%%%d==0, N%%%d==0, K%%%d==0 (or %d/%d/%d).\n"
               "  Neither kernel has a partial-tile path. Pick a shape that divides.\n",
               M, N, K, cfg_main::BM, cfg_main::BN, cfg_main::BK, cfg_alt::BM, cfg_alt::BN, cfg_alt::BK);
        return 1;
    }

    std::vector<int8_t> hA((size_t)M * K), hB((size_t)N * K);
    std::mt19937 rng(1234);
    std::uniform_int_distribution<int> d(-64, 63);
    for (auto& x : hA) x = (int8_t)d(rng);
    for (auto& x : hB) x = (int8_t)d(rng);
    int8_t *dA, *dB; int* dC;
    (void)hipMalloc(&dA, hA.size()); (void)hipMalloc(&dB, hB.size());
    (void)hipMalloc(&dC, (size_t)M * N * 4);
    (void)hipMemcpy(dA, hA.data(), hA.size(), hipMemcpyHostToDevice);
    (void)hipMemcpy(dB, hB.data(), hB.size(), hipMemcpyHostToDevice);

    const char* which = dispatch(dA, dB, dC, M, N, K);
    (void)hipDeviceSynchronize();
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) { printf("launch failed: %s\n", hipGetErrorString(e)); return 1; }

    if (verify) {
        std::vector<int> hC((size_t)M * N);
        (void)hipMemcpy(hC.data(), dC, (size_t)M * N * 4, hipMemcpyDeviceToHost);
        const int NS = getenv("NSAMPLE") ? atoi(getenv("NSAMPLE")) : 2048;
        std::mt19937 pick(99); int bad = 0;
        for (int t = 0; t < NS; ++t) {
            const int i = pick() % M, j = pick() % N;
            long ref = 0;
            for (int k = 0; k < K; ++k) ref += (long)hA[(size_t)i * K + k] * (long)hB[(size_t)j * K + k];
            if ((long)hC[(size_t)i * N + j] != ref) {
                if (bad < 4) printf("  bad [%d,%d] got=%d ref=%ld\n", i, j, hC[(size_t)i * N + j], ref);
                ++bad;
            }
        }
        printf("  gate: bad=%d/%d (exact int32)\n", bad, NS);
        if (bad) return 1;
    }

    hipEvent_t t0, t1; (void)hipEventCreate(&t0); (void)hipEventCreate(&t1);
    for (int i = 0; i < 20; ++i) dispatch(dA, dB, dC, M, N, K);
    (void)hipDeviceSynchronize();
    (void)hipEventRecord(t0);
    for (int i = 0; i < iters; ++i) dispatch(dA, dB, dC, M, N, K);
    (void)hipEventRecord(t1); (void)hipDeviceSynchronize();
    float ms; (void)hipEventElapsedTime(&ms, t0, t1); ms /= iters;
    const double ops = 2.0 * M * N * K;
    printf("HK int8 GEMM  %dx%dx%d  tile=%s  %.2f TOPS  %.4f ms  (%.1f%% of the 334.9 TOPS measured WMMA peak)\n",
           M, N, K, which, ops / (ms * 1e-3) / 1e12, ms, ops / (ms * 1e-3) / 1e12 / 334.9 * 100);
    return 0;
}
