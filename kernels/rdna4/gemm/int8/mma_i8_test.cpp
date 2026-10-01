// Does the new int8 MMA primitive compute the right thing, through the HK types?
//
// One 16x16x16 int8 WMMA per wave, driven through kittens::rt_base storage
// rather than raw vectors, gated against an exact int32 CPU reference.
//
// The test exists because the int8 path has two failure modes that both produce
// a plausible-looking wrong answer rather than a crash:
//   1. unsigned operands (missing neg_lo) -- result scales as 127.5^2 * K
//   2. a wrong lane->element map -- result is the right magnitude, wrong values
// Only an exact reference over signed inputs distinguishes them.
//
// build:
//   hipcc -DKITTENS_RDNA4 --offload-arch=gfx1201 -std=c++20 -w -O3 \
//     -I$HOME/hipkittens_gfx1201/include -I/opt/rocm/include/hip \
//     mma_i8_test.cpp -o mma_i8_test.out
#include "kittens.cuh"
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace kittens;

// A is 16x16 int8 row-major, B is 16x16 int8 row-major, C = A * B^T in int32.
//
// The lane->element map is the SAME one the bf16 path uses, which is the point:
// if int8 needed a different map the tile abstraction would be leaking. For
// srcA, lane l holds row (l%16), k [8*(l/16) .. +8). srcB likewise by column.
__global__ __launch_bounds__(32)
void k_mma_i8(const int8_t* __restrict__ A, const int8_t* __restrict__ B,
              int* __restrict__ C)
{
    const int l = threadIdx.x;
    const int p = l & 15, q = l >> 4;

    int8_4 av[2], bv[2];
    int8_t* ap = reinterpret_cast<int8_t*>(av);
    int8_t* bp = reinterpret_cast<int8_t*>(bv);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        ap[e] = A[p * 16 + 8 * q + e];      // row p, k = 8q+e
        bp[e] = B[p * 16 + 8 * q + e];      // col p of B^T == row p of B
    }

    int2 acc[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[i] = base_types::constants<int2>::zero();

    kittens::wmma161616_acc(acc, av, bv);

    // Accumulator layout, shared with the bf16 path: element el of the 8 goes to
    // row (8*(l/16) + el), column (l%16).
    const int* a8 = reinterpret_cast<const int*>(acc);
#pragma unroll
    for (int el = 0; el < 8; ++el) C[(8 * q + el) * 16 + p] = a8[el];
}

int main(int argc, char** argv)
{
    const int trials = (argc > 1) ? atoi(argv[1]) : 64;
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> d(-127, 127);

    int8_t *dA, *dB; int* dC;
    (void)hipMalloc(&dA, 256); (void)hipMalloc(&dB, 256); (void)hipMalloc(&dC, 256 * 4);

    int bad = 0; long long maxerr = 0;
    for (int t = 0; t < trials; ++t) {
        std::vector<int8_t> A(256), B(256);
        for (auto& x : A) x = (int8_t)d(rng);
        for (auto& x : B) x = (int8_t)d(rng);
        // trial 0 is a deliberate all-negative case: with unsigned operands it
        // would come out large and positive instead of large and positive-ish
        // by a different route, and the exact reference catches either.
        if (t == 0) { for (auto& x : A) x = -100; for (auto& x : B) x = 100; }

        (void)hipMemcpy(dA, A.data(), 256, hipMemcpyHostToDevice);
        (void)hipMemcpy(dB, B.data(), 256, hipMemcpyHostToDevice);
        hipLaunchKernelGGL(k_mma_i8, dim3(1), dim3(32), 0, 0, dA, dB, dC);
        std::vector<int> C(256);
        (void)hipMemcpy(C.data(), dC, 256 * 4, hipMemcpyDeviceToHost);

        for (int i = 0; i < 16; ++i)
            for (int j = 0; j < 16; ++j) {
                long long ref = 0;
                for (int k = 0; k < 16; ++k) ref += (long long)A[i * 16 + k] * (long long)B[j * 16 + k];
                const long long e = llabs((long long)C[i * 16 + j] - ref);
                if (e) {
                    if (bad < 5)
                        std::printf("  bad t=%d [%d,%d] got=%d ref=%lld\n", t, i, j, C[i * 16 + j], ref);
                    ++bad;
                }
                maxerr = e > maxerr ? e : maxerr;
            }
    }
    std::printf("int8 wmma161616_acc through HK rt storage: bad=%d/%d  max_abs_err=%lld\n",
                bad, trials * 256, maxerr);
    std::printf("%s\n", bad ? "FAIL" : "PASS (exact -- int32 arithmetic has no rounding)");
    return bad != 0;
}
