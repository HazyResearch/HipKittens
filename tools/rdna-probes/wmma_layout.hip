// Empirically determine the register layouts of RDNA3 WMMA.
//
// Method (gauge-aware): the absolute labelling of A's rows, B's cols and the
// reduction index k is a gauge freedom -- what a GEMM actually needs is a
// mutually consistent triple (A lane/elem -> row, B lane/elem -> col,
// D lane/reg -> (row,col)).  We recover it with one-hot probes:
//
//   * A-probe: A one-hot at (lane L, elem I), B = all ones, C = 0.
//     D[r][c] = sum_k A[r][k] * 1 = 1 for the single row r that (L,I) feeds,
//     for every c.  So the set of (lane,reg) slots that come back non-zero IS
//     row r of D.  Probes producing the same slot-set share a row.
//   * B-probe: mirror, giving columns of D.
//   * k-probe: A one-hot and B one-hot -> non-zero iff both land on the same
//     k.  Confirms A's and B's elem indices use the same k ordering, which is
//     all a GEMM needs (any common permutation of k cancels out).
//
// Two things that will silently give garbage if ignored:
//   * On gfx11 wave32 the A and B operands must be *duplicated* across lanes
//     0-15 and 16-31.  A one-hot in only one half is undefined, so every probe
//     sets both lane L and lane L^16.
//   * Operands are fed from global memory rather than built with a runtime
//     vector index, so nothing here depends on how the compiler lowers a
//     dynamic insertelement.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <map>
#include <cstring>

#define HIP_CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
    exit(1);} } while (0)

typedef _Float16 half16 __attribute__((ext_vector_type(16)));
typedef _Float16 half8  __attribute__((ext_vector_type(8)));
typedef float    float8 __attribute__((ext_vector_type(8)));
typedef float    float4v __attribute__((ext_vector_type(4)));

#if defined(PROBE_GFX11_W32)
#define VARIANT "gfx11 wave32"
#define WAVE 32
#define AELEMS 16
#define DREGS 8
#define DUP_HALVES 1   // lanes 0-15 and 16-31 must carry identical A/B data
typedef half16 avec_t;
typedef float8 dvec_t;
__device__ static inline dvec_t wmma(avec_t a, avec_t b, dvec_t c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}
#elif defined(PROBE_GFX11_W64)
#define VARIANT "gfx11 wave64"
#define WAVE 64
#define AELEMS 16
#define DREGS 4
#define DUP_HALVES 1
typedef half16  avec_t;
typedef float4v dvec_t;
__device__ static inline dvec_t wmma(avec_t a, avec_t b, dvec_t c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w64(a, b, c);
}
#elif defined(PROBE_GFX12_W32)
#define VARIANT "gfx12 wave32"
#define WAVE 32
#define AELEMS 8
#define DREGS 8
#define DUP_HALVES 0   // gfx12 splits k across the two lane halves instead
typedef half8  avec_t;
typedef float8 dvec_t;
__device__ static inline dvec_t wmma(avec_t a, avec_t b, dvec_t c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
}
#elif defined(PROBE_GFX12_W64)
#define VARIANT "gfx12 wave64"
#define WAVE 64
#define AELEMS 8
#define DREGS 4
#define DUP_HALVES 0
typedef half8   avec_t;
typedef float4v dvec_t;
__device__ static inline dvec_t wmma(avec_t a, avec_t b, dvec_t c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w64_gfx12(a, b, c);
}
#else
#error "pick a PROBE_* variant"
#endif

// Number of distinct one-hot probe positions.  With duplicated halves only
// lanes 0-15 are independent; the probe writes lane L and L^16 together.
static constexpr int PLANES = DUP_HALVES ? 16 : WAVE;
static constexpr int NPROBE = PLANES * AELEMS;

// operands live in global memory: [NPROBE][WAVE][AELEMS]
__global__ __launch_bounds__(WAVE) void probe_kernel(const _Float16 *__restrict__ A,
                                                     const _Float16 *__restrict__ B,
                                                     float *__restrict__ out) {
    int lane = threadIdx.x;
    for (int p = 0; p < NPROBE; ++p) {
        avec_t a, b;
        const _Float16 *ap = A + ((size_t)p * WAVE + lane) * AELEMS;
        const _Float16 *bp = B + ((size_t)p * WAVE + lane) * AELEMS;
#pragma unroll
        for (int i = 0; i < AELEMS; ++i) { a[i] = ap[i]; b[i] = bp[i]; }
        dvec_t c;
#pragma unroll
        for (int r = 0; r < DREGS; ++r) c[r] = 0.0f;
        dvec_t d = wmma(a, b, c);
#pragma unroll
        for (int r = 0; r < DREGS; ++r)
            out[(size_t)p * WAVE * DREGS + lane * DREGS + r] = d[r];
    }
}

struct Mask {
    unsigned long long w[8] = {0, 0, 0, 0, 0, 0, 0, 0};  // up to 512 slots
    bool operator<(const Mask &o) const { return memcmp(w, o.w, sizeof(w)) < 0; }
    bool empty() const {
        for (int i = 0; i < 8; ++i) if (w[i]) return false;
        return true;
    }
    int popcount() const {
        int n = 0;
        for (int i = 0; i < 8; ++i) n += __builtin_popcountll(w[i]);
        return n;
    }
    void set(int i) { w[i >> 6] |= 1ull << (i & 63); }
    bool test(int i) const { return (w[i >> 6] >> (i & 63)) & 1; }
};

int main() {
    const size_t nop  = (size_t)NPROBE * WAVE * AELEMS;
    const size_t nout = (size_t)NPROBE * WAVE * DREGS;

    std::vector<_Float16> hA(nop, (_Float16)0.0f), hB(nop, (_Float16)0.0f);
    // mode 0 = A one-hot / B ones; mode 1 = A ones / B one-hot
    auto fill = [&](std::vector<_Float16> &onehot, std::vector<_Float16> &ones) {
        for (int p = 0; p < NPROBE; ++p) {
            int pl = p / AELEMS, pe = p % AELEMS;
            for (int lane = 0; lane < WAVE; ++lane) {
                bool hit = DUP_HALVES ? ((lane & 15) == pl) : (lane == pl);
                for (int i = 0; i < AELEMS; ++i) {
                    onehot[((size_t)p * WAVE + lane) * AELEMS + i] =
                        (_Float16)((hit && i == pe) ? 1.0f : 0.0f);
                    ones[((size_t)p * WAVE + lane) * AELEMS + i] = (_Float16)1.0f;
                }
            }
        }
    };

    _Float16 *dA, *dB;
    float *d_out;
    HIP_CHECK(hipMalloc(&dA, nop * sizeof(_Float16)));
    HIP_CHECK(hipMalloc(&dB, nop * sizeof(_Float16)));
    HIP_CHECK(hipMalloc(&d_out, nout * sizeof(float)));
    std::vector<float> h(nout);

    std::map<Mask, int> classid[2];
    std::vector<Mask> classmask[2];
    std::vector<int> probeclass[2];

    for (int mode = 0; mode < 2; ++mode) {
        if (mode == 0) fill(hA, hB); else fill(hB, hA);
        HIP_CHECK(hipMemcpy(dA, hA.data(), nop * sizeof(_Float16), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dB, hB.data(), nop * sizeof(_Float16), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(d_out, 0, nout * sizeof(float)));
        hipLaunchKernelGGL(probe_kernel, dim3(1), dim3(WAVE), 0, 0, dA, dB, d_out);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipMemcpy(h.data(), d_out, nout * sizeof(float), hipMemcpyDeviceToHost));

        probeclass[mode].assign(NPROBE, -1);
        for (int p = 0; p < NPROBE; ++p) {
            Mask m;
            for (int lane = 0; lane < WAVE; ++lane)
                for (int r = 0; r < DREGS; ++r)
                    if (h[(size_t)p * WAVE * DREGS + lane * DREGS + r] != 0.0f)
                        m.set(lane * DREGS + r);
            if (m.empty()) continue;
            auto it = classid[mode].find(m);
            int id;
            if (it == classid[mode].end()) {
                id = (int)classmask[mode].size();
                classid[mode][m] = id;
                classmask[mode].push_back(m);
            } else id = it->second;
            probeclass[mode][p] = id;
        }
    }

    printf("=== %s | wave=%d | A/B elems per lane=%d | D regs per lane=%d ===\n",
           VARIANT, WAVE, AELEMS, DREGS);
    printf("distinct A-classes (should be 16 = rows of D): %zu\n", classmask[0].size());
    printf("distinct B-classes (should be 16 = cols of D): %zu\n", classmask[1].size());
    printf("slots lit per A-class: %s\n",
           classmask[0].empty() ? "n/a" : std::to_string(classmask[0][0].popcount()).c_str());
    printf("slots lit per B-class: %s\n",
           classmask[1].empty() ? "n/a" : std::to_string(classmask[1][0].popcount()).c_str());

    printf("\n--- D / C layout: lane -> (reg: row,col) ---\n");
    for (int lane = 0; lane < WAVE; ++lane) {
        printf("lane %2d:", lane);
        for (int r = 0; r < DREGS; ++r) {
            int row = -1, col = -1;
            for (size_t i = 0; i < classmask[0].size(); ++i)
                if (classmask[0][i].test(lane * DREGS + r)) row = (int)i;
            for (size_t i = 0; i < classmask[1].size(); ++i)
                if (classmask[1][i].test(lane * DREGS + r)) col = (int)i;
            printf("  r%d=(%2d,%2d)", r, row, col);
        }
        printf("\n");
    }

    printf("\n--- A layout: probe lane -> row fed, per elem slot (elem == k) ---\n");
    for (int pl = 0; pl < PLANES; ++pl) {
        printf("lane %2d%s:", pl, DUP_HALVES ? "(+16)" : "     ");
        for (int e = 0; e < AELEMS; ++e) printf(" %3d", probeclass[0][pl * AELEMS + e]);
        printf("\n");
    }
    printf("\n--- B layout: probe lane -> col fed, per elem slot (elem == k) ---\n");
    for (int pl = 0; pl < PLANES; ++pl) {
        printf("lane %2d%s:", pl, DUP_HALVES ? "(+16)" : "     ");
        for (int e = 0; e < AELEMS; ++e) printf(" %3d", probeclass[1][pl * AELEMS + e]);
        printf("\n");
    }
    return 0;
}
