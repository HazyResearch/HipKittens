/**
 * MXFP8 GEMM — 4-wave TN kernel (occ 1) with block scaling
 *
 *   - 4 warps (256 threads), launch_bounds(256, 1) → occupancy 1
 *   - 256×256 output tile, WARPS_M=2, WARPS_N=2, each warp 128×128
 *   - Pre-packed MFMA scales via repack_scales_for_mfma_kernel
 *   - Direct global→VGPR scale loads, no LDS staging for scales
 *   - 64 scaled MFMAs per K-iteration (4 quadrants × 16)
 */

#include "kittens.cuh"
#include <random>
#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

using namespace kittens;

#define SIZE 8192

#define HipCheckError() do { \
    hipError_t err = hipGetLastError(); \
    if (err != hipSuccess) { fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); exit(1); } \
} while(0)

struct TimingResult {
    float best_time_ms, avg_time_ms;
    double best_tflops, avg_tflops;
    int timing_iterations;
};


// Scaled MMA: 16 base-tile MFMAs covering 64×64 output
template<typename RT_C, typename RT_A, typename RT_B>
__device__ __forceinline__ void mma_ABt_scaled_16(
    RT_C &c, const RT_A &a, const RT_B &b,
    const fp8e8m0_4 *sa, const fp8e8m0_4 *sb) {
    __builtin_amdgcn_sched_barrier(0);
    mma_ABt_base_scaled<0, 0>(c.tiles[0][0], a.tiles[0][0], b.tiles[0][0], c.tiles[0][0], sa, sb);
    mma_ABt_base_scaled<0, 1>(c.tiles[0][1], a.tiles[0][0], b.tiles[1][0], c.tiles[0][1], sa, sb);
    mma_ABt_base_scaled<0, 2>(c.tiles[0][2], a.tiles[0][0], b.tiles[2][0], c.tiles[0][2], sa, sb);
    mma_ABt_base_scaled<0, 3>(c.tiles[0][3], a.tiles[0][0], b.tiles[3][0], c.tiles[0][3], sa, sb);
    mma_ABt_base_scaled<1, 0>(c.tiles[1][0], a.tiles[1][0], b.tiles[0][0], c.tiles[1][0], sa, sb);
    mma_ABt_base_scaled<1, 1>(c.tiles[1][1], a.tiles[1][0], b.tiles[1][0], c.tiles[1][1], sa, sb);
    mma_ABt_base_scaled<1, 2>(c.tiles[1][2], a.tiles[1][0], b.tiles[2][0], c.tiles[1][2], sa, sb);
    mma_ABt_base_scaled<1, 3>(c.tiles[1][3], a.tiles[1][0], b.tiles[3][0], c.tiles[1][3], sa, sb);
    mma_ABt_base_scaled<2, 0>(c.tiles[2][0], a.tiles[2][0], b.tiles[0][0], c.tiles[2][0], sa, sb);
    mma_ABt_base_scaled<2, 1>(c.tiles[2][1], a.tiles[2][0], b.tiles[1][0], c.tiles[2][1], sa, sb);
    mma_ABt_base_scaled<2, 2>(c.tiles[2][2], a.tiles[2][0], b.tiles[2][0], c.tiles[2][2], sa, sb);
    mma_ABt_base_scaled<2, 3>(c.tiles[2][3], a.tiles[2][0], b.tiles[3][0], c.tiles[2][3], sa, sb);
    mma_ABt_base_scaled<3, 0>(c.tiles[3][0], a.tiles[3][0], b.tiles[0][0], c.tiles[3][0], sa, sb);
    mma_ABt_base_scaled<3, 1>(c.tiles[3][1], a.tiles[3][0], b.tiles[1][0], c.tiles[3][1], sa, sb);
    mma_ABt_base_scaled<3, 2>(c.tiles[3][2], a.tiles[3][0], b.tiles[2][0], c.tiles[3][2], sa, sb);
    mma_ABt_base_scaled<3, 3>(c.tiles[3][3], a.tiles[3][0], b.tiles[3][0], c.tiles[3][3], sa, sb);
    __builtin_amdgcn_sched_barrier(0);
}
// Kernel constants
constexpr int NUM_WARPS  = 4;
constexpr int WARPS_ROW  = 2;
constexpr int WARPS_COL  = 2;
constexpr int BLOCK_ROW  = 256;
constexpr int BLOCK_COL  = 256;
constexpr int BLOCK_K    = 128;
constexpr int HALF_ROW   = BLOCK_ROW / 2;
constexpr int HALF_COL   = BLOCK_COL / 2;
constexpr int REG_M      = HALF_ROW / WARPS_ROW;
constexpr int REG_N      = HALF_COL / WARPS_COL;

using G = kittens::group<NUM_WARPS>;


// Kernel
template <int M, int N, int K>
__global__ __launch_bounds__(256, 1)
void mxfp8_gemm_4wave_kernel(
    const kittens::gl<fp8e4m3, 1, 1, M, K> A,
    const kittens::gl<fp8e4m3, 1, 1, N, K> B,
    const kittens::gl<float,   1, 1, M, N> C,
    const uint32_t *__restrict__ scale_A_iter,
    const uint32_t *__restrict__ scale_B_iter) {
    constexpr int k_iters = K / BLOCK_K;

    using ST_A = st_fp8e4m3<HALF_ROW, BLOCK_K, st_16x128_s>;
    using ST_B = st_fp8e4m3<HALF_COL, BLOCK_K, st_16x128_s>;

    using RT_A = rt_fp8e4m3<REG_M, BLOCK_K>;
    using RT_B = rt_fp8e4m3<REG_N, BLOCK_K>;
    using RT_C = rt_fl<REG_M, REG_N, col_l, rt_16x16_s>;

    __shared__ ST_A As[2][2];
    __shared__ ST_B Bs[2][2];

    RT_C cA, cB, cC, cD;
    zero(cA); zero(cB); zero(cC); zero(cD);

    constexpr int tiles_M = M / BLOCK_ROW;
    constexpr int tiles_N = N / BLOCK_COL;
    const int NUM_XCDS    = 8;
    const int WGM         = 8;
    int wgid             = chiplet_transform_chunked(blockIdx.x, gridDim.x, NUM_XCDS, WGM * WGM);
    int num_wgid_in_group = WGM * tiles_N;
    int group_id         = wgid / num_wgid_in_group;
    int first_pid_m      = group_id * WGM;
    int group_size_m     = min(tiles_M - first_pid_m, WGM);
    int block_row        = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    int block_col        = (wgid % num_wgid_in_group) / group_size_m;
    int block_m          = block_row * BLOCK_ROW;
    int block_n          = block_col * BLOCK_COL;

    int warp_m = warpid() / WARPS_COL;
    int warp_n = warpid() % WARPS_COL;

    int a_row_h0  = warp_m * REG_M;
    int a_row_h1  = HALF_ROW + warp_m * REG_M;
    int b_row_h0  = warp_n * REG_N;
    int b_row_h1  = HALF_COL + warp_n * REG_N;

    int curr = 0, next = 1;

    // Prologue: load first two K-tiles
    __builtin_amdgcn_sched_barrier(0);

    RT_A a[2];
    RT_B b[2];

    G::load(As[curr][0], A, {0, 0, block_row * 2,     0});
    G::load(Bs[curr][0], B, {0, 0, block_col * 2,     0});
    G::load(Bs[curr][1], B, {0, 0, block_col * 2 + 1, 0});
    G::load(As[curr][1], A, {0, 0, block_row * 2 + 1, 0});

    G::load(As[next][0], A, {0, 0, block_row * 2,     1});
    G::load(Bs[next][0], B, {0, 0, block_col * 2,     1});
    G::load(Bs[next][1], B, {0, 0, block_col * 2 + 1, 1});
    G::load(As[next][1], A, {0, 0, block_row * 2 + 1, 1});

    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(7)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    auto a_subtile_0 = kittens::subtile_inplace<REG_M, BLOCK_K>(As[curr][0], {warp_m, 0});
    load(a[0], a_subtile_0);

    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(6)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    auto b_subtile_0 = kittens::subtile_inplace<REG_N, BLOCK_K>(Bs[curr][0], {warp_n, 0});
    load(b[0], b_subtile_0);

    // Main loop
    // Merged quadrants: cA+cB in one 32-MFMA phase (same a[0], both b halves)
    // cC+cD in another 32-MFMA phase (same a[1], both b halves)
    // Safe at occ 1 — no co-resident WG starvation risk
    int lid = laneid();

    #pragma unroll 1
    for (int k = 0; k < k_iters - 2; k++, curr ^= 1, next ^= 1) {
        asm volatile("s_waitcnt vmcnt(4)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        fp8e8m0_4 sa_h0 = (fp8e8m0_4)scale_A_iter[k * M + block_m + a_row_h0 + lid];
        fp8e8m0_4 sb_h0 = (fp8e8m0_4)scale_B_iter[k * N + block_n + b_row_h0 + lid];
        fp8e8m0_4 sa_h1 = (fp8e8m0_4)scale_A_iter[k * M + block_m + a_row_h1 + lid];
        fp8e8m0_4 sb_h1 = (fp8e8m0_4)scale_B_iter[k * N + block_n + b_row_h1 + lid];

        auto bs_subtile_1 = kittens::subtile_inplace<REG_N, BLOCK_K>(Bs[curr][1], {warp_n, 0});
        load(b[1], bs_subtile_1);
        G::load(As[curr][0], A, {0, 0, block_row * 2, k + 2});
        G::load(Bs[curr][0], B, {0, 0, block_col * 2, k + 2});
        asm volatile("s_waitcnt lgkmcnt(0)");

        mma_ABt_scaled_16(cA, a[0], b[0], &sa_h0, &sb_h0);

        auto a_subtile_1 = kittens::subtile_inplace<REG_M, BLOCK_K>(As[curr][1], {warp_m, 0});
        load(a[1], a_subtile_1);
        G::load(Bs[curr][1], B, {0, 0, block_col * 2 + 1, k + 2});
        asm volatile("s_waitcnt lgkmcnt(0)");
        G::load(As[curr][1], A, {0, 0, block_row * 2 + 1, k + 2});

        mma_ABt_scaled_16(cB, a[0], b[1], &sa_h0, &sb_h1);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto a_subtile_0_next = kittens::subtile_inplace<REG_M, BLOCK_K>(As[next][0], {warp_m, 0});
        load(a[0], a_subtile_0_next);

        mma_ABt_scaled_16(cC, a[1], b[0], &sa_h1, &sb_h0);

        auto b_subtile_0_next = kittens::subtile_inplace<REG_N, BLOCK_K>(Bs[next][0], {warp_n, 0});
        load(b[0], b_subtile_0_next);

        mma_ABt_scaled_16(cD, a[1], b[1], &sa_h1, &sb_h1);
    }

    // Epilogue k = k_iters - 2
    {
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");

        int k = k_iters - 2;
        fp8e8m0_4 sa_h0 = (fp8e8m0_4)scale_A_iter[k * M + block_m + a_row_h0 + lid];
        fp8e8m0_4 sb_h0 = (fp8e8m0_4)scale_B_iter[k * N + block_n + b_row_h0 + lid];
        fp8e8m0_4 sa_h1 = (fp8e8m0_4)scale_A_iter[k * M + block_m + a_row_h1 + lid];
        fp8e8m0_4 sb_h1 = (fp8e8m0_4)scale_B_iter[k * N + block_n + b_row_h1 + lid];

        auto b_subtile_1 = kittens::subtile_inplace<REG_N, BLOCK_K>(Bs[curr][1], {warp_n, 0});
        load(b[1], b_subtile_1);

        mma_ABt_scaled_16(cA, a[0], b[0], &sa_h0, &sb_h0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto a_subtile_1 = kittens::subtile_inplace<REG_M, BLOCK_K>(As[curr][1], {warp_m, 0});
        load(a[1], a_subtile_1);

        mma_ABt_scaled_16(cB, a[0], b[1], &sa_h0, &sb_h1);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(2)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto a_subtile_0_next = kittens::subtile_inplace<REG_M, BLOCK_K>(As[next][0], {warp_m, 0});
        load(a[0], a_subtile_0_next);

        mma_ABt_scaled_16(cC, a[1], b[0], &sa_h1, &sb_h0);

        auto b_subtile_0_next = kittens::subtile_inplace<REG_N, BLOCK_K>(Bs[next][0], {warp_n, 0});
        load(b[0], b_subtile_0_next);

        mma_ABt_scaled_16(cD, a[1], b[1], &sa_h1, &sb_h1);

        curr ^= 1; next ^= 1;
    }

    // Final epilogue
    {
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");

        int k = k_iters - 1;
        fp8e8m0_4 sa_h0 = (fp8e8m0_4)scale_A_iter[k * M + block_m + a_row_h0 + lid];
        fp8e8m0_4 sb_h0 = (fp8e8m0_4)scale_B_iter[k * N + block_n + b_row_h0 + lid];
        fp8e8m0_4 sa_h1 = (fp8e8m0_4)scale_A_iter[k * M + block_m + a_row_h1 + lid];
        fp8e8m0_4 sb_h1 = (fp8e8m0_4)scale_B_iter[k * N + block_n + b_row_h1 + lid];

        auto b_subtile_1 = kittens::subtile_inplace<REG_N, BLOCK_K>(Bs[curr][1], {warp_n, 0});
        load(b[1], b_subtile_1);

        mma_ABt_scaled_16(cA, a[0], b[0], &sa_h0, &sb_h0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto a_subtile_1 = kittens::subtile_inplace<REG_M, BLOCK_K>(As[curr][1], {warp_m, 0});
        load(a[1], a_subtile_1);

        mma_ABt_scaled_16(cB, a[0], b[1], &sa_h0, &sb_h1);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        mma_ABt_scaled_16(cC, a[1], b[0], &sa_h1, &sb_h0);
        mma_ABt_scaled_16(cD, a[1], b[1], &sa_h1, &sb_h1);
    }
    __builtin_amdgcn_sched_barrier(0);

    store(C, cA, {0, 0, block_row * WARPS_ROW * 2 + warp_m, block_col * WARPS_COL * 2 + warp_n});
    store(C, cB, {0, 0, block_row * WARPS_ROW * 2 + warp_m, block_col * WARPS_COL * 2 + WARPS_COL + warp_n});
    store(C, cC, {0, 0, block_row * WARPS_ROW * 2 + WARPS_ROW + warp_m, block_col * WARPS_COL * 2 + warp_n});
    store(C, cD, {0, 0, block_row * WARPS_ROW * 2 + WARPS_ROW + warp_m, block_col * WARPS_COL * 2 + WARPS_COL + warp_n});
}


// GPU kernels for quantization and scale packing
__global__ void quantize_fp8_kernel(
    const float *__restrict__ src, fp8e4m3 *__restrict__ dst,
    uint8_t *__restrict__ scales, int rows, int K) {
    int scale_K      = K / 32;
    int idx          = blockIdx.x * blockDim.x + threadIdx.x;
    int total_blocks = rows * scale_K;
    if (idx >= total_blocks) return;

    int row  = idx / scale_K;
    int kb   = idx % scale_K;

    float mx = 0.0f;
    for (int i = 0; i < 32; i++)
        mx = fmaxf(mx, fabsf(src[row * K + kb * 32 + i]));

    uint8_t s = 0;
    if (mx > 0.0f) {
        int e = __float2int_rd(log2f(mx)) + 127;
        s = (uint8_t)max(0, min(254, e));
    }
    scales[row * scale_K + kb] = s;

    float inv = ldexpf(1.0f, 127 - (int)s);
    for (int i = 0; i < 32; i++)
        dst[row * K + kb * 32 + i] = fp8e4m3(src[row * K + kb * 32 + i] * inv);
}

__global__ void pack_scales_kernel(
    const uint8_t *__restrict__ scales, uint32_t *__restrict__ packed,
    int dim, int scale_K, int k_iters, int block_k_per_iter) {
    int idx     = blockIdx.x * blockDim.x + threadIdx.x;
    int total   = k_iters * dim;
    if (idx >= total) return;

    int ki      = idx / dim;
    int row     = idx % dim;
    int kb_base = ki * block_k_per_iter;

    uint32_t p = 0;
    for (int j = 0; j < 4; j++)
        p |= (uint32_t)scales[row * scale_K + kb_base + j] << (j * 8);
    packed[ki * dim + row] = p;
}

__global__ void repack_scales_for_mfma_kernel(
    const uint32_t *__restrict__ scale_iter,
    uint32_t *__restrict__ scale_mfma,
    int dim, int k_iters) {
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = k_iters * dim;
    if (idx >= total) return;

    int ki    = idx / dim;
    int row   = idx % dim;
    int r16   = row % 16;
    int k_sub = (row / 16) % 4;
    int tile  = row / 64;

    uint32_t packed = 0;
    for (int g = 0; g < 4; g++) {
        int src_row = tile * 64 + g * 16 + r16;
        if (src_row < dim) {
            uint32_t src_val = scale_iter[ki * dim + src_row];
            uint8_t  byte_val = (src_val >> (k_sub * 8)) & 0xFF;
            packed |= (uint32_t)byte_val << (g * 8);
        }
    }
    scale_mfma[ki * dim + row] = packed;
}


// Host: CPU reference for correctness
static uint8_t compute_e8m0_scale(const float *vals, int count) {
    float mx = 0.0f;
    for (int i = 0; i < count; i++) mx = std::max(mx, std::abs(vals[i]));
    if (mx == 0.0f) return 0;
    return (uint8_t)std::clamp((int)std::floor(std::log2(mx)) + 127, 0, 254);
}


// Host: quantize + pack + repack
template <int M, int N, int K>
void prepare_mxfp8_data(
    std::vector<fp8e4m3> &a_q, std::vector<fp8e4m3> &b_q,
    std::vector<uint8_t> &sa_raw, std::vector<uint8_t> &sb_raw,
    std::vector<uint32_t> &sa_mfma, std::vector<uint32_t> &sb_mfma,
    uint32_t seed = 42) {
    constexpr int scale_K  = K / 32;
    constexpr int k_iters  = K / BLOCK_K;

    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 0.5f);

    std::vector<float> a_f(M * K), b_f(N * K);
    for (auto &v : a_f) v = dist(rng);
    for (auto &v : b_f) v = dist(rng);

    a_q.resize(M * K); b_q.resize(N * K);
    sa_raw.resize(M * scale_K); sb_raw.resize(N * scale_K);

    for (int row = 0; row < M; row++)
        for (int kb = 0; kb < scale_K; kb++) {
            float blk[32];
            for (int i = 0; i < 32; i++) blk[i] = a_f[row * K + kb * 32 + i];
            uint8_t s = compute_e8m0_scale(blk, 32);
            sa_raw[row * scale_K + kb] = s;
            float inv = std::ldexp(1.0f, 127 - (int)s);
            for (int i = 0; i < 32; i++) a_q[row * K + kb * 32 + i] = fp8e4m3(blk[i] * inv);
        }
    for (int row = 0; row < N; row++)
        for (int kb = 0; kb < scale_K; kb++) {
            float blk[32];
            for (int i = 0; i < 32; i++) blk[i] = b_f[row * K + kb * 32 + i];
            uint8_t s = compute_e8m0_scale(blk, 32);
            sb_raw[row * scale_K + kb] = s;
            float inv = std::ldexp(1.0f, 127 - (int)s);
            for (int i = 0; i < 32; i++) b_q[row * K + kb * 32 + i] = fp8e4m3(blk[i] * inv);
        }

    // Pack iteration-major, then repack for MFMA on GPU
    std::vector<uint32_t> sa_iter(k_iters * M), sb_iter(k_iters * N);
    for (int ki = 0; ki < k_iters; ki++) {
        int kb_base = ki * 4;
        for (int row = 0; row < M; row++) {
            uint32_t p = 0;
            for (int j = 0; j < 4; j++) p |= (uint32_t)sa_raw[row * scale_K + kb_base + j] << (j * 8);
            sa_iter[ki * M + row] = p;
        }
        for (int row = 0; row < N; row++) {
            uint32_t p = 0;
            for (int j = 0; j < 4; j++) p |= (uint32_t)sb_raw[row * scale_K + kb_base + j] << (j * 8);
            sb_iter[ki * N + row] = p;
        }
    }

    uint32_t *d_sa, *d_sb, *d_sa_mfma, *d_sb_mfma;
    hipMalloc(&d_sa, (size_t)k_iters * M * sizeof(uint32_t));
    hipMalloc(&d_sb, (size_t)k_iters * N * sizeof(uint32_t));
    hipMalloc(&d_sa_mfma, (size_t)k_iters * M * sizeof(uint32_t));
    hipMalloc(&d_sb_mfma, (size_t)k_iters * N * sizeof(uint32_t));
    hipMemcpy(d_sa, sa_iter.data(), (size_t)k_iters * M * sizeof(uint32_t), hipMemcpyHostToDevice);
    hipMemcpy(d_sb, sb_iter.data(), (size_t)k_iters * N * sizeof(uint32_t), hipMemcpyHostToDevice);
    int rp_blk_a = (k_iters * M + 255) / 256;
    int rp_blk_b = (k_iters * N + 255) / 256;
    repack_scales_for_mfma_kernel<<<rp_blk_a, 256>>>(d_sa, d_sa_mfma, M, k_iters);
    repack_scales_for_mfma_kernel<<<rp_blk_b, 256>>>(d_sb, d_sb_mfma, N, k_iters);
    hipDeviceSynchronize();

    sa_mfma.resize(k_iters * M); sb_mfma.resize(k_iters * N);
    hipMemcpy(sa_mfma.data(), d_sa_mfma, (size_t)k_iters * M * sizeof(uint32_t), hipMemcpyDeviceToHost);
    hipMemcpy(sb_mfma.data(), d_sb_mfma, (size_t)k_iters * N * sizeof(uint32_t), hipMemcpyDeviceToHost);
    hipFree(d_sa); hipFree(d_sb); hipFree(d_sa_mfma); hipFree(d_sb_mfma);
}


// Correctness
template <int M, int N, int K>
bool run_correctness() {
    constexpr int scale_K  = K / 32;
    constexpr int k_iters  = K / BLOCK_K;
    constexpr int grid     = (M / BLOCK_ROW) * (N / BLOCK_COL);

    std::vector<fp8e4m3> a_q, b_q;
    std::vector<uint8_t> sa_raw, sb_raw;
    std::vector<uint32_t> sa_mfma, sb_mfma;
    prepare_mxfp8_data<M, N, K>(a_q, b_q, sa_raw, sb_raw, sa_mfma, sb_mfma);

    printf("  Computing CPU reference...\n");
    constexpr size_t MN = (size_t)M * N;
    std::vector<float> c_ref(MN, 0.0f);
    #pragma omp parallel for collapse(2)
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
            c_ref[(size_t)i * N + j] = acc;
        }

    fp8e4m3 *d_a, *d_b; float *d_c; uint32_t *d_sa, *d_sb;
    hipMalloc(&d_a, (size_t)M * K);
    hipMalloc(&d_b, (size_t)N * K);
    hipMalloc(&d_c, MN * sizeof(float));
    hipMalloc(&d_sa, (size_t)k_iters * M * sizeof(uint32_t));
    hipMalloc(&d_sb, (size_t)k_iters * N * sizeof(uint32_t));
    hipMemcpy(d_a, a_q.data(), (size_t)M * K, hipMemcpyHostToDevice);
    hipMemcpy(d_b, b_q.data(), (size_t)N * K, hipMemcpyHostToDevice);
    hipMemcpy(d_sa, sa_mfma.data(), (size_t)k_iters * M * sizeof(uint32_t), hipMemcpyHostToDevice);
    hipMemcpy(d_sb, sb_mfma.data(), (size_t)k_iters * N * sizeof(uint32_t), hipMemcpyHostToDevice);
    hipMemset(d_c, 0, MN * sizeof(float));

    gl<fp8e4m3, 1, 1, M, K> A_gl(d_a, nullptr, nullptr, nullptr, nullptr);
    gl<fp8e4m3, 1, 1, N, K> B_gl(d_b, nullptr, nullptr, nullptr, nullptr);
    gl<float,   1, 1, M, N> C_gl(d_c, nullptr, nullptr, nullptr, nullptr);
    mxfp8_gemm_4wave_kernel<M, N, K><<<grid, 256>>>(A_gl, B_gl, C_gl, d_sa, d_sb);
    hipDeviceSynchronize();

    std::vector<float> c_gpu(MN);
    hipMemcpy(c_gpu.data(), d_c, MN * sizeof(float), hipMemcpyDeviceToHost);

    float c_max = 0;
    for (size_t i = 0; i < MN; i++) c_max = std::max(c_max, std::abs(c_ref[i]));
    float atol = c_max * 0.001f;
    int fail_count = 0;
    for (size_t i = 0; i < MN; i++)
        if (std::abs(c_gpu[i] - c_ref[i]) > atol) fail_count++;

    hipFree(d_a); hipFree(d_b); hipFree(d_c); hipFree(d_sa); hipFree(d_sb);

    printf("  atol (0.1%%): %.2e -> %d / %zu failures -> %s\n",
           atol, fail_count, MN, fail_count == 0 ? "PASS" : "FAIL");
    return fail_count == 0;
}


// Benchmark
constexpr int ROTATING_BUFFER_COUNT = 4;

template <int M, int N, int K>
TimingResult run_benchmark(int warmup_iters, int timing_iters) {
    constexpr int k_iters  = K / BLOCK_K;
    constexpr int grid     = (M / BLOCK_ROW) * (N / BLOCK_COL);

    std::vector<fp8e4m3> a_q, b_q;
    std::vector<uint8_t> sa_raw, sb_raw;
    std::vector<uint32_t> sa_mfma, sb_mfma;

    fp8e4m3 *d_a, *d_b; float *d_c; uint32_t *d_sa, *d_sb;
    hipMalloc(&d_a,  (size_t)ROTATING_BUFFER_COUNT * M * K);
    hipMalloc(&d_b,  (size_t)ROTATING_BUFFER_COUNT * N * K);
    hipMalloc(&d_c,  (size_t)M * N * sizeof(float));
    hipMalloc(&d_sa, (size_t)ROTATING_BUFFER_COUNT * k_iters * M * sizeof(uint32_t));
    hipMalloc(&d_sb, (size_t)ROTATING_BUFFER_COUNT * k_iters * N * sizeof(uint32_t));
    HipCheckError();

    for (int buf = 0; buf < ROTATING_BUFFER_COUNT; buf++) {
        prepare_mxfp8_data<M, N, K>(a_q, b_q, sa_raw, sb_raw, sa_mfma, sb_mfma, 42 + buf);
        hipMemcpy(d_a  + (size_t)buf * M * K, a_q.data(), (size_t)M * K, hipMemcpyHostToDevice);
        hipMemcpy(d_b  + (size_t)buf * N * K, b_q.data(), (size_t)N * K, hipMemcpyHostToDevice);
        hipMemcpy(d_sa + (size_t)buf * k_iters * M, sa_mfma.data(), (size_t)k_iters * M * sizeof(uint32_t), hipMemcpyHostToDevice);
        hipMemcpy(d_sb + (size_t)buf * k_iters * N, sb_mfma.data(), (size_t)k_iters * N * sizeof(uint32_t), hipMemcpyHostToDevice);
    }
    HipCheckError();

    for (int i = 0; i < warmup_iters; i++) {
        int buf = i % ROTATING_BUFFER_COUNT;
        gl<fp8e4m3, 1, 1, M, K> A(d_a + (size_t)buf * M * K, nullptr, nullptr, nullptr, nullptr);
        gl<fp8e4m3, 1, 1, N, K> B(d_b + (size_t)buf * N * K, nullptr, nullptr, nullptr, nullptr);
        gl<float,   1, 1, M, N> C(d_c, nullptr, nullptr, nullptr, nullptr);
        mxfp8_gemm_4wave_kernel<M, N, K><<<grid, 256>>>(A, B, C,
            d_sa + (size_t)buf * k_iters * M, d_sb + (size_t)buf * k_iters * N);
        hipDeviceSynchronize();
    }

    hipEvent_t t0, t1;
    hipEventCreate(&t0); hipEventCreate(&t1);
    std::vector<float> times;
    for (int r = 0; r < timing_iters; r++) {
        int buf = r % ROTATING_BUFFER_COUNT;
        gl<fp8e4m3, 1, 1, M, K> A(d_a + (size_t)buf * M * K, nullptr, nullptr, nullptr, nullptr);
        gl<fp8e4m3, 1, 1, N, K> B(d_b + (size_t)buf * N * K, nullptr, nullptr, nullptr, nullptr);
        gl<float,   1, 1, M, N> C(d_c, nullptr, nullptr, nullptr, nullptr);
        hipEventRecord(t0);
        mxfp8_gemm_4wave_kernel<M, N, K><<<grid, 256>>>(A, B, C,
            d_sa + (size_t)buf * k_iters * M, d_sb + (size_t)buf * k_iters * N);
        hipEventRecord(t1);
        hipEventSynchronize(t1);
        float ms; hipEventElapsedTime(&ms, t0, t1);
        times.push_back(ms);
    }

    float best = *std::min_element(times.begin(), times.end());
    float avg  = 0; for (float t : times) avg += t; avg /= times.size();
    double ops = 2.0 * M * N * K;

    hipEventDestroy(t0); hipEventDestroy(t1);
    hipFree(d_a); hipFree(d_b); hipFree(d_c); hipFree(d_sa); hipFree(d_sb);

    return { best, avg, ops / (best * 1e-3) / 1e12, ops / (avg * 1e-3) / 1e12, timing_iters };
}

int main() {
    constexpr int M             = SIZE;
    constexpr int N             = SIZE;
    constexpr int K             = SIZE;
    constexpr int warmup_iters  = 500;
    constexpr int timing_iters  = 100;

    printf("=== MXFP8 TN 4-wave GEMM ===\n");
    printf("Matrix dimensions: %dx%dx%d\n", M, N, K);
    printf("Warmup: %d, Timing: %d\n\n", warmup_iters, timing_iters);

    printf("Correctness check:\n");
    bool pass = run_correctness<M, N, K>();

    if (!pass) {
        printf("\nCorrectness FAILED — skipping benchmark.\n");
        return 1;
    }

    printf("\nRunning benchmark...\n");
    TimingResult res = run_benchmark<M, N, K>(warmup_iters, timing_iters);

    printf("\n=== PERFORMANCE RESULTS ===\n");
    printf("Kernel time (best): %.3f ms,  TFLOPS: %.2f\n", res.best_time_ms, res.best_tflops);
    printf("Kernel time (avg ): %.3f ms,  TFLOPS: %.2f\n", res.avg_time_ms, res.avg_tflops);
    printf("\nCorrectness: PASSED\n");

    return 0;
}
