#include "kittens.cuh"
#include <random>
#include <omp.h>
#include <chrono>

using namespace kittens;

#include "../profile_utils.cpp"
#include "utils.cpp"

#define SIZE 16384

constexpr int NUM_WARPS = 4;

using G = kittens::group<NUM_WARPS>;

/**
 * @brief One K-step worth of work for a single c[i][j] accumulator: 32 base
 * MFMAs (4x4 base accumulator tiles, 2 K-MFMAs each since rt_16x64 has K=64
 * and BLOCK_K=128), interleaved with one global->shared prefetch (load_one<i>)
 * and the corresponding shared->register fill (load_one<i, ...>) for the next
 * cluster's inputs.
 *
 * Mirrors fp8fp32/FP8_4wave/4_wave.cu::do_interleaved_cluster, with 2x as many
 * MFMAs per (n, m) pair to cover INT8's K=64 MFMA vs. fp8's K=128.
 */
template<typename ST_GL, typename GL_GL, typename ST, typename RT, typename RT_A, typename RT_B, typename RT_C, ducks::coord::tile COORD=coord<ST_GL>>
__device__ inline static void do_interleaved_cluster(ST_GL& dst_gl, const GL_GL& src_gl, COORD idx, RT& dst, const ST& src, RT_A& a, RT_B& b, RT_C& c) {
    __builtin_amdgcn_sched_barrier(0);
    mma_ABt_one(c, a, b, c, 0, 0, 0); mma_ABt_one(c, a, b, c, 0, 0, 1);
    __builtin_amdgcn_sched_barrier(0);

    precomputed_addresses addresses = precompute_addresses(dst_gl, src_gl, idx);

    __builtin_amdgcn_sched_barrier(0);
    mma_ABt_one(c, a, b, c, 0, 1, 0); mma_ABt_one(c, a, b, c, 0, 1, 1);
    __builtin_amdgcn_sched_barrier(0);

    uint32_t swizzled_offsets[2];
    prefill_swizzled_offsets<2>(dst, src, swizzled_offsets);

    load_one<0>(dst_gl, src_gl, addresses);
    load_one<0, 0, 0>(dst, src, swizzled_offsets);

    __builtin_amdgcn_sched_barrier(0);
    mma_ABt_one(c, a, b, c, 0, 2, 0); mma_ABt_one(c, a, b, c, 0, 2, 1);
    __builtin_amdgcn_sched_barrier(0);

    load_one<0, 0, 1>(dst, src, swizzled_offsets);

    __builtin_amdgcn_sched_barrier(0);
    mma_ABt_one(c, a, b, c, 0, 3, 0); mma_ABt_one(c, a, b, c, 0, 3, 1);
    __builtin_amdgcn_sched_barrier(0);

    load_one<1>(dst_gl, src_gl, addresses);
    load_one<1, 0, 0>(dst, src, swizzled_offsets);

    __builtin_amdgcn_sched_barrier(0);
    mma_ABt_one(c, a, b, c, 1, 0, 0); mma_ABt_one(c, a, b, c, 1, 0, 1);
    mma_ABt_one(c, a, b, c, 1, 1, 0); mma_ABt_one(c, a, b, c, 1, 1, 1);
    __builtin_amdgcn_sched_barrier(0);

    load_one<1, 0, 1>(dst, src, swizzled_offsets);

    __builtin_amdgcn_sched_barrier(0);
    mma_ABt_one(c, a, b, c, 1, 2, 0); mma_ABt_one(c, a, b, c, 1, 2, 1);
    mma_ABt_one(c, a, b, c, 1, 3, 0); mma_ABt_one(c, a, b, c, 1, 3, 1);
    __builtin_amdgcn_sched_barrier(0);

    load_one<2>(dst_gl, src_gl, addresses);
    load_one<2, 0, 0>(dst, src, swizzled_offsets);

    __builtin_amdgcn_sched_barrier(0);
    mma_ABt_one(c, a, b, c, 2, 0, 0); mma_ABt_one(c, a, b, c, 2, 0, 1);
    mma_ABt_one(c, a, b, c, 2, 1, 0); mma_ABt_one(c, a, b, c, 2, 1, 1);
    __builtin_amdgcn_sched_barrier(0);

    load_one<2, 0, 1>(dst, src, swizzled_offsets);

    __builtin_amdgcn_sched_barrier(0);
    mma_ABt_one(c, a, b, c, 2, 2, 0); mma_ABt_one(c, a, b, c, 2, 2, 1);
    mma_ABt_one(c, a, b, c, 2, 3, 0); mma_ABt_one(c, a, b, c, 2, 3, 1);
    __builtin_amdgcn_sched_barrier(0);

    load_one<3>(dst_gl, src_gl, addresses);
    load_one<3, 0, 0>(dst, src, swizzled_offsets);

    __builtin_amdgcn_sched_barrier(0);
    mma_ABt_one(c, a, b, c, 3, 0, 0); mma_ABt_one(c, a, b, c, 3, 0, 1);
    mma_ABt_one(c, a, b, c, 3, 1, 0); mma_ABt_one(c, a, b, c, 3, 1, 1);
    __builtin_amdgcn_sched_barrier(0);

    load_one<3, 0, 1>(dst, src, swizzled_offsets);

    __builtin_amdgcn_sched_barrier(0);
    mma_ABt_one(c, a, b, c, 3, 2, 0); mma_ABt_one(c, a, b, c, 3, 2, 1);
    mma_ABt_one(c, a, b, c, 3, 3, 0); mma_ABt_one(c, a, b, c, 3, 3, 1);
    __builtin_amdgcn_sched_barrier(0);
}

template <int M, int N, int K>
__global__ __launch_bounds__(256, 1) void matmul_device(const kittens::gl<int8, 1, 1, M, K> A, const kittens::gl<int8, 1, 1, N, K> B, const kittens::gl<int, 1, 1, M, N> C) {
    constexpr int WARPS_COL = 2;
    constexpr int WARPS_ROW = 2;
    constexpr int BLOCK_SIZE_ROW = 256;
    constexpr int BLOCK_SIZE_COL = 256;
    constexpr int BLOCK_K = 128;
    constexpr int k_step = BLOCK_K;
    constexpr int k_iters = K / BLOCK_K;

    using ST_A = st_int8<BLOCK_SIZE_ROW / 2, BLOCK_K, st_16x128_s>;
    using ST_B = st_int8<BLOCK_SIZE_COL / 2, BLOCK_K, st_16x128_s>;

    using GL_A = kittens::gl<int8, 1, 1, M, K>;
    using GL_B = kittens::gl<int8, 1, 1, N, K>;
    using GL_C = kittens::gl<int,  1, 1, M, N>;

    // rt_16x64 base => K=64 MFMA. BLOCK_K=128 => RT::width=2.
    using RT_A = rt_int8<BLOCK_SIZE_ROW / 2 / WARPS_ROW, k_step, row_l, rt_16x64_s>; // 64x128 = 4x2
    using RT_B = rt_int8<BLOCK_SIZE_COL / 2 / WARPS_COL, k_step, row_l, rt_16x64_s>; // 64x128 = 4x2
    using RT_C = rt_int <BLOCK_SIZE_ROW / 2 / WARPS_ROW, BLOCK_SIZE_COL / 2 / WARPS_COL, col_l, rt_16x16_s>; // 64x64 = 4x4

    __shared__ ST_A As[2][2];
    __shared__ ST_B Bs[2][2];

    RT_C c[2][2];

    // XCD-aware block-id swizzle for L2 locality (same scheme as FP8/MXFP4).
    int wgid = blockIdx.x;
    const int NUM_WGS = gridDim.x;
    const int NUM_XCDS = 8;
    wgid = (wgid % NUM_XCDS) * (NUM_WGS / NUM_XCDS) + (wgid / NUM_XCDS);
    const int WGM = 4;
    const int num_pid_m = (M + BLOCK_SIZE_ROW - 1) / BLOCK_SIZE_ROW;
    const int num_pid_n = (N + BLOCK_SIZE_COL - 1) / BLOCK_SIZE_COL;
    const int num_wgid_in_group = WGM * num_pid_n;
    const int group_id = wgid / num_wgid_in_group;
    const int first_pid_m = group_id * WGM;
    const int group_size_m = min(num_pid_m - first_pid_m, WGM);
    const int pid_m = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    const int pid_n = (wgid % num_wgid_in_group) / group_size_m;
    const int block_row = pid_m;
    const int block_col = pid_n;

    const int warp_m = (warpid() / WARPS_COL);
    const int warp_n = (warpid() % WARPS_COL);

    int curr = 0, next = 1;

    {
    __builtin_amdgcn_sched_barrier(0);
    RT_A a[2];
    RT_B b[2];

    // Prologue: kick off A,B,C inputs for K-tile 0 (curr) and K-tile 1 (next).
    G::load(As[curr][0], A, {0, 0, block_row*WARPS_ROW,     0});
    G::load(Bs[curr][0], B, {0, 0, block_col*WARPS_COL,     0});
    G::load(Bs[curr][1], B, {0, 0, block_col*WARPS_COL + 1, 0});
    G::load(As[curr][1], A, {0, 0, block_row*WARPS_ROW + 1, 0});

    zero(c[0][0]); zero(c[0][1]); zero(c[1][0]); zero(c[1][1]);

    G::load(As[next][0], A, {0, 0, block_row*WARPS_ROW,     1});
    G::load(Bs[next][0], B, {0, 0, block_col*WARPS_COL,     1});
    G::load(Bs[next][1], B, {0, 0, block_col*WARPS_COL + 1, 1});
    G::load(As[next][1], A, {0, 0, block_row*WARPS_ROW + 1, 1});

    // Each G::load issues 4 buffer_load_lds per warp -> 4 vmcnt slots / call.
    // After 8 loads we have 32 outstanding; wait until 28 remain to release
    // As[curr][0], then 24 to release Bs[curr][0].
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(28)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    auto a_subtile_0 = kittens::subtile_inplace<BLOCK_SIZE_ROW / 2 / WARPS_ROW, k_step>(As[curr][0], {warp_m, 0});
    load(a[0], a_subtile_0);

    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(24)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    auto b_subtile_0 = kittens::subtile_inplace<BLOCK_SIZE_COL / 2 / WARPS_COL, k_step>(Bs[curr][0], {warp_n, 0});
    load(b[0], b_subtile_0);

    // Main K loop: each iteration consumes (curr) and prefetches (k+2 -> curr
    // by the time we loop). Four mma_ABt calls per iteration, one per c[i][j],
    // each paired with one shared->register fill and one global->shared prefetch.
    #pragma unroll
    for (int k = 0; k < k_iters - 2; ++k, curr ^= 1, next ^= 1) {
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(16)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        auto bs_subtile_1 = kittens::subtile_inplace<BLOCK_SIZE_COL / 2 / WARPS_COL, k_step>(Bs[curr][1], {warp_n, 0});
        G::load(As[curr][0], A, {0, 0, block_row*WARPS_ROW,     k + 2});
        load(b[1], bs_subtile_1);
        mma_ABt(c[0][0], a[0], b[0], c[0][0]);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto a_subtile_1 = kittens::subtile_inplace<BLOCK_SIZE_ROW / 2 / WARPS_ROW, k_step>(As[curr][1], {warp_m, 0});
        G::load(Bs[curr][0], B, {0, 0, block_col*WARPS_COL,     k + 2});
        load(a[1], a_subtile_1);
        mma_ABt(c[0][1], a[0], b[1], c[0][1]);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(16)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto a_subtile_0 = kittens::subtile_inplace<BLOCK_SIZE_ROW / 2 / WARPS_ROW, k_step>(As[next][0], {warp_m, 0});
        G::load(Bs[curr][1], B, {0, 0, block_col*WARPS_COL + 1, k + 2});
        load(a[0], a_subtile_0);
        mma_ABt(c[1][0], a[1], b[0], c[1][0]);

        auto b_subtile_0 = kittens::subtile_inplace<BLOCK_SIZE_COL / 2 / WARPS_COL, k_step>(Bs[next][0], {warp_n, 0});
        G::load(As[curr][1], A, {0, 0, block_row*WARPS_ROW + 1, k + 2});
        load(b[0], b_subtile_0);
        mma_ABt(c[1][1], a[1], b[1], c[1][1]);
    }

    { // EPILOGUE: K-tile k_iters - 2 (one prefetch still outstanding for k_iters - 1).
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(8)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto bs_subtile_1 = kittens::subtile_inplace<BLOCK_SIZE_COL / 2 / WARPS_COL, k_step>(Bs[curr][1], {warp_n, 0});
        load(b[1], bs_subtile_1);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[0][0], a[0], b[0], c[0][0]);
        __builtin_amdgcn_sched_barrier(0);

        asm volatile("s_waitcnt lgkmcnt(0)");
        auto a_subtile_1 = kittens::subtile_inplace<BLOCK_SIZE_ROW / 2 / WARPS_ROW, k_step>(As[curr][1], {warp_m, 0});
        load(a[1], a_subtile_1);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[0][1], a[0], b[1], c[0][1]);
        __builtin_amdgcn_sched_barrier(0);

        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");

        auto a_subtile_0 = kittens::subtile_inplace<BLOCK_SIZE_ROW / 2 / WARPS_ROW, k_step>(As[next][0], {warp_m, 0});
        load(a[0], a_subtile_0);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[1][0], a[1], b[0], c[1][0]);
        __builtin_amdgcn_sched_barrier(0);

        auto b_subtile_0 = kittens::subtile_inplace<BLOCK_SIZE_COL / 2 / WARPS_COL, k_step>(Bs[next][0], {warp_n, 0});
        load(b[0], b_subtile_0);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[1][1], a[1], b[1], c[1][1]);
        __builtin_amdgcn_sched_barrier(0);

        curr ^= 1;
        next ^= 1;
    }

    { // EPILOGUE: K-tile k_iters - 1 (drain).
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto bs_subtile_1 = kittens::subtile_inplace<BLOCK_SIZE_COL / 2 / WARPS_COL, k_step>(Bs[curr][1], {warp_n, 0});
        load(b[1], bs_subtile_1);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[0][0], a[0], b[0], c[0][0]);
        __builtin_amdgcn_sched_barrier(0);

        asm volatile("s_waitcnt lgkmcnt(0)");
        auto a_subtile_1 = kittens::subtile_inplace<BLOCK_SIZE_ROW / 2 / WARPS_ROW, k_step>(As[curr][1], {warp_m, 0});
        load(a[1], a_subtile_1);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[0][1], a[0], b[1], c[0][1]);
        __builtin_amdgcn_sched_barrier(0);

        asm volatile("s_waitcnt lgkmcnt(0)");

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[1][0], a[1], b[0], c[1][0]);
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[1][1], a[1], b[1], c[1][1]);
        __builtin_amdgcn_sched_barrier(0);
    }
    __builtin_amdgcn_sched_barrier(0);
    } // RT_A/RT_B scope ends; release their VGPRs before the store.

    store(C, c[0][0], {0, 0, (block_row * WARPS_ROW)     * 2 + warp_m, (block_col * WARPS_COL)     * 2 + warp_n});
    store(C, c[0][1], {0, 0, (block_row * WARPS_ROW)     * 2 + warp_m, (block_col * WARPS_COL + 1) * 2 + warp_n});
    store(C, c[1][0], {0, 0, (block_row * WARPS_ROW + 1) * 2 + warp_m, (block_col * WARPS_COL)     * 2 + warp_n});
    store(C, c[1][1], {0, 0, (block_row * WARPS_ROW + 1) * 2 + warp_m, (block_col * WARPS_COL + 1) * 2 + warp_n});
}

// -------- host driver --------

template <int M, int N, int K>
TimingResult matmul_host(std::vector<int8>& a, std::vector<int8>& b, std::vector<int>& c,
                         int warmup_iters, int timing_iters) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int threads_per_warp = 64;
    constexpr int warps_per_cu = 4;
    constexpr int threads_per_block = threads_per_warp * warps_per_cu;
    constexpr int threadblocks = (M / BLOCK_SIZE) * (N / BLOCK_SIZE);

    c.resize((size_t)M * N);

    int8 *d_a, *d_b;
    int  *d_c;
    hipMalloc(&d_a, (size_t)M * K * sizeof(int8));
    hipMalloc(&d_b, (size_t)N * K * sizeof(int8));
    hipMalloc(&d_c, (size_t)M * N * sizeof(int));
    HipCheckError();

    random_init<M, N, K>(a, b, 42);
    hipMemcpy(d_a, a.data(), (size_t)M * K * sizeof(int8), hipMemcpyHostToDevice);
    hipMemcpy(d_b, b.data(), (size_t)N * K * sizeof(int8), hipMemcpyHostToDevice);
    HipCheckError();

    kittens::gl<int8, 1, 1, M, K> A(d_a, nullptr, nullptr, nullptr, nullptr);
    kittens::gl<int8, 1, 1, N, K> B(d_b, nullptr, nullptr, nullptr, nullptr);
    kittens::gl<int,  1, 1, M, N> C(d_c, nullptr, nullptr, nullptr, nullptr);

    for (int i = 0; i < warmup_iters; i++) {
        hipMemset(d_c, 0, (size_t)M * N * sizeof(int));
        matmul_device<M, N, K><<<threadblocks, threads_per_block>>>(A, B, C);
        HipCheckError();
    }

    hipEvent_t start_event, stop_event;
    hipEventCreate(&start_event);
    hipEventCreate(&stop_event);

    std::vector<float> times_ms;
    times_ms.reserve(timing_iters);
    for (int r = 0; r < timing_iters; ++r) {
        hipMemset(d_c, 0, (size_t)M * N * sizeof(int));
        hipEventRecord(start_event, 0);
        matmul_device<M, N, K><<<threadblocks, threads_per_block>>>(A, B, C);
        hipEventRecord(stop_event, 0);
        hipEventSynchronize(stop_event);
        float ms = 0.f;
        hipEventElapsedTime(&ms, start_event, stop_event);
        times_ms.push_back(ms);
        HipCheckError();
    }

    float sum_ms = 0.f, best_ms = 1e30f;
    for (float t : times_ms) { sum_ms += t; best_ms = std::min(best_ms, t); }
    float avg_ms = sum_ms / times_ms.size();

    double total_ops = 2.0 * M * N * K;
    double best_tops = (total_ops / (best_ms * 1e-3)) / 1e12;
    double avg_tops  = (total_ops / (avg_ms  * 1e-3)) / 1e12;

    hipEventDestroy(start_event);
    hipEventDestroy(stop_event);

    hipMemcpy(c.data(), d_c, (size_t)M * N * sizeof(int), hipMemcpyDeviceToHost);
    HipCheckError();

    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
    HipCheckError();

    return {best_ms, avg_ms, best_tops, avg_tops, timing_iters};
}

int main() {
    constexpr int M = SIZE;
    constexpr int N = SIZE;
    constexpr int K = SIZE;
    constexpr int warmup_iters = 50;
    constexpr int timing_iters = 50;

    printf("INT8 x INT8 -> INT32 GEMM (M=%d, N=%d, K=%d)\n", M, N, K);
    printf("Warmup iterations: %d, Timing iterations: %d\n\n", warmup_iters, timing_iters);

    std::vector<int8> a_host((size_t)M * K);
    std::vector<int8> b_host((size_t)N * K);
    std::vector<int>  c_host((size_t)M * N);
    std::vector<int>  c_ref ((size_t)M * N);

    printf("Running optimized kernel (matmul_device)...\n");
    TimingResult host_timing = matmul_host<M, N, K>(a_host, b_host, c_host, warmup_iters, timing_iters);

    printf("Running CPU reference...\n");
    auto t0 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            int acc = 0;
            for (int k = 0; k < K; ++k) {
                acc += int(a_host[(size_t)row * K + k]) * int(b_host[(size_t)col * K + k]);
            }
            c_ref[(size_t)row * N + col] = acc;
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ref_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    bool success = true;
    int first_bad = -1;
    for (size_t i = 0; i < (size_t)M * N; ++i) {
        if (c_host[i] != c_ref[i]) {
            if (success) first_bad = (int)i;
            success = false;
        }
    }

    printf("\n=== PERFORMANCE RESULTS ===\n");
    printf("CPU reference: %.3f ms\n", ref_ms);
    printf("Optimized kernel (matmul_device):\n");
    printf("  Kernel time (best): %.3f ms,  TOPS: %.2f\n", host_timing.best_time_ms, host_timing.best_tops);
    printf("  Kernel time (avg ): %.3f ms,  TOPS: %.2f\n", host_timing.avg_time_ms,  host_timing.avg_tops);

    if (success) {
        printf("\nCorrectness: PASSED\n");
    } else {
        int r = first_bad / N, cc = first_bad % N;
        printf("\nCorrectness: FAILED at (%d, %d): got %d, ref %d\n",
               r, cc, c_host[first_bad], c_ref[first_bad]);
    }

    return success ? 0 : 1;
}
