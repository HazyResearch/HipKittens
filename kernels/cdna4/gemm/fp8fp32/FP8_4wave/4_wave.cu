#include "kittens.cuh"
#include <random>
#include <omp.h>
#include <chrono>

using namespace kittens;

#include "../profile_utils.cpp"
#include "utils.cpp"

#define SIZE 8192

constexpr int NUM_WARPS = 4;

using G = kittens::group<NUM_WARPS>;

template<typename ST_GL, typename GL_GL, typename ST, typename RT, typename RT_A, typename RT_B, typename RT_C, ducks::coord::tile COORD=coord<ST_GL>>
__device__ inline static void do_interleaved_cluster(ST_GL& dst_gl, const GL_GL& src_gl, COORD idx, RT& dst, const ST& src, RT_A& a, RT_B& b, RT_C& c) {
    __builtin_amdgcn_sched_barrier(0);
    mma_ABt_one(c, a, b, c, 0, 0, 0);
    __builtin_amdgcn_sched_barrier(0);
    
    precomputed_addresses addresses = precompute_addresses(dst_gl, src_gl, idx);

    __builtin_amdgcn_sched_barrier(0);
    mma_ABt_one(c, a, b, c, 0, 1, 0);
    __builtin_amdgcn_sched_barrier(0);

    uint32_t swizzled_offsets[2];
    prefill_swizzled_offsets<2>(dst, src, swizzled_offsets);

    load_one<0>(dst_gl, src_gl, addresses);
    load_one<0, 0, 0>(dst, src, swizzled_offsets);

    __builtin_amdgcn_sched_barrier(0);
    mma_ABt_one(c, a, b, c, 0, 2, 0);
    __builtin_amdgcn_sched_barrier(0);

    load_one<0, 0, 1>(dst, src, swizzled_offsets);

    __builtin_amdgcn_sched_barrier(0);
    mma_ABt_one(c, a, b, c, 0, 3, 0);
    __builtin_amdgcn_sched_barrier(0);

    load_one<1>(dst_gl, src_gl, addresses);
    load_one<1, 0, 0>(dst, src, swizzled_offsets);
    __builtin_amdgcn_sched_barrier(0);
    mma_ABt_one(c, a, b, c, 1, 0, 0);
    mma_ABt_one(c, a, b, c, 1, 1, 0);
    __builtin_amdgcn_sched_barrier(0);

    load_one<1, 0, 1>(dst, src, swizzled_offsets);
    __builtin_amdgcn_sched_barrier(0);
    mma_ABt_one(c, a, b, c, 1, 2, 0);
    mma_ABt_one(c, a, b, c, 1, 3, 0);
    __builtin_amdgcn_sched_barrier(0);

    load_one<2>(dst_gl, src_gl, addresses);
    load_one<2, 0, 0>(dst, src, swizzled_offsets);
    __builtin_amdgcn_sched_barrier(0);
    mma_ABt_one(c, a, b, c, 2, 0, 0);
    mma_ABt_one(c, a, b, c, 2, 1, 0);
    __builtin_amdgcn_sched_barrier(0);

    load_one<2, 0, 1>(dst, src, swizzled_offsets);
    __builtin_amdgcn_sched_barrier(0);
    mma_ABt_one(c, a, b, c, 2, 2, 0);
    mma_ABt_one(c, a, b, c, 2, 3, 0);
    __builtin_amdgcn_sched_barrier(0);

    load_one<3>(dst_gl, src_gl, addresses);
    load_one<3, 0, 0>(dst, src, swizzled_offsets);
    __builtin_amdgcn_sched_barrier(0);
    mma_ABt_one(c, a, b, c, 3, 0, 0);
    mma_ABt_one(c, a, b, c, 3, 1, 0);
    __builtin_amdgcn_sched_barrier(0);

    load_one<3, 0, 1>(dst, src, swizzled_offsets);
    __builtin_amdgcn_sched_barrier(0);
    mma_ABt_one(c, a, b, c, 3, 2, 0);
    mma_ABt_one(c, a, b, c, 3, 3, 0);
    __builtin_amdgcn_sched_barrier(0);
}

template <int M, int N, int K>
__global__ __launch_bounds__(256, 1) void matmul_device(const kittens::gl<fp8e4m3, 1, 1, M, K> A, const kittens::gl<fp8e4m3, 1, 1, N, K> B, const kittens::gl<bf16, 1, 1, M, N> C) {
    constexpr int WARPS_COL = 2;
    constexpr int WARPS_ROW = 2;
    constexpr int BLOCK_SIZE_ROW = 256;
    constexpr int BLOCK_SIZE_COL = 256;
    constexpr int BLOCK_K = 128;
    constexpr int k_step = BLOCK_K;
    constexpr int blocks_row = M / BLOCK_SIZE_ROW; // Number of blocks along output matrix row dim
    constexpr int blocks_col = N / BLOCK_SIZE_COL; // Number of blocks along output matrix col dim
    constexpr int total_blocks_needed = blocks_row * blocks_col;
    constexpr int k_iters = K / BLOCK_K; // K iterations

    using ST_A = st_fp8e4m3<BLOCK_SIZE_ROW / 2, BLOCK_K, st_16x128_s>;
    using ST_B = st_fp8e4m3<BLOCK_SIZE_COL / 2, BLOCK_K, st_16x128_s>;

    using GL_A = kittens::gl<fp8e4m3, 1, 1, M, K>;
    using GL_B = kittens::gl<fp8e4m3, 1, 1, N, K>;
    using GL_C = kittens::gl<bf16, 1, 1, M, N>;

    using RT_A = rt_fp8e4m3<BLOCK_SIZE_ROW / 2 / WARPS_ROW, k_step>; // 128x128 = 4x2
    using RT_B = rt_fp8e4m3<BLOCK_SIZE_COL / 2 / WARPS_COL, k_step>; // 128x128 = 4x2
    using RT_C = rt_fl<BLOCK_SIZE_ROW / 2 / WARPS_ROW, BLOCK_SIZE_COL / 2 / WARPS_COL, col_l, rt_16x16_s>; // 128x128 = 4x4

    __shared__ ST_A As[2][2];
    __shared__ ST_B Bs[2][2];

    RT_C c[2][2];

    int global_block_id = blockIdx.x;

    // Original WGID.
    int wgid = global_block_id;
    const int NUM_WGS = gridDim.x;
    const int NUM_XCDS = 8;
    const int CUS_PER_XCD = 32;
    const int NUM_CUS = CUS_PER_XCD * NUM_XCDS;
    // Swizzle chiplet so that wgids are in the same XCD.
    wgid = (wgid % NUM_XCDS) * (NUM_WGS / NUM_XCDS) + (wgid / NUM_XCDS);
    // Swizzle for better L2 within the same XCD.
    const int WGM = 4;
    const int num_pid_m = (M + BLOCK_SIZE_ROW - 1) / BLOCK_SIZE_ROW;
    const int num_pid_n = (N + BLOCK_SIZE_COL - 1) / BLOCK_SIZE_COL;
    int num_wgid_in_group = WGM * num_pid_n;
    int group_id = wgid / num_wgid_in_group;
    int first_pid_m = group_id * WGM;
    int group_size_m = min(num_pid_m - first_pid_m, WGM);
    int pid_m = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    int pid_n = (wgid % num_wgid_in_group) / group_size_m;
    // Assign the tile's row/column based on the pid_m and pid_n.
    const int row = pid_m; // blockIdx.x
    const int col = pid_n; // blockIdx.y

    int curr = 0, next = 1;

    // Convert linear block ID to 2D coordinates
    int block_row = row;
    int block_col = col;
    int block_m = block_row * BLOCK_SIZE_ROW;
    int block_n = block_col * BLOCK_SIZE_COL;

    // Warp arrangement within threadblock
    int warp_m = (warpid() / WARPS_COL);
    int warp_n = (warpid() % WARPS_COL);

    {
    __builtin_amdgcn_sched_barrier(0);
    RT_A a[2];
    RT_B b[2];

    G::load(As[curr][0], A, {0, 0, block_row*WARPS_ROW, 0});
    G::load(Bs[curr][0], B, {0, 0, block_col*WARPS_COL, 0});
    G::load(Bs[curr][1], B, {0, 0, block_col*WARPS_COL+1, 0});
    G::load(As[curr][1], A, {0, 0, block_row*WARPS_ROW+1, 0});

    zero(c[0][0]);
    zero(c[0][1]);
    zero(c[1][0]);
    zero(c[1][1]);

    G::load(As[next][0], A, {0, 0, block_row*WARPS_ROW, 1});
    G::load(Bs[next][0], B, {0, 0, block_col*WARPS_COL, 1});
    G::load(Bs[next][1], B, {0, 0, block_col*WARPS_COL+1, 1});
    G::load(As[next][1], A, {0, 0, block_row*WARPS_ROW+1, 1});

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

    #pragma unroll
    for (int k = 0; k < k_iters - 2; ++k, curr ^= 1, next ^= 1) {
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(16)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        auto bs_subtile_1 = kittens::subtile_inplace<BLOCK_SIZE_COL / 2 / WARPS_COL, k_step>(Bs[curr][1], {warp_n, 0});
        do_interleaved_cluster(As[curr][0], A, {0, 0, block_row*WARPS_ROW, k + 2}, b[1], bs_subtile_1, a[0], b[0], c[0][0]);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto a_subtile_1 = kittens::subtile_inplace<BLOCK_SIZE_ROW / 2 / WARPS_ROW, k_step>(As[curr][1], {warp_m, 0});
        do_interleaved_cluster(Bs[curr][0], B, {0, 0, block_col*WARPS_COL, k + 2}, a[1], a_subtile_1, a[0], b[1], c[0][1]);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(16)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto a_subtile_0 = kittens::subtile_inplace<BLOCK_SIZE_ROW / 2 / WARPS_ROW, k_step>(As[next][0], {warp_m, 0});
        do_interleaved_cluster(Bs[curr][1], B, {0, 0, block_col*WARPS_COL+1, k + 2}, a[0], a_subtile_0, a[1], b[0], c[1][0]);

        auto b_subtile_0 = kittens::subtile_inplace<BLOCK_SIZE_COL / 2 / WARPS_COL, k_step>(Bs[next][0], {warp_n, 0});
        do_interleaved_cluster(As[curr][1], A, {0, 0, block_row*WARPS_ROW+1, k + 2}, b[0], b_subtile_0, a[1], b[1], c[1][1]);
    }

    { // EPILOGUE: k = k_iters - 2
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(16)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto b_subtile_1 = kittens::subtile_inplace<BLOCK_SIZE_COL / 2 / WARPS_COL, k_step>(Bs[curr][1], {warp_n, 0});
        load(b[1], b_subtile_1);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[0][0], a[0], b[0], c[0][0]);
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto a_subtile_1 = kittens::subtile_inplace<BLOCK_SIZE_ROW / 2 / WARPS_ROW, k_step>(As[curr][1], {warp_m, 0});
        load(a[1], a_subtile_1);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[0][1], a[0], b[1], c[0][1]);
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(8)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

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

    { // EPILOGUE: k = k_iters - 1
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto b_subtile_1 = kittens::subtile_inplace<BLOCK_SIZE_COL / 2 / WARPS_COL, k_step>(Bs[curr][1], {warp_n, 0});
        load(b[1], b_subtile_1);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[0][0], a[0], b[0], c[0][0]);
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto a_subtile_1 = kittens::subtile_inplace<BLOCK_SIZE_ROW / 2 / WARPS_ROW, k_step>(As[curr][1], {warp_m, 0});
        load(a[1], a_subtile_1);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[0][1], a[0], b[1], c[0][1]);
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[1][0], a[1], b[0], c[1][0]);
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[1][1], a[1], b[1], c[1][1]);
        __builtin_amdgcn_sched_barrier(0);
    }
    __builtin_amdgcn_sched_barrier(0);
    }

    store(C, c[0][0], {0, 0, (block_row * WARPS_ROW) * 2 + warp_m, (block_col * WARPS_COL) * 2 + warp_n});
    store(C, c[0][1], {0, 0, (block_row * WARPS_ROW) * 2 + warp_m, (block_col * WARPS_COL + 1) * 2 + warp_n});
    store(C, c[1][0], {0, 0, (block_row * WARPS_ROW + 1) * 2 + warp_m, (block_col * WARPS_COL) * 2 + warp_n});
    store(C, c[1][1], {0, 0, (block_row * WARPS_ROW + 1) * 2 + warp_m, (block_col * WARPS_COL + 1) * 2 + warp_n});
}

template <int M, int N, int K>
__global__ __launch_bounds__(256, 1) void matmul_device_1024(const kittens::gl<fp8e4m3, 1, 1, M, K> A, const kittens::gl<fp8e4m3, 1, 1, N, K> B, const kittens::gl<bf16, 1, 1, M, N> C) {
    constexpr int WARPS_COL = 2;
    constexpr int WARPS_ROW = 2;
    constexpr int BLOCK_SIZE_ROW = 64;
    constexpr int BLOCK_SIZE_COL = 64;
    constexpr int BLOCK_K = 128;
    constexpr int k_step = BLOCK_K;
    constexpr int blocks_row = M / BLOCK_SIZE_ROW; // Number of blocks along output matrix row dim
    constexpr int blocks_col = N / BLOCK_SIZE_COL; // Number of blocks along output matrix col dim
    constexpr int total_blocks_needed = blocks_row * blocks_col;
    constexpr int k_iters = K / BLOCK_K; // K iterations

    using ST_A = st_fp8e4m3<BLOCK_SIZE_ROW / 2, BLOCK_K, st_16x128_s>;
    using ST_B = st_fp8e4m3<BLOCK_SIZE_COL / 2, BLOCK_K, st_16x128_s>;

    using GL_A = kittens::gl<fp8e4m3, 1, 1, M, K>;
    using GL_B = kittens::gl<fp8e4m3, 1, 1, N, K>;
    using GL_C = kittens::gl<bf16, 1, 1, M, N>;

    using RT_A = rt_fp8e4m3<BLOCK_SIZE_ROW / 2 / WARPS_ROW, k_step>; // 128x128 = 4x2
    using RT_B = rt_fp8e4m3<BLOCK_SIZE_COL / 2 / WARPS_COL, k_step>; // 128x128 = 4x2
    using RT_C = rt_fl<BLOCK_SIZE_ROW / 2 / WARPS_ROW, BLOCK_SIZE_COL / 2 / WARPS_COL, col_l, rt_16x16_s>; // 128x128 = 4x4

    __shared__ ST_A As[2][2];
    __shared__ ST_B Bs[2][2];

    RT_C c[2][2];

    int global_block_id = blockIdx.x;

    // Original WGID.
    int wgid = global_block_id;
    const int NUM_WGS = gridDim.x;
    const int NUM_XCDS = 8;
    const int CUS_PER_XCD = 32;
    const int NUM_CUS = CUS_PER_XCD * NUM_XCDS;
    // Swizzle chiplet so that wgids are in the same XCD.
    wgid = (wgid % NUM_XCDS) * (NUM_WGS / NUM_XCDS) + (wgid / NUM_XCDS);
    // Swizzle for better L2 within the same XCD.
    const int WGM = 4;
    const int num_pid_m = (M + BLOCK_SIZE_ROW - 1) / BLOCK_SIZE_ROW;
    const int num_pid_n = (N + BLOCK_SIZE_COL - 1) / BLOCK_SIZE_COL;
    int num_wgid_in_group = WGM * num_pid_n;
    int group_id = wgid / num_wgid_in_group;
    int first_pid_m = group_id * WGM;
    int group_size_m = min(num_pid_m - first_pid_m, WGM);
    int pid_m = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    int pid_n = (wgid % num_wgid_in_group) / group_size_m;
    // Assign the tile's row/column based on the pid_m and pid_n.
    const int row = pid_m; // blockIdx.x
    const int col = pid_n; // blockIdx.y

    int curr = 0, next = 1;

    // Convert linear block ID to 2D coordinates
    int block_row = row;
    int block_col = col;
    int block_m = block_row * BLOCK_SIZE_ROW;
    int block_n = block_col * BLOCK_SIZE_COL;

    // Warp arrangement within threadblock
    int warp_m = (warpid() / WARPS_COL);
    int warp_n = (warpid() % WARPS_COL);

    {
    __builtin_amdgcn_sched_barrier(0);
    RT_A a[2];
    RT_B b[2];

    G::load(As[curr][0], A, {0, 0, block_row*WARPS_ROW, 0});
    G::load(Bs[curr][0], B, {0, 0, block_col*WARPS_COL, 0});
    G::load(Bs[curr][1], B, {0, 0, block_col*WARPS_COL+1, 0});
    G::load(As[curr][1], A, {0, 0, block_row*WARPS_ROW+1, 0});

    zero(c[0][0]);
    zero(c[0][1]);
    zero(c[1][0]);
    zero(c[1][1]);

    G::load(As[next][0], A, {0, 0, block_row*WARPS_ROW, 1});
    G::load(Bs[next][0], B, {0, 0, block_col*WARPS_COL, 1});
    G::load(Bs[next][1], B, {0, 0, block_col*WARPS_COL+1, 1});
    G::load(As[next][1], A, {0, 0, block_row*WARPS_ROW+1, 1});

    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(7)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    auto a_subtile_0 = kittens::subtile_inplace<BLOCK_SIZE_ROW / 2 / WARPS_ROW, k_step>(As[curr][0], {warp_m, 0});
    load(a[0], a_subtile_0);

    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(6)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    auto b_subtile_0 = kittens::subtile_inplace<BLOCK_SIZE_COL / 2 / WARPS_COL, k_step>(Bs[curr][0], {warp_n, 0});
    load(b[0], b_subtile_0);

    #pragma unroll
    for (int k = 0; k < k_iters - 2; ++k, curr ^= 1, next ^= 1) {
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(4)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // template<typename ST_GL, typename GL_GL, typename ST, typename RT, typename RT_A, typename RT_B, typename RT_C, ducks::coord::tile COORD=coord<ST_GL>>
        auto bs_subtile_1 = kittens::subtile_inplace<BLOCK_SIZE_COL / 2 / WARPS_COL, k_step>(Bs[curr][1], {warp_n, 0});
        // do_interleaved_cluster(As[curr][0], A, {0, 0, block_row*WARPS_ROW, k + 2}, b[1], bs_subtile_1, a[0], b[0], c[0][0]);
        G::load(As[curr][0], A, {0, 0, block_row*WARPS_ROW, k + 2});
        load(b[1], bs_subtile_1);
        mma_ABt(c[0][0], a[0], b[0], c[0][0]);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto a_subtile_1 = kittens::subtile_inplace<BLOCK_SIZE_ROW / 2 / WARPS_ROW, k_step>(As[curr][1], {warp_m, 0});
        // do_interleaved_cluster(Bs[curr][0], B, {0, 0, block_col*WARPS_COL, k + 2}, a[1], a_subtile_1, a[0], b[1], c[0][1]);
        G::load(Bs[curr][0], B, {0, 0, block_col*WARPS_COL, k + 2});
        load(a[1], a_subtile_1);
        mma_ABt(c[0][1], a[0], b[1], c[0][1]);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto a_subtile_0 = kittens::subtile_inplace<BLOCK_SIZE_ROW / 2 / WARPS_ROW, k_step>(As[next][0], {warp_m, 0});
        // do_interleaved_cluster(Bs[curr][1], B, {0, 0, block_col*WARPS_COL+1, k + 2}, a[0], a_subtile_0, a[1], b[0], c[1][0]);
        G::load(Bs[curr][1], B, {0, 0, block_col*WARPS_COL+1, k + 2});
        load(a[0], a_subtile_0);
        mma_ABt(c[1][0], a[1], b[0], c[1][0]);

        auto b_subtile_0 = kittens::subtile_inplace<BLOCK_SIZE_COL / 2 / WARPS_COL, k_step>(Bs[next][0], {warp_n, 0});
        // do_interleaved_cluster(As[curr][1], A, {0, 0, block_row*WARPS_ROW+1, k + 2}, b[0], b_subtile_0, a[1], b[1], c[1][1]);
        G::load(As[curr][1], A, {0, 0, block_row*WARPS_ROW+1, k + 2});
        load(b[0], b_subtile_0);
        mma_ABt(c[1][1], a[1], b[1], c[1][1]);
    }

    { // EPILOGUE: k = k_iters - 2
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto b_subtile_1 = kittens::subtile_inplace<BLOCK_SIZE_COL / 2 / WARPS_COL, k_step>(Bs[curr][1], {warp_n, 0});
        load(b[1], b_subtile_1);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[0][0], a[0], b[0], c[0][0]);
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto a_subtile_1 = kittens::subtile_inplace<BLOCK_SIZE_ROW / 2 / WARPS_ROW, k_step>(As[curr][1], {warp_m, 0});
        load(a[1], a_subtile_1);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[0][1], a[0], b[1], c[0][1]);
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(2)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

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

    { // EPILOGUE: k = k_iters - 1
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto b_subtile_1 = kittens::subtile_inplace<BLOCK_SIZE_COL / 2 / WARPS_COL, k_step>(Bs[curr][1], {warp_n, 0});
        load(b[1], b_subtile_1);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[0][0], a[0], b[0], c[0][0]);
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto a_subtile_1 = kittens::subtile_inplace<BLOCK_SIZE_ROW / 2 / WARPS_ROW, k_step>(As[curr][1], {warp_m, 0});
        load(a[1], a_subtile_1);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[0][1], a[0], b[1], c[0][1]);
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[1][0], a[1], b[0], c[1][0]);
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[1][1], a[1], b[1], c[1][1]);
        __builtin_amdgcn_sched_barrier(0);
    }
    __builtin_amdgcn_sched_barrier(0);
    }

    store(C, c[0][0], {0, 0, (block_row * WARPS_ROW) * 2 + warp_m, (block_col * WARPS_COL) * 2 + warp_n});
    store(C, c[0][1], {0, 0, (block_row * WARPS_ROW) * 2 + warp_m, (block_col * WARPS_COL + 1) * 2 + warp_n});
    store(C, c[1][0], {0, 0, (block_row * WARPS_ROW + 1) * 2 + warp_m, (block_col * WARPS_COL) * 2 + warp_n});
    store(C, c[1][1], {0, 0, (block_row * WARPS_ROW + 1) * 2 + warp_m, (block_col * WARPS_COL + 1) * 2 + warp_n});
}

template <int M, int N, int K>
__global__ __launch_bounds__(256, 1) void matmul_device_2048(const kittens::gl<fp8e4m3, 1, 1, M, K> A, const kittens::gl<fp8e4m3, 1, 1, N, K> B, const kittens::gl<bf16, 1, 1, M, N> C) {
    constexpr int WARPS_COL = 2;
    constexpr int WARPS_ROW = 2;
    constexpr int BLOCK_SIZE_ROW = 128;
    constexpr int BLOCK_SIZE_COL = 128;
    constexpr int BLOCK_K = 128;
    constexpr int k_step = BLOCK_K;
    constexpr int blocks_row = M / BLOCK_SIZE_ROW; // Number of blocks along output matrix row dim
    constexpr int blocks_col = N / BLOCK_SIZE_COL; // Number of blocks along output matrix col dim
    constexpr int total_blocks_needed = blocks_row * blocks_col;
    constexpr int k_iters = K / BLOCK_K; // K iterations

    using ST_A = st_fp8e4m3<BLOCK_SIZE_ROW / 2, BLOCK_K, st_16x128_s>;
    using ST_B = st_fp8e4m3<BLOCK_SIZE_COL / 2, BLOCK_K, st_16x128_s>;

    using GL_A = kittens::gl<fp8e4m3, 1, 1, M, K>;
    using GL_B = kittens::gl<fp8e4m3, 1, 1, N, K>;
    using GL_C = kittens::gl<bf16, 1, 1, M, N>;

    using RT_A = rt_fp8e4m3<BLOCK_SIZE_ROW / 2 / WARPS_ROW, k_step>; // 128x128 = 4x2
    using RT_B = rt_fp8e4m3<BLOCK_SIZE_COL / 2 / WARPS_COL, k_step>; // 128x128 = 4x2
    using RT_C = rt_fl<BLOCK_SIZE_ROW / 2 / WARPS_ROW, BLOCK_SIZE_COL / 2 / WARPS_COL, col_l, rt_16x16_s>; // 128x128 = 4x4

    __shared__ ST_A As[2][2];
    __shared__ ST_B Bs[2][2];

    RT_C c[2][2];

    int global_block_id = blockIdx.x;

    // Original WGID.
    int wgid = global_block_id;
    const int NUM_WGS = gridDim.x;
    const int NUM_XCDS = 8;
    const int CUS_PER_XCD = 32;
    const int NUM_CUS = CUS_PER_XCD * NUM_XCDS;
    // Swizzle chiplet so that wgids are in the same XCD.
    wgid = (wgid % NUM_XCDS) * (NUM_WGS / NUM_XCDS) + (wgid / NUM_XCDS);
    // Swizzle for better L2 within the same XCD.
    const int WGM = 4;
    const int num_pid_m = (M + BLOCK_SIZE_ROW - 1) / BLOCK_SIZE_ROW;
    const int num_pid_n = (N + BLOCK_SIZE_COL - 1) / BLOCK_SIZE_COL;
    int num_wgid_in_group = WGM * num_pid_n;
    int group_id = wgid / num_wgid_in_group;
    int first_pid_m = group_id * WGM;
    int group_size_m = min(num_pid_m - first_pid_m, WGM);
    int pid_m = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    int pid_n = (wgid % num_wgid_in_group) / group_size_m;
    // Assign the tile's row/column based on the pid_m and pid_n.
    const int row = pid_m; // blockIdx.x
    const int col = pid_n; // blockIdx.y

    int curr = 0, next = 1;

    // Convert linear block ID to 2D coordinates
    int block_row = row;
    int block_col = col;
    int block_m = block_row * BLOCK_SIZE_ROW;
    int block_n = block_col * BLOCK_SIZE_COL;

    // Warp arrangement within threadblock
    int warp_m = (warpid() / WARPS_COL);
    int warp_n = (warpid() % WARPS_COL);

    {
    __builtin_amdgcn_sched_barrier(0);
    RT_A a[2];
    RT_B b[2];

    G::load(As[curr][0], A, {0, 0, block_row*WARPS_ROW, 0});
    G::load(Bs[curr][0], B, {0, 0, block_col*WARPS_COL, 0});
    G::load(Bs[curr][1], B, {0, 0, block_col*WARPS_COL+1, 0});
    G::load(As[curr][1], A, {0, 0, block_row*WARPS_ROW+1, 0});

    zero(c[0][0]);
    zero(c[0][1]);
    zero(c[1][0]);
    zero(c[1][1]);

    G::load(As[next][0], A, {0, 0, block_row*WARPS_ROW, 1});
    G::load(Bs[next][0], B, {0, 0, block_col*WARPS_COL, 1});
    G::load(Bs[next][1], B, {0, 0, block_col*WARPS_COL+1, 1});
    G::load(As[next][1], A, {0, 0, block_row*WARPS_ROW+1, 1});

    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(14)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    auto a_subtile_0 = kittens::subtile_inplace<BLOCK_SIZE_ROW / 2 / WARPS_ROW, k_step>(As[curr][0], {warp_m, 0});
    load(a[0], a_subtile_0);

    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(12)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    auto b_subtile_0 = kittens::subtile_inplace<BLOCK_SIZE_COL / 2 / WARPS_COL, k_step>(Bs[curr][0], {warp_n, 0});
    load(b[0], b_subtile_0);

    #pragma unroll
    for (int k = 0; k < k_iters - 2; ++k, curr ^= 1, next ^= 1) {
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(8)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // template<typename ST_GL, typename GL_GL, typename ST, typename RT, typename RT_A, typename RT_B, typename RT_C, ducks::coord::tile COORD=coord<ST_GL>>
        auto bs_subtile_1 = kittens::subtile_inplace<BLOCK_SIZE_COL / 2 / WARPS_COL, k_step>(Bs[curr][1], {warp_n, 0});
        // do_interleaved_cluster(As[curr][0], A, {0, 0, block_row*WARPS_ROW, k + 2}, b[1], bs_subtile_1, a[0], b[0], c[0][0]);
        G::load(As[curr][0], A, {0, 0, block_row*WARPS_ROW, k + 2});
        load(b[1], bs_subtile_1);
        mma_ABt(c[0][0], a[0], b[0], c[0][0]);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto a_subtile_1 = kittens::subtile_inplace<BLOCK_SIZE_ROW / 2 / WARPS_ROW, k_step>(As[curr][1], {warp_m, 0});
        // do_interleaved_cluster(Bs[curr][0], B, {0, 0, block_col*WARPS_COL, k + 2}, a[1], a_subtile_1, a[0], b[1], c[0][1]);
        G::load(Bs[curr][0], B, {0, 0, block_col*WARPS_COL, k + 2});
        load(a[1], a_subtile_1);
        mma_ABt(c[0][1], a[0], b[1], c[0][1]);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(8)");
        __builtin_amdgcn_s_barrier();
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto a_subtile_0 = kittens::subtile_inplace<BLOCK_SIZE_ROW / 2 / WARPS_ROW, k_step>(As[next][0], {warp_m, 0});
        // do_interleaved_cluster(Bs[curr][1], B, {0, 0, block_col*WARPS_COL+1, k + 2}, a[0], a_subtile_0, a[1], b[0], c[1][0]);
        G::load(Bs[curr][1], B, {0, 0, block_col*WARPS_COL+1, k + 2});
        load(a[0], a_subtile_0);
        mma_ABt(c[1][0], a[1], b[0], c[1][0]);

        auto b_subtile_0 = kittens::subtile_inplace<BLOCK_SIZE_COL / 2 / WARPS_COL, k_step>(Bs[next][0], {warp_n, 0});
        // do_interleaved_cluster(As[curr][1], A, {0, 0, block_row*WARPS_ROW+1, k + 2}, b[0], b_subtile_0, a[1], b[1], c[1][1]);
        G::load(As[curr][1], A, {0, 0, block_row*WARPS_ROW+1, k + 2});
        load(b[0], b_subtile_0);
        mma_ABt(c[1][1], a[1], b[1], c[1][1]);
    }

    { // EPILOGUE: k = k_iters - 2
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(8)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto b_subtile_1 = kittens::subtile_inplace<BLOCK_SIZE_COL / 2 / WARPS_COL, k_step>(Bs[curr][1], {warp_n, 0});
        load(b[1], b_subtile_1);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[0][0], a[0], b[0], c[0][0]);
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto a_subtile_1 = kittens::subtile_inplace<BLOCK_SIZE_ROW / 2 / WARPS_ROW, k_step>(As[curr][1], {warp_m, 0});
        load(a[1], a_subtile_1);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[0][1], a[0], b[1], c[0][1]);
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

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

    { // EPILOGUE: k = k_iters - 1
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto b_subtile_1 = kittens::subtile_inplace<BLOCK_SIZE_COL / 2 / WARPS_COL, k_step>(Bs[curr][1], {warp_n, 0});
        load(b[1], b_subtile_1);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[0][0], a[0], b[0], c[0][0]);
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        auto a_subtile_1 = kittens::subtile_inplace<BLOCK_SIZE_ROW / 2 / WARPS_ROW, k_step>(As[curr][1], {warp_m, 0});
        load(a[1], a_subtile_1);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[0][1], a[0], b[1], c[0][1]);
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[1][0], a[1], b[0], c[1][0]);
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        mma_ABt(c[1][1], a[1], b[1], c[1][1]);
        __builtin_amdgcn_sched_barrier(0);
    }
    __builtin_amdgcn_sched_barrier(0);
    }

    store(C, c[0][0], {0, 0, (block_row * WARPS_ROW) * 2 + warp_m, (block_col * WARPS_COL) * 2 + warp_n});
    store(C, c[0][1], {0, 0, (block_row * WARPS_ROW) * 2 + warp_m, (block_col * WARPS_COL + 1) * 2 + warp_n});
    store(C, c[1][0], {0, 0, (block_row * WARPS_ROW + 1) * 2 + warp_m, (block_col * WARPS_COL) * 2 + warp_n});
    store(C, c[1][1], {0, 0, (block_row * WARPS_ROW + 1) * 2 + warp_m, (block_col * WARPS_COL + 1) * 2 + warp_n});
}

// Rotating buffer configuration (global constant)
constexpr int ROTATING_BUFFER_COUNT = ((((1024*1024)/SIZE)*512)/SIZE)/2; // 500 MiB

// Random initialization function
template <int M, int N, int K>
void random_init(std::vector<fp8e4m3>& a_host, std::vector<fp8e4m3>& b_host, uint32_t seed = 42) {
    std::mt19937 gen(seed); // Seed for reproducibility
    std::normal_distribution<float> dis(-1.0f, 1.0f);
    for (int i = 0; i < M*K; i++) {
        a_host[i] = fp8e4m3(dis(gen));
    }
    for (int i = 0; i < N*K; i++) {
        b_host[i] = fp8e4m3(dis(gen));
    }
}

template <int M, int N, int K, int CUs>
TimingResult matmul_host(std::vector<fp8e4m3>& a, std::vector<fp8e4m3>& b, std::vector<bf16>& c,
                        int warmup_iters = 3, int timing_iters = 20) {
    constexpr int BLOCK_SIZE = M == 1024 ? 64 : M == 2048 ? 128 : 256;
    constexpr int threads_per_warp = 64;
    constexpr int warps_per_cu = 4;
    constexpr int threads_per_block = threads_per_warp * warps_per_cu;
    constexpr int threadblocks = M / BLOCK_SIZE * N / BLOCK_SIZE;

    // Use global rotating buffer configuration
    constexpr int block_count = ROTATING_BUFFER_COUNT;
    
    // Ensure input vectors have correct size
    if (a.size() != M * K) {
        fprintf(stderr, "Error: Input vector 'a' size %zu does not match expected M*K=%d\n", a.size(), M*K);
        return {0, 0, 0, 0, 0};
    }
    if (b.size() != N * K) {
        fprintf(stderr, "Error: Input vector 'b' size %zu does not match expected N*K=%d\n", b.size(), N*K);
        return {0, 0, 0, 0, 0};
    }
    
    // Resize output vector
    c.resize(M * N);
    
    // Allocate device memory (A and B use rotating buffers, C is single buffer)
    fp8e4m3 *d_a, *d_b;
    bf16 *d_c;
    hipMalloc(&d_a, block_count * M*K*sizeof(fp8e4m3));
    hipMalloc(&d_b, block_count * N*K*sizeof(fp8e4m3));
    hipMalloc(&d_c, M*N*sizeof(bf16));  // Single buffer (output only, no rotation needed)
    HipCheckError();
    
    // Pre-initialize all buffer sections with random data on host
    printf("Initializing %d rotating buffer sections (%zu MB total, A+B only)...\n",
           block_count,
           (block_count * (M*K*sizeof(fp8e4m3) + N*K*sizeof(fp8e4m3)) + M*N*sizeof(bf16)) / (1024*1024));

    for (int block = 0; block < block_count; ++block) {
        // Generate random data with different seed for each buffer
        random_init<M, N, K>(a, b, 42 + block);
        // Print the maximum value from each of a and b for this block
        fp8e4m3 max_a = a[0];
        fp8e4m3 max_b = b[0];
        #pragma omp parallel for
        for (int i = 1; i < M*K; ++i) {
            if ((float)a[i] > (float)max_a) max_a = a[i];
        }
        #pragma omp parallel for
        for (int i = 1; i < N*K; ++i) {
            if ((float)b[i] > (float)max_b) max_b = b[i];
        }
        printf("Block %d: max(a) = %f, max(b) = %f\n", block, (float)max_a, (float)max_b);

        // Copy to offset position in device memory
        hipMemcpy(d_a + block * M * K, a.data(), M*K*sizeof(fp8e4m3), hipMemcpyHostToDevice);
        hipMemcpy(d_b + block * N * K, b.data(), N*K*sizeof(fp8e4m3), hipMemcpyHostToDevice);
    }
    HipCheckError();
    printf("Buffer initialization complete.\n");
    
    // Warmup iterations using rotating buffers (A and B only)
    for (int i = 0; i < warmup_iters; i++) {
        int block_idx = i % block_count;
        fp8e4m3* d_a_current = d_a + block_idx * M * K;
        fp8e4m3* d_b_current = d_b + block_idx * N * K;

        hipMemset(d_c, 0, M*N*sizeof(bf16));

        kittens::gl<fp8e4m3, 1, 1, M, K> A(d_a_current, nullptr, nullptr, nullptr, nullptr);
        kittens::gl<fp8e4m3, 1, 1, N, K> B(d_b_current, nullptr, nullptr, nullptr, nullptr);
        kittens::gl<bf16, 1, 1, M, N> C(d_c, nullptr, nullptr, nullptr, nullptr);

        if constexpr (M == 1024 && N == 1024 && K == 1024) {
            matmul_device_1024<M, N, K><<<threadblocks, threads_per_block>>>(A, B, C);
        } else if constexpr (M == 2048 && N == 2048 && K == 2048) {
            matmul_device_2048<M, N, K><<<threadblocks, threads_per_block>>>(A, B, C);
        } else {
            matmul_device<M, N, K><<<threadblocks, threads_per_block>>>(A, B, C);
        }
        HipCheckError();
        hipDeviceSynchronize();
    }
    
    // Create HIP events for precise kernel timing
    hipEvent_t start_event, stop_event;
    hipEventCreate(&start_event);
    hipEventCreate(&stop_event);
    
    // Timed kernel-only loop with rotating buffers (A and B only)
    std::vector<float> times_ms;
    times_ms.reserve(timing_iters);
    for (int r = 0; r < timing_iters; ++r) {
        // Calculate which buffer section to use for inputs (modulo rotation)
        int block_idx = r % block_count;
        fp8e4m3* d_a_current = d_a + block_idx * M * K;
        fp8e4m3* d_b_current = d_b + block_idx * N * K;

        // Zero out the single output buffer
        hipMemset(d_c, 0, M*N*sizeof(bf16));

        // Create gl wrappers (A and B rotate, C is always the same)
        kittens::gl<fp8e4m3, 1, 1, M, K> A_current(d_a_current, nullptr, nullptr, nullptr, nullptr);
        kittens::gl<fp8e4m3, 1, 1, N, K> B_current(d_b_current, nullptr, nullptr, nullptr, nullptr);
        kittens::gl<bf16, 1, 1, M, N> C_current(d_c, nullptr, nullptr, nullptr, nullptr);

        if constexpr (M == 1024 && N == 1024 && K == 1024) {
            hipEventRecord(start_event, 0);
            matmul_device_1024<M, N, K><<<threadblocks, threads_per_block>>>(A_current, B_current, C_current);
            hipEventRecord(stop_event, 0);
            hipEventSynchronize(stop_event);
        } else if constexpr (M == 2048 && N == 2048 && K == 2048) {
            hipEventRecord(start_event, 0);
            matmul_device_2048<M, N, K><<<threadblocks, threads_per_block>>>(A_current, B_current, C_current);
            hipEventRecord(stop_event, 0);
            hipEventSynchronize(stop_event);
        } else {
            hipEventRecord(start_event, 0);
            matmul_device<M, N, K><<<threadblocks, threads_per_block>>>(A_current, B_current, C_current);
            hipEventRecord(stop_event, 0);
            hipEventSynchronize(stop_event);
        }
        float ms = 0.0f;
        hipEventElapsedTime(&ms, start_event, stop_event);
        times_ms.push_back(ms);
        HipCheckError();
    }
    
    // Calculate best and average times
    float sum_ms = 0.f, best_ms = 1e30f;
    for (float t : times_ms) { 
        sum_ms += t; 
        best_ms = std::min(best_ms, t); 
    }
    float avg_ms = sum_ms / times_ms.size();
    
    // Calculate TFLOPS (2*M*N*K operations)
    double total_ops = 2.0 * M * N * K;
    double best_tflops = (total_ops / (best_ms * 1e-3)) / 1e12;
    double avg_tflops = (total_ops / (avg_ms * 1e-3)) / 1e12;
    
    // Cleanup events
    hipEventDestroy(start_event);
    hipEventDestroy(stop_event);
    HipCheckError();

    // Copy result back to host (single C buffer, no offset needed)
    hipMemcpy(c.data(), d_c, M*N*sizeof(bf16), hipMemcpyDeviceToHost);
    HipCheckError();
    
    // Free device memory
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
    HipCheckError();
    
    return {best_ms, avg_ms, best_tflops, avg_tflops, timing_iters};
}

int main() {
    // Reduced problem size for faster timing
    constexpr int M = SIZE;  // 256 threadblocks needed for 2048x2048
    constexpr int N = SIZE;  
    constexpr int K = SIZE;  // Smaller K for reasonable timing
    constexpr int CUs = 256; // 256 threadblocks (1 outer iteration)
    
    // Timing parameters to keep total runtime reasonable  
    constexpr int warmup_iters = 500;
    constexpr int timing_iters = 100;

    printf("Matrix dimensions: %dx%dx%d, CUs: %d\n", M, N, K, CUs);
    printf("Warmup iterations: %d, Timing iterations: %d\n\n", warmup_iters, timing_iters);

    // Initialize input matrices (will be initialized inside matmul_host)
    std::vector<fp8e4m3> a_host(M*K);
    std::vector<fp8e4m3> b_host(N*K);
    std::vector<bf16> c_ref(M*N);
    std::vector<bf16> c_host(M*N);

    // Compute test result with timing (data will be generated with rotating buffers)
    printf("Running optimized kernel (matmul_device)...\n");
    TimingResult host_timing = matmul_host<M, N, K, CUs>(a_host, b_host, c_host, warmup_iters, timing_iters);

    // Initialize data for reference computation (match the buffer used in last timing iteration)
    int last_buffer_idx = (timing_iters - 1) % ROTATING_BUFFER_COUNT;
    random_init<M, N, K>(a_host, b_host, 42 + last_buffer_idx);

    // Compute reference result with timing
    printf("Running reference kernel (matmul_device_ref)...\n");
    // TimingResult ref_timing = matmul_ref<M, N, K, CUs>(a_host, b_host, c_ref, warmup_iters, timing_iters);
    // Reference CPU implementation of matmul with OpenMP
    auto matmul_ref = [](const std::vector<fp8e4m3>& a_host, const std::vector<fp8e4m3>& b_host, std::vector<bf16>& c_ref, int timing_iters) -> TimingResult {
        TimingResult result{};
        // Only time the main loop, not the warmup
        double best_time_ms = 1e30, total_time_ms = 0.0;
        int M = a_host.size() / K;
        int N = b_host.size() / K;
        // Timing
        for (int t = 0; t < timing_iters; ++t) {
            auto t0 = std::chrono::high_resolution_clock::now();
            #pragma omp parallel for
            for (int row = 0; row < M; ++row) {
                for (int col = 0; col < N; ++col) {
                    float acc = 0.0f;
                    for (int k = 0; k < K; ++k) {
                        acc += float(a_host[row * K + k]) * float(b_host[col * K + k]);
                    }
                    c_ref[row * N + col] = bf16(acc);
                }
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (ms < best_time_ms) best_time_ms = ms;
            total_time_ms += ms;
        }
        double avg_time_ms = total_time_ms / timing_iters;
        double tflops = 2.0 * M * N * K / 1e12;
        result.best_time_ms = best_time_ms;
        result.avg_time_ms = avg_time_ms;
        result.best_tflops = tflops / (best_time_ms / 1000.0);
        result.avg_tflops = tflops / (avg_time_ms / 1000.0);
        return result;
    };

    TimingResult ref_timing = matmul_ref(a_host, b_host, c_ref, 1);

    bool success = true;
    // Compare GPU result (c_host) with CPU reference (c_ref)
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            // c_host is row major: [row*N + col]
            // c_ref is row major: [row*N + col]
            float c_val = float(c_host[row * N + col]);
            float c_ref_val = float(c_ref[row * N + col]);
            float diff = std::abs(c_val - c_ref_val);
            float threshold = c_ref_val * 0.01f;
            if (diff > threshold) {
                printf("Mismatch at (row=%d, col=%d): c_host = %f, c_ref = %f, diff = %f\n", row, col, c_val, c_ref_val, diff);
                success = false;
                break;
            }
        }
        if (!success) {
            break;
        }
    }
    // Performance comparison and results
    printf("\n=== PERFORMANCE RESULTS ===\n");
    
    printf("Reference kernel (matmul_device_ref):\n");
    printf("  Kernel time (best): %.3f ms,  TFLOPS: %.2f\n", ref_timing.best_time_ms, ref_timing.best_tflops);
    printf("  Kernel time (avg ): %.3f ms,  TFLOPS: %.2f\n", ref_timing.avg_time_ms, ref_timing.avg_tflops);
    
    printf("\nOptimized kernel (matmul_device):\n");
    printf("  Kernel time (best): %.3f ms,  TFLOPS: %.2f\n", host_timing.best_time_ms, host_timing.best_tflops);
    printf("  Kernel time (avg ): %.3f ms,  TFLOPS: %.2f\n", host_timing.avg_time_ms, host_timing.avg_tflops);
    
    printf("\nSpeedup (best): %.2fx\n", ref_timing.best_time_ms / host_timing.best_time_ms);
    printf("Speedup (avg ): %.2fx\n", ref_timing.avg_time_ms / host_timing.avg_time_ms);
    
    if (success) {
        printf("\nCorrectness: PASSED\n");
    } else {
        printf("\nCorrectness: FAILED\n");
    }

    return 0;
}

