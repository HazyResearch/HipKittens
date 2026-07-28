/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * License for AMD contributions = MIT. See LICENSE for more information
*************************************************************************/

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;

#include "./utils.cpp"

constexpr int NUM_WARPS   = 8;
constexpr int WARPS_ROW   = 2;
constexpr int WARPS_COL   = 4;
constexpr int BLOCK_M     = 128;
constexpr int BLOCK_N     = 256;
constexpr int BLOCK_K     = 128;
constexpr int REG_M       = BLOCK_M / 4;
constexpr int REG_N       = BLOCK_N / 4;
constexpr int MFMA_K      = 32;
constexpr int SCALE_BLOCK = 128;
constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
constexpr size_t SMEM_BYTES = (BLOCK_M * BLOCK_K + BLOCK_N * BLOCK_K) * sizeof(fp8e4m3);
static_assert(SMEM_BYTES <= 64 * 1024, "SMEM_BYTES exceeds gfx942 LDS size (64 KB)");

using G = kittens::group<NUM_WARPS>;

template <int M, int N, int K, typename OType>
struct micro_globals {
    gl<fp8e4m3, 1, 1, M, K> a;
    gl<fp8e4m3, 1, 1, N, K> b;
    gl<OType,    1, 1, M, N> c;
    gl<float,    1, 1, -1, -1> scale_a;
    gl<float,    1, 1, -1, -1> scale_b;
    dim3 grid()  { return dim3((N / BLOCK_N) * (M / BLOCK_M)); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return SMEM_BYTES; }
};

template <int M, int N, int K, bool IS_1D2D, typename OType>
__global__ __launch_bounds__(NUM_THREADS, 2)
void micro_tk(const micro_globals<M, N, K, OType> g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st<fp8e4m3, BLOCK_M, BLOCK_K> (&As) = al.allocate<st<fp8e4m3, BLOCK_M, BLOCK_K>>();
    st<fp8e4m3, BLOCK_N, BLOCK_K> (&Bs) = al.allocate<st<fp8e4m3, BLOCK_N, BLOCK_K>>();

    rt<fp8e4m3, REG_M, MFMA_K> at[5];
    rt<fp8e4m3, REG_N, MFMA_K> bt[3];
    rt_fl<REG_M, REG_N, ducks::rt_layout::col> C_accum[2];
    rt_fl<REG_M, REG_N, ducks::rt_layout::col> partial[2];
    for (int i = 0; i < 2; i++) { zero(C_accum[i]); }

    int wgid = (blockIdx.y * gridDim.x) + blockIdx.x;
    const int NUM_WGS = gridDim.x * gridDim.y;
    constexpr int WGM = 4;
    wgid = chiplet_transform_chunked(wgid, NUM_WGS, NUM_XCDS, WGM * WGM);

    const int num_pid_m = ceil_div(M, BLOCK_M);
    const int num_pid_n = ceil_div(N, BLOCK_N);
    int num_wgid_in_group = WGM * num_pid_n;
    int group_id = wgid / num_wgid_in_group;
    int first_pid_m = group_id * WGM;
    int group_size_m = min(num_pid_m - first_pid_m, WGM);
    int pid_m = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    int pid_n = (wgid % num_wgid_in_group) / group_size_m;
    const int row = pid_m;
    const int col = pid_n;

    const int warp_id = warpid();
    const int warp_row = warp_id / WARPS_COL;
    const int warp_col = warp_id % WARPS_COL;

    volatile int num_k_steps_v = ceil_div(K, BLOCK_K);
    const int num_k_steps = num_k_steps_v;

    const float *sa_block = g.scale_a.raw_ptr + row * BLOCK_M;

    const int local_m0 = warp_row * REG_M;
    const int local_m1 = (warp_row + WARPS_ROW) * REG_M;

    const uint32_t sa_range = (uint32_t)((M - row * BLOCK_M) * 4);

    const int sb_block0 = col * (BLOCK_N / SCALE_BLOCK) + warp_col / 2;
    const float *sb_base = g.scale_b.raw_ptr + sb_block0 * num_k_steps;
    int32x4_lds_t sb_srsrc = make_buf_res((const void*)sb_base, (uint32_t)num_k_steps * 4);

    const int sb_col0 = col * BLOCK_N;
    const float *sb_block = g.scale_b.raw_ptr + sb_col0;
    const int local_n = warp_col * REG_N;
    const uint32_t sb_range = (uint32_t)((N - sb_col0) * 4);

    G::load(As, g.a, {0, 0, row, 0});
    G::load(Bs, g.b, {0, 0, col, 0});

    float sb_cur = llvm_amdgcn_s_buffer_load_f32(sb_srsrc, 0, 0);
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();

    if (warp_row == 1) {
        __builtin_amdgcn_s_barrier();
    }

    #pragma unroll 2
    for (int k_step = 0; k_step < num_k_steps - 1; ++k_step) {

        constexpr int A_ELEMS_PER_THREAD = (BLOCK_M * BLOCK_K) / NUM_THREADS;
        constexpr int B_ELEMS_PER_THREAD = (BLOCK_N * BLOCK_K) / NUM_THREADS;
        float4 a_buffer_next[A_ELEMS_PER_THREAD * sizeof(fp8e4m3) / sizeof(float4)];
        float4 b_buffer_next[B_ELEMS_PER_THREAD * sizeof(fp8e4m3) / sizeof(float4)];

        zero(partial[0]); zero(partial[1]);

        float sa_reg0[REG_M / 16 * 4];
        float sa_reg1[REG_M / 16 * 4];

        load_global_to_register_buffer<2, false, NUM_THREADS>(b_buffer_next, B_ELEMS_PER_THREAD, g.b, {0, 0, col, k_step + 1}, Bs);
        float sb_next;
        float sb_reg[REG_N / 16];
        load(at[0], subtile_inplace<REG_M, MFMA_K>(As, {warp_row, 0}));
        load(at[1], subtile_inplace<REG_M, MFMA_K>(As, {warp_row + WARPS_ROW, 0}));
        load(bt[0], subtile_inplace<REG_N, MFMA_K>(Bs, {warp_col, 0}));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(partial[0], at[0], bt[0], partial[0]);
        mma_ABt(partial[1], at[1], bt[0], partial[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        load(bt[1], subtile_inplace<REG_N, MFMA_K>(Bs, {warp_col, 1}));
        load(at[2], subtile_inplace<REG_M, MFMA_K>(As, {warp_row, 1}));
        load(at[3], subtile_inplace<REG_M, MFMA_K>(As, {warp_row + WARPS_ROW, 1}));
        load(bt[0], subtile_inplace<REG_N, MFMA_K>(Bs, {warp_col, 2}));
        load(at[0], subtile_inplace<REG_M, MFMA_K>(As, {warp_row, 2}));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(partial[0], at[2], bt[1], partial[0]);
        mma_ABt(partial[1], at[3], bt[1], partial[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        load_global_to_register_buffer<2, false, NUM_THREADS>(a_buffer_next, A_ELEMS_PER_THREAD, g.a, {0, 0, row, k_step + 1}, As);
        load(at[1], subtile_inplace<REG_M, MFMA_K>(As, {warp_row + WARPS_ROW, 2}));
        load(bt[2], subtile_inplace<REG_N, MFMA_K>(Bs, {warp_col, 3}));
        load(at[4], subtile_inplace<REG_M, MFMA_K>(As, {warp_row, 3}));
        load(at[3], subtile_inplace<REG_M, MFMA_K>(As, {warp_row + WARPS_ROW, 3}));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_s_setprio(1);
        mma_ABt(partial[0], at[0], bt[0], partial[0]);
        mma_ABt(partial[1], at[1], bt[0], partial[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        asm volatile("s_waitcnt lgkmcnt(0)");
        store_register_buffer_to_shared<NUM_THREADS>(As, a_buffer_next);
        load_scale_global_reg<REG_M / 16>(sa_reg0, sa_block + k_step * M, local_m0, sa_range);
        load_scale_global_reg<REG_M / 16>(sa_reg1, sa_block + k_step * M, local_m1, sa_range);
        if constexpr (IS_1D2D)
            sb_next = llvm_amdgcn_s_buffer_load_f32(sb_srsrc, (k_step + 1) * 4, 0);
        else
            load_scaleB_global_reg<REG_N / 16>(sb_reg, sb_block + k_step * N, local_n, sb_range);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_s_setprio(1);
        mma_ABt(partial[0], at[4], bt[2], partial[0]);
        mma_ABt(partial[1], at[3], bt[2], partial[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        store_register_buffer_to_shared<NUM_THREADS>(Bs, b_buffer_next);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        if constexpr (IS_1D2D) {
            apply_block_scale_1d2d(C_accum[0], partial[0], sa_reg0, sb_cur);
            apply_block_scale_1d2d(C_accum[1], partial[1], sa_reg1, sb_cur);
            sb_cur = sb_next;
        } else {
            apply_block_scale_1d1d(C_accum[0], partial[0], sa_reg0, sb_reg);
            apply_block_scale_1d1d(C_accum[1], partial[1], sa_reg1, sb_reg);
        }
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

    }

    zero(partial[0]); zero(partial[1]);
    __builtin_amdgcn_sched_barrier(0);
    load(bt[0], subtile_inplace<REG_N, MFMA_K>(Bs, {warp_col, 0}));
    load(at[0], subtile_inplace<REG_M, MFMA_K>(As, {warp_row, 0}));
    load(at[1], subtile_inplace<REG_M, MFMA_K>(As, {warp_row + WARPS_ROW, 0}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    __builtin_amdgcn_s_setprio(1);
    mma_ABt(partial[0], at[0], bt[0], partial[0]);
    mma_ABt(partial[1], at[1], bt[0], partial[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    load(bt[1], subtile_inplace<REG_N, MFMA_K>(Bs, {warp_col, 1}));
    load(at[2], subtile_inplace<REG_M, MFMA_K>(As, {warp_row, 1}));
    load(at[3], subtile_inplace<REG_M, MFMA_K>(As, {warp_row + WARPS_ROW, 1}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    __builtin_amdgcn_s_setprio(1);
    mma_ABt(partial[0], at[2], bt[1], partial[0]);
    mma_ABt(partial[1], at[3], bt[1], partial[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    load(bt[0], subtile_inplace<REG_N, MFMA_K>(Bs, {warp_col, 2}));
    load(at[0], subtile_inplace<REG_M, MFMA_K>(As, {warp_row, 2}));
    load(at[1], subtile_inplace<REG_M, MFMA_K>(As, {warp_row + WARPS_ROW, 2}));
    load(bt[1], subtile_inplace<REG_N, MFMA_K>(Bs, {warp_col, 3}));
    load(at[2], subtile_inplace<REG_M, MFMA_K>(As, {warp_row, 3}));
    load(at[3], subtile_inplace<REG_M, MFMA_K>(As, {warp_row + WARPS_ROW, 3}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    __builtin_amdgcn_s_setprio(1);
    mma_ABt(partial[0], at[0], bt[0], partial[0]);
    mma_ABt(partial[1], at[1], bt[0], partial[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    __builtin_amdgcn_s_setprio(1);
    mma_ABt(partial[0], at[2], bt[1], partial[0]);
    mma_ABt(partial[1], at[3], bt[1], partial[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    {
        const int last = num_k_steps - 1;
        float sa_reg0[REG_M / 16 * 4];
        float sa_reg1[REG_M / 16 * 4];
        load_scale_global_reg<REG_M / 16>(sa_reg0, sa_block + last * M, local_m0, sa_range);
        load_scale_global_reg<REG_M / 16>(sa_reg1, sa_block + last * M, local_m1, sa_range);
        if constexpr (IS_1D2D) {
            apply_block_scale_1d2d(C_accum[0], partial[0], sa_reg0, sb_cur);
            apply_block_scale_1d2d(C_accum[1], partial[1], sa_reg1, sb_cur);
        } else {
            float sb_reg[REG_N / 16];
            load_scaleB_global_reg<REG_N / 16>(sb_reg, sb_block + last * N, local_n, sb_range);
            apply_block_scale_1d1d(C_accum[0], partial[0], sa_reg0, sb_reg);
            apply_block_scale_1d1d(C_accum[1], partial[1], sa_reg1, sb_reg);
        }
    }

    if (warp_row == 0) {
        __builtin_amdgcn_s_barrier();
    }

    store_output(g.c.raw_ptr, C_accum[0], row * 4 + warp_row,             col * 4 + warp_col, M, N);
    store_output(g.c.raw_ptr, C_accum[1], row * 4 + warp_row + WARPS_ROW, col * 4 + warp_col, M, N);
}

constexpr int PY_M = 8192;
constexpr int PY_N = 8192;
constexpr int PY_K = 8192;

template <typename OType>
using py_globals = micro_globals<PY_M, PY_N, PY_K, OType>;

template <bool IS_1D2D, typename OType>
void dispatch_micro(py_globals<OType> g) {
    hipFuncSetAttribute((void *)micro_tk<PY_M, PY_N, PY_K, IS_1D2D, OType>,
                        hipFuncAttributeMaxDynamicSharedMemorySize, g.dynamic_shared_memory());
    micro_tk<PY_M, PY_N, PY_K, IS_1D2D, OType>
        <<<g.grid(), g.block(), g.dynamic_shared_memory()>>>(g);
    hipDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "cdna3 blockwise fp8 gemm";
    py::bind_function<dispatch_micro<true, float>>(m, "gemm_1d2d",
        &py_globals<float>::a, &py_globals<float>::b, &py_globals<float>::c,
        &py_globals<float>::scale_a, &py_globals<float>::scale_b);
    py::bind_function<dispatch_micro<false, float>>(m, "gemm_1d1d",
        &py_globals<float>::a, &py_globals<float>::b, &py_globals<float>::c,
        &py_globals<float>::scale_a, &py_globals<float>::scale_b);
    py::bind_function<dispatch_micro<true, bf16>>(m, "gemm_1d2d_bf16",
        &py_globals<bf16>::a, &py_globals<bf16>::b, &py_globals<bf16>::c,
        &py_globals<bf16>::scale_a, &py_globals<bf16>::scale_b);
    py::bind_function<dispatch_micro<false, bf16>>(m, "gemm_1d1d_bf16",
        &py_globals<bf16>::a, &py_globals<bf16>::b, &py_globals<bf16>::c,
        &py_globals<bf16>::scale_a, &py_globals<bf16>::scale_b);
}
