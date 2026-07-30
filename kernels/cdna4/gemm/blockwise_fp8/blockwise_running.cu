#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

using namespace kittens;

#include "./utils.cpp"

template <typename OType, bool IS_1D2D>
__global__ __launch_bounds__(NUM_THREADS, 2)
void micro_tk(micro_globals<kittens::fp8e4m3, kittens::fp8e4m3, OType> g) {
    const auto A = g.a;
    const auto B = g.b;
    const auto C = g.c;
    const float *scale_A = g.scale_a.raw_ptr;
    const float *scale_B = g.scale_b.raw_ptr;
    const int M = (int)g.c.rows(), N = (int)g.c.cols(), K = (int)g.a.cols();
    const int k_iters = K / BLOCK_K;
    const int scale_K = K / SCALE_BLOCK;

    using ST_A = kittens::st_fp8e4m3<HALF_ROW, BLOCK_K, kittens::st_16x128_s>;
    using ST_B = kittens::st_fp8e4m3<HALF_COL, BLOCK_K, kittens::st_16x128_s>;
    using RT_A = kittens::rt_fp8e4m3<REG_M, MFMA_K>;
    using RT_B = kittens::rt_fp8e4m3<REG_N, MFMA_K>;
    using RT_C = kittens::rt_fl<REG_M, REG_N, kittens::col_l, kittens::rt_16x16_s>;

    __shared__ ST_A As[2][2];
    __shared__ ST_B Bs[2][2];
    __shared__ __align__(16) float smem_sa_prev[BLOCK_M];
    __shared__ __align__(16) float smem_sa_curr[BLOCK_M];
    __shared__ __align__(16) float smem_a_ratio[2][BLOCK_M];

    RT_A a;
    RT_B b0, b1;
    RT_C cA, cB, cC, cD;
    kittens::zero(cA); kittens::zero(cB); kittens::zero(cC); kittens::zero(cD);

    const int tiles_M = (M + BLOCK_M - 1) / BLOCK_M;
    const int tiles_N = (N + BLOCK_N - 1) / BLOCK_N;
    constexpr int WGM = 8;
    int wgid = kittens::chiplet_transform_chunked((int)blockIdx.x, (int)gridDim.x, kittens::NUM_XCDS, WGM * WGM);
    const int num_wgid_in_group = WGM * tiles_N;
    const int group_id = wgid / num_wgid_in_group;
    const int first_pid_m = group_id * WGM;
    const int group_size_m = min(tiles_M - first_pid_m, WGM);
    const int block_row = __builtin_amdgcn_readfirstlane(first_pid_m + ((wgid % num_wgid_in_group) % group_size_m));
    const int block_col = __builtin_amdgcn_readfirstlane((wgid % num_wgid_in_group) / group_size_m);
    const int block_m = block_row * BLOCK_M;
    const int block_n = block_col * BLOCK_N;

    const int warp_m = __builtin_amdgcn_readfirstlane(kittens::warpid() / WARPS_COL);
    const int warp_n = __builtin_amdgcn_readfirstlane(kittens::warpid() % WARPS_COL);
    const int tid = threadIdx.x;

    const int sb_h0_idx = block_n / SCALE_BLOCK;
    const int sb_h1_idx = sb_h0_idx + 1;
    const int local_n0 = block_n + warp_n * REG_N;
    const int local_n1 = block_n + HALF_COL + warp_n * REG_N;
    const int a_row_h0 = warp_m * REG_M;
    const int a_row_h1 = HALF_ROW + warp_m * REG_M;

    using T = kittens::fp8e4m3;
    constexpr int bpt = ST_A::underlying_subtile_bytes_per_thread;
    constexpr int bpm = bpt * NUM_THREADS;
    constexpr int copies_A = HALF_ROW * BLOCK_K * sizeof(T) / bpm;
    constexpr int copies_B = HALF_COL * BLOCK_K * sizeof(T) / bpm;
    uint32_t sw_A[copies_A], sw_B[copies_B];
    G::prefill_swizzled_offsets(As[0][0], A, sw_A);
    G::prefill_swizzled_offsets(Bs[0][0], B, sw_B);

    const T *a_base = (const T *)&A[{0, 0, 0, 0}];
    const T *b_base = (const T *)&B[{0, 0, 0, 0}];
    const int a_row_stride = A.template stride<2>() * sizeof(T);
    const int b_row_stride = B.template stride<2>() * sizeof(T);
    const bool a_partial_m = (block_row + 1) * BLOCK_M > M;
    const bool b_partial_n = (block_col + 1) * BLOCK_N > N;
    kittens::i32x4 a_srd = a_partial_m
        ? kittens::make_srsrc(a_base, M * a_row_stride, 0)
        : kittens::make_srsrc(a_base, M * a_row_stride, a_row_stride);
    kittens::i32x4 b_srd = b_partial_n
        ? kittens::make_srsrc(b_base, N * b_row_stride, 0)
        : kittens::make_srsrc(b_base, N * b_row_stride, b_row_stride);

    const int wid = kittens::warpid() % NUM_WARPS;
    constexpr int elem_per_warp = (16 / sizeof(T)) * kittens::WARP_THREADS;
    uint32_t a_lds[2][2], b_lds[2][2];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++) {
            a_lds[i][j] = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
                reinterpret_cast<uintptr_t>(&As[i][j].data[0]) + wid * elem_per_warp * sizeof(T)));
            b_lds[i][j] = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
                reinterpret_cast<uintptr_t>(&Bs[i][j].data[0]) + wid * elem_per_warp * sizeof(T)));
        }

    const float *sb0_p = scale_B + sb_h0_idx * scale_K;
    const float *sb1_p = scale_B + sb_h1_idx * scale_K;

    kittens::i32x4 sa_srd = kittens::make_srsrc((const void *)scale_A, (uint32_t)(scale_K * M * (int)sizeof(float)));
    const int sa_warp = tid / kittens::WARP_THREADS;
    const int sa_lane = tid % kittens::WARP_THREADS;
    const uint32_t sa_curr_lds_warp = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(&smem_sa_curr[sa_warp * kittens::WARP_THREADS])));

    int tic = 0, toc = 1;
    int rtic = 0, rtoc = 1;

    G::load(Bs[tic][0], B, {0, 0, block_col * 2, 0}, sw_B, b_srd, b_base, b_lds[tic][0]);
    G::load(As[tic][0], A, {0, 0, block_row * 2, 0}, sw_A, a_srd, a_base, a_lds[tic][0]);
    G::load(Bs[tic][1], B, {0, 0, block_col * 2 + 1, 0}, sw_B, b_srd, b_base, b_lds[tic][1]);
    G::load(As[tic][1], A, {0, 0, block_row * 2 + 1, 0}, sw_A, a_srd, a_base, a_lds[tic][1]);

    if (warp_m == 1) __builtin_amdgcn_s_barrier();
    asm volatile("s_waitcnt vmcnt(4)");
    __builtin_amdgcn_s_barrier();

    G::load(As[toc][0], A, {0, 0, block_row * 2, 1}, sw_A, a_srd, a_base, a_lds[toc][0]);
    G::load(Bs[toc][0], B, {0, 0, block_col * 2, 1}, sw_B, b_srd, b_base, b_lds[toc][0]);
    G::load(Bs[toc][1], B, {0, 0, block_col * 2 + 1, 1}, sw_B, b_srd, b_base, b_lds[toc][1]);
    asm volatile("s_waitcnt vmcnt(6)");
    __builtin_amdgcn_s_barrier();

    if (tid < BLOCK_M) smem_sa_prev[tid] = 1.0f;
    load_scales_to_curr<BLOCK_M>(sa_srd, sa_curr_lds_warp, tid, 0, M, block_m, sa_warp, sa_lane);
    float curr_sb_h0, curr_sb_h1, prev_sb_h0, prev_sb_h1;
    ColScale<RT_C::width> curr_cs0, curr_cs1, prev_cs0, prev_cs1;
    if constexpr (IS_1D2D) {
        curr_sb_h0 = load_scaleB_scalar(sb0_p, 0);
        curr_sb_h1 = load_scaleB_scalar(sb1_p, 0);
    } else {
        curr_cs0 = load_scaleB_col<RT_C>(scale_B + 0 * N, local_n0, N);
        curr_cs1 = load_scaleB_col<RT_C>(scale_B + 0 * N, local_n1, N);
    }
    asm volatile("s_waitcnt vmcnt(0)");
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();

    compute_a_ratios_and_promote<BLOCK_M>(tid, smem_sa_prev, smem_sa_curr, smem_a_ratio[rtic]);
    __builtin_amdgcn_s_barrier();

    if constexpr (IS_1D2D) {
        prev_sb_h0 = 1.0f; prev_sb_h1 = 1.0f;
    } else {
        #pragma unroll
        for (int j = 0; j < RT_C::width; j++) {
            prev_cs0.v[j] = 1.0f;
            prev_cs1.v[j] = 1.0f;
        }
    }

    #pragma unroll 2
    for (int k = 0; k < k_iters - 2; k++, tic ^= 1, toc ^= 1, rtic ^= 1, rtoc ^= 1) {
        float sb_ratio_h0; ColScale<RT_C::width> cr0;
        if constexpr (IS_1D2D) sb_ratio_h0 = prev_sb_h0 / curr_sb_h0;
        else                   cr0 = col_scale_ratio(prev_cs0, curr_cs0);

        auto rr_h0 = load_row_ratio<RT_C>(smem_a_ratio[rtic], a_row_h0);
        load_scales_to_curr<BLOCK_M>(sa_srd, sa_curr_lds_warp, tid, k + 1, M, block_m, sa_warp, sa_lane);
        float nxt_sb_h0, nxt_sb_h1;
        ColScale<RT_C::width> nxt_cs0, nxt_cs1;
        if constexpr (IS_1D2D) {
            nxt_sb_h0 = load_scaleB_scalar(sb0_p, k + 1);
            nxt_sb_h1 = load_scaleB_scalar(sb1_p, k + 1);
        } else {
            nxt_cs0 = load_scaleB_col<RT_C>(scale_B + (k + 1) * N, local_n0, N);
            nxt_cs1 = load_scaleB_col<RT_C>(scale_B + (k + 1) * N, local_n1, N);
        }

        kittens::load(b0, kittens::subtile_inplace<REG_N, BLOCK_K>(Bs[tic][0], {warp_n, 0}));
        kittens::load(a,  kittens::subtile_inplace<REG_M, BLOCK_K>(As[tic][0], {warp_m, 0}));
        G::load(As[toc][1], A, {0, 0, block_row * 2 + 1, k + 1}, sw_A, a_srd, a_base, a_lds[toc][1]);
        asm volatile("s_waitcnt lgkmcnt(8)");
        __builtin_amdgcn_s_barrier();

        if constexpr (IS_1D2D) apply_row_ratio_sb(cA, rr_h0, sb_ratio_h0);
        else                   apply_row_col_ratio(cA, rr_h0, cr0);
        __builtin_amdgcn_s_setprio(2);
        kittens::mma_ABt(cA, a, b0, cA);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        float sb_ratio_h1; ColScale<RT_C::width> cr1;
        if constexpr (IS_1D2D) sb_ratio_h1 = prev_sb_h1 / curr_sb_h1;
        else                   cr1 = col_scale_ratio(prev_cs1, curr_cs1);
        kittens::load(b1, kittens::subtile_inplace<REG_N, BLOCK_K>(Bs[tic][1], {warp_n, 0}));
        G::load(As[tic][0], A, {0, 0, block_row * 2, k + 2}, sw_A, a_srd, a_base, a_lds[tic][0]);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        if constexpr (IS_1D2D) apply_row_ratio_sb(cB, rr_h0, sb_ratio_h1);
        else                   apply_row_col_ratio(cB, rr_h0, cr1);
        __builtin_amdgcn_s_setprio(2);
        kittens::mma_ABt(cB, a, b1, cB);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        kittens::load(a, kittens::subtile_inplace<REG_M, BLOCK_K>(As[tic][1], {warp_m, 0}));
        G::load(Bs[tic][0], B, {0, 0, block_col * 2, k + 2}, sw_B, b_srd, b_base, b_lds[tic][0]);
        auto rr_h1 = load_row_ratio<RT_C>(smem_a_ratio[rtic], a_row_h1);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        if constexpr (IS_1D2D) apply_row_ratio_sb(cC, rr_h1, sb_ratio_h0);
        else                   apply_row_col_ratio(cC, rr_h1, cr0);
        __builtin_amdgcn_s_setprio(2);
        kittens::mma_ABt(cC, a, b0, cC);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        G::load(Bs[tic][1], B, {0, 0, block_col * 2 + 1, k + 2}, sw_B, b_srd, b_base, b_lds[tic][1]);
        asm volatile("s_waitcnt vmcnt(6)");
        compute_a_ratios_and_promote<BLOCK_M>(tid, smem_sa_prev, smem_sa_curr, smem_a_ratio[rtoc]);
        __builtin_amdgcn_s_barrier();

        if constexpr (IS_1D2D) apply_row_ratio_sb(cD, rr_h1, sb_ratio_h1);
        else                   apply_row_col_ratio(cD, rr_h1, cr1);
        __builtin_amdgcn_s_setprio(2);
        kittens::mma_ABt(cD, a, b1, cD);
        __builtin_amdgcn_s_setprio(0);

        if constexpr (IS_1D2D) {
            prev_sb_h0 = curr_sb_h0; prev_sb_h1 = curr_sb_h1;
            curr_sb_h0 = nxt_sb_h0; curr_sb_h1 = nxt_sb_h1;
        } else {
            prev_cs0 = curr_cs0; prev_cs1 = curr_cs1;
            curr_cs0 = nxt_cs0; curr_cs1 = nxt_cs1;
        }
        __builtin_amdgcn_s_barrier();
    }

    {
        int k = k_iters - 2;
        float sb_ratio_h0, sb_ratio_h1;
        ColScale<RT_C::width> cr0, cr1;
        if constexpr (IS_1D2D) {
            sb_ratio_h0 = prev_sb_h0 / curr_sb_h0;
            sb_ratio_h1 = prev_sb_h1 / curr_sb_h1;
        } else {
            cr0 = col_scale_ratio(prev_cs0, curr_cs0);
            cr1 = col_scale_ratio(prev_cs1, curr_cs1);
        }
        load_scales_to_curr<BLOCK_M>(sa_srd, sa_curr_lds_warp, tid, k + 1, M, block_m, sa_warp, sa_lane);
        float nxt_sb_h0, nxt_sb_h1;
        ColScale<RT_C::width> nxt_cs0, nxt_cs1;
        if constexpr (IS_1D2D) {
            nxt_sb_h0 = load_scaleB_scalar(sb0_p, k + 1);
            nxt_sb_h1 = load_scaleB_scalar(sb1_p, k + 1);
        } else {
            nxt_cs0 = load_scaleB_col<RT_C>(scale_B + (k + 1) * N, local_n0, N);
            nxt_cs1 = load_scaleB_col<RT_C>(scale_B + (k + 1) * N, local_n1, N);
        }
        asm volatile("s_waitcnt vmcnt(0)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        kittens::load(b0, kittens::subtile_inplace<REG_N, BLOCK_K>(Bs[tic][0], {warp_n, 0}));
        kittens::load(a,  kittens::subtile_inplace<REG_M, BLOCK_K>(As[tic][0], {warp_m, 0}));
        G::load(As[toc][1], A, {0, 0, block_row * 2 + 1, k + 1}, sw_A, a_srd, a_base, a_lds[toc][1]);
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        if constexpr (IS_1D2D) apply_row_ratio_sb(cA, load_row_ratio<RT_C>(smem_a_ratio[rtic], a_row_h0), sb_ratio_h0);
        else                   apply_row_col_ratio(cA, load_row_ratio<RT_C>(smem_a_ratio[rtic], a_row_h0), cr0);
        __builtin_amdgcn_s_setprio(2);
        kittens::mma_ABt(cA, a, b0, cA);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        kittens::load(b1, kittens::subtile_inplace<REG_N, BLOCK_K>(Bs[tic][1], {warp_n, 0}));
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        if constexpr (IS_1D2D) apply_row_ratio_sb(cB, load_row_ratio<RT_C>(smem_a_ratio[rtic], a_row_h0), sb_ratio_h1);
        else                   apply_row_col_ratio(cB, load_row_ratio<RT_C>(smem_a_ratio[rtic], a_row_h0), cr1);
        __builtin_amdgcn_s_setprio(2);
        kittens::mma_ABt(cB, a, b1, cB);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        kittens::load(a, kittens::subtile_inplace<REG_M, BLOCK_K>(As[tic][1], {warp_m, 0}));
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        if constexpr (IS_1D2D) apply_row_ratio_sb(cC, load_row_ratio<RT_C>(smem_a_ratio[rtic], a_row_h1), sb_ratio_h0);
        else                   apply_row_col_ratio(cC, load_row_ratio<RT_C>(smem_a_ratio[rtic], a_row_h1), cr0);
        __builtin_amdgcn_s_setprio(2);
        kittens::mma_ABt(cC, a, b0, cC);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        kittens::load(b0, kittens::subtile_inplace<REG_N, BLOCK_K>(Bs[toc][0], {warp_n, 0}));
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        if constexpr (IS_1D2D) apply_row_ratio_sb(cD, load_row_ratio<RT_C>(smem_a_ratio[rtic], a_row_h1), sb_ratio_h1);
        else                   apply_row_col_ratio(cD, load_row_ratio<RT_C>(smem_a_ratio[rtic], a_row_h1), cr1);
        compute_a_ratios_and_promote<BLOCK_M>(tid, smem_sa_prev, smem_sa_curr, smem_a_ratio[rtoc]);
        __builtin_amdgcn_s_setprio(2);
        kittens::mma_ABt(cD, a, b1, cD);
        __builtin_amdgcn_s_setprio(0);

        if constexpr (IS_1D2D) {
            prev_sb_h0 = curr_sb_h0; prev_sb_h1 = curr_sb_h1;
            curr_sb_h0 = nxt_sb_h0; curr_sb_h1 = nxt_sb_h1;
        } else {
            prev_cs0 = curr_cs0; prev_cs1 = curr_cs1;
            curr_cs0 = nxt_cs0; curr_cs1 = nxt_cs1;
        }
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        tic ^= 1; toc ^= 1; rtic ^= 1; rtoc ^= 1;
    }

    {
        float sb_ratio_h0, sb_ratio_h1;
        ColScale<RT_C::width> cr0, cr1;
        if constexpr (IS_1D2D) {
            sb_ratio_h0 = prev_sb_h0 / curr_sb_h0;
            sb_ratio_h1 = prev_sb_h1 / curr_sb_h1;
        } else {
            cr0 = col_scale_ratio(prev_cs0, curr_cs0);
            cr1 = col_scale_ratio(prev_cs1, curr_cs1);
        }
        asm volatile("s_waitcnt vmcnt(0)");
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();

        kittens::load(a, kittens::subtile_inplace<REG_M, BLOCK_K>(As[tic][0], {warp_m, 0}));
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        if constexpr (IS_1D2D) apply_row_ratio_sb(cA, load_row_ratio<RT_C>(smem_a_ratio[rtic], a_row_h0), sb_ratio_h0);
        else                   apply_row_col_ratio(cA, load_row_ratio<RT_C>(smem_a_ratio[rtic], a_row_h0), cr0);
        __builtin_amdgcn_s_setprio(2);
        kittens::mma_ABt(cA, a, b0, cA);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        kittens::load(b1, kittens::subtile_inplace<REG_N, BLOCK_K>(Bs[tic][1], {warp_n, 0}));
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        asm volatile("s_waitcnt lgkmcnt(0)");
        if constexpr (IS_1D2D) apply_row_ratio_sb(cB, load_row_ratio<RT_C>(smem_a_ratio[rtic], a_row_h0), sb_ratio_h1);
        else                   apply_row_col_ratio(cB, load_row_ratio<RT_C>(smem_a_ratio[rtic], a_row_h0), cr1);
        __builtin_amdgcn_s_setprio(2);
        kittens::mma_ABt(cB, a, b1, cB);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        kittens::load(a, kittens::subtile_inplace<REG_M, BLOCK_K>(As[tic][1], {warp_m, 0}));
        __builtin_amdgcn_s_barrier();

        asm volatile("s_waitcnt lgkmcnt(0)");
        if constexpr (IS_1D2D) apply_row_ratio_sb(cC, load_row_ratio<RT_C>(smem_a_ratio[rtic], a_row_h1), sb_ratio_h0);
        else                   apply_row_col_ratio(cC, load_row_ratio<RT_C>(smem_a_ratio[rtic], a_row_h1), cr0);
        __builtin_amdgcn_s_setprio(2);
        kittens::mma_ABt(cC, a, b0, cC);
        __builtin_amdgcn_s_setprio(0);

        if constexpr (IS_1D2D) apply_row_ratio_sb(cD, load_row_ratio<RT_C>(smem_a_ratio[rtic], a_row_h1), sb_ratio_h1);
        else                   apply_row_col_ratio(cD, load_row_ratio<RT_C>(smem_a_ratio[rtic], a_row_h1), cr1);
        kittens::mma_ABt(cD, a, b1, cD);
        __builtin_amdgcn_s_barrier();
    }

    const int m_off0 = block_row * BLOCK_M + warp_m * REG_M;
    const int m_off1 = block_row * BLOCK_M + HALF_ROW + warp_m * REG_M;
    const int n_off0 = block_col * BLOCK_N + warp_n * REG_N;
    const int n_off1 = block_col * BLOCK_N + HALF_COL + warp_n * REG_N;

    OType *c_ptr = C.raw_ptr;

    auto finish = [&](RT_C &c, int a_row, float sb, const ColScale<RT_C::width> &cs, int m_off, int n_off) {
        if constexpr (IS_1D2D) apply_row_ratio_sb(c, load_row_ratio<RT_C>(smem_sa_prev, a_row), sb);
        else                   apply_row_col_ratio(c, load_row_ratio<RT_C>(smem_sa_prev, a_row), cs);
        store_output(c_ptr, c, m_off, n_off, M, N);
    };
    finish(cA, a_row_h0, curr_sb_h0, curr_cs0, m_off0, n_off0);
    finish(cB, a_row_h0, curr_sb_h1, curr_cs1, m_off0, n_off1);
    finish(cC, a_row_h1, curr_sb_h0, curr_cs0, m_off1, n_off0);
    finish(cD, a_row_h1, curr_sb_h1, curr_cs1, m_off1, n_off1);
}

template <typename OType>
using py_globals = micro_globals_fp8<OType>;

template <bool IS_1D2D, typename OType>
void dispatch_micro(py_globals<OType> g) {
    micro_tk<OType, IS_1D2D><<<g.grid(), g.block()>>>(g);
    hipDeviceSynchronize();
}

PYBIND11_MODULE(tk_running, m) {
    m.doc() = "cdna4 blockwise fp8 gemm (running-scale)";
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
