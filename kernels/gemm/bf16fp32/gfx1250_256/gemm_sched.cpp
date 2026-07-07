/**
 * @file gemm_sched.cpp
 * @brief 256x256 gfx1250 GEMM — MI350 ping-pong + fine-grained scheduling.
 *
 * Same decomposition as `gemm_naive.cpp` (As[2][2], Bs[2][2], tic/toc, tile+=2).
 * This rung replaces coarse wait_async/sync with MI350-style partial waits,
 * s_setprio around MMA, and sched_barrier placement.
 *
 * Reference: `256_256_64_32_with16x32.cpp`
 *   vmcnt  -> sync::wait_async<N>   (asynccnt; N needs calibration per issue batch)
 *   lgkmcnt -> sync::wait_ds<N>
 */

#include "common.h"

using namespace kittens;
using namespace gfx1250_gemm_256;

namespace {

using Acc_tile = rt_fl<HALF_WAVE_M, HALF_WAVE_N, col_l, rt_16x16_s>;

__device__ __forceinline__ void lds_store_acc16(
    float* __restrict__ acc, int gr_base, int gc_base,
    const rt_base<float, ducks::rt_layout::col, ducks::rt_shape::rt_16x16>& tile)
{
    const int L = laneid(), half = L / 16, col = L % 16, gc = gc_base + col;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        acc[(gr_base + 2 * k     + 8 * half) * HALF_WAVE_N + gc] = tile.data[k].x;
        acc[(gr_base + 2 * k + 1 + 8 * half) * HALF_WAVE_N + gc] = tile.data[k].y;
    }
}

__device__ __forceinline__ void lds_load_acc16(
    rt_base<float, ducks::rt_layout::col, ducks::rt_shape::rt_16x16>& tile,
    const float* __restrict__ acc, int gr_base, int gc_base)
{
    const int L = laneid(), half = L / 16, col = L % 16, gc = gc_base + col;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        tile.data[k].x = acc[(gr_base + 2 * k     + 8 * half) * HALF_WAVE_N + gc];
        tile.data[k].y = acc[(gr_base + 2 * k + 1 + 8 * half) * HALF_WAVE_N + gc];
    }
}

__device__ __forceinline__ void lds_store_acc(float* dst, const Acc_tile& C)
{
    #pragma unroll
    for (int n = 0; n < HALF_WAVE_M / 16; ++n)
        #pragma unroll
        for (int m = 0; m < HALF_WAVE_N / 16; ++m)
            lds_store_acc16(dst, n * 16, m * 16, C.tiles[n][m]);
}

__device__ __forceinline__ void lds_load_acc(Acc_tile& C, const float* src)
{
    #pragma unroll
    for (int n = 0; n < HALF_WAVE_M / 16; ++n)
        #pragma unroll
        for (int m = 0; m < HALF_WAVE_N / 16; ++m)
            lds_load_acc16(C.tiles[n][m], src, n * 16, m * 16);
}

__device__ __forceinline__ void store_acc16(
    bf16* __restrict__ c_global,
    int gr_base, int gc_base, int N,
    const rt_base<float, ducks::rt_layout::col, ducks::rt_shape::rt_16x16>& tile)
{
    const int L    = laneid();
    const int half = L / 16;
    const int col  = L % 16;
    const int gc   = gc_base + col;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int gr0 = gr_base + 2 * k     + 8 * half;
        const int gr1 = gr_base + 2 * k + 1 + 8 * half;
        c_global[gr0 * N + gc] = base_types::convertor<bf16, float>::convert(tile.data[k].x);
        c_global[gr1 * N + gc] = base_types::convertor<bf16, float>::convert(tile.data[k].y);
    }
}

__device__ __forceinline__ void store_acc64x32(
    bf16* __restrict__ c_global, int wgr_base, int wgc_base, int N,
    const rt_fl<HALF_WAVE_M, HALF_WAVE_N, col_l, rt_16x16_s>& C)
{
    #pragma unroll
    for (int n = 0; n < HALF_WAVE_M / 16; ++n)
        #pragma unroll
        for (int m = 0; m < HALF_WAVE_N / 16; ++m)
            store_acc16(c_global, wgr_base + n * 16, wgc_base + m * 16, N, C.tiles[n][m]);
}

__device__ __forceinline__ void barrier() { sync::sync(); }

__device__ __forceinline__ void sched_barrier() { __builtin_amdgcn_sched_barrier(0); }

template<typename... Args>
__device__ __forceinline__ void mma_prio(Args&&... args) {
    __builtin_amdgcn_s_setprio(1);
    mma_ABt(std::forward<Args>(args)...);
    __builtin_amdgcn_s_setprio(0);
}

} // namespace

__global__ GEMM256_KERNEL_ATTR
void gemm_sched_kernel(const gemm_globals g, int M, int N, int K)
{
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al(reinterpret_cast<int*>(&__shm[0]));

    A_quarter (&As)[2][2] = al.allocate<A_quarter, 2, 2>();
    B_quarter (&Bs)[2][2] = al.allocate<B_quarter, 2, 2>();
    float (&C_lds)[NUM_WAVES][2][2][HALF_WAVE_M][HALF_WAVE_N] =
        al.allocate<float, NUM_WAVES, 2, 2, HALF_WAVE_M, HALF_WAVE_N>();

    rt_bf<HALF_WAVE_M, K_STEP, row_l, rt_16x32_s> A_tile;
    rt_bf<HALF_WAVE_N, K_STEP, row_l, rt_16x32_s> B_tile_0;
    rt_bf<HALF_WAVE_N, K_STEP, row_l, rt_16x32_s> B_tile_1;
    Acc_tile C_tile;
    bool c_init[2][2] = {{false, false}, {false, false}};

    const int row      = blockIdx.x;
    const int col      = blockIdx.y;
    const int warp_row = warpid() / WARPS_N;
    const int warp_col = warpid() % WARPS_N;
    const int num_tiles = K / K_STEP;

    const int a_origin = warp_row * HALF_WAVE_M * K_STEP;
    const int b_origin = warp_col * HALF_WAVE_N * K_STEP;
    float (&my_c)[2][2][HALF_WAVE_M][HALF_WAVE_N] = C_lds[warpid()];

    auto mma_q = [&](int br, int bc, const auto& A, const auto& B) {
        if (!c_init[br][bc]) { zero(C_tile); c_init[br][bc] = true; }
        else lds_load_acc(C_tile, &my_c[br][bc][0][0]);
        mma_prio(C_tile, A, B, C_tile);
        lds_store_acc(&my_c[br][bc][0][0], C_tile);
    };

    int tic = 0;
    int toc = 1;

    auto g2s_a = [&](int buf, int half, int ktile) {
        load_async<NUM_THREADS>(As[buf][half], g.a, {0, 0, row * 2 + half, ktile}, K);
    };
    auto g2s_b = [&](int buf, int half, int ktile) {
        load_async<NUM_THREADS>(Bs[buf][half], g.b, {0, 0, col * 2 + half, ktile}, K);
    };

    // --- prologue (MI350 lines 116-133) ---
    g2s_b(tic, 0, 0);
    g2s_a(tic, 0, 0);
    g2s_b(tic, 1, 0);
    g2s_a(tic, 1, 0);

    if (warp_row == 1)
        barrier();

    sync::wait_async<0>();  // TODO: map MI350 vmcnt(4) once asynccnt/issue batch is counted
    barrier();

    if (num_tiles >= 2) {
        g2s_b(toc, 0, 1);
        g2s_a(toc, 0, 1);
        g2s_b(toc, 1, 1);
        sync::wait_async<0>();  // TODO: vmcnt(6)
        barrier();
    }

    #pragma unroll 1
    for (int tile = 0; tile < num_tiles - 2; tile += 2) {
        load(B_tile_0, Bs[0][0], b_origin);
        load(A_tile, As[0][0], a_origin);
        g2s_a(1, 1, tile + 1);
        sync::wait_ds<8>();
        barrier();
        sync::wait_ds<0>();
        mma_q(0, 0, A_tile, B_tile_0);
        barrier();
        sched_barrier();

        load(B_tile_1, Bs[0][1], b_origin);
        g2s_b(0, 0, tile + 2);
        barrier();
        sync::wait_ds<0>();
        mma_q(0, 1, A_tile, B_tile_1);
        barrier();

        load(A_tile, As[0][1], a_origin);
        g2s_a(0, 0, tile + 2);
        barrier();
        sync::wait_ds<0>();
        mma_q(1, 0, A_tile, B_tile_0);
        barrier();
        sched_barrier();

        load(B_tile_0, Bs[1][0], b_origin);
        g2s_b(0, 1, tile + 2);
        sync::wait_async<0>();  // TODO: vmcnt(6)
        barrier();
        mma_q(1, 1, A_tile, B_tile_1);
        barrier();

        load(A_tile, As[1][0], a_origin);
        g2s_a(0, 1, tile + 2);
        sync::wait_ds<8>();
        barrier();
        sync::wait_ds<0>();
        mma_q(0, 0, A_tile, B_tile_0);
        barrier();
        sched_barrier();

        load(B_tile_1, Bs[1][1], b_origin);
        g2s_b(1, 0, tile + 3);
        barrier();
        sync::wait_ds<0>();
        mma_q(0, 1, A_tile, B_tile_1);
        barrier();

        load(A_tile, As[1][1], a_origin);
        g2s_a(1, 0, tile + 3);
        barrier();
        sync::wait_ds<0>();
        mma_q(1, 0, A_tile, B_tile_0);
        barrier();
        sched_barrier();

        g2s_b(1, 1, tile + 3);
        sync::wait_async<0>();  // TODO: vmcnt(6)
        barrier();
        mma_q(1, 1, A_tile, B_tile_1);
        barrier();
    }

    if (num_tiles >= 2) {
        const int tile = num_tiles - 2;

        load(B_tile_0, Bs[tic][0], b_origin);
        load(A_tile, As[tic][0], a_origin);
        g2s_a(toc, 1, tile + 1);
        barrier();
        sync::wait_ds<0>();
        mma_q(0, 0, A_tile, B_tile_0);
        barrier();

        load(B_tile_1, Bs[tic][1], b_origin);
        barrier();
        sync::wait_ds<0>();
        mma_q(0, 1, A_tile, B_tile_1);
        barrier();

        load(A_tile, As[tic][1], a_origin);
        sync::wait_async<0>();  // TODO: vmcnt(4)
        barrier();
        sync::wait_ds<0>();
        mma_q(1, 0, A_tile, B_tile_0);
        mma_q(1, 1, A_tile, B_tile_1);
        barrier();
        tic ^= 1;
        toc ^= 1;
    }

    {
        load(B_tile_0, Bs[tic][0], b_origin);
        load(A_tile, As[tic][0], a_origin);
        if (num_tiles >= 2)
            sync::wait_async<0>();  // TODO: vmcnt(2)
        barrier();
        sync::wait_ds<0>();
        mma_q(0, 0, A_tile, B_tile_0);
        barrier();

        load(B_tile_1, Bs[tic][1], b_origin);
        if (num_tiles >= 2)
            sync::wait_async<0>();  // TODO: vmcnt(0)
        barrier();
        sync::wait_ds<0>();
        mma_q(0, 1, A_tile, B_tile_1);
        barrier();

        load(A_tile, As[tic][1], a_origin);
        barrier();
        sync::wait_ds<0>();
        mma_q(1, 0, A_tile, B_tile_0);
        mma_q(1, 1, A_tile, B_tile_1);
        barrier();
    }

    if (warp_row == 0)
        barrier();

    bf16* c_base = reinterpret_cast<bf16*>(&g.c[{0, 0, 0, 0}]);
    const int row_tile = row * 2;
    const int col_tile = col * 2;

    lds_load_acc(C_tile, &my_c[0][0][0][0]);
    store_acc64x32(c_base,
        (row_tile * WARPS_M + warp_row) * HALF_WAVE_M,
        (col_tile * WARPS_N + warp_col) * HALF_WAVE_N,
        N, C_tile);
    lds_load_acc(C_tile, &my_c[0][1][0][0]);
    store_acc64x32(c_base,
        (row_tile * WARPS_M + warp_row) * HALF_WAVE_M,
        (col_tile * WARPS_N + WARPS_N + warp_col) * HALF_WAVE_N,
        N, C_tile);
    lds_load_acc(C_tile, &my_c[1][0][0][0]);
    store_acc64x32(c_base,
        (row_tile * WARPS_M + WARPS_M + warp_row) * HALF_WAVE_M,
        (col_tile * WARPS_N + warp_col) * HALF_WAVE_N,
        N, C_tile);
    lds_load_acc(C_tile, &my_c[1][1][0][0]);
    store_acc64x32(c_base,
        (row_tile * WARPS_M + WARPS_M + warp_row) * HALF_WAVE_M,
        (col_tile * WARPS_N + WARPS_N + warp_col) * HALF_WAVE_N,
        N, C_tile);
}

void dispatch(gemm_globals g)
{
    const size_t mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute(reinterpret_cast<const void*>(gemm_sched_kernel),
                        hipFuncAttributeMaxDynamicSharedMemorySize,
                        static_cast<int>(mem_size));
    gemm_sched_kernel<<<g.grid(), g.block(), mem_size, g.stream>>>(
        g, g.M(), g.N(), g.K());
}

#include "harness.h"
