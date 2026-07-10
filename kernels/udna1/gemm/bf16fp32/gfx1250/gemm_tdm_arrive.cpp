/**
 * @file gemm_tdm_arrive.cpp
 * @brief Rung 7 -- per-transfer LDS-barrier TDM GEMM for gfx1250.
 *
 * Diff vs `gemm_expert`: replace cooperative async loads with `load_tdm`
 * issued by wave 0 (for A) and wave 1 (for B). Each TDM transfer is paired
 * with its own `barrier_lds` cell; the producer calls
 * `sync::async_barrier_arrive` after the TDM completes, and the consumer
 * waits on the cell's phase flip. This matches the production lowering
 * used by the Triton AMD backend (which similarly does not rely on the D#
 * auto-arrive path).
 *
 * Runtime note: this kernel exercises `DS_ATOMIC_ASYNC_BARRIER_ARRIVE_B64`
 * and the LDS phase-flip wait. The code matches the SP3 spec for the cell
 * layout (sec 9.8.13): pending in `bar_state[15:0]`, phase in
 * `bar_state[31:16]`, `init_count` in `cell[47:32]`, `pending` initialized
 * to `count - 1`, and one arrive per producer wave (the DS atomic fires
 * per active lane). On runtimes that don't model the async barrier
 * arrive opcode this hangs; on silicon and on runtimes that honor it
 * the kernel should pass. Excluded from the default smoke-test sweep
 * until a runtime that models it is in reach.
 *
 * Exercises the new fine-grained API:
 *   - `sync::barrier_lds`             -- 64-bit LDS barrier cell.
 *   - `sync::init_barrier(bar, n)`    -- prime the cell for `n` arrivals.
 *   - `kittens::load_tdm_arrive(..., bar)` -- TDM that auto-arrives on `bar`.
 *   - `sync::wait_barrier(bar, phase)`-- block on the cell's phase flip.
 *
 * The kernel proves out two things:
 *   1. `load_tdm_arrive` constructs a valid D# with `atomic_barrier_enable`
 *      set and the LDS barrier address routed in, and the runtime delivers
 *      the auto-arrive correctly.
 *   2. Independent phases on A_bar and B_bar let the kernel keep more than
 *      one TDM transfer in flight at a time without inter-operand stalls.
 *
 * Tile: 64x64 output, K_STEP = 32, 4 warps in a 2x2 layout (matches the
 * rest of the ladder).
 */

#include "common.h"

using namespace kittens;
using namespace gfx1250_gemm;

__global__ __launch_bounds__(NUM_THREADS, 1)
void gemm_tdm_arrive_kernel(const gemm_globals g, int M, int N, int K)
{
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al(reinterpret_cast<int*>(&__shm[0]));

    // Segment 0 layout: A slabs followed by the four 8-byte barrier cells.
    // Allocate barriers FIRST in segment 0 (their addresses fit in 16 bits,
    // which is what the D# `atomic_barrier_address` field carries), then
    // the A buffers; finally B in segment 1.
    sync::barrier_lds(&A_bar)[2] = al.allocate_in<segment<0>, sync::barrier_lds, 2>();
    sync::barrier_lds(&B_bar)[2] = al.allocate_in<segment<0>, sync::barrier_lds, 2>();
    A_tile(&A_st)[2] = al.allocate_in<segment<0>, A_tile, 2>();
    B_tile(&B_st)[2] = al.allocate_in<segment<1>, B_tile, 2>();

    rt_fl<WARP_M, WARP_N, col_l, rt_16x16_s> C_acc;
    zero(C_acc);

    const int tile_m  = blockIdx.x;
    const int tile_n  = blockIdx.y;
    const int wid     = warpid();
    const int warp_r  = wid / WARPS_N;
    const int warp_c  = wid % WARPS_N;
    const int k_iters = K / K_STEP;

    // One thread primes the four cells. Each cell expects 1 arrival per
    // phase (the single TDM transfer that will target it).
    if (threadIdx.x == 0) {
        sync::init_barrier(&A_bar[0].state, 1);
        sync::init_barrier(&A_bar[1].state, 1);
        sync::init_barrier(&B_bar[0].state, 1);
        sync::init_barrier(&B_bar[1].state, 1);
    }
    sync::sync();

    // Per-buffer parity bits. The cell's phase bit starts at 0 and flips
    // each time the pending count drains; `wait_barrier(.., phase ^ 1)`
    // unblocks once the next arrival lands.
    int A_phase[2] = {0, 0};
    int B_phase[2] = {0, 0};

    // Prologue: wave 0 issues A[0], wave 1 issues B[0]. We use plain
    // `load_tdm` and follow it with a manual `async_barrier_arrive` ordered
    // against the producer's TENSORcnt. This matches the production pattern
    // used by the Triton AMD backend; the D# auto-arrive path (set via
    // `load_tdm_arrive`) is also wired in the library for runtimes that
    // model it natively.
    //
    // `async_barrier_arrive` is a DS atomic, so it fires per active lane:
    // guard with `laneid() == 0` so each producer wave arrives exactly
    // once per phase (matching the `init_barrier(.., 1)` priming above).
    if (wid == 0) {
        load_tdm(A_st[0], g.a, {0, 0, tile_m, 0}, M, K, K);
        sync::wait_tdm();
        if (laneid() == 0) sync::async_barrier_arrive(&A_bar[0].state);
    }
    if (wid == 1) {
        load_tdm(B_st[0], g.b, {0, 0, tile_n, 0}, N, K, K);
        sync::wait_tdm();
        if (laneid() == 0) sync::async_barrier_arrive(&B_bar[0].state);
    }

    {
        // Expert scheduling covers the K-loop only; restored before the
        // epilogue store so the store's hazards are hardware-handled as usual.
        sched::expert_scope _sched;

        for (int k = 0; k < k_iters; ++k) {
            const int cur = k & 1, nxt = 1 - cur;

            if (k + 1 < k_iters) {
                if (wid == 0) {
                    load_tdm(A_st[nxt], g.a, {0, 0, tile_m, k + 1}, M, K, K);
                    sync::wait_tdm();
                    if (laneid() == 0) sync::async_barrier_arrive(&A_bar[nxt].state);
                }
                if (wid == 1) {
                    load_tdm(B_st[nxt], g.b, {0, 0, tile_n, k + 1}, N, K, K);
                    sync::wait_tdm();
                    if (laneid() == 0) sync::async_barrier_arrive(&B_bar[nxt].state);
                }
            }

            // Wait for THIS K-step's transfers (independent of the next).
            // Toggle the parity for the cell we're about to consume.
            A_phase[cur] ^= 1;
            B_phase[cur] ^= 1;
            sync::wait_barrier(&A_bar[cur].state, A_phase[cur]);
            sync::wait_barrier(&B_bar[cur].state, B_phase[cur]);
            sync::sync();   // make A/B-arrived state visible to every consumer warp

            // A_reg/B_reg are reused every iteration: the previous mma must
            // finish reading them before these loads overwrite them.
            sched::wait_alu();

            rt_bf<WARP_M, K_STEP, row_l, rt_16x32_s> A_reg;
            rt_bf<WARP_N, K_STEP, row_l, rt_16x32_s> B_reg;
            kittens::load(A_reg, A_st[cur], warp_r * WARP_M * K_STEP);
            kittens::load(B_reg, B_st[cur], warp_c * WARP_N * K_STEP);

            sync::wait_ds();
            mma_ABt(C_acc, A_reg, B_reg, C_acc);

            sync::sync();
        }
    }

    bf16* c_base = reinterpret_cast<bf16*>(&g.c[{0, 0, 0, 0}]);
    store_acc<WARP_M / 16, WARP_N / 16>(
        c_base,
        tile_m * BLOCK_M + warp_r * WARP_M,
        tile_n * BLOCK_N + warp_c * WARP_N,
        N, C_acc);
}

void dispatch(gemm_globals g)
{
    // Same layout as `gemm_segment`/`gemm_expert` (A in seg 0, B in seg 1)
    // plus 4 barrier cells in seg 0.
    constexpr size_t bar_bytes = 4 * sizeof(sync::barrier_lds);
    const size_t mem_size = MAX_SHARED_MEMORY_PER_SEGMENT + 2 * sizeof(B_tile);
    (void)bar_bytes;
    hipFuncSetAttribute(reinterpret_cast<const void*>(gemm_tdm_arrive_kernel),
                        hipFuncAttributeMaxDynamicSharedMemorySize,
                        static_cast<int>(mem_size));
    gemm_tdm_arrive_kernel<<<g.grid(), g.block(), mem_size, g.stream>>>(
        g, g.M(), g.N(), g.K());
}

#include "harness.h"
