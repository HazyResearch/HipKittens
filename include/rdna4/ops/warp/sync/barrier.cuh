/**
 * @file
 * @brief Synchronization primitives for gfx12 (gfx1201).
 *
 * gfx12 replaces the unified `s_waitcnt` with per-counter waits and exposes
 * a split barrier that lets a warp signal "data ready" before waiting on
 * the corresponding consumer barrier. This header wraps them behind a
 * `kittens::sync::*` API, as inline assembly where clang has no builtin.
 *
 * gfx1250's async global->LDS copies, TDM and hardware LDS barrier cells
 * do not exist on gfx1201, so `wait_async`, `wait_tdm` and the
 * `barrier_lds` cell API are absent from this tree.
 */

#pragma once


#include "../../../common/common.cuh"

namespace kittens {
namespace sync {

/* ----------  SPLIT BARRIER (BLOCK-WIDE)  ---------- */

/**
 * @brief Signal a block-wide split barrier.
 *
 * Lowers to `s_barrier_signal -1`. May be issued from any warp and returns
 * immediately; only `wait()` blocks until every warp in the block has signalled.
 */
__device__ __forceinline__ void arrive() {
    __builtin_amdgcn_s_barrier_signal(-1);
}

/**
 * @brief Wait on a block-wide split barrier.
 *
 * Lowers to `s_barrier_wait -1`. Blocks until every warp in the block has
 * called `arrive()` since the last completion of this barrier.
 */
__device__ __forceinline__ void wait() {
    __builtin_amdgcn_s_barrier_wait(-1);
}

/**
 * @brief Block-wide barrier (signal + wait), draining LDS first.
 *
 * Semantically equivalent to `__syncthreads()`.
 *
 * The `s_wait_dscnt 0` is load-bearing and must not be removed. On gfx12,
 * `s_barrier_signal` does **not** drain outstanding LDS traffic -- LDS ops are
 * tracked by their own DScnt counter and the barrier is indifferent to it.
 * Without the wait, a warp can signal the barrier while its global->shared
 * stores are still in flight; every other warp then passes the barrier and
 * reads stale LDS. The result is not a crash but garbage operands, which in a
 * GEMM surfaces as `inf` in the accumulator.
 *
 * This is the defect that made the first gfx1201 GEMM port return wrong numbers
 * at every problem size -- including a single 64x64x32 block with one K step --
 * while the shared-memory address mapping itself round-tripped perfectly.
 * Races of this kind survive isolated round-trip tests precisely because a
 * lightly loaded kernel usually wins them.
 *
 * Issued as inline asm rather than by calling wait_ds<0>() because the
 * per-counter wait helpers are declared later in this header.
 *
 * Prefer the split form (`arrive()` / work / `wait()`) when the window can be
 * filled -- but then draining LDS becomes the caller's responsibility.
 */
__device__ __forceinline__ void sync() {
    asm volatile("s_wait_dscnt 0" ::: "memory");
    arrive();
    wait();
}

/* ----------  GFX12+ PER-COUNTER WAITS  ---------- */
//
// Each gfx1250 wait counter is 6 bits; the `N` template parameter is the
// maximum number of in-flight ops that may remain after the wait. The default
// `N=0` drains the counter completely (semantically a full sync of that class).
// Use a non-zero `N` to keep a `K`-deep pipeline running, draining one slot
// at a time as new ops are issued.

/**
 * @brief Wait for outstanding global (and texture) loads, leaving up to N in flight.
 *
 * Lowers to `s_wait_loadcnt N`. Required after a `global_load_async_to_lds`
 * or any `global_load_*` whose results are about to be read.
 *
 * @note Clang 23 does not yet expose `__builtin_amdgcn_s_wait_loadcnt`;
 *       we emit the instruction directly with `N` as an immediate operand.
 */
template<int N = 0>
__device__ __forceinline__ void wait_load() {
    static_assert(N >= 0 && N < 64, "loadcnt is 6-bit; max 63");
    asm volatile("s_wait_loadcnt %0" :: "i"(N) : "memory");
}

/**
 * @brief Wait for outstanding global stores, leaving up to N in flight.
 *
 * Lowers to `s_wait_storecnt N`.
 */
template<int N = 0>
__device__ __forceinline__ void wait_store() {
    static_assert(N >= 0 && N < 64, "storecnt is 6-bit; max 63");
    asm volatile("s_wait_storecnt %0" :: "i"(N) : "memory");
}

/**
 * @brief Wait for outstanding LDS (DS_*) operations, leaving up to N in flight.
 *
 * Lowers to `s_wait_dscnt N`. Required between LDS writes (or `ds_load_b*`
 * issues) and a dependent VALU/WMMA consumer.
 */
template<int N = 0>
__device__ __forceinline__ void wait_ds() {
    static_assert(N >= 0 && N < 64, "dscnt is 6-bit; max 63");
    asm volatile("s_wait_dscnt %0" :: "i"(N) : "memory");
}

/**
 * @brief Wait for outstanding kernel-message ops, leaving up to N in flight.
 *
 * Lowers to `s_wait_kmcnt N`.
 */
template<int N = 0>
__device__ __forceinline__ void wait_km() {
    static_assert(N >= 0 && N < 64, "kmcnt is 6-bit; max 63");
    asm volatile("s_wait_kmcnt %0" :: "i"(N) : "memory");
}

/**
 * @brief Memory fence covering both global loads and LDS ops.
 *
 * Convenience for the common "producer side" pattern: ensure all in-flight
 * loads have settled into LDS before signalling consumers.
 */
__device__ __forceinline__ void fence() {
    wait_load<0>();
    wait_ds<0>();
}
} // namespace sync
} // namespace kittens

