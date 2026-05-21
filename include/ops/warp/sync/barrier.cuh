/**
 * @file
 * @brief Synchronization primitives for gfx1250.
 *
 * gfx1250 splits the unified `s_waitcnt` into one wait per traffic class
 * (load / store / ds / km / async / tensor) and replaces `__syncthreads()`
 * with a signal/wait pair (`s_barrier_signal -1` / `s_barrier_wait -1`).
 * Producer/consumer kernels can also arrive at an LDS-resident barrier
 * cell from an async-ordered path (`DS_ATOMIC_ASYNC_BARRIER_ARRIVE_B64`)
 * and wait on the cell's phase flip.
 *
 * This header exposes:
 *
 *   kittens::sync::{arrive, wait, sync}      -- block-wide split barrier
 *   kittens::sync::wait_{load, store, ds, km, async, tensor}<N>
 *                                             -- per-counter waits
 *   kittens::sync::fence()                    -- loads + LDS drain
 *
 *   kittens::semaphore                        -- 64-bit LDS barrier cell
 *   kittens::init_semaphore(sem, arrivals)    -- prime a semaphore
 *   kittens::arrive(sem)                      -- async-ordered arrive
 *   kittens::wait(sem, phase)                 -- phase-parity wait
 *
 * Where clang exposes a builtin we use it directly; the per-counter waits
 * that have no builtin (load, store, ds, km) emit inline asm with the
 * count as an `i` immediate.
 */

#pragma once

#ifdef KITTENS_UDNA1

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
 * @brief Block-wide barrier (signal + wait).
 *
 * Semantically equivalent to `__syncthreads()`. Prefer the split form
 * (`arrive()` followed by independent work followed by `wait()`) when the
 * window between signalling and waiting can be filled with non-dependent
 * instructions.
 */
__device__ __forceinline__ void sync() {
    arrive();
    wait();
}

/* ----------  PER-COUNTER WAITS  ---------- */
//
// Each counter is 6 bits; the `N` template parameter is the maximum number
// of in-flight ops that may remain after the wait. `N = 0` (the default)
// drains the counter completely. Use a non-zero `N` to keep a K-deep
// pipeline running, draining one slot at a time as new ops are issued.

/**
 * @brief Wait for outstanding global (and texture) loads, leaving up to N in flight.
 *
 * Lowers to `s_wait_loadcnt N`. Required after any `global_load_*` whose
 * results are about to be read by a dependent VALU/WMMA op.
 *
 * @note Clang 23 does not expose `__builtin_amdgcn_s_wait_loadcnt`; the
 *       instruction is emitted directly with `N` as an `i` immediate.
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
 * @brief Wait for outstanding async global->LDS transfers, leaving up to N in flight.
 *
 * Lowers to `s_wait_asynccnt N`. Drains anything started by
 * `__builtin_amdgcn_(global|cluster)_load_async_to_lds_*`.
 */
template<int N = 0>
__device__ __forceinline__ void wait_async() {
    static_assert(N >= 0 && N < 64, "asynccnt is 6-bit; max 63");
    __builtin_amdgcn_s_wait_asynccnt(N);
}

/**
 * @brief Wait for outstanding TDM transfers, leaving up to N in flight.
 *
 * Lowers to `s_wait_tensorcnt N`. Drains anything started by
 * `__builtin_amdgcn_tensor_load_to_lds` or `tensor_store_from_lds`.
 *
 * @code
 *   tdm::load_async(buf[0], ...);
 *   tdm::load_async(buf[1], ...);
 *   tdm::load_async(buf[2], ...);
 *   for (int k = 0; k + 3 < K; ++k) {
 *       sync::wait_tensor<2>();           // drain one slot, two stay in flight
 *       consume(buf[k % 3]);
 *       tdm::load_async(buf[k % 3], ...);
 *   }
 *   sync::wait_tensor<0>();               // drain the tail
 * @endcode
 */
template<int N = 0>
__device__ __forceinline__ void wait_tensor() {
    static_assert(N >= 0 && N < 64, "tensorcnt is 6-bit; max 63");
    __builtin_amdgcn_s_wait_tensorcnt(N);
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

/* ----------  LDS BARRIER CELL (SEMAPHORE)  ---------- */
//
// 64-bit LDS-resident barrier cell, per SP3 §9.8.13
// (DS_ATOMIC_ASYNC_BARRIER_ARRIVE_B64). The cell is packed as:
//
//   bits 63..48 : reserved (zero)
//   bits 47..32 : init_count        (reload value at phase flip)
//   bits 31..0  : bar_state, itself packed as [phase | pending_count]
//                 with the boundary at bit 16:
//                   bits 31..16 : phase            (counter, parity alternates)
//                   bits 15..0  : pending_count
//
// Each arrive subtracts 1 from bar_state. When pending rolls under (its
// MSB becomes 1), the hardware reloads bar_state to
//   (new_phase << 16) | init_count
// and wakes any wave sleeping on the cell.
//
// To expect N arrivals per phase: pending and init_count both start at
// N - 1. Phase decrements per flip, so its LSB alternates 0,1,0,1... --
// the classic parity-flip pattern.

/**
 * @brief 64-bit LDS-resident semaphore.
 *
 * Allocate as `__shared__ kittens::semaphore sem;` (or inside an LDS
 * segment via `shared_allocator::allocate_in<segment<I>, semaphore, N>()`)
 * and prime once with `init_semaphore(sem, arrivals)` before any
 * `arrive()` call.
 */
struct alignas(8) semaphore { uint64_t state; };

/**
 * @brief Initialize a semaphore to expect `arrivals` arrivals per phase.
 *
 * Writes the packed cell layout `[init_count(47:32) | phase=0(31:16) |
 * pending(15:0)]` with `pending = init_count = arrivals - 1`.
 */
__device__ __forceinline__ void init_semaphore(semaphore& sem, uint32_t arrivals) {
    const uint32_t pending  = arrivals - 1;
    const uint32_t init_cnt = arrivals - 1;
    sem.state =  uint64_t(pending  & 0xFFFFu)
              | (uint64_t(0)                  << 16)   // phase = 0
              | (uint64_t(init_cnt & 0xFFFFu) << 32);
}

/**
 * @brief Arrive at a semaphore from an async-ordered path.
 *
 * Lowers to `DS_ATOMIC_ASYNC_BARRIER_ARRIVE_B64`. The DS atomic fires per
 * active lane, so producer waves must guard the call with
 * `if (laneid() == 0)` to avoid 32-fold over-arrival on wave-32.
 *
 * Use this when arriving manually after `sync::wait_tensor<0>()`. For the
 * hardware auto-arrive path (set in the TDM descriptor itself) use the
 * `tdm::load_async` overload that takes a `semaphore&`.
 */
__device__ __forceinline__ void arrive(semaphore& sem) {
    uintptr_t lds_uint = reinterpret_cast<uintptr_t>(&sem.state);
    __builtin_amdgcn_ds_atomic_async_barrier_arrive_b64(
        reinterpret_cast<long __attribute__((address_space(3)))*>(lds_uint));
}

/**
 * @brief Block on `sem` until its phase LSB matches `phase`.
 *
 * Phase decrements once per flip, so its low bit alternates 0,1,0,1...
 * Callers maintain a parity bit per semaphore and pass it inverted before
 * each wait (`expected = (phase ^= 1)`). The hardware wakes sleeping
 * waves on phase flip; `s_sleep 1` yields the SIMD between polls.
 */
__device__ __forceinline__ void wait(semaphore& sem, int phase) {
    const uint32_t lds_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&sem.state));
    while (true) {
        uint64_t v;
        asm volatile("ds_load_b64 %0, %1 offset:0"
            : "=v"(v) : "v"(lds_addr) : "memory");
        // Phase lives in the high 16 bits of the low 32-bit bar_state.
        const uint32_t bar_state = static_cast<uint32_t>(v);
        const int phase_lsb = int((bar_state >> 16) & 1);
        if (phase_lsb == phase) break;
        __builtin_amdgcn_s_sleep(1);
    }
}

} // namespace kittens

#endif // KITTENS_UDNA1
