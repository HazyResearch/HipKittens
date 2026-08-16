/**
 * @file
 * @brief Synchronization primitives for gfx1250.
 *
 * gfx1250 replaces the unified `s_waitcnt` with per-counter waits and exposes
 * a split barrier that lets a warp signal "data ready" before waiting on
 * the corresponding consumer barrier. This header wraps every primitive
 * behind a clean `kittens::sync::*` API; the actual instructions are emitted
 * via clang `__builtin_amdgcn_*` builtins where available, and inline
 * assembly otherwise (per-counter waits other than `asynccnt`/`tensorcnt`
 * are not yet exposed as clang builtins as of LLVM 23).
 */

#pragma once


#include "../../../common/common.cuh"

namespace kittens {
namespace sync {

/* ----------  NAMED BARRIERS  ---------- */

/**
 * @brief The number of distinct named barriers a gfx1250 workgroup actually gets.
 *
 */
static constexpr int NUM_NAMED_BARRIERS = 4;

/**
 * @brief A named barrier id that cannot be out of range, because the type won't compile.
 *
 * Use this instead of a raw integer anywhere a named barrier id is needed:
 * @code
 *   using F0 = kittens::sync::named_barrier<1>;
 *   using D0 = kittens::sync::named_barrier<4>;
 *   // using D1 = kittens::sync::named_barrier<5>;  // <-- does not compile
 * @endcode
 *
 * @tparam ID barrier id in `[1, NUM_NAMED_BARRIERS]`.
 */
template<int ID>
struct named_barrier {
    static_assert(ID >= 1,
        "named barrier id must be >= 1: id 0 is BARRIER_ID_NULL, for which the ISA "
        "specifies that signal and wait have no effect.");
    static_assert(ID <= NUM_NAMED_BARRIERS,
        "named barrier id exceeds the number of distinct barriers this part provides (4). "
        "Ids above 4 alias onto 1..4 -- ((id-1) mod 4)+1 -- so they do not fault or hang, "
        "they silently share a barrier with a lower id and the protocol produces wrong "
        "answers at full speed. Re-scope the layout to at most 4 barriers; see "
        "sync::NUM_NAMED_BARRIERS.");
    static constexpr int id = ID;

    /** Program the member count and reset the signal count. */
    template<int COUNT>
    __device__ __forceinline__ static void init() {
        static_assert(COUNT >= 1 && COUNT <= 127,
            "named barrier member count must be in [1,127]: the field is 7 bits, and "
            "s_barrier_init writes it ONLY IF NONZERO -- a count of 0 silently leaves "
            "whatever the previous dispatch left behind (measured: barrier state is "
            "NOT reset across dispatches on this part).");
        asm volatile("s_mov_b32 m0, %0\n\t"
                     "s_barrier_init m0"
                     :: "s"((COUNT << 16) | ID) : "memory");
    }

    /** Arrive. Non-blocking, and does not require membership: the ISA states a wave
     *  "may signal any barrier owned by its own workgroup, whether or not the wave
     *  has joined" -- which is what makes asymmetric membership possible at all. */
    __device__ __forceinline__ static void signal() {
        asm volatile("s_barrier_signal %0" :: "i"(ID) : "memory");
    }

    /** Become a member. Writes only wave-local state (`NAMED_BARRIER`), so it is
     *  invisible to `state()` and a readback cannot confirm it took effect. It is
     *  required because `s_barrier_wait` is a NOP for a wave that has not joined.
     *  Does not block. */
    __device__ __forceinline__ static void join() {
        asm volatile("s_mov_b32 m0, %0\n\t"
                     "s_barrier_join m0" :: "s"(ID) : "memory");
    }

    /** Read the barrier state. The `s_wait_kmcnt` is required and is inside the asm
     *  block deliberately: the ISA says "Increments KMCNT on issue... use
     *  S_WAIT_KMCNT to determine when D0 can be read", and LLVM will not supply it
     *  because it does not model opaque inline-asm bodies. Omitting it reads stale
     *  state. Decode with `barrier_member_count` / `barrier_signal_count`. */
    __device__ __forceinline__ static unsigned state() {
        unsigned s;
        asm volatile("s_get_barrier_state %0, %1\n\t"
                     "s_wait_kmcnt 0x0" : "=s"(s) : "i"(ID) : "memory");
        return s;
    }

    /* No `wait()` is provided, deliberately. `s_barrier_wait` is an unbounded
     * device-side wait with no bounded escape, so a wedged barrier wedges the
     * device; anything that needs to block should do so explicitly at the call
     * site. Note also that a named `s_barrier_wait`'s literal is a barrier class,
     * not an id: which barrier a wave waits on comes from its last `join()`. */
};

/** Decode `named_barrier<>::state()`. Fields per sp3 MI400 3.4.48 (p.87):
 *  `{5'0, NUM_NAMED_BARRIERS/4 [26:24], 1'0, signalCnt [22:16], 5'0, memberCnt [10:4], 3'0, valid [0]}`.
 *  Note `[26:24]` reads 0 on this part even though four barriers are live, so it
 *  does not report the allocation -- do not use it to discover the barrier count. */
__host__ __device__ __forceinline__ unsigned barrier_valid(unsigned s)        { return  s        & 0x1u;  }
__host__ __device__ __forceinline__ unsigned barrier_member_count(unsigned s) { return (s >>  4) & 0x7Fu; }
__host__ __device__ __forceinline__ unsigned barrier_signal_count(unsigned s) { return (s >> 16) & 0x7Fu; }

/**
 * @brief Compile-time check that a set of named barrier ids is collision-free.
 *
 * Belt and braces for a multi-barrier protocol: `named_barrier` catches an id that
 * is out of range, and this catches a *set* that would alias even when every member
 * is individually legal -- which cannot happen at `NUM_NAMED_BARRIERS = 4` with
 * distinct ids, but would the moment anyone raises the constant without re-probing.
 * Instantiate it once next to the protocol's id definitions so the layout is
 * checked where it is declared rather than where it is used.
 */
template<int... IDS>
struct named_barrier_set {
    static_assert(((IDS >= 1 && IDS <= NUM_NAMED_BARRIERS) && ...),
        "every named barrier id in the set must be in [1, NUM_NAMED_BARRIERS].");
    static constexpr int count = sizeof...(IDS);
    static_assert(count <= NUM_NAMED_BARRIERS,
        "this protocol asks for more named barriers than the part provides distinct "
        "ones, so at least two of its logical barriers WILL alias onto the same "
        "hardware barrier. That fails silently, not loudly. Reduce the barrier count "
        "-- e.g. a two-edge producer/consumer protocol needs 2*STAGES barriers, so "
        "STAGES must be <= 2 on this part.");
};

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
 * Not equivalent to `__syncthreads()`, which also drains LDS before signalling.
 * This is the rendezvous alone, pair it with the `wait_*` for whatever the other 
 * warps are about to read.
 *
 * Prefer the split form (`arrive()`, then independent work, then `wait()`) when
 * the window between signalling and waiting can be filled with instructions
 * that do not depend on the barrier.
 */
__device__ __forceinline__ void sync() {
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
 */
template<int N = 0>
__device__ __forceinline__ void wait_tdm() {
    static_assert(N >= 0 && N < 64, "tensorcnt is 6-bit; max 63");
    __builtin_amdgcn_s_wait_tensorcnt(N);
}

/* ----------  LDS BARRIER CELLS (FOR TDM / ASYNC ARRIVE)  ---------- */
//
// 64-bit LDS-resident barrier cell, per SP3 section 9.8.13
// (DS_ATOMIC_ASYNC_BARRIER_ARRIVE_B64). The cell is packed as:
//
//   bits 63..48 : reserved (zero)
//   bits 47..32 : init_count        (reload value at phase flip)
//   bits 31..0  : bar_state, itself packed as [phase | pending_count]
//                 with the WIDTH boundary at bit 29:
//                   bits 31..29 : phase            (counter, parity alternates)
//                   bits 28..0  : pending_count    (29-bit)
//
//
// Each arrive subtracts 1 from bar_state. When pending rolls under (its
// MSB becomes 1), the hardware reloads bar_state to
//   (new_phase << 29) | init_count
// and wakes any wave sleeping on the cell.
//
// To expect N arrivals per phase: pending starts at N-1 and init_count is
// N-1. Phase decrements per flip, so its LSB alternates 0,1,0,1... -- the
// classic parity-flip pattern.
static constexpr int BARRIER_PHASE_BIT = 29;

/**
 * @brief 64-bit LDS barrier cell.
 *
 * Allocate as `__shared__ kittens::sync::barrier_lds bar;` and prime once
 * with `init_barrier(&bar.state, count)` before any arrive. `count` is the
 * number of arrivals required to flip the phase.
 */
struct alignas(8) barrier_lds { uint64_t state; };

/// @brief Initialize an LDS barrier cell to expect `count` arrivals per phase.
__device__ __forceinline__ void init_barrier(uint64_t* bar, uint32_t count) {
    // pending = count - 1, phase = 0, init_count = count - 1.
    const uint32_t pending  = count - 1;
    const uint32_t init_cnt = count - 1;
    *bar =  uint64_t(pending  & 0x1FFFFFFFu)                 // bits 28..0 pending (29-bit)
         | (uint64_t(0)                  << BARRIER_PHASE_BIT) // bit 29  phase = 0
         | (uint64_t(init_cnt & 0xFFFFu) << 32);            // bits 47..32 init_count
}

/**
 * @brief Block on `bar` until its phase LSB matches `expected_phase`.
 *
 * Phase decrements once per flip, so its low bit alternates 0,1,0,1...
 * Callers maintain a parity bit per barrier and pass it inverted before
 * each wait (`expected = (phase ^= 1)`). The hardware wakes sleeping
 * waves on phase flip; `s_sleep 1` yields the SIMD between polls.
 */
__device__ __forceinline__ void wait_barrier(uint64_t* bar, int expected_phase) {
    const uint32_t lds_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(bar));
    while (true) {
        uint64_t v;
        asm volatile("ds_load_b64 %0, %1 offset:0"
            : "=v"(v) : "v"(lds_addr) : "memory");
        const uint32_t bar_state = static_cast<uint32_t>(v);
        const int phase_lsb = int((bar_state >> BARRIER_PHASE_BIT) & 1);
        if (phase_lsb == expected_phase) break;
        __builtin_amdgcn_s_sleep(1);
    }
}

/**
 * @brief Arrive at an LDS barrier cell from an async-ordered path.
 *
 * Lowers to `DS_ATOMIC_ASYNC_BARRIER_ARRIVE_B64`. Use this to manually
 * arrive at a cell (the auto-arrive form is encoded in the TDM descriptor
 * via `tdm::arrive`).
 */
__device__ __forceinline__ void async_barrier_arrive(uint64_t* lds_counter) {
    uintptr_t lds_uint = reinterpret_cast<uintptr_t>(lds_counter);
    __builtin_amdgcn_ds_atomic_async_barrier_arrive_b64(
        reinterpret_cast<long __attribute__((address_space(3)))*>(lds_uint));
}

} // namespace sync
} // namespace kittens

