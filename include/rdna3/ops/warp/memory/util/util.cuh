/**
 * @file
 * @brief General memory utilities not specialized for either tiles or vectors.
 */
#pragma once

#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/amd_detail/hip_ldg.h>

namespace kittens {

enum class coherency {
    cache_all = 0,
    cache_global = 1,
    cache_stream = 2,
    non_temporal = 3
};

/* ----------   Shared memory utilities  ---------- */
__device__ inline float2 load_shared_vec(uint32_t lds_off) {
    float2 result;
    asm volatile(
        "ds_read_b64 %0, %1\n"
        "s_waitcnt lgkmcnt(0)\n"
        : "=v"(result)              // Output: store result in float2
        : "v"(lds_off)              // Input: LDS offset to read from
        : "memory"
    );
    return result;
}

__device__ inline void store_shared_vec(uint32_t lds_off, float2 val) {
    asm volatile(
        "ds_write_b64 %0, %1\n"
        :
        : "v"(lds_off), "v"(val)
        : "memory"
    );
}

__device__ inline float2 load_global_vec2(const float2* gptr) {
    float2 v;
    // Use global_load_dwordx2 which is more cache-friendly than flat_load
    asm volatile(
        "global_load_dwordx2 %0, %1, off\n"
        "s_waitcnt vmcnt(0)\n"
        : "=v"(v) 
        : "v"(gptr)
        : "memory"
    );
    return v;   
}

__device__ inline float4 load_global_vec4(const float4* gptr) {
    float4 v;
    // Use global_load_dwordx4 which is more cache-friendly than flat_load
    asm volatile(
        "global_load_dwordx4 %0, %1, off\n"
        "s_waitcnt vmcnt(0)\n"
        : "=v"(v) 
        : "v"(gptr)
        : "memory"
    );
    return v;   
}

#ifndef KITTENS_ROW_BOUNDS
#define KITTENS_ROW_BOUNDS 1
#endif

/**
 * @brief How many of a tile's TILE_ROWS rows, starting at element coordinate
 *        `uc`, lie inside `g` along `axis` -- clamped to [0, TILE_ROWS].
 *
 * The row predicate every global load and store below applies: past it a load
 * reads zero and a store writes nothing, so a tile hanging off the end of a
 * tensor is legal instead of a read past the allocation. It is one scalar
 * compare, and for a whole tile every caller takes its unpredicated path.
 * Where the loads are buffer ops the predicate is the buffer's range and the
 * instruction count does not change, which is what keeps a hand-placed vmcnt
 * a compile-time immediate.
 *
 * KITTENS_ROW_BOUNDS=0 treats every tile as whole -- the behaviour before the
 * predicate existed, kept so the two can be timed against each other.
 */
template<int axis, int TILE_ROWS, typename GL, typename COORD>
__device__ inline int rows_in_bounds(const GL &g, const COORD &uc) {
#if KITTENS_ROW_BOUNDS
    const int start = axis == 0 ? uc.b : axis == 1 ? uc.d : uc.r;
    const int left = int(g.template shape<axis>()) - start;
    return left <= 0 ? 0 : (left < TILE_ROWS ? left : TILE_ROWS);
#else
    return TILE_ROWS;
#endif
}

using i32x4 = int32_t __attribute__((ext_vector_type(4)));
struct buffer_resource {
    uint64_t ptr;
    uint32_t range;
    uint32_t config;
};

__device__ inline buffer_resource make_buffer_resource(uint64_t ptr, uint32_t range, uint32_t config) {
    return {ptr, range, config};
}

/*
 * gfx11 buffer resource descriptor (V#).
 *
 * Word 3 is *not* the gfx9 config word the CDNA tree uses. Measured on gfx1100
 * (see tools/rdna-probes/srsrc3.hip), a descriptor is only usable if both hold:
 *
 *   OOB_SELECT (bits 29:28) == 3.  At any smaller value NumRecords is counted in
 *     units of STRIDE, and STRIDE is 0 for a raw buffer -- so *every* access is
 *     out of bounds. Loads return zero and stores are dropped, silently, with no
 *     fault and no warning. Both gfx9 words (0x00020000 from global_to_register
 *     and 0x00110000 from the old make_srsrc) fail exactly this way on gfx1100.
 *   FORMAT (bits 18:12) != 0.  Non-format buffer ops ignore what the format is,
 *     but 0 is BUF_FMT_INVALID and invalidates the descriptor: 0x30000000 reads
 *     back zeros while 0x30004000 works.
 *
 * 0x31014000 satisfies both and matches what LLVM emits for gfx10/gfx11.
 */
static constexpr uint32_t GFX11_BUFFER_CONFIG = 0x31014000u;

/**
 * @brief Build a raw buffer descriptor over `range_bytes` starting at `ptr`.
 *
 * Accesses at or past `range_bytes` are clamped by the hardware: loads read 0,
 * stores are dropped. That bounds check is the whole reason to prefer buffer
 * ops over plain pointers here -- it makes ragged edge tiles free.
 *
 * Unlike the CDNA version this takes no row stride. The stride field only means
 * anything for *structured* buffers, and turning swizzling on would change the
 * address computation out from under the raw byte offsets every caller passes.
 */
__device__ inline i32x4 make_srsrc(const void* ptr, uint32_t range_bytes) {
    std::uintptr_t as_int = reinterpret_cast<std::uintptr_t>(ptr);   // width = sizeof(void*)
    std::uint64_t  as_u64 = static_cast<std::uint64_t>(as_int);    // widen if host is 32-bit
    buffer_resource rsrc = make_buffer_resource(as_u64, range_bytes, GFX11_BUFFER_CONFIG);
    i32x4 r = *reinterpret_cast<const i32x4*>(&rsrc);
    // A buffer instruction takes its descriptor in SGPRs. Every caller builds
    // it from a tile coordinate, which is uniform across the wave, but LLVM
    // cannot always see that -- a coordinate derived from the warp index looks
    // per-lane -- and then wraps each buffer op in a waterfall loop. On the
    // GEMM's bounded epilogue that was 64 loops and 8% of the kernel.
    r.x = __builtin_amdgcn_readfirstlane(r.x);
    r.y = __builtin_amdgcn_readfirstlane(r.y);
    r.z = __builtin_amdgcn_readfirstlane(r.z);
    r.w = __builtin_amdgcn_readfirstlane(r.w);
    return r;
}

__device__ uint16_t llvm_amdgcn_raw_buffer_load_b16(i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.load.i16");

__device__ uint32_t llvm_amdgcn_raw_buffer_load_b32(i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.load.i32");

__device__ uint64_t llvm_amdgcn_raw_buffer_load_b64(i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.load.i64");

__device__ __uint128_t llvm_amdgcn_raw_buffer_load_b128(i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.load.i128");

__device__ void llvm_amdgcn_raw_buffer_store_b8(uint8_t vdata, i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.store.i8");

__device__ void llvm_amdgcn_raw_buffer_store_b16(uint16_t vdata, i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.store.i16");

__device__ void llvm_amdgcn_raw_buffer_store_b32(uint32_t vdata, i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.store.i32");

__device__ void llvm_amdgcn_raw_buffer_store_b64(uint64_t vdata, i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.store.i64");

__device__ void llvm_amdgcn_raw_buffer_store_b128(__uint128_t vdata, i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.store.i128");


__device__ inline float2 load_global_vec2_async(const float2* gptr) {
    float2 v;
    // Use global_load_dwordx2 which is more cache-friendly than flat_load
    asm volatile(
        "global_load_dwordx2 %0, %1, off\n"
        : "=v"(v) 
        : "v"(gptr)
        : "memory"
    );
    return v;   
}

__device__ inline float4 load_global_vec4_async(const float4* gptr) {
    float4 v;
    // Use global_load_dwordx4 which is more cache-friendly than flat_load
    asm volatile(
        "global_load_dwordx4 %0, %1, off\n"
        : "=v"(v) 
        : "v"(gptr)
        : "memory"
    );
    return v;   
}

__device__ inline void store_global_b128_async(void* gptr, __uint128_t value) {
    asm volatile(
        "global_store_dwordx4 %0, %1, off\n"
        :
        : "v"(gptr), "v"(value)
        : "memory"
    );
}

__device__ inline float2 load_shared_vec_async(uint32_t lds_off) {
    float2 result;
    asm volatile(
        "ds_read_b64 %0, %1\n"
        // "s_waitcnt lgkmcnt(0)\n"
        : "=v"(result)              // Output: store result in float2
        : "v"(lds_off)              // Input: LDS offset to read from
        : "memory"
    );
    return result;
}

/**
 * @brief Wait until at most `N` LDS operations are still outstanding.
 *
 * LDS returns in order, so lds_wait<N>() after issuing a batch of M+N reads
 * retires exactly the first M of them. That is what lets a caller issue the
 * reads for K-slice k+1, then wait only on slice k's, and run slice k's math
 * with the next slice's reads still in flight.
 */
template<int N=0> __device__ inline void lds_wait() {
    static_assert(N >= 0 && N <= 63, "lgkmcnt is 6 bits on gfx11");
    asm volatile("s_waitcnt lgkmcnt(%0)" :: "i"(N) : "memory");
}

namespace detail {
typedef float bind_v4f __attribute__((ext_vector_type(4)));

// The offset is a template parameter, not a loop index, so every access is a
// constant offset in the IR that SROA sees -- an object reached through a
// runtime offset would be forced to scratch, which on a hand-waited kernel is
// worse than the bug this is fixing.
template<int I, int N, typename T> __device__ inline void bind_regs(T &x) {
    if constexpr (I < N) {
        bind_v4f v;
        __builtin_memcpy(&v, (const char *)&x + I * sizeof(bind_v4f), sizeof(v));
        asm volatile("" : "+v"(v));
        __builtin_memcpy((char *)&x + I * sizeof(bind_v4f), &v, sizeof(v));
        bind_regs<I + 1, N>(x);
    }
}
}  // namespace detail

/**
 * @brief Re-establish the register dependence that s_waitcnt does not carry.
 *
 * `load_shared_vec4_async` writes its destination through an inline-asm *output*
 * operand, so LLVM believes the value is ready the instant the asm statement
 * ends. `lds_wait` is a separate volatile asm sharing no operands with it, and
 * `v_wmma` is a pure builtin with no memory effect -- so nothing in the IR stops
 * the scheduler from hoisting the consuming WMMA above the wait, and it does,
 * nondeterministically, as soon as register pressure shifts. It is not a spill:
 * ScratchSize stays 0 while the answer changes run to run.
 *
 * Binding the destination after the wait puts the missing edge back. The bind
 * cannot move above the wait (volatile asm is ordered against volatile asm) and
 * the consumer cannot move above the bind (it reads the bind's outputs). It
 * costs nothing at runtime: "+v" ties each output to its own input register, so
 * no instruction is emitted.
 *
 * Any tile loaded with `load<false>` and waited on by hand has to go through
 * this (or through `lds_wait_for`, which does both) before it is used.
 */
template<typename T> __device__ inline void lds_bind(T &x) {
    static_assert(sizeof(T) % sizeof(detail::bind_v4f) == 0,
                  "lds_bind takes whole 128-bit register groups");
    detail::bind_regs<0, sizeof(T) / sizeof(detail::bind_v4f)>(x);
}

/**
 * @brief `lds_wait<N>()` followed by `lds_bind` on each tile it retires.
 *
 * The form to reach for: the wait alone is not enough to order a WMMA against
 * the reads it is waiting on. See lds_bind.
 */
template<int N=0, typename... Ts> __device__ inline void lds_wait_for(Ts&... xs) {
    lds_wait<N>();
    (lds_bind(xs), ...);
}

/**
 * @brief Wait until at most `N` vector-memory loads are still outstanding.
 *
 * The vmcnt analogue of lds_wait. Buffer loads return in issue order, so a
 * caller that keeps D batches of `B` loads in flight retires the oldest batch
 * with vm_wait<(D-1)*B>() and leaves the newer D-1 batches moving. That is how
 * the GEMM gets a global prefetch distance of more than one K-tile without
 * more than one barrier.
 */
template<int N=0> __device__ inline void vm_wait() {
    static_assert(N >= 0 && N <= 63, "vmcnt is 6 bits on gfx11");
    asm volatile("s_waitcnt vmcnt(%0)" :: "i"(N) : "memory");
}

/*
 * 128-bit LDS access.
 *
 * These have to be inline asm for the same reason the b64 pair above does, and
 * the reason is worth stating because it is not obvious from the C++: a shared
 * tile reached through shared_allocator is a *generic* pointer as far as the
 * compiler is concerned, so writing `*(float4*)p` gets you flat_load_b128, not
 * ds_read_b128.  Both address the same bytes, but flat goes out through the
 * memory pipe and counts against vmcnt.  Taking the 32-bit LDS offset and
 * naming the instruction is what pins the access to the LDS pipe.
 *
 * Neither waits: the callers issue a batch and then a single s_waitcnt.
 */
// The asm operands are the native vector type rather than HIP's float4: a
// 16-byte HIP_vector_type is not something clang will place in a register for a
// "v" *input* constraint ("indirect register inputs"), though it accepts it as
// an output.  The memcpys are free; both types are four consecutive dwords.
typedef float raw_v4f __attribute__((ext_vector_type(4)));

__device__ inline float4 load_shared_vec4_async(uint32_t lds_off) {
    raw_v4f v;
    asm volatile(
        "ds_read_b128 %0, %1\n"
        : "=v"(v)
        : "v"(lds_off)
        : "memory"
    );
    float4 result;
    __builtin_memcpy(&result, &v, sizeof(v));
    return result;
}

__device__ inline void store_shared_vec4(uint32_t lds_off, float4 val) {
    raw_v4f v;
    __builtin_memcpy(&v, &val, sizeof(v));
    asm volatile(
        "ds_write_b128 %0, %1\n"
        :
        : "v"(lds_off), "v"(v)
        : "memory"
    );
}

/* ----------   To prevent generic addressing  ---------- */

template<typename T> struct move {
    __device__ static inline void lds(T& dst, uint32_t src);
    __device__ static inline void sts(uint32_t dst, const T& src);
    __device__ static inline void ldg(T& dst, T* src);
    __device__ static inline void stg(T* dst, const T& src);
};

// meant to be used only with shared tiles and shared vectors
namespace detail {
template<typename T> struct size_info {
    static constexpr uint32_t bytes    = sizeof(std::remove_reference_t<T>);
};
template<ducks::st::all ST> struct size_info<ST> {
    static constexpr uint32_t elements = ST::num_elements;
    static constexpr uint32_t bytes    = ST::num_elements * sizeof(typename ST::dtype);
};
template<ducks::sv::all SV> struct size_info<SV> {
    static constexpr uint32_t elements = SV::length;
    static constexpr uint32_t bytes    = SV::length * sizeof(typename SV::dtype);
};
}
template<typename... Args>                       inline constexpr uint32_t size_bytes             = 0; // base case
template<typename T, typename... Args>           inline constexpr uint32_t size_bytes<T, Args...> = detail::size_info<T>::bytes + size_bytes<Args...>; // recursive case

} // namespace kittens
