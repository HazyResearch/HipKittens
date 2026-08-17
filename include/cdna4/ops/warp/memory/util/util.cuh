/**
 * @file
 * @brief General memory utilities not specialized for either tiles or vectors.
 */
#pragma once

#include "../../../../common/common.cuh"
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
    typedef float __attribute__((ext_vector_type(2))) v2f32_t;
    typedef const v2f32_t __attribute__((address_space(3))) * v2f32_lds_cptr_t;
    v2f32_t raw = *reinterpret_cast<v2f32_lds_cptr_t>(lds_off);
    float2 r; r.x = raw[0]; r.y = raw[1];
    return r;
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
    return *gptr;
}

__device__ inline float4 load_global_vec4(const float4* gptr) {
    return *gptr;
}

__device__ inline buffer_resource make_buffer_resource(uint64_t ptr, uint32_t range, uint32_t config) {
    return {ptr, range, config};
}
__device__ inline __amdgpu_buffer_rsrc_t make_srsrc(const void* ptr, uint32_t range_bytes, uint32_t row_stride_bytes = 0) {
    short stride_field = 0;
    if (row_stride_bytes) {
        stride_field = (short)((row_stride_bytes & 0x3FFF) | 0x4000 | 0x8000);
    }
    return __builtin_amdgcn_make_buffer_rsrc(
        const_cast<void*>(ptr), stride_field,
        (uint64_t)range_bytes, 0x110000);
}

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

using as3_uint32_ptr = uint32_t __attribute__((address_space(3)))*;
using int32x4_t = int32_t __attribute__((ext_vector_type(4)));

// Legacy intrinsic declaration kept for kernels that bypass the library load() path.
extern "C" __device__ void 
llvm_amdgcn_raw_buffer_load_lds(int32x4_t rsrc,
                                as3_uint32_ptr lds_ptr,
                                int size,
                                int voffset, 
                                int soffset, 
                                int offset,
                                int aux) __asm("llvm.amdgcn.raw.buffer.load.lds");

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
