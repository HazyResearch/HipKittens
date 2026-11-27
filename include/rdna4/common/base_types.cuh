/**
 * @file
 * @brief Declarations, manipulations, and wrappers for basic types.
 * 
 * This file is a bunch of utilities for going back and forth between different types.
 * 
 * Many of them are for the compiler, so as to clean up the code. It unfortunately
 * seems necessary when we have types we really care about that are less than word width.
 */

#pragma once

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_fp4.h>
#include <hip/amd_detail/amd_hip_ocp_types.h>
#include <hip/hip_runtime.h>
#include <string>
#include <bit>

typedef uint32_t __amd_fp8x4_storage_t;

namespace kittens {

// /**
//  * @brief Bfloat16 floating-point type.
//  */
using bf16 = __hip_bfloat16;
/**
 * @brief Half-precision floating-point type.
 */
using half = __half;
// /**
//  * @brief Packed word of two bfloat16 floating-point values.
//  */
using bf16_2 = __hip_bfloat162;
/**
 * @brief Packed word of two half-precision floating-point values.
 */
using half_2 = __half2;
/**
 * @brief float8 floating-point type.
 */
using fp8e4m3 = __hip_fp8_e4m3;
/**
 * @brief Packed word of two float8 floating-point values.
 */
using fp8e4m3_2 = __hip_fp8x2_e4m3;
/**
 * @brief Packed word of four float8 floating-point values.
 */
using fp8e4m3_4 = __hip_fp8x4_e4m3;

// int8 for v_wmma_i32_16x16x16_iu8. At wave32 a 16x16 int8 tile is 8 elements
// per lane = 8 bytes = 2 VGPRs, which is exactly the instruction's srcA/srcB
// width, so a 4-packed element type gives packed_per_thread == 2 and the
// rt_base storage needs no special-casing. The i32 accumulator reuses int2,
// giving packed_per_thread == 4 and 8 VGPRs -- the instruction's srcC/D width.
// NOTE: no packing<>/constants<> specialisation for the SCALAR int8. `int8_t` is
// `signed char`, and `fp8e8m0` is already a typedef of the same type -- the tree
// has specialised on it, so a second one is a redefinition, not an addition.
// int8_4 is a distinct struct and can be specialised freely. The MMA overloads
// take int8_4 directly, so nothing here needs the scalar trait.
using int8 = int8_t;
struct alignas(4) int8_4 { int8_t x, y, z, w; };
/**
 * @brief 8-bit exponent-only block-scaling scale type.
 */
using fp8e8m0 = __amd_scale_t;
/**
 * @brief Packed word of two 8-bit exponent-only block-scaling scale values.
 */
using fp8e8m0_2 = __amd_fp8x2_storage_t;
/**
 * @brief Packed word of four 8-bit exponent-only block-scaling scale values.
 */
using fp8e8m0_4 = __amd_fp8x4_storage_t;
/**
 * @brief FP4 E2M1 floating-point type.
 */
using fp4e2m1   = __hip_fp4_e2m1;
/**
 * @brief Packed word of two FP4 E2M1 floating-point values.
 */
using fp4e2m1_2 = __hip_fp4x2_e2m1;
/**
 * @brief Packed word of four FP4 E2M1 floating-point values.
 */
using fp4e2m1_4 = __hip_fp4x4_e2m1;

namespace ducks {
/**
 * @namespace base_types
 *
 * @brief A namespace for concepts for basic data types.
 */
namespace base_types {

template<typename T>
concept T2 = std::is_same_v<T, float2> || std::is_same_v<T, bf16_2> || std::is_same_v<T, half_2> || std::is_same_v<T, fp8e4m3_4> || std::is_same_v<T, int8_4> || std::is_same_v<T, int2>
    || std::is_same_v<T, fp4e2m1_4>;
template<typename T>
concept T1 = std::is_same_v<T, float>  || std::is_same_v<T, bf16  > || std::is_same_v<T, half> || std::is_same_v<T, fp8e4m3> || std::is_same_v<T, int>
    || std::is_same_v<T, fp4e2m1>;

} // namespace base_types
} // namespace ducks

/**
 * @namespace base_types
 *
 * @brief A namespace for ThunderKittens basic data types.
 */
namespace base_types {

/**
 * @brief Provides compile-time constants for different types.
 *
 * @tparam T The type for which to provide constants.
 */
template<typename T> struct constants {
    /**
     * @brief Zero
     * @return Constexpr zero with type T
     */
    static __device__ inline constexpr T zero()      { return T{0}; }
    /**
     * @brief Ones
     * @return Constexpr ones with type T
     */
    static __device__ inline constexpr T ones()       { return T{1}; }
    /**
     * @brief Positive infinity. Particularly useful for initializing before a min op.
     * @return Constexpr positive infinity with type T
     */
    static __device__ inline constexpr T pos_infty() { return T{INFINITY}; } // I'll find a better way at some point but this appears to work.
    /**
     * @brief Negative infinity. Particularly useful for initializing before a max op.
     * @return Constexpr negative infinity with type T
     */
    static __device__ inline constexpr T neg_infty() { return T{-INFINITY}; }
};
template<> struct constants<float2> {
    static __device__ inline constexpr float2 zero()      { return float2{0.f, 0.f}; }
    static __device__ inline constexpr float2 ones()       { return float2{1.f, 1.f}; }
    static __device__ inline constexpr float2 pos_infty() { return float2{constants<float>::pos_infty(), constants<float>::pos_infty()}; }
    static __device__ inline constexpr float2 neg_infty() { return float2{constants<float>::neg_infty(), constants<float>::neg_infty()}; }
};
template<> struct constants<bf16> {
    static __device__ inline constexpr bf16 zero()      { return std::bit_cast<bf16>(uint16_t(0x0000)); } // unfortunately __float2bf16_rn is not constexpr
    static __device__ inline constexpr bf16 ones()       { return std::bit_cast<bf16>(uint16_t(0x3F80)); }
    static __device__ inline constexpr bf16 pos_infty() { return std::bit_cast<bf16>(uint16_t(0x7F80)); }
    static __device__ inline constexpr bf16 neg_infty() { return std::bit_cast<bf16>(uint16_t(0xFF80)); }
};
template<> struct constants<bf16_2> {
    static __device__ inline bf16_2 zero()      { return bf16_2{constants<bf16>::zero(),      constants<bf16>::zero()};      }
    static __device__ inline bf16_2 ones()       { return bf16_2{constants<bf16>::ones(),       constants<bf16>::ones()};       }
    static __device__ inline bf16_2 pos_infty() { return bf16_2{constants<bf16>::pos_infty(), constants<bf16>::pos_infty()}; }
    static __device__ inline bf16_2 neg_infty() { return bf16_2{constants<bf16>::neg_infty(), constants<bf16>::neg_infty()}; }
};
template<> struct constants<half> {
    static __device__ inline constexpr half zero()      { return std::bit_cast<half>(uint16_t(0x0000)); }
    static __device__ inline constexpr half ones()       { return std::bit_cast<half>(uint16_t(0x3C00)); }
    static __device__ inline constexpr half pos_infty() { return std::bit_cast<half>(uint16_t(0x7C00)); }
    static __device__ inline constexpr half neg_infty() { return std::bit_cast<half>(uint16_t(0xFC00)); }
};
template<> struct constants<half_2> {
    static __device__ inline constexpr half_2 zero()      { return std::bit_cast<half_2>(uint32_t(0x00000000)); }
    static __device__ inline constexpr half_2 ones()       { return std::bit_cast<half_2>(uint32_t(0x3C003C00)); }
    static __device__ inline constexpr half_2 pos_infty() { return std::bit_cast<half_2>(uint32_t(0x7C007C00)); }
    static __device__ inline constexpr half_2 neg_infty() { return std::bit_cast<half_2>(uint32_t(0xFC00FC00)); }
};
template<> struct constants<int8_4> {
    static __device__ inline constexpr int8_4 zero() { return std::bit_cast<int8_4>(uint32_t(0x00000000)); }
    static __device__ inline constexpr int8_4 one()  { return std::bit_cast<int8_4>(uint32_t(0x01010101)); }
};
template<> struct constants<fp8e4m3> {
    static __device__ inline constexpr fp8e4m3 zero() { return std::bit_cast<fp8e4m3>(uint8_t(0x00)); }
    static __device__ inline constexpr fp8e4m3 one() { return std::bit_cast<fp8e4m3>(uint8_t(0x38)); }
};
template<> struct constants<fp8e4m3_2> {
    static __device__ inline constexpr fp8e4m3_2 zero() { return std::bit_cast<fp8e4m3_2>(uint16_t(0x0000)); }
    static __device__ inline constexpr fp8e4m3_2 one() { return std::bit_cast<fp8e4m3_2>(uint16_t(0x3838)); }
};
template<> struct constants<fp8e4m3_4> {
    static __device__ inline constexpr fp8e4m3_4 zero() { return std::bit_cast<fp8e4m3_4>(uint32_t(0x00000000)); }
    static __device__ inline constexpr fp8e4m3_4 one() { return std::bit_cast<fp8e4m3_4>(uint32_t(0x38383838)); }
};
template<> struct constants<fp8e8m0> {
    static __device__ inline constexpr fp8e8m0 zero() { return std::bit_cast<fp8e8m0>(uint8_t(0x00)); } // not actually 0
    static __device__ inline constexpr fp8e8m0 one() { return std::bit_cast<fp8e8m0>(uint8_t(0x7F)); }
};
template<> struct constants<fp8e8m0_2> {
    static __device__ inline constexpr fp8e8m0_2 zero() { return std::bit_cast<fp8e8m0_2>(uint16_t(0x0000)); } // not actually 0
    static __device__ inline constexpr fp8e8m0_2 one() { return std::bit_cast<fp8e8m0_2>(uint16_t(0x7F7F)); }
};
template<> struct constants<fp8e8m0_4> {
    static __device__ inline constexpr fp8e8m0_4 zero() { return std::bit_cast<fp8e8m0_4>(uint32_t(0x00000000)); } // not actually 0
    static __device__ inline constexpr fp8e8m0_4 one() { return std::bit_cast<fp8e8m0_4>(uint32_t(0x7F7F7F7F)); }
};
template<> struct constants<fp4e2m1> {
    static __device__ inline constexpr fp4e2m1 zero() { return std::bit_cast<fp4e2m1>(uint8_t(0x00)); }
    static __device__ inline constexpr fp4e2m1 one()  { return std::bit_cast<fp4e2m1>(uint8_t(0x02)); }
};
template<> struct constants<fp4e2m1_4> {
    static __device__ inline constexpr fp4e2m1_4 zero() { return std::bit_cast<fp4e2m1_4>(uint16_t(0x0000)); }
    static __device__ inline constexpr fp4e2m1_4 one()  { return std::bit_cast<fp4e2m1_4>(uint16_t(0x2222)); }
};
template<> struct constants<int> {
    static __device__ inline constexpr int zero()      { return 0; }
    static __device__ inline constexpr int ones()       { return 1; }
};
template<> struct constants<int2> {
    static __device__ inline constexpr int2 zero()      { return int2{0, 0}; }
    static __device__ inline constexpr int2 ones()       { return int2{1, 1}; }
};

/**
 * @brief Provides information about packing of elements for a given type.
 *
 * @tparam T The type for which to provide packing information.
 */
template<typename T> struct packing {
    /**
     * @brief The number of elements packed together.
     *
     * @return constexpr int representing number of elements within the type.
     */
    static __host__ __device__ inline constexpr int num() { return 1; }
    /**
     * @brief Packs a single T element twice (replicated) into its packed type.
     *
     * @param i[in] The element to pack.
     * @return The packed type.
     */
    static __device__ inline constexpr T pack(const auto &i);
};
template<> struct packing<bf16> {
    static __host__ __device__ inline constexpr int num() { return 1; }
    using unpacked_type = bf16;
    using packed_type = bf16_2;
    static __device__ inline bf16_2 pack(const bf16 &i) { return bf16_2{i, i}; }
};
template<> struct packing<bf16_2> {
    static __host__ __device__ inline constexpr int num() { return 2; }
    using unpacked_type = bf16;
    using packed_type = bf16_2;
    static __device__ inline bf16_2 pack(const bf16 &i) { return bf16_2{i, i}; } // this replication makes code cleaner later.
};
template<> struct packing<half> {
    static __host__ __device__ inline constexpr int num() { return 1; }
    using unpacked_type = half;
    using packed_type = half_2;
    static __device__ inline constexpr half_2 pack(const half &i) { return half_2{i, i}; }
};
template<> struct packing<half_2> {
    static __host__ __device__ inline constexpr int num() { return 2; }
    using unpacked_type = half;
    using packed_type = half_2;
    static __device__ inline constexpr half_2 pack(const half &i) { return half_2{i, i}; } // this replication makes code cleaner later.
};
template<> struct packing<float> {
    static __host__ __device__ inline constexpr int num() { return 1; }
    using unpacked_type = float;
    using packed_type = float2;
    static __device__ inline constexpr float2 pack(const float &i) { return float2{i, i}; }
};
template<> struct packing<float2> {
    static __host__ __device__ inline constexpr int num() { return 2; }
    using unpacked_type = float;
    using packed_type = float2;
    static __device__ inline constexpr float2 pack(const float &i) { return float2{i, i}; } // this replication makes code cleaner later.
};
template<> struct packing<int> {
    static __host__ __device__ inline constexpr int num() { return 1; }
    using unpacked_type = int;
    using packed_type = int2;
    static __device__ inline constexpr int2 pack(const int &i) { return int2{i, i}; } // this replication makes code cleaner later.
};
template<> struct packing<int2> {
    static __host__ __device__ inline constexpr int num() { return 2; }
    using unpacked_type = int;
    using packed_type = int2;
    static __device__ inline constexpr int2 pack(const int &i) { return int2{i, i}; } // this replication makes code cleaner later.
};
template<> struct packing<float4> {
    static __host__ __device__ inline constexpr int num() { return 4; }
};
template<> struct packing<int4> {
    static __host__ __device__ inline constexpr int num() { return 4; }
};
template<> struct packing<fp8e4m3> {
    static __host__ __device__ inline constexpr int num() { return 1; }
    using unpacked_type = fp8e4m3;
    using packed_type = fp8e4m3_4;
};
template<> struct packing<fp8e4m3_4> {
    static __host__ __device__ inline constexpr int num() { return 4; }
    using unpacked_type = fp8e4m3;
    using packed_type = fp8e4m3_4;
};
template<> struct packing<fp8e8m0> {
    static __host__ __device__ inline constexpr int num() { return 1; }
    using unpacked_type = fp8e8m0;
    using packed_type = fp8e8m0_4;
};
template<> struct packing<fp8e8m0_4> {
    static __host__ __device__ inline constexpr int num() { return 4; }
    using unpacked_type = fp8e8m0;
    using packed_type = fp8e8m0_4;
};
template<> struct packing<fp4e2m1> {
    static __host__ __device__ inline constexpr int num() { return 1; }
    using unpacked_type = fp4e2m1;
    using packed_type = fp4e2m1_4;
};
template<> struct packing<fp4e2m1_4> {
    static __host__ __device__ inline constexpr int num() { return 4; }
    using unpacked_type = fp4e2m1;
    using packed_type = fp4e2m1_4;
};

/**
 * @brief Provides templated functionality to convert between different types.
 *
 * @tparam T The target type for conversion.
 * @tparam U The source type for conversion.
 */
template<typename T, typename U> struct convertor {
    /**
     * @brief Converts a value of type U to type T.
     *
     * @param u[in] The value of type U to convert.
     * @return T The converted value of type T.
     */
    static __host__ __device__ inline T convert(const U & u) {
        return (T)u;
    }
};
template<> struct convertor<float, bf16> {
    static __host__ __device__ inline float convert(const bf16 & u) {
        return 	__bfloat162float(u);
    }
};
// template<> struct convertor<bf16, float> {
//     static __host__ __device__ inline bf16 convert(const float & u) {
//         return 	__float2bfloat16(u);
//     }
// };
template<> struct convertor<bf16, float> {
    static __host__ __device__ inline bf16 convert(const float &u) {
        // Fast unsafe conversion (truncation only)
        return std::bit_cast<bf16>(
            static_cast<uint16_t>(
                std::bit_cast<uint32_t>(u) >> 16
            )
        );
    }
};
template<> struct convertor<float2, bf16_2> {
    static __host__ __device__ inline float2 convert(const bf16_2 & u) {
        return 	__bfloat1622float2(u);
    }
};

template<> struct convertor<bf16_2, float2> {
    // gfx1201 has NO f32 -> bf16 conversion instruction. This specialisation
    // previously emitted `v_cvt_pk_bf16_f32` as inline asm, inherited from the
    // gfx1250 tree this one was derived from. Verified against the assembler:
    //
    //     gfx950  (CDNA4)  : assembles
    //     gfx1250          : assembles
    //     gfx1201 (RDNA4)  : "instruction not supported on this GPU"
    //
    // So every caller of this specialisation failed to BUILD for gfx1201. It
    // went unnoticed because the GEMM never instantiates it -- its only other
    // user is `packed_shfl<bf16_2>` in util.cuh, which nothing in kernels/rdna4
    // called until attention needed the fp32 softmax result narrowed to bf16.
    //
    // ROUNDING MODE: round-to-nearest-even, via HIP's __float2bfloat16. Note
    // this deliberately does NOT match the scalar `convertor<bf16, float>`
    // directly above, which truncates and says so. Truncation is biased toward
    // zero, and in flash attention that bias does not cancel: the softmax
    // denominator is accumulated in fp32 from unrounded values while P is
    // narrowed to bf16, so a systematic downward bias on P survives into O
    // rather than dividing out. RNE is unbiased and costs ~13 VALU per pair
    // against ~1 for truncation (gfx1201 has no hardware help either way).
    // If that cost shows up in a profile, the fast path is 5 VALU -- two
    // v_bfe_u32, two v_add3_u32, one v_perm_b32 -- by dropping the NaN
    // handling, which is safe for softmax output but not in general.
    static __host__ __device__ inline bf16_2 convert(const float2 &u) {
        bf16_2 r;
        r.x = __float2bfloat16(u.x);
        r.y = __float2bfloat16(u.y);
        return r;
    }
};


template<> struct convertor<float, half> {
    static __host__ __device__ inline float convert(const half & u) {
        return __half2float(u);
    }
};
template<> struct convertor<half, float> {
    static __host__ __device__ inline half convert(const float & u) {
        return __float2half(u);
    }
};
template<> struct convertor<float2, half_2> {
    static __host__ __device__ inline float2 convert(const half_2 & u) {
        return __half22float2(u);
    }
};
template<> struct convertor<half_2, float2> {
    static __host__ __device__ inline half_2 convert(const float2 & u) {
        return __float22half2_rn(u);
    }
};
template<> struct convertor<bf16, half> {
    static __host__ __device__ inline bf16 convert(const half & u) {
        return __float2bfloat16(__half2float(u));
    }
};
template<> struct convertor<half, bf16> {
    static __host__ __device__ inline half convert(const bf16 & u) {
        return __float2half(__bfloat162float(u));
    }
};
template<> struct convertor<bf16_2, half_2> {
    static __host__ __device__ inline bf16_2 convert(const half_2 & u) {
        return __float22bfloat162_rn(__half22float2(u));
    }
};
template<> struct convertor<half_2, bf16_2> {
    static __host__ __device__ inline half_2 convert(const bf16_2 & u) {
        return __float22half2_rn(__bfloat1622float2(u));
    }
};
template<> struct convertor<fp8e4m3_4, float4> {
    static __host__ __device__ inline fp8e4m3_4 convert(const float4& u) {
        return fp8e4m3_4(u);
    }
};
template<> struct convertor<float4, fp8e4m3_4> {
    static __host__ __device__ inline float4 convert(const fp8e4m3_4& u) {
        fp8e4m3 *vals = reinterpret_cast<fp8e4m3*>(const_cast<fp8e4m3_4*>(&u));
        return make_float4(float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]));
    }
};
template<> struct convertor<fp8e4m3_2, float2> {
    static __host__ __device__ inline fp8e4m3_2 convert(const float2& u) {
        return fp8e4m3_2(u);
    }
};
template<> struct convertor<float2, fp8e4m3_2> {
    static __host__ __device__ inline float2 convert(const fp8e4m3_2& u) {
        fp8e4m3 *vals = reinterpret_cast<fp8e4m3*>(const_cast<fp8e4m3_2*>(&u));
        return make_float2(float(vals[0]), float(vals[1]));
    }
};
template<> struct convertor<fp8e4m3, float> {
    static __host__ __device__ inline fp8e4m3 convert(const float & u) {
        return fp8e4m3(u);
    }
};
template<> struct convertor<float, fp8e4m3> {
    static __host__ __device__ inline float convert(const fp8e4m3 & u) {
        return float(u);
    }
};
template<> struct convertor<fp4e2m1, float> {
    static __host__ __device__ inline fp4e2m1 convert(const float & u) {
        return fp4e2m1(u);
    }
};
template<> struct convertor<float, fp4e2m1> {
    static __host__ __device__ inline float convert(const fp4e2m1 & u) {
        return float(u);
    }
};
template<> struct convertor<fp4e2m1_4, float4> {
    static __host__ __device__ inline fp4e2m1_4 convert(const float4& u) {
        return fp4e2m1_4(u);
    }
};
template<> struct convertor<float4, fp4e2m1_4> {
    static __host__ __device__ inline float4 convert(const fp4e2m1_4& u) {
        return float4(u);
    }
};
}
}
