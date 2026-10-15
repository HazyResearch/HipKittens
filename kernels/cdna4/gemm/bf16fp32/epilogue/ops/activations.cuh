#pragma once
#include "epilogue_args.cuh"

// SiLU (swish): silu(x) = x * sigmoid(x) = x / (1 + e^-x).
// Computed as x * rcp(1 + exp2(-x * log2(e))) with the hardware v_exp_f32 and v_rcp_f32:
// 5 instructions on gfx950, versus 14 for the IEEE-divide form. Measured max relative error
// 2e-6 for x in [-40, 40], far below bf16's 3.9e-3 rounding step. For very negative x,
// exp2 overflows to inf and rcp(inf) = 0, so the result is -0 rather than a tiny denormal.
namespace epilogue_ops {
struct fast_silu {
    static constexpr float NEG_LOG2E = -1.4426950408889634f;
    template<typename T> static __device__ inline T op(const T& x);
};
template<> __device__ inline float fast_silu::op<float>(const float& x) {
    return x * __builtin_amdgcn_rcpf(1.0f + __builtin_amdgcn_exp2f(x * NEG_LOG2E));
}
template<> __device__ inline float2 fast_silu::op<float2>(const float2& x) {
    return float2{ op<float>(x.x), op<float>(x.y) };
}
}  // namespace epilogue_ops

// In-place SiLU on a register tile.
template<typename T>
__device__ inline void silu_op(T& x) {
    unary_map<epilogue_ops::fast_silu, T>(x, x);
}
