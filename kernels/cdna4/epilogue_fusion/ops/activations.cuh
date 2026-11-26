#pragma once
#include "epilogue_args.cuh"

// silu(x) = x * rcp(1 + exp2(-x * log2(e))) via v_exp_f32/v_rcp_f32: 5 instructions vs 14 for IEEE divide,
// max rel. error 2e-6 on [-40, 40] (bf16 step 3.9e-3). Very negative x gives -0.
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

template<typename T>
__device__ inline void silu_op(T& x) {
    unary_map<epilogue_ops::fast_silu, T>(x, x);
}
