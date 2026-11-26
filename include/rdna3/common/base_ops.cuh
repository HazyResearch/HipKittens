/**
 * @file
 * @brief Basic operations on generic types.
 */

#pragma once

#include <limits>
#include "base_types.cuh"

namespace kittens {

/**
 * @namespace base_ops
 *
 * @brief A namespace for operations on basic data types.
 */
namespace base_ops {

/* ----------  CONST OPS  ---------- */

/**
 * @brief Represents the zero constant operation.
 *
 * This operation returns the zero value of the specified type.
 *
 * @tparam T The data type for which to return the zero value.
 * @return The zero value of type T.
 */
struct zero {
    template<typename T, typename... args> __device__ static inline constexpr T op(args... _) { return base_types::constants<T>::zero();      }
};
/**
 * @brief Represents the one constant operation.
 *
 * This operation returns the one value of the specified type.
 *
 * @tparam T The data type for which to return the one value.
 * @return The one value of type T.
 */
struct one {
    template<typename T, typename... args> __device__ static inline constexpr T op(args... _) { return base_types::constants<T>::one();       }
};
/**
 * @brief Represents the positive infinity constant operation.
 *
 * This operation returns the positive infinity value of the specified type.
 *
 * @tparam T The data type for which to return the positive infinity value.
 * @return The positive infinity value of type T.
 */
struct pos_infty {
    template<typename T, typename... args> __device__ static inline constexpr T op(args... _) { return base_types::constants<T>::pos_infty(); }
};
/**
 * @brief Represents the negative infinity constant operation.
 *
 * This operation returns the negative infinity value of the specified type.
 *
 * @tparam T The data type for which to return the negative infinity value.
 * @return The negative infinity value of type T.
 */
struct neg_infty {
    template<typename T, typename... args> __device__ static inline constexpr T op(args... _) { return base_types::constants<T>::neg_infty(); }
};


/* ----------  UNARY OPS  ---------- */

/**
 * @brief Exponential function operation.
 *
 * This operation calculates the exponential of the input value.
 *
 * @tparam T The data type of the input and output values.
 * @param x[in] The input value.
 * @return The exponential of the input value.
 */
struct exp {
    template<typename T> static __device__ inline T op(const T &x) { return exp(x); }
};
template<> __device__ inline float  exp::op<float> (const float &x ) { return __expf(x);                        }
template<> __device__ inline float2 exp::op<float2>(const float2 &x) { return float2{__expf(x.x), __expf(x.y)}; }
template<> __device__ inline bf16   exp::op<bf16>  (const bf16 &x  ) { return hexp(x);                          }
template<> __device__ inline bf16_2 exp::op<bf16_2>(const bf16_2 &x) { return h2exp(x);                         }
template<> __device__ inline half   exp::op<half>  (const half &x  ) { return hexp(x);                          }
template<> __device__ inline half_2 exp::op<half_2>(const half_2 &x) { return h2exp(x);                         }

// /**
//  * @brief Exponential function operation, in base 2
//  *
//  * This operation calculates the exponential of the input value, in base 2.
//  *
//  * @tparam T The data type of the input and output values.
//  * @param x[in] The input value.
//  * @return The exponential of the input value.
//  */
// struct exp2 {
//     template<typename T> static __device__ inline T op(const T &x) { return exp2f(x); }
// };
// template<> __device__ inline float  exp2::op<float> (const float &x ) { return exp2f(x);                        }
// template<> __device__ inline float2 exp2::op<float2>(const float2 &x) { return float2{exp2f(x.x), exp2f(x.y)}; }
// template<> __device__ inline bf16   exp2::op<bf16>  (const bf16 &x  ) { return hexp2(x);                          }
// template<> __device__ inline bf16_2 exp2::op<bf16_2>(const bf16_2 &x) { return h2exp2(x);                         }
// template<> __device__ inline half   exp2::op<half>  (const half &x  ) { return hexp2(x);                          }
// template<> __device__ inline half_2 exp2::op<half_2>(const half_2 &x) { return h2exp2(x);                         }


/**
 * @brief Base-2 exponential operation using `__builtin_amdgcn_exp2_f32`
 *
 * Maps directly to `v_exp_f32_e32` on AMD, for highest performance.
 * Expects `x` to be in a safe numerical range (e.g., [-64, 88]).
 */
 struct exp2 {
    template <typename T>
    static __device__ inline T op(const T &x) {
        return exp2f(x);  // fallback
    }
};

// Force hardware v_exp_f32 for float
template<>
__device__ inline float exp2::op<float>(const float &x) {
    return __builtin_amdgcn_exp2f(x);  // Emits v_exp_f32_e32
}

// Force hardware v_exp_f32 for float2
template<>
__device__ inline float2 exp2::op<float2>(const float2 &x) {
    return {
        __builtin_amdgcn_exp2f(x.x),
        __builtin_amdgcn_exp2f(x.y)
    };
}

// Delegate to low-precision approximations
template<> __device__ inline half    exp2::op<half>(const half &x)     { return hexp2(x);  }
template<> __device__ inline half_2  exp2::op<half_2>(const half_2 &x) { return h2exp2(x); }
template<> __device__ inline bf16    exp2::op<bf16>(const bf16 &x)     { return hexp2(x);  }
template<> __device__ inline bf16_2  exp2::op<bf16_2>(const bf16_2 &x) { return h2exp2(x); }



/**
 * @brief Natural log function operation.
 *
 * This operation calculates the natural logarithm of the input value.
 *
 * @tparam T The data type of the input and output values.
 * @param x[in] The input value.
 * @return The natural logarithm of the input value.
 */
struct log {
    template<typename T> static __device__ inline T op(const T &x) { return log(x); }
};
template<> __device__ inline float  log::op<float> (const float &x ) { return __logf(x);                        }
template<> __device__ inline float2 log::op<float2>(const float2 &x) { return float2{__logf(x.x), __logf(x.y)}; }
template<> __device__ inline bf16   log::op<bf16>  (const bf16 &x  ) { return hlog(x);                          }
template<> __device__ inline bf16_2 log::op<bf16_2>(const bf16_2 &x) { return h2log(x);                         }
template<> __device__ inline half   log::op<half>  (const half &x  ) { return hlog(x);                          }
template<> __device__ inline half_2 log::op<half_2>(const half_2 &x) { return h2log(x);                         }
/**
 * @brief Logarithm base 2 operation.
 *
 * This operation calculates the logarithm base 2 of the input value.
 *
 * @tparam T The data type of the input and output values.
 * @param x[in] The input value.
 * @return The logarithm base 2 of the input value.
 */
struct log2 {
    template<typename T> static __device__ inline T op(const T &x) { return log2(x); }
};
template<> __device__ inline float  log2::op<float> (const float &x ) { return __log2f(x);                        }
template<> __device__ inline float2 log2::op<float2>(const float2 &x) { return float2{__log2f(x.x), __log2f(x.y)}; }
template<> __device__ inline bf16   log2::op<bf16>  (const bf16 &x  ) { return hlog2(x);                          }
template<> __device__ inline bf16_2 log2::op<bf16_2>(const bf16_2 &x) { return h2log2(x);                         }
template<> __device__ inline half   log2::op<half>  (const half &x  ) { return hlog2(x);                          }
template<> __device__ inline half_2 log2::op<half_2>(const half_2 &x) { return h2log2(x);                         }
/**
 * @brief Absolute value operation.
 *
 * This operation calculates the absolute value of the input.
 *
 * @tparam T The data type of the input and output values.
 * @param x[in] The input value.
 * @return The absolute value of the input.
 */
struct abs {
    template<typename T> static __device__ inline T op(const T &x) { return abs(x); }
};
template<> __device__ inline float  abs::op<float> (const float &x ) { return fabsf(x);                       }
template<> __device__ inline float2 abs::op<float2>(const float2 &x) { return float2{fabsf(x.x), fabsf(x.y)}; }
template<> __device__ inline bf16   abs::op<bf16>  (const bf16 &x  ) { return __habs(x);                      }
template<> __device__ inline bf16_2 abs::op<bf16_2>(const bf16_2 &x) { return __habs2(x);                     }
template<> __device__ inline half   abs::op<half>  (const half &x  ) { return __habs(x);                      }
template<> __device__ inline half_2 abs::op<half_2>(const half_2 &x) { return __habs2(x);                     }
/**
 * @brief Rectified Linear Unit (ReLU) operation.
 *
 * This operation applies the ReLU function to the input, which is the
 * maximum of zero and the input value.
 *
 * @tparam T The data type of the input and output values.
 * @param x[in] The input value.
 * @return The result of ReLU function applied to the input.
 */
struct relu {
    template<typename T> static __device__ inline T op(const T &x) { return max(x, base_types::constants<T>::zero()); }
};
template<> __device__ inline float  relu::op<float> (const float &x ) { return max(x, 0.f);                                  }
template<> __device__ inline float2 relu::op<float2>(const float2 &x) { return float2{max(x.x, 0.f), max(x.y, 0.f)};         }
template<> __device__ inline bf16   relu::op<bf16>  (const bf16 &x  ) { return __hmax(x, base_types::constants<bf16>::zero());    }
template<> __device__ inline bf16_2 relu::op<bf16_2>(const bf16_2 &x) { return __hmax2(x, base_types::constants<bf16_2>::zero()); }
template<> __device__ inline half   relu::op<half>  (const half &x  ) { return __hmax(x, base_types::constants<half>::zero());    }
template<> __device__ inline half_2 relu::op<half_2>(const half_2 &x) { return half_2{__hmax(x.x, base_types::constants<half>::zero()),
                                                                                     __hmax(x.y, base_types::constants<half>::zero())}; }

/**
 * @brief The unit step: 1 where x > 0, else 0. With a subtraction in front it
 * is a comparison as a multiplier -- a dropout's keep mask, x * step(r - p).
 */
struct step {
    template<typename T> static __device__ inline T op(const T &x);
};
template<> __device__ inline float  step::op<float> (const float &x ) { return x > 0.f ? 1.f : 0.f; }
template<> __device__ inline float2 step::op<float2>(const float2 &x) { return float2{x.x > 0.f ? 1.f : 0.f,
                                                                                     x.y > 0.f ? 1.f : 0.f}; }

constexpr float SQRT_2_OVER_PI   = 0.7978845608028654f;
constexpr float GELU_COEFF       = 0.044715f;
constexpr float GELU_INNER_COEFF = GELU_COEFF * SQRT_2_OVER_PI;
constexpr float DGELU_COEFF      = 3.0f * GELU_COEFF * SQRT_2_OVER_PI;

/// tanh via the hardware exp2, which gfx11 has as a full-rate transcendental.
/// The clamp keeps e2x finite; beyond +/-20 the result is 1 to within float.
static __device__ inline float fast_tanh(float x) {
    x = fmaxf(fminf(x, 20.f), -20.f);
    float e2x = __builtin_amdgcn_exp2f(x * 2.8853900817779268f);
    return (e2x - 1.0f) * __frcp_rn(e2x + 1.0f);
}

/**
 * @brief Gaussian Error Linear Unit (GELU) activation.
 *
 * Computes the GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
 *
 * @tparam T The data type of the input and output values.
 * @param x[in] The input value.
 * @return The GELU activation applied to the input.
 */
struct gelu {
    template<typename T> static __device__ inline T op(const T &x);
};
template<> __device__ inline float gelu::op<float>(const float &x) {
    return x * (0.5f + 0.5f * fast_tanh(x * (SQRT_2_OVER_PI + GELU_INNER_COEFF * x * x)));
}
template<> __device__ inline float2 gelu::op<float2>(const float2 &x) {
    return float2{gelu::op<float>(x.x), gelu::op<float>(x.y)};
}

/**
 * @brief Derivative of the GELU activation.
 *
 * Computes the derivative of the GELU approximation with respect to the input.
 *
 * @tparam T The data type of the input and output values.
 * @param x[in] The input value.
 * @return The derivative of GELU evaluated at the input.
 */
struct dgelu {
    template<typename T> static __device__ inline T op(const T &x);
};
template<> __device__ inline float dgelu::op<float>(const float &x) {
    float tanh_out = fast_tanh(SQRT_2_OVER_PI * x * (1.f + GELU_COEFF * x * x));
    return 0.5f * x * ((1.f - tanh_out * tanh_out) * (SQRT_2_OVER_PI + DGELU_COEFF * x * x)) +
           0.5f * (1.f + tanh_out);
}
template<> __device__ inline float2 dgelu::op<float2>(const float2 &x) {
    return float2{dgelu::op<float>(x.x), dgelu::op<float>(x.y)};
}

/**
 * @brief Square root.
 *
 * Float only, deliberately: the normalization kernels that need this
 * accumulate in fp32 regardless of the tensor's storage type, and a packed
 * 16-bit sqrt would invite computing a variance in bf16, which has eight
 * mantissa bits and loses the small differences a variance is made of.
 *
 * @tparam T The data type of the input and output values.
 * @param x[in] The input value.
 * @return The square root of the input.
 */
struct sqrt {
    template<typename T> static __device__ inline T op(const T &x);
};
template<> __device__ inline float  sqrt::op<float> (const float &x ) { return __builtin_amdgcn_sqrtf(x); }
template<> __device__ inline float2 sqrt::op<float2>(const float2 &x) {
    return float2{__builtin_amdgcn_sqrtf(x.x), __builtin_amdgcn_sqrtf(x.y)};
}

/**
 * @brief Reciprocal square root, 1/sqrt(x).
 *
 * `v_rsq_f32` in one instruction, which is the whole reason this exists as an
 * op rather than as a divide by a sqrt: the tail of every RMSNorm and
 * LayerNorm is one of these per row, and the divide form is three instructions
 * and a different rounding.
 *
 * ~1 ulp, like every other transcendental here (see exp, which uses
 * `v_exp_f32` for the same reason). Do not use it where the exact IEEE result
 * matters.
 *
 * @tparam T The data type of the input and output values.
 * @param x[in] The input value.
 * @return 1/sqrt of the input.
 */
struct rsqrt {
    template<typename T> static __device__ inline T op(const T &x);
};
template<> __device__ inline float  rsqrt::op<float> (const float &x ) { return __builtin_amdgcn_rsqf(x); }
template<> __device__ inline float2 rsqrt::op<float2>(const float2 &x) {
    return float2{__builtin_amdgcn_rsqf(x.x), __builtin_amdgcn_rsqf(x.y)};
}

/**
 * @brief Sigmoid linear unit, x * sigmoid(x), also called swish.
 *
 * Written as x / (1 + exp(-x)) using the hardware exp2, i.e. exp(-x) is
 * exp2(-x * log2(e)). Fusing it saves the three temporaries the composed form
 * needs, which on a kernel that is already register-bound is the difference
 * between one occupancy step and the next.
 *
 * @tparam T The data type of the input and output values.
 * @param x[in] The input value.
 * @return x * sigmoid(x).
 */
struct silu {
    template<typename T> static __device__ inline T op(const T &x);
};
template<> __device__ inline float silu::op<float>(const float &x) {
    return x * __builtin_amdgcn_rcpf(1.f + __builtin_amdgcn_exp2f(-x * 1.44269504088896340736f));
}
template<> __device__ inline float2 silu::op<float2>(const float2 &x) {
    return float2{silu::op<float>(x.x), silu::op<float>(x.y)};
}

/**
 * @brief Copy operation.
 *
 * This operation returns the input value unchanged.
 *
 * @tparam T The data type of the input and output values.
 * @param a[in] The input value.
 * @return The same value as the input.
 */
struct copy { // for non-compile-time setters.
    template<typename T> static __device__ inline T op(const T &a) { return a; }
};


/* ----------  BINARY OPS  ---------- */

/**
 * @brief Copy2 operation.
 *
 * This operation returns the second input value unchanged.
 *
 * @tparam T The data type of the input and output values.
 * @param a[in] The first input value (ignored).
 * @param b[in] The second input value.
 * @return The same value as the second input.
 */
struct copy2 { // this turns out to be a slightly hacky op that makes some code cleaner :/
    template<typename T> static __device__ inline T op(const T &a, const T &b) { return b; }
};
/**
 * @brief Sum operation.
 *
 * This operation calculates the sum of two input values.
 *
 * @tparam T The data type of the input and output values.
 * @param a[in] The first input value.
 * @param b[in] The second input value.
 * @return The sum of the input values.
 */
struct sum {
    template<typename T> static __device__ inline T op(const T &a, const T &b) { return a+b; }
};
template<> __device__ inline float2 sum::op<float2>(const float2 &a, const float2 &b) { return float2{a.x+b.x, a.y+b.y}; }
template<> __device__ inline bf16   sum::op<bf16>  (const bf16   &a, const bf16   &b) { return __hadd(a, b);             }
template<> __device__ inline bf16_2 sum::op<bf16_2>(const bf16_2 &a, const bf16_2 &b) { return __hadd2(a, b);            }
template<> __device__ inline half   sum::op<half>  (const half   &a, const half   &b) { return __hadd(a, b);             }
template<> __device__ inline half_2 sum::op<half_2>(const half_2 &a, const half_2 &b) { return __hadd2(a, b);            }
/**
 * @brief Subtraction operation.
 *
 * This operation calculates the difference between two input values.
 *
 * @tparam T The data type of the input and output values.
 * @param a[in] The first input value.
 * @param b[in] The second input value.
 * @return The difference between the input values.
 */
struct sub {
    template<typename T> static __device__ inline T op(const T &a, const T &b) { return a-b; }
};
template<> __device__ inline float2 sub::op<float2>(const float2 &a, const float2 &b) { return float2{a.x-b.x, a.y-b.y}; }
template<> __device__ inline bf16   sub::op<bf16>  (const bf16   &a, const bf16   &b) { return __hsub(a, b);             }
template<> __device__ inline bf16_2 sub::op<bf16_2>(const bf16_2 &a, const bf16_2 &b) { return __hsub2(a, b);            }
template<> __device__ inline half   sub::op<half>  (const half   &a, const half   &b) { return __hsub(a, b);             }
template<> __device__ inline half_2 sub::op<half_2>(const half_2 &a, const half_2 &b) { return __hsub2(a, b);            }
/**
 * @brief Multiplication operation.
 *
 * This operation calculates the product of two input values.
 *
 * @tparam T The data type of the input and output values.
 * @param a[in] The first input value.
 * @param b[in] The second input value.
 * @return The product of the input values.
 */
struct mul {
    template<typename T> static __device__ inline T op(const T &a, const T &b) { return a*b; }
};
template<> __device__ inline float2 mul::op<float2>(const float2 &a, const float2 &b) { return float2{a.x*b.x, a.y*b.y}; }
template<> __device__ inline bf16   mul::op<bf16>  (const bf16   &a, const bf16   &b) { return __hmul(a, b);             }
template<> __device__ inline bf16_2 mul::op<bf16_2>(const bf16_2 &a, const bf16_2 &b) { return __hmul2(a, b);            }
template<> __device__ inline half   mul::op<half>  (const half   &a, const half   &b) { return __hmul(a, b);             }
template<> __device__ inline half_2 mul::op<half_2>(const half_2 &a, const half_2 &b) { return __hmul2(a, b);            }
/**
 * @brief Division operation.
 *
 * This operation calculates the quotient of two input values.
 *
 * @tparam T The data type of the input and output values.
 * @param a[in] The first input value.
 * @param b[in] The second input value.
 * @return The quotient of the input values.
 */
struct div {
    template<typename T> static __device__ inline T op(const T &a, const T &b) { return a/b; }
};
template<> __device__ inline float2 div::op<float2>(const float2 &a, const float2 &b) { return float2{a.x/b.x, a.y/b.y}; }
template<> __device__ inline bf16   div::op<bf16>  (const bf16   &a, const bf16   &b) { return __hdiv(a, b);             }
template<> __device__ inline bf16_2 div::op<bf16_2>(const bf16_2 &a, const bf16_2 &b) { return __h2div(a, b);            } // this op is a special snowflake
template<> __device__ inline half   div::op<half>  (const half   &a, const half   &b) { return __hdiv(a, b);             }
template<> __device__ inline half_2 div::op<half_2>(const half_2 &a, const half_2 &b) { return __h2div(a, b);            }
/**
 * @brief Maximum operation.
 *
 * This operation calculates the maximum of two input values.
 *
 * @tparam T The data type of the input and output values.
 * @param a[in] The first input value.
 * @param b[in] The second input value.
 * @return The maximum of the input values.
 */
 struct max {
    template<typename T> static __device__ inline T op(const T &a, const T &b) { return ::max(a, b); }
};
template<>  __device__ inline float2 max::op<float2>(const float2 &a, const float2 &b) { return float2{::max(a.x, b.x), ::max(a.y, b.y)};  }
template<>  __device__ inline bf16   max::op<bf16>  (const bf16   &a, const bf16   &b) { return __hmax(a, b);                              }
template<>  __device__ inline bf16_2 max::op<bf16_2>(const bf16_2 &a, const bf16_2 &b) { return __hmax2(a, b);                             }
template<>  __device__ inline half   max::op<half>  (const half   &a, const half   &b) { return __hmax(a, b);                              }
template<>  __device__ inline half_2 max::op<half_2>(const half_2 &a, const half_2 &b) { return half_2{__hmax(a.x, b.x), __hmax(a.y, b.y)}; }
/**
 * @brief Minimum operation.
 *
 * This operation calculates the minimum of two input values.
 *
 * @tparam T The data type of the input and output values.
 * @param a[in] The first input value.
 * @param b[in] The second input value.
 * @return The minimum of the input values.
 */
struct min {
    template<typename T> static __device__ inline T op(const T &a, const T &b) { return ::min(a, b); }
};
template<>  __device__ inline float2 min::op<float2>(const float2 &a, const float2 &b) { return float2{::min(a.x, b.x), ::min(a.y, b.y)};   }
template<>  __device__ inline bf16   min::op<bf16>  (const bf16   &a, const bf16   &b) { return __hmin(a, b);                               }
template<>  __device__ inline bf16_2 min::op<bf16_2>(const bf16_2 &a, const bf16_2 &b) { return __hmin2(a, b);                              }
template<>  __device__ inline half   min::op<half>  (const half   &a, const half   &b) { return __hmin(a, b);                               }
template<>  __device__ inline half_2 min::op<half_2>(const half_2 &a, const half_2 &b) { return half_2{__hmin(a.x, b.x), __hmin(a.y, b.y)};  }


/* ----------  TERNARY OPS  ---------- */

/**
 * @brief Fused multiply-add operation A * B + C.
 *
 * This operation performs a fused multiply-add, computing (A * B) + C with only one rounding.
 *
 * @tparam T The data type of the input and output values.
 * @param a[in] The first input value.
 * @param b[in] The second input value.
 * @param c[in] The third input value to be added.
 * @return The result of the fused multiply-add operation.
 */
struct fma_AxBtC {
    template<typename T> static __device__ inline T op(const T &a, const T &b, const T &c) {
        return sum::op<T>(mul::op<T>(a, b), c);
    }
};
/**
 * @brief Fused multiply-add operation A * C + B.
 *
 * This operation performs a fused multiply-add, computing (A * C) + B with only one rounding.
 * This is particularly useful for attention mechanisms in neural networks.
 *
 * @tparam T The data type of the input and output values.
 * @param a[in] The first input value.
 * @param b[in] The third input value to be added.
 * @param c[in] The second input value.
 * @return The result of the fused multiply-add operation.
 */
struct fma_AxCtB { // this is the one needed for attention
    template<typename T> static __device__ inline T op(const T &a, const T &b, const T &c) {
        return sum::op<T>(mul::op<T>(a, c), b);
    }
};

} // namespace base_ops

} // namespace kittens
