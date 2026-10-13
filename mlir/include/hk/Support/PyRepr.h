#ifndef HK_SUPPORT_PYREPR_H
#define HK_SUPPORT_PYREPR_H

#include "mlir/IR/Types.h"

#include <string>

namespace mlir::hk {

/// How the Python frontend spells a type in a message: rt<bf16,16x64,col>,
/// st<bf16,128x64>[2], rv<fp32,16,col>, scalar<i32>, ...
std::string pyRepr(Type t);

/// Python's repr(float): the shortest string that round-trips, in fixed
/// notation for decimal exponents in (-4, 16] and exponent notation
/// otherwise (1e-06, 1e+16), with ".0" on integral fixed values. NaN and the
/// infinities are "nan", "inf", "-inf".
std::string pyFloat(double v);

} // namespace mlir::hk

#endif // HK_SUPPORT_PYREPR_H
