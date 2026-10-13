#ifndef HK_TARGET_CPP_HKTOCPP_H
#define HK_TARGET_CPP_HKTOCPP_H

#include "hk/Dialect/HK/IR/HKDialect.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace mlir::hk {

/// The kernel as HipKittens C++: a header, the globals struct with its grid,
/// the __global__ and the host `launch`. Everything except the module
/// boundary, which python/hk/codegen/scaffold.py wraps around it.
///
/// The kernel's module must hold its grid function (`hk.grid_of`). Values are
/// named as the frontend named them, deduplicated in program order; values
/// with no name -- a literal, a width conversion -- are written inline.
LogicalResult translateKernelToCpp(func::FuncOp kernel,
                                   ArrayRef<std::string> includes,
                                   raw_ostream &os);

/// `hk-to-cpp` for hk-translate: every kernel in the module, in order.
void registerToCppTranslation();

} // namespace mlir::hk

#endif // HK_TARGET_CPP_HKTOCPP_H
