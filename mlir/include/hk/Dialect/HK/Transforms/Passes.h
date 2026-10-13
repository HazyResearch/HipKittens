#ifndef HK_DIALECT_HK_TRANSFORMS_PASSES_H
#define HK_DIALECT_HK_TRANSFORMS_PASSES_H

#include "hk/Dialect/HK/IR/HKDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::hk {

#define GEN_PASS_DECL
#include "hk/Dialect/HK/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "hk/Dialect/HK/Transforms/Passes.h.inc"

/// hk-verify, hk-lds-alloc, hk-lds-pipeline, hk-check-budgets, in that order,
/// on every kernel in the module. Registered as `hk-kernel-pipeline`.
void buildKernelPipeline(OpPassManager &pm);
void registerKernelPipeline();

/// The kernel `func.func`s of a module run through these; others are left.
bool isKernel(func::FuncOp f);

/// "\n    at file:line in func", or empty: how every hk error ends.
std::string atLoc(Operation *op);

} // namespace mlir::hk

#endif // HK_DIALECT_HK_TRANSFORMS_PASSES_H
