#ifndef HK_DIALECT_HK_IR_HKDIALECT_H
#define HK_DIALECT_HK_IR_HKDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "hk/Dialect/HK/IR/HKOpsDialect.h.inc"
#include "hk/Dialect/HK/IR/HKEnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "hk/Dialect/HK/IR/HKOpsTypes.h.inc"

#define GET_OP_CLASSES
#include "hk/Dialect/HK/IR/HKOps.h.inc"

namespace mlir::hk {

/// hk's name for an element type: bf16, fp16, fp32, i32, i8, fp8e4m3,
/// fp8e5m2. Empty for a type hk has no name for.
llvm::StringRef dtypeName(Type t);

/// The spelling include/rdna3 uses: bf16, half, float, int, int8, ...
llvm::StringRef dtypeCpp(Type t);

/// Bits per element.
unsigned dtypeBits(Type t);

/// The element type hk calls `name`, or null.
Type dtypeFromName(MLIRContext *ctx, llvm::StringRef name);

/// True for the two OCP fp8 formats.
bool isFp8(Type t);

/// Kernel attributes, on the `func.func` that is the kernel.
namespace attrs {
inline constexpr llvm::StringLiteral kKernel = "hk.kernel";
inline constexpr llvm::StringLiteral kArch = "hk.arch";
inline constexpr llvm::StringLiteral kWarps = "hk.warps";
inline constexpr llvm::StringLiteral kConsts = "hk.consts";
inline constexpr llvm::StringLiteral kLdsBytes = "hk.lds_bytes";
inline constexpr llvm::StringLiteral kLdsStatic = "hk.lds_static";
inline constexpr llvm::StringLiteral kWarnings = "hk.warnings";
inline constexpr llvm::StringLiteral kGridOf = "hk.grid_of";
inline constexpr llvm::StringLiteral kArgName = "hk.name";
inline constexpr llvm::StringLiteral kUnroll = "hk.unroll";
inline constexpr llvm::StringLiteral kOpaque = "hk.opaque";
inline constexpr llvm::StringLiteral kIvName = "hk.iv_name";
} // namespace attrs

/// The value name an op or block argument carries: the outermost NameLoc of
/// its location. Empty when it has none.
llvm::StringRef valueNameOf(Location loc);

/// "file:line in func" for a location built by the hk frontend --
/// NameLoc(name, FusedLoc<"func">[file:line]) -- or empty.
std::string pythonLocOf(Location loc);

} // namespace mlir::hk

#endif // HK_DIALECT_HK_IR_HKDIALECT_H
