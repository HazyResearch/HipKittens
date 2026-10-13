#include "hk/Dialect/HK/IR/HKDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::hk;

#include "hk/Dialect/HK/IR/HKOpsDialect.cpp.inc"
#include "hk/Dialect/HK/IR/HKEnums.cpp.inc"

void HKDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "hk/Dialect/HK/IR/HKOps.cpp.inc"
      >();
  registerTypes();
}

StringRef mlir::hk::dtypeName(Type t) {
  if (t.isBF16())
    return "bf16";
  if (t.isF16())
    return "fp16";
  if (t.isF32())
    return "fp32";
  if (t.isInteger(32))
    return "i32";
  if (t.isInteger(8))
    return "i8";
  if (isa<Float8E4M3FNType>(t))
    return "fp8e4m3";
  if (isa<Float8E5M2Type>(t))
    return "fp8e5m2";
  return "";
}

StringRef mlir::hk::dtypeCpp(Type t) {
  if (t.isBF16())
    return "bf16";
  if (t.isF16())
    return "half";
  if (t.isF32())
    return "float";
  if (t.isInteger(32))
    return "int";
  if (t.isInteger(8))
    return "int8";
  if (isa<Float8E4M3FNType>(t))
    return "fp8e4m3";
  if (isa<Float8E5M2Type>(t))
    return "fp8e5m2";
  return "";
}

unsigned mlir::hk::dtypeBits(Type t) { return t.getIntOrFloatBitWidth(); }

bool mlir::hk::isFp8(Type t) {
  return isa<Float8E4M3FNType, Float8E5M2Type>(t);
}

Type mlir::hk::dtypeFromName(MLIRContext *ctx, StringRef name) {
  Builder b(ctx);
  return llvm::StringSwitch<Type>(name)
      .Case("bf16", b.getBF16Type())
      .Case("fp16", b.getF16Type())
      .Case("fp32", b.getF32Type())
      .Case("i32", b.getI32Type())
      .Case("i8", b.getI8Type())
      .Case("fp8e4m3", Float8E4M3FNType::get(ctx))
      .Case("fp8e5m2", Float8E5M2Type::get(ctx))
      .Default(Type());
}

StringRef mlir::hk::valueNameOf(Location loc) {
  if (auto n = dyn_cast<NameLoc>(loc))
    return n.getName().getValue();
  return "";
}

std::string mlir::hk::pythonLocOf(Location loc) {
  if (auto n = dyn_cast<NameLoc>(loc))
    loc = n.getChildLoc();
  auto fused = dyn_cast<FusedLoc>(loc);
  if (!fused)
    return "";
  auto func = dyn_cast_or_null<StringAttr>(fused.getMetadata());
  if (!func || fused.getLocations().empty())
    return "";
  auto flc = dyn_cast<FileLineColLoc>(fused.getLocations().front());
  if (!flc)
    return "";
  return (flc.getFilename().getValue() + ":" + Twine(flc.getLine()) + " in " +
          func.getValue())
      .str();
}
