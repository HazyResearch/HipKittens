#include "hk/Dialect/HK/IR/HKDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::hk;

#define GET_TYPEDEF_CLASSES
#include "hk/Dialect/HK/IR/HKOpsTypes.cpp.inc"

void HKDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "hk/Dialect/HK/IR/HKOpsTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Parsing helpers
//===----------------------------------------------------------------------===//

static ParseResult parseRowCol(AsmParser &p, bool &col) {
  StringRef kw;
  SMLoc at = p.getCurrentLocation();
  if (p.parseKeyword(&kw))
    return failure();
  if (kw == "row")
    col = false;
  else if (kw == "col")
    col = true;
  else
    return p.emitError(at, "expected 'row' or 'col', got '") << kw << "'";
  return success();
}

static ParseResult parseExtents(AsmParser &p, unsigned &rows, unsigned &cols) {
  SmallVector<int64_t, 2> dims;
  SMLoc at = p.getCurrentLocation();
  if (p.parseDimensionList(dims, /*allowDynamic=*/false,
                           /*withTrailingX=*/false))
    return failure();
  if (dims.size() != 2)
    return p.emitError(at, "expected ROWSxCOLS");
  rows = dims[0];
  cols = dims[1];
  return success();
}

static LogicalResult checkExtent(function_ref<InFlightDiagnostic()> emitError,
                                 StringRef what, StringRef n, unsigned v) {
  if (v == 0 || v % 16)
    return emitError() << what << " " << n << "=" << v
                       << " must be a positive multiple of 16";
  return success();
}

static bool knownDtype(Type t) { return !dtypeName(t).empty(); }

//===----------------------------------------------------------------------===//
// rt
//===----------------------------------------------------------------------===//

Type RegTileType::parse(AsmParser &p) {
  Type et;
  unsigned rows, cols;
  bool col;
  SMLoc at = p.getNameLoc();
  if (p.parseLess() || p.parseType(et) || p.parseComma() ||
      parseExtents(p, rows, cols) || p.parseComma() || parseRowCol(p, col) ||
      p.parseGreater())
    return {};
  return RegTileType::getChecked([&] { return p.emitError(at); },
                                 p.getContext(), et, rows, cols, col);
}

void RegTileType::print(AsmPrinter &p) const {
  p << "<" << getElementType() << ", " << getRows() << "x" << getCols() << ", "
    << getLayout() << ">";
}

LogicalResult RegTileType::verify(function_ref<InFlightDiagnostic()> emitError,
                                  Type et, unsigned rows, unsigned cols, bool) {
  if (!knownDtype(et))
    return emitError() << "register tile of " << et
                       << ": not an element type include/rdna3 has";
  if (dtypeBits(et) < 16 && !isFp8(et))
    return emitError() << "there is no " << dtypeName(et)
                       << " register tile: rt_base exists only for 16- and "
                          "32-bit elements (and fp8 on gfx1201)";
  if (failed(checkExtent(emitError, "register tile", "rows", rows)) ||
      failed(checkExtent(emitError, "register tile", "cols", cols)))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// st
//===----------------------------------------------------------------------===//

Type SharedTileType::parse(AsmParser &p) {
  Type et;
  unsigned rows, cols, count = 1;
  SMLoc at = p.getNameLoc();
  if (p.parseLess() || p.parseType(et) || p.parseComma() ||
      parseExtents(p, rows, cols))
    return {};
  if (succeeded(p.parseOptionalComma()) && p.parseInteger(count))
    return {};
  if (p.parseGreater())
    return {};
  return SharedTileType::getChecked([&] { return p.emitError(at); },
                                    p.getContext(), et, rows, cols, count);
}

void SharedTileType::print(AsmPrinter &p) const {
  p << "<" << getElementType() << ", " << getRows() << "x" << getCols();
  if (getCount() != 1)
    p << ", " << getCount();
  p << ">";
}

LogicalResult
SharedTileType::verify(function_ref<InFlightDiagnostic()> emitError, Type et,
                       unsigned rows, unsigned cols, unsigned count) {
  if (!knownDtype(et))
    return emitError() << "shared tile of " << et;
  if (failed(checkExtent(emitError, "shared tile", "rows", rows)) ||
      failed(checkExtent(emitError, "shared tile", "cols", cols)))
    return failure();
  if (count == 0)
    return emitError() << "shared tile count must be positive";
  return success();
}

uint64_t SharedTileType::getNumBytes() const {
  return uint64_t(getRows()) * getCols() * (dtypeBits(getElementType()) / 8) *
         getCount();
}

//===----------------------------------------------------------------------===//
// rv
//===----------------------------------------------------------------------===//

Type RegVecType::parse(AsmParser &p) {
  Type et;
  unsigned rows, cols;
  bool tileCol, kindCol, uniform = false;
  if (p.parseLess() || p.parseType(et) || p.parseComma() ||
      parseExtents(p, rows, cols) || p.parseComma() ||
      parseRowCol(p, tileCol) || p.parseComma() || parseRowCol(p, kindCol))
    return {};
  if (succeeded(p.parseOptionalComma())) {
    if (p.parseKeyword("uniform"))
      return {};
    uniform = true;
  }
  if (p.parseGreater())
    return {};
  return RegVecType::get(p.getContext(), et, rows, cols, tileCol, kindCol,
                         uniform);
}

void RegVecType::print(AsmPrinter &p) const {
  p << "<" << getElementType() << ", " << getTileRows() << "x" << getTileCols()
    << ", " << (getTileColLayout() ? "col" : "row") << ", " << getKind();
  if (getUniform())
    p << ", uniform";
  p << ">";
}

//===----------------------------------------------------------------------===//
// sv
//===----------------------------------------------------------------------===//

Type SharedVecType::parse(AsmParser &p) {
  Type et;
  unsigned length, count = 1;
  SMLoc at = p.getNameLoc();
  if (p.parseLess() || p.parseType(et) || p.parseComma() ||
      p.parseInteger(length))
    return {};
  if (succeeded(p.parseOptionalComma()) && p.parseInteger(count))
    return {};
  if (p.parseGreater())
    return {};
  return SharedVecType::getChecked([&] { return p.emitError(at); },
                                   p.getContext(), et, length, count);
}

void SharedVecType::print(AsmPrinter &p) const {
  p << "<" << getElementType() << ", " << getLength();
  if (getCount() != 1)
    p << ", " << getCount();
  p << ">";
}

LogicalResult
SharedVecType::verify(function_ref<InFlightDiagnostic()> emitError, Type et,
                      unsigned length, unsigned count) {
  if (failed(checkExtent(emitError, "shared vector", "length", length)))
    return failure();
  if (count == 0)
    return emitError() << "shared vector count must be positive";
  return success();
}

uint64_t SharedVecType::getNumBytes() const {
  return uint64_t(getLength()) * getCount() * (dtypeBits(getElementType()) / 8);
}

//===----------------------------------------------------------------------===//
// coord
//===----------------------------------------------------------------------===//

Type CoordType::parse(AsmParser &p) {
  StringRef kw;
  SMLoc at = p.getCurrentLocation();
  if (p.parseLess() || p.parseKeyword(&kw) || p.parseGreater())
    return {};
  if (kw != "tile" && kw != "element") {
    p.emitError(at, "coord unit must be 'tile' or 'element'");
    return {};
  }
  return CoordType::get(p.getContext(), kw == "element");
}

void CoordType::print(AsmPrinter &p) const {
  p << "<" << (getElement() ? "element" : "tile") << ">";
}

//===----------------------------------------------------------------------===//
// granules
//===----------------------------------------------------------------------===//

Type GranulesType::parse(AsmParser &p) {
  Type et;
  unsigned rows, cols;
  SMLoc at = p.getNameLoc();
  if (p.parseLess() || p.parseType(et) || p.parseComma() ||
      parseExtents(p, rows, cols) || p.parseGreater())
    return {};
  return GranulesType::getChecked([&] { return p.emitError(at); },
                                  p.getContext(), et, rows, cols);
}

void GranulesType::print(AsmPrinter &p) const {
  p << "<" << getElementType() << ", " << getRows() << "x" << getCols() << ">";
}

/// Transcribed from st<T,R,C>::swizzle_bytes, 2-byte case.
unsigned GranulesType::getSwizzleBytes() const {
  unsigned width = getCols() / 16;
  return width % 4 == 0 ? 128 : width % 2 == 0 ? 64 : 32;
}

LogicalResult GranulesType::verify(function_ref<InFlightDiagnostic()> emitError,
                                   Type et, unsigned rows, unsigned cols) {
  if (dtypeBits(et) != 16)
    return emitError() << "granules: the decomposition is derived and checked "
                          "for 2-byte tiles only";
  unsigned width = cols / 16;
  unsigned swz = width % 4 == 0 ? 128 : width % 2 == 0 ? 64 : 32;
  if (swz < 64)
    return emitError() << "granules: swizzle_bytes is " << swz
                       << "; needs >= 32 columns of a 2-byte tile";
  return success();
}

//===----------------------------------------------------------------------===//
// stage
//===----------------------------------------------------------------------===//

Type StageType::parse(AsmParser &p) {
  Type et;
  unsigned rows, cols, threads, depth = 1;
  SMLoc at = p.getNameLoc();
  if (p.parseLess() || p.parseType(et) || p.parseComma() ||
      parseExtents(p, rows, cols) || p.parseComma() || p.parseInteger(threads))
    return {};
  if (succeeded(p.parseOptionalComma()) && p.parseInteger(depth))
    return {};
  if (p.parseGreater())
    return {};
  return StageType::getChecked([&] { return p.emitError(at); },
                               p.getContext(), et, rows, cols, threads, depth);
}

void StageType::print(AsmPrinter &p) const {
  p << "<" << getElementType() << ", " << getRows() << "x" << getCols() << ", "
    << getThreads();
  if (getDepth() != 1)
    p << ", " << getDepth();
  p << ">";
}

/// Transcribed from stage_calls in global_to_shared.cuh. Must agree exactly:
/// a buffer one float4 short is an out-of-bounds write to the stack.
unsigned StageType::getCalls() const {
  unsigned perFloat4 = 16 / (dtypeBits(getElementType()) / 8);
  unsigned n = getRows() * getCols();
  unsigned chunks = n / perFloat4;
  return (chunks + getThreads() - 1) / getThreads();
}

LogicalResult StageType::verify(function_ref<InFlightDiagnostic()> emitError,
                                Type, unsigned, unsigned, unsigned threads,
                                unsigned depth) {
  if (threads == 0 || threads % 32)
    return emitError() << "stage buffer threads=" << threads
                       << " must be a multiple of 32";
  if (depth == 0)
    return emitError() << "stage buffer depth must be positive";
  return success();
}
