#include "hk/Dialect/HK/IR/HKDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;
using namespace mlir::hk;

#define GET_OP_CLASSES
#include "hk/Dialect/HK/IR/HKOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// An op that makes a tile has a result; one that writes a tile has `into`.
/// Never both, never neither.
static LogicalResult verifyDest(Operation *op, Value dst, Value result) {
  if (dst && result)
    return op->emitOpError("has both a destination and a result");
  if (!dst && !result)
    return op->emitOpError("needs a destination (`into %x`) or a result");
  return success();
}

static Type destType(Value dst, Value result) {
  return dst ? dst.getType() : result.getType();
}

static Type elementOf(Type t) {
  if (auto r = dyn_cast<RegTileType>(t))
    return r.getElementType();
  if (auto v = dyn_cast<RegVecType>(t))
    return v.getElementType();
  return {};
}

/// Two types that compile to the same C++: rv's identity is (element,
/// length, kind, layout), not the tile it carries.
static bool sameCpp(Type a, Type b) {
  auto va = dyn_cast<RegVecType>(a);
  auto vb = dyn_cast<RegVecType>(b);
  if (va && vb)
    return va.sameCpp(vb);
  return a == b;
}

static LogicalResult checkMapDtype(Operation *op, Type t) {
  static const llvm::StringSet<> fp32Only = {"gelu", "dgelu", "sqrt", "rsqrt",
                                             "silu"};
  StringRef name = op->getName().stripDialect();
  StringRef dn = dtypeName(elementOf(t));
  bool ok = fp32Only.contains(name)
                ? dn == "fp32"
                : (dn == "bf16" || dn == "fp16" || dn == "fp32");
  if (!ok)
    return op->emitOpError() << "has no " << dn
                             << " implementation in include/rdna3";
  return success();
}

//===----------------------------------------------------------------------===//
// Indices
//===----------------------------------------------------------------------===//

LogicalResult AnyNeOp::verify() {
  if (!getLhs().getType().sameCpp(getRhs().getType()))
    return emitOpError("compares vectors of different shapes");
  return success();
}

//===----------------------------------------------------------------------===//
// Global memory
//===----------------------------------------------------------------------===//

LogicalResult LoadOp::verify() {
  if (failed(verifyDest(*this, getDst(), getResult())))
    return failure();
  if (getBounded() && !isa<RegTileType>(destType(getDst(), getResult())))
    return emitOpError("bounded predicates tile rows; the destination is not a tile");
  return success();
}

LogicalResult StoreOp::verify() {
  if (getBounded() && !isa<RegTileType>(getValue().getType()))
    return emitOpError("bounded predicates tile rows; the value is not a tile");
  return success();
}

LogicalResult StoreScalarOp::verify() {
  if (!getIdx().getType().getElement())
    return emitOpError("takes an element coordinate");
  if (!getValue().getType().getUniform())
    return emitOpError("needs a uniform vector: every entry the same value");
  return success();
}

LogicalResult LoadScalarOp::verify() {
  if (!getIdx().getType().getElement())
    return emitOpError("takes an element coordinate");
  if (!getSrc().getType().getElementType().isInteger(32))
    return emitOpError("reads an int32 global");
  return success();
}

LogicalResult WgArriveOp::verify() {
  if (!getIdx().getType().getElement())
    return emitOpError("takes an element coordinate");
  if (!getCounter().getType().getElementType().isInteger(32))
    return emitOpError("counts in an int32 global");
  return success();
}

//===----------------------------------------------------------------------===//
// LDS
//===----------------------------------------------------------------------===//

LogicalResult SharedAtOp::verify() {
  auto buf = getBuf().getType();
  if (buf.getCount() == 1)
    return emitOpError("indexes a single LDS tile, not a stack of them");
  if (getResult().getType() != buf.element())
    return emitOpError("result must be one buffer of the stack");
  return success();
}

LogicalResult SubtileOp::verify() {
  auto src = getSrc().getType();
  auto res = getResult().getType();
  if (src.getCount() != 1)
    return emitOpError("subtile of a stack; index it first");
  if (res.getElementType() != src.getElementType() || res.getCount() != 1)
    return emitOpError("a subtile keeps the element type");
  if (src.getRows() % res.getRows() || src.getCols() % res.getCols())
    return emitOpError("the window does not tile its parent");
  return success();
}

LogicalResult LoadSharedOp::verify() {
  if (failed(verifyDest(*this, getDst(), getResult())))
    return failure();
  auto t = cast<RegTileType>(destType(getDst(), getResult()));
  auto st = getSrc().getType();
  if (t.getRows() != st.getRows() || t.getCols() != st.getCols())
    return emitOpError("register and shared tiles must have the same extents");
  if (!getWait() && getLdsLoads() == 0)
    return emitOpError("wait = false on a pair that does not vectorise");
  return success();
}

LogicalResult GranulesOp::verify() {
  auto st = getSrc().getType();
  if (st.getCount() != 1)
    return emitOpError("granules of a stack; index it first");
  if (getResult().getType().getTile() != st)
    return emitOpError("granules must be built for the tile they address");
  return success();
}

LogicalResult LoadFragOp::verify() {
  if (failed(verifyDest(*this, getDst(), getResult())))
    return failure();
  auto t = cast<RegTileType>(destType(getDst(), getResult()));
  auto gr = getGranules().getType();
  if (t.getRows() != 16 || t.getCols() != 16 || t.getColLayout() ||
      t.getElementType() != gr.getElementType())
    return emitOpError("reads a 16x16 row tile of the shared tile's element type");
  if (getRow() < 0 || 16 * getRow() + 16 > gr.getRows() || getCol() < 0 ||
      16 * getCol() + 16 > gr.getCols())
    return emitOpError("fragment coordinate out of the tile");
  return success();
}

LogicalResult StoreFragOp::verify() {
  auto t = getValue().getType();
  auto gr = getGranules().getType();
  if (t.getRows() != 16 || t.getCols() != 16 || t.getColLayout() ||
      t.getElementType() != gr.getElementType())
    return emitOpError("writes a 16x16 row tile of the shared tile's element type");
  if (bool(getRowIndex()) == getRow().has_value())
    return emitOpError("takes a constant row or a runtime row, exactly one");
  return success();
}

LogicalResult LdsWaitForOp::verify() {
  if (getTiles().empty())
    return emitOpError("needs the tiles it retires");
  return success();
}

//===----------------------------------------------------------------------===//
// Staging
//===----------------------------------------------------------------------===//

LogicalResult StageBufferOp::verify() {
  if (getResult().getType().getTile() != getDst().getType().element())
    return emitOpError("stages a tile of a different shape than its destination");
  return success();
}

LogicalResult StageLoadOp::verify() {
  if (getSlot() < 0 || getSlot() >= getBuf().getType().getDepth())
    return emitOpError("slot out of the buffer's depth");
  return success();
}

LogicalResult StageCommitOp::verify() {
  auto dst = getDst().getType();
  auto buf = getBuf().getType();
  if (dst.getCount() != 1)
    return emitOpError("commits into a stack; index it");
  if (buf.getTile() != dst)
    return emitOpError("buffer stages a different tile than the destination");
  if (getSlot() < 0 || getSlot() >= buf.getDepth())
    return emitOpError("slot out of the buffer's depth");
  return success();
}

LogicalResult SetPrioOp::verify() {
  if (getLevel() < 0 || getLevel() > 3)
    return emitOpError("the priority field is 2 bits, 0..3");
  return success();
}

//===----------------------------------------------------------------------===//
// Creation and maps
//===----------------------------------------------------------------------===//

#define HK_CREATE_VERIFY(OP)                                                   \
  LogicalResult OP::verify() {                                                 \
    return verifyDest(*this, getDst(), getResult());                           \
  }
HK_CREATE_VERIFY(ZeroOp)
HK_CREATE_VERIFY(FullOp)
HK_CREATE_VERIFY(NegInftyOp)
#undef HK_CREATE_VERIFY

template <typename OpT>
static LogicalResult verifyBinary(OpT op) {
  if (failed(verifyDest(op, op.getDst(), op.getResult())))
    return failure();
  Type lhs = op.getLhs().getType();
  if (failed(checkMapDtype(op, lhs)))
    return failure();
  Value rhs = op.getRhs();
  if (bool(rhs) == op.getScalar().has_value())
    return op.emitOpError("takes a right operand or a literal scalar, exactly one");
  if (rhs && !rhs.getType().isInteger(32) && !sameCpp(rhs.getType(), lhs))
    return op.emitOpError("operand types differ; elementwise ops do not broadcast");
  if (!sameCpp(destType(op.getDst(), op.getResult()), lhs))
    return op.emitOpError("result type differs from the operands'");
  return success();
}

#define HK_BINARY_VERIFY(OP)                                                   \
  LogicalResult OP::verify() { return verifyBinary(*this); }
HK_BINARY_VERIFY(AddOp)
HK_BINARY_VERIFY(SubOp)
HK_BINARY_VERIFY(MulOp)
HK_BINARY_VERIFY(DivOp)
HK_BINARY_VERIFY(MaxOp)
HK_BINARY_VERIFY(MinOp)
#undef HK_BINARY_VERIFY

template <typename OpT>
static LogicalResult verifyUnary(OpT op) {
  if (failed(verifyDest(op, op.getDst(), op.getResult())))
    return failure();
  if (failed(checkMapDtype(op, op.getSrc().getType())))
    return failure();
  if (!sameCpp(destType(op.getDst(), op.getResult()), op.getSrc().getType()))
    return op.emitOpError("result type differs from the operand's");
  return success();
}

#define HK_UNARY_VERIFY(OP)                                                    \
  LogicalResult OP::verify() { return verifyUnary(*this); }
HK_UNARY_VERIFY(ExpOp)
HK_UNARY_VERIFY(Exp2Op)
HK_UNARY_VERIFY(LogOp)
HK_UNARY_VERIFY(Log2Op)
HK_UNARY_VERIFY(AbsOp)
HK_UNARY_VERIFY(ReluOp)
HK_UNARY_VERIFY(CopyOp)
HK_UNARY_VERIFY(GeluOp)
HK_UNARY_VERIFY(DgeluOp)
HK_UNARY_VERIFY(SqrtOp)
HK_UNARY_VERIFY(RsqrtOp)
HK_UNARY_VERIFY(SiluOp)
#undef HK_UNARY_VERIFY

//===----------------------------------------------------------------------===//
// Reductions and broadcasts
//===----------------------------------------------------------------------===//

template <typename OpT>
static LogicalResult verifyReduce(OpT op, bool colKind) {
  if (failed(verifyDest(op, op.getDst(), op.getResult())))
    return failure();
  auto src = op.getSrc().getType();
  auto want = RegVecType::get(op.getContext(), src.getElementType(),
                              src.getRows(), src.getCols(), src.getColLayout(),
                              colKind, false);
  auto got = cast<RegVecType>(destType(op.getDst(), op.getResult()));
  if (!got.sameCpp(want))
    return op.emitOpError() << "reducing " << src << " gives " << want
                            << ", not " << got;
  if (op.getAccumulate() && !op.getDst())
    return op.emitOpError("accumulate needs a destination to fold into");
  return success();
}

#define HK_REDUCE_VERIFY(OP, COL)                                              \
  LogicalResult OP::verify() { return verifyReduce(*this, COL); }
HK_REDUCE_VERIFY(RowMaxOp, true)
HK_REDUCE_VERIFY(RowMinOp, true)
HK_REDUCE_VERIFY(RowSumOp, true)
HK_REDUCE_VERIFY(RowProdOp, true)
HK_REDUCE_VERIFY(ColMaxOp, false)
HK_REDUCE_VERIFY(ColMinOp, false)
HK_REDUCE_VERIFY(ColSumOp, false)
HK_REDUCE_VERIFY(ColProdOp, false)
#undef HK_REDUCE_VERIFY

template <typename OpT>
static LogicalResult verifyBroadcast(OpT op, bool rowOp) {
  if (failed(verifyDest(op, op.getDst(), op.getResult())))
    return failure();
  auto src = op.getSrc().getType();
  auto vec = op.getVec().getType();
  // A row op takes one value per row: the column vector row_sum gives.
  if (vec.getColKind() != rowOp)
    return op.emitOpError() << "takes a " << (rowOp ? "col" : "row")
                            << " vector";
  if (vec.getTileRows() != src.getRows() || vec.getTileCols() != src.getCols())
    return op.emitOpError("the vector came off a tile of another shape");
  if (destType(op.getDst(), op.getResult()) != src)
    return op.emitOpError("result type differs from the tile's");
  return success();
}

#define HK_BROADCAST_VERIFY(OP, ROW)                                           \
  LogicalResult OP::verify() { return verifyBroadcast(*this, ROW); }
HK_BROADCAST_VERIFY(AddRowOp, true)
HK_BROADCAST_VERIFY(SubRowOp, true)
HK_BROADCAST_VERIFY(MulRowOp, true)
HK_BROADCAST_VERIFY(DivRowOp, true)
HK_BROADCAST_VERIFY(AddColOp, false)
HK_BROADCAST_VERIFY(SubColOp, false)
HK_BROADCAST_VERIFY(MulColOp, false)
HK_BROADCAST_VERIFY(DivColOp, false)
#undef HK_BROADCAST_VERIFY

template <typename OpT>
static LogicalResult verifyBroadcastFill(OpT op, bool row) {
  auto vec = op.getVec().getType();
  auto res = op.getResult().getType();
  if (vec.getColKind() != row)
    return op.emitOpError() << "takes a " << (row ? "col" : "row") << " vector";
  if (vec.getTileRows() != res.getRows() || vec.getTileCols() != res.getCols())
    return op.emitOpError("the vector came off a tile of another shape");
  return success();
}

LogicalResult BroadcastRowOp::verify() { return verifyBroadcastFill(*this, true); }
LogicalResult BroadcastColOp::verify() { return verifyBroadcastFill(*this, false); }

LogicalResult RetypeVecOp::verify() {
  auto a = getSrc().getType();
  auto b = getResult().getType();
  if (a.getElementType() != b.getElementType() ||
      a.getLength() != b.getLength() ||
      a.getTileColLayout() != b.getTileColLayout() ||
      a.getColKind() != b.getColKind())
    return emitOpError("an rv's C++ type is (element, length, layout); these differ");
  return success();
}

//===----------------------------------------------------------------------===//
// Masks
//===----------------------------------------------------------------------===//

template <typename OpT>
static LogicalResult verifyFill(OpT op) {
  if (failed(verifyDest(op, op.getDst(), op.getResult())))
    return failure();
  if (destType(op.getDst(), op.getResult()) != op.getSrc().getType())
    return op.emitOpError("a fill keeps shape, layout and element type");
  return success();
}

#define HK_FILL_VERIFY(OP)                                                     \
  LogicalResult OP::verify() { return verifyFill(*this); }
HK_FILL_VERIFY(LeftFillOp)
HK_FILL_VERIFY(RightFillOp)
HK_FILL_VERIFY(UpperFillOp)
HK_FILL_VERIFY(LowerFillOp)
HK_FILL_VERIFY(TriuOp)
HK_FILL_VERIFY(TrilOp)
#undef HK_FILL_VERIFY

template <typename OpT>
static LogicalResult verifyCausal(OpT op) {
  if (failed(verifyDest(op, op.getDst(), op.getResult())))
    return failure();
  auto src = cast<RegTileType>(op.getSrc().getType());
  if (src.getRows() != src.getCols())
    return op.emitOpError("a diagonal only means something on a square tile");
  if (destType(op.getDst(), op.getResult()) != src)
    return op.emitOpError("result type differs from the tile's");
  return success();
}

LogicalResult MakeCausalOp::verify() { return verifyCausal(*this); }
LogicalResult MakeCausalTOp::verify() { return verifyCausal(*this); }

//===----------------------------------------------------------------------===//
// Conversions
//===----------------------------------------------------------------------===//

LogicalResult CastOp::verify() {
  if (failed(verifyDest(*this, getDst(), getResult())))
    return failure();
  auto src = dyn_cast<RegTileType>(getSrc().getType());
  auto dst = dyn_cast<RegTileType>(destType(getDst(), getResult()));
  if (!src || !dst)
    return emitOpError("casts register tiles");
  if (src.getRows() != dst.getRows() || src.getCols() != dst.getCols() ||
      src.getColLayout() != dst.getColLayout())
    return emitOpError("a cast keeps shape and layout");
  return success();
}

LogicalResult TransposeOp::verify() {
  if (failed(verifyDest(*this, getDst(), getResult())))
    return failure();
  auto src = dyn_cast<RegTileType>(getSrc().getType());
  if (!src || destType(getDst(), getResult()) != src.transposed())
    return emitOpError("transposes a tile into the mirrored shape and opposite layout");
  return success();
}

LogicalResult TransposeSepOp::verify() {
  if (failed(verifyDest(*this, getDst(), getResult())))
    return failure();
  auto src = dyn_cast<RegTileType>(getSrc().getType());
  if (!src)
    return emitOpError("transposes register tiles");
  auto want = RegTileType::get(getContext(), src.getElementType(),
                               src.getCols(), src.getRows(),
                               src.getColLayout());
  if (destType(getDst(), getResult()) != want)
    return emitOpError("transpose_sep mirrors the shape and keeps the layout");
  if (getDst() && getDst() == getSrc())
    return emitOpError("needs separate registers for source and destination");
  return success();
}

//===----------------------------------------------------------------------===//
// WMMA
//===----------------------------------------------------------------------===//

namespace {
struct MmaSpec {
  bool aCol, bCol;
  // Which extent of A gives D.rows, of B gives D.cols, and the reduction.
  bool dRowsFromACols, dColsFromBRows, aRedIsRows, bRedIsCols;
};
} // namespace

template <typename OpT>
static LogicalResult verifyMma(OpT op, MmaSpec s) {
  if (failed(verifyDest(op, op.getDst(), op.getResult())))
    return failure();
  auto a = op.getA().getType(), b = op.getB().getType(),
       c = op.getC().getType();
  bool fp8 = isFp8(a.getElementType()) && isFp8(b.getElementType());
  if (a.getElementType() != b.getElementType() && !fp8)
    return op.emitOpError("takes one operand type, not a mixed pair");
  if (!fp8 && dtypeBits(a.getElementType()) != 16)
    return op.emitOpError("takes bf16 or fp16 operands (fp8 on gfx1201)");
  if (!c.getElementType().isF32() || !c.getColLayout())
    return op.emitOpError("accumulates into a col-layout fp32 tile");
  if (a.getColLayout() != s.aCol || b.getColLayout() != s.bCol)
    return op.emitOpError("operand layouts do not match this variant");
  unsigned redA = s.aRedIsRows ? a.getRows() : a.getCols();
  unsigned redB = s.bRedIsCols ? b.getCols() : b.getRows();
  if (redA != redB)
    return op.emitOpError("reduction extents disagree");
  unsigned rows = s.dRowsFromACols ? a.getCols() : a.getRows();
  unsigned cols = s.dColsFromBRows ? b.getRows() : b.getCols();
  auto want = RegTileType::get(op.getContext(), c.getElementType(), rows, cols,
                               true);
  if (c != want || destType(op.getDst(), op.getResult()) != want)
    return op.emitOpError() << "A " << a << " times B " << b << " is " << want;
  return success();
}

LogicalResult MmaABOp::verify() {
  return verifyMma(*this, {false, true, false, false, false, false});
}
LogicalResult MmaABtOp::verify() {
  return verifyMma(*this, {false, false, false, true, false, true});
}
LogicalResult MmaAtBOp::verify() {
  return verifyMma(*this, {true, true, true, false, true, false});
}
LogicalResult MmaAtBtOp::verify() {
  return verifyMma(*this, {true, false, true, true, true, true});
}
