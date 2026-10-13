// hk -> HipKittens C++.
//
// The output is meant to be read: when a kernel spills or mis-schedules, the
// generated .cpp is what goes to the resource check and the disassembler. So
// real names, one type alias per tile type, and the same structure a
// hand-written kernel in kernels/rdna3 has.
//
// Tiles are buffers in the dialect, so every op is already in the
// destination-passing form include/rdna3 exposes: an op with a result
// declares it, an op with `into %dst` writes a local that already exists.

#include "hk/Target/Cpp/HKToCpp.h"

#include "hk/Support/PyRepr.h"
#include "hk/Target/Target.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/TypeSwitch.h"

#include <cmath>
#include <map>
#include <vector>

using namespace mlir;
using namespace mlir::hk;

namespace {

std::string join(ArrayRef<std::string> xs, StringRef sep = ", ") {
  std::string out;
  for (auto [i, x] : llvm::enumerate(xs)) {
    if (i)
      out += sep;
    out += x;
  }
  return out;
}

/// A float literal for C++: Python's repr, except where that is not a token.
std::string floatLit(double v) {
  if (std::isnan(v))
    return "NAN";
  if (std::isinf(v))
    return v > 0 ? "INFINITY" : "-INFINITY";
  return pyFloat(v);
}

std::string rawCpp(Type t) {
  if (auto r = dyn_cast<RegTileType>(t))
    return "kittens::rt<" + dtypeCpp(r.getElementType()).str() + ", " +
           std::to_string(r.getRows()) + ", " + std::to_string(r.getCols()) +
           ", kittens::ducks::rt_layout::" + r.getLayout().str() + ">";
  if (auto s = dyn_cast<SharedTileType>(t))
    return "kittens::st<" + dtypeCpp(s.getElementType()).str() + ", " +
           std::to_string(s.getRows()) + ", " + std::to_string(s.getCols()) +
           ">";
  if (auto v = dyn_cast<RegVecType>(t))
    return "typename " + rawCpp(v.getTile()) + "::" + v.getKind().str() +
           "_vec";
  if (auto s = dyn_cast<SharedVecType>(t))
    return "kittens::sv<" + dtypeCpp(s.getElementType()).str() + ", " +
           std::to_string(s.getLength()) + ">";
  if (auto g = dyn_cast<GlobalType>(t))
    return "kittens::gl<" + dtypeCpp(g.getElementType()).str() +
           ", -1, -1, -1, -1>";
  if (auto g = dyn_cast<GranulesType>(t))
    return "kittens::lds_granules<" + rawCpp(g.getTile()) + ">";
  if (isa<StageType>(t))
    return "float4";
  if (t.isIntOrFloat())
    return dtypeCpp(t).str();
  return "/*?*/";
}

Type plainOf(Type t) {
  if (auto v = dyn_cast<RegVecType>(t))
    return v.plain();
  return t;
}

/// What makes two types one alias: the C++ they spell. An rv's identity is
/// (element, length, kind, layout), not the tile it carries.
std::string typeKey(Type t) {
  t = plainOf(t);
  if (auto v = dyn_cast<RegVecType>(t))
    return "rv|" + dtypeName(v.getElementType()).str() + "|" +
           std::to_string(v.getLength()) + "|" + v.getKind().str() + "|" +
           (v.getTileColLayout() ? "col" : "row");
  std::string s;
  llvm::raw_string_ostream os(s);
  os << t;
  return s;
}

std::string aliasName(Type t) {
  if (auto r = dyn_cast<RegTileType>(t))
    return "rt_" + dtypeName(r.getElementType()).str() + "_" +
           std::to_string(r.getRows()) + "x" + std::to_string(r.getCols()) +
           "_" + r.getLayout().str();
  if (auto v = dyn_cast<RegVecType>(t))
    return "rv_" + dtypeName(v.getElementType()).str() + "_" +
           std::to_string(v.getTileRows()) + "x" +
           std::to_string(v.getTileCols()) + "_" +
           (v.getTileColLayout() ? "col" : "row") + "_" + v.getKind().str();
  auto s = cast<SharedTileType>(t);
  return "st_" + dtypeName(s.getElementType()).str() + "_" +
         std::to_string(s.getRows()) + "x" + std::to_string(s.getCols());
}

std::string pyValue(Attribute a) {
  if (auto b = dyn_cast<BoolAttr>(a))
    return b.getValue() ? "True" : "False";
  if (auto i = dyn_cast<IntegerAttr>(a))
    return std::to_string(i.getInt());
  if (auto f = dyn_cast<FloatAttr>(a))
    return pyFloat(f.getValueAsDouble());
  if (auto s = dyn_cast<StringAttr>(a))
    return s.getValue().str();
  return "?";
}

class Emitter {
public:
  Emitter(func::FuncOp kernel) : kernel(kernel) {}

  LogicalResult emit(ArrayRef<std::string> includes, raw_ostream &os) {
    auto arch = kernel->getAttrOfType<StringAttr>(attrs::kArch);
    tgt = getTarget(arch ? arch.getValue() : "gfx1100");
    if (!tgt)
      return kernel.emitError("unknown arch");
    if (failed(findGrid()))
      return failure();
    assignNames();
    header(includes);
    if (failed(globalsStruct()))
      return failure();
    if (failed(kernelBody()))
      return failure();
    launch();
    for (auto &l : lines)
      os << l << "\n";
    return failure(bad);
  }

  void applyNames() {
    assignNames();
    auto under = [&](const std::string &n, Location loc) -> Location {
      if (auto nl = dyn_cast<NameLoc>(loc))
        loc = nl.getChildLoc();
      return NameLoc::get(StringAttr::get(kernel.getContext(), n), loc);
    };
    for (auto &[v, n] : names) {
      if (auto ba = dyn_cast<BlockArgument>(v)) {
        ba.setLoc(under(n, ba.getLoc()));
        continue;
      }
      // One location per op: a multi-result op keeps MLIR's numbering.
      Operation *op = v.getDefiningOp();
      if (op->getNumResults() == 1)
        op->setLoc(under(n, op->getLoc()));
    }
  }

private:
  func::FuncOp kernel, grid;
  const TargetInfo *tgt = nullptr;
  std::vector<std::string> lines;
  int indent = 1;
  int opaque = 0;
  bool bad = false;
  DenseMap<Value, std::string> names;
  llvm::MapVector<std::string, std::pair<Type, std::string>,
                  std::map<std::string, unsigned>>
      aliases;

  // -- helpers ----------------------------------------------------------

  void w(const std::string &line = "") {
    if (line.empty())
      lines.emplace_back();
    else
      lines.push_back(std::string(4 * indent, ' ') + line);
  }

  StringRef kernelName() { return kernel.getSymName(); }
  int64_t intAttr(StringRef n, int64_t d = 0) {
    if (auto a = kernel->getAttrOfType<IntegerAttr>(n))
      return a.getInt();
    return d;
  }
  int64_t threads() { return intAttr(attrs::kWarps, 1) * 32; }
  int64_t ldsBytes() { return intAttr(attrs::kLdsBytes); }

  StringRef argName(func::FuncOp f, unsigned i) {
    if (auto n = f.getArgAttrOfType<StringAttr>(i, attrs::kArgName))
      return n.getValue();
    return "";
  }

  void fail(Operation *op, const Twine &msg) {
    op->emitError(msg);
    bad = true;
  }

  LogicalResult findGrid() {
    auto mod = kernel->getParentOfType<ModuleOp>();
    if (mod)
      for (auto f : mod.getOps<func::FuncOp>())
        if (auto of = f->getAttrOfType<FlatSymbolRefAttr>(attrs::kGridOf))
          if (of.getValue() == kernelName())
            grid = f;
    if (!grid)
      return kernel.emitError("no grid function (hk.grid_of) for this kernel");
    return success();
  }

  /// Deterministic names, in program order: a base name the first time,
  /// base1, base2, ... after. The frontend's naming, so the textual IR and
  /// the C++ call a value the same thing.
  void assignNames() {
    llvm::StringMap<int> counts;
    auto bind = [&](Value v, StringRef base) {
      int n = counts[base]++;
      names[v] = n == 0 ? base.str() : (base + Twine(n)).str();
    };
    kernel.getBody().walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (op == kernel.getOperation())
        return;
      if (auto f = dyn_cast<scf::ForOp>(op)) {
        StringRef iv = "arg";
        if (auto a = f->getAttrOfType<StringAttr>(attrs::kIvName))
          iv = a.getValue();
        else if (auto n = valueNameOf(f.getInductionVar().getLoc());
                 !n.empty())
          iv = n;
        bind(f.getInductionVar(), iv);
        auto carried = f->getAttrOfType<ArrayAttr>("hk.carried_names");
        for (auto [i, a] : llvm::enumerate(f.getRegionIterArgs())) {
          StringRef n = valueNameOf(a.getLoc());
          if (carried && i < carried.size())
            n = cast<StringAttr>(carried[i]).getValue();
          bind(a, n.empty() ? "arg" : n);
        }
        return;
      }
      if (auto wl = dyn_cast<scf::WhileOp>(op)) {
        auto carried = wl->getAttrOfType<ArrayAttr>("hk.carried_names");
        for (auto [i, a] : llvm::enumerate(wl.getBeforeArguments())) {
          StringRef n = valueNameOf(a.getLoc());
          if (carried && i < carried.size())
            n = cast<StringAttr>(carried[i]).getValue();
          bind(a, n.empty() ? "arg" : n);
        }
        return;
      }
      if (auto i = dyn_cast<scf::IfOp>(op)) {
        auto rn = i->getAttrOfType<ArrayAttr>("hk.result_names");
        for (auto [k, r] : llvm::enumerate(i.getResults())) {
          StringRef n = rn && k < rn.size()
                            ? cast<StringAttr>(rn[k]).getValue()
                            : StringRef("v");
          bind(r, n);
        }
        return;
      }
      for (Value r : op->getResults()) {
        StringRef n = valueNameOf(op->getLoc());
        if (!n.empty())
          bind(r, n);
      }
    });
    // A loop's results are its iter_args' variables.
    kernel.walk([&](scf::ForOp f) {
      for (auto [r, a] : llvm::zip(f.getResults(), f.getRegionIterArgs()))
        names[r] = names[a];
    });
    // A while's condition forwards its arguments unchanged (emitWhile
    // refuses anything else), so the body's arguments and the loop's results
    // are the same locals as the test's.
    kernel.walk([&](scf::WhileOp wl) {
      for (auto [a, b] :
           llvm::zip(wl.getAfterArguments(), wl.getBeforeArguments()))
        names[a] = names[b];
      for (auto [r, b] : llvm::zip(wl.getResults(), wl.getBeforeArguments()))
        names[r] = names[b];
    });
  }

  std::string name(Value v) {
    if (auto ba = dyn_cast<BlockArgument>(v)) {
      if (ba.getOwner() == &kernel.getBody().front())
        return "g." + argName(kernel, ba.getArgNumber()).str();
    }
    auto it = names.find(v);
    if (it != names.end())
      return it->second;
    return inlineExpr(v);
  }

  /// A value nobody named: a literal, or a width conversion of one that is.
  std::string inlineExpr(Value v) {
    Operation *def = v.getDefiningOp();
    if (!def) {
      bad = true;
      return "/*unnamed block argument*/";
    }
    if (auto c = dyn_cast<arith::ConstantOp>(def)) {
      if (auto i = dyn_cast<IntegerAttr>(c.getValue())) {
        if (i.getType().isInteger(1))
          return i.getValue().getBoolValue() ? "1" : "0";
        return std::to_string(i.getInt());
      }
      if (auto f = dyn_cast<FloatAttr>(c.getValue()))
        return floatLit(f.getValueAsDouble());
    }
    if (isa<arith::ExtUIOp, arith::ExtSIOp, arith::TruncIOp,
            arith::IndexCastOp>(def))
      return name(def->getOperand(0));
    if (auto e = dyn_cast<ExtentOp>(def))
      return name(e.getTensor()) + "." +
             stringifyExtent(e.getAxis()).str() + "()";
    if (auto s = scalarBinary(def))
      return "(" + name(def->getOperand(0)) + " " + *s + " " +
             name(def->getOperand(1)) + ")";
    fail(def, "value has no name and is not a literal");
    return "/*?*/";
  }

  /// The C++ operator of a host-expression arith op, for inline use.
  std::optional<std::string> scalarBinary(Operation *op) {
    return llvm::TypeSwitch<Operation *, std::optional<std::string>>(op)
        .Case([](arith::AddIOp) { return "+"; })
        .Case([](arith::SubIOp) { return "-"; })
        .Case([](arith::MulIOp) { return "*"; })
        .Case([](arith::DivSIOp) { return "/"; })
        .Case([](arith::RemSIOp) { return "%"; })
        .Case([](arith::AndIOp) { return "&"; })
        .Case([](arith::OrIOp) { return "|"; })
        .Case([](arith::XOrIOp) { return "^"; })
        .Default([](Operation *) { return std::nullopt; });
  }

  std::string cpp(Type t) {
    auto it = aliases.find(typeKey(t));
    if (it != aliases.end())
      return it->second.second;
    return rawCpp(t);
  }

  // -- top level -------------------------------------------------------

  void header(ArrayRef<std::string> includes) {
    indent = 0;
    w("// Generated by hk (HipKittens Python IR). Do not edit by hand;");
    w("// edit the kernel that produced it and regenerate.");
    w("//");
    w("// kernel : " + kernelName().str());
    w("// arch   : " + tgtArch() + "  warps=" +
      std::to_string(intAttr(attrs::kWarps, 1)) +
      " threads=" + std::to_string(threads()));
    if (auto consts = kernel->getAttrOfType<DictionaryAttr>(attrs::kConsts);
        consts && !consts.empty()) {
      std::vector<std::string> kv;
      for (NamedAttribute na : consts)
        kv.push_back(na.getName().str() + "=" + pyValue(na.getValue()));
      w("// const  : " + join(kv));
    }
    w("// LDS    : " + std::to_string(ldsBytes()) + " B");
    if (auto warns = kernel->getAttrOfType<ArrayAttr>(attrs::kWarnings))
      for (Attribute a : warns) {
        StringRef s = cast<StringAttr>(a).getValue();
        SmallVector<StringRef> parts;
        s.split(parts, '\n');
        for (StringRef p : parts)
          w("// WARN   : " + p.trim().str());
      }
    w();
    for (auto &inc : includes)
      w("#include " + inc);
    w();
    w("using namespace kittens;");
    w();
  }

  std::string tgtArch() {
    auto arch = kernel->getAttrOfType<StringAttr>(attrs::kArch);
    return arch ? arch.getValue().str() : "gfx1100";
  }

  LogicalResult globalsStruct() {
    w("struct globals {");
    indent = 1;
    for (unsigned i = 0; i < kernel.getNumArguments(); ++i)
      w(rawCpp(kernel.getArgument(i).getType()) + " " +
        argName(kernel, i).str() + ";");
    w();
    w("dim3 grid() const {");
    indent = 2;
    w("const globals &g = *this;");
    auto ret = cast<func::ReturnOp>(grid.getBody().front().getTerminator());
    if (ret.getNumOperands() != 3)
      return grid.emitError("a grid is three dimensions");
    std::vector<std::string> dims;
    for (Value v : ret.getOperands())
      dims.push_back(gridExpr(v));
    w("return dim3(" + join(dims) + ");");
    indent = 1;
    w("}");
    w("dim3 block() const { return dim3(" + std::to_string(threads()) + "); }");
    if (ldsBytes())
      w("size_t dynamic_shared_memory() const { return " +
        std::to_string(ldsBytes()) + "; }");
    indent = 0;
    w("};");
    w();
    return success();
  }

  std::string gridExpr(Value v) {
    if (auto ba = dyn_cast<BlockArgument>(v))
      return "g." + argName(grid, ba.getArgNumber()).str();
    Operation *def = v.getDefiningOp();
    if (auto c = dyn_cast<arith::ConstantOp>(def))
      return std::to_string(cast<IntegerAttr>(c.getValue()).getInt());
    if (auto e = dyn_cast<ExtentOp>(def))
      return gridExpr(e.getTensor()) + "." +
             stringifyExtent(e.getAxis()).str() + "()";
    if (auto s = scalarBinary(def))
      return "(" + gridExpr(def->getOperand(0)) + " " + *s + " " +
             gridExpr(def->getOperand(1)) + ")";
    fail(def, "a grid dimension combines extents, constants and + - * / %");
    return "0";
  }

  void collectAliases() {
    auto note = [&](Type t, auto &self) -> void {
      t = plainOf(t);
      if (auto v = dyn_cast<RegVecType>(t))
        self(v.getTile(), self);
      if (!isa<RegTileType, SharedTileType, RegVecType>(t))
        return;
      std::string key = typeKey(t);
      if (!aliases.count(key))
        aliases[key] = {t, aliasName(t)};
    };
    kernel.getBody().walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (op == kernel.getOperation())
        return;
      for (Value v : op->getResults())
        note(v.getType(), note);
      for (Value v : op->getOperands())
        note(v.getType(), note);
      for (Region &r : op->getRegions())
        for (Block &b : r)
          for (BlockArgument a : b.getArguments())
            note(a.getType(), note);
    });
    if (aliases.empty())
      return;
    for (auto &kv : aliases) {
      Type t = kv.second.first;
      if (auto v = dyn_cast<RegVecType>(t))
        w("using " + kv.second.second + " = " +
          aliases[typeKey(v.getTile())].second + "::" + v.getKind().str() +
          "_vec;");
      else
        w("using " + kv.second.second + " = " + rawCpp(t) + ";");
    }
    w();
  }

  LogicalResult kernelBody() {
    collectAliases();
    w("__global__ __launch_bounds__(" + std::to_string(threads()) + ", 1)");
    w("void " + kernelName().str() + "_kernel(const globals g) {");
    indent = 1;
    if (ldsBytes()) {
      w("extern __shared__ alignment_dummy __shm[];");
      w("shared_allocator al((int*)&__shm[0]);");
      w();
    }
    emitBlock(kernel.getBody().front());
    indent = 0;
    w("}");
    w();
    return success();
  }

  void launch() {
    std::string n = kernelName().str();
    w("static void launch_on(const globals &g, hipStream_t stream) {");
    indent = 1;
    if (ldsBytes()) {
      std::string lds = std::to_string(ldsBytes());
      w("hipFuncSetAttribute((void *)&" + n +
        "_kernel, hipFuncAttributeMaxDynamicSharedMemorySize, " + lds + ");");
      w(n + "_kernel<<<g.grid(), g.block(), " + lds + ", stream>>>(g);");
    } else {
      w(n + "_kernel<<<g.grid(), g.block(), 0, stream>>>(g);");
    }
    indent = 0;
    w("}");
    w();
    w("static void launch(const globals &g) { launch_on(g, 0); }");
    w();
  }

  // -- ops ----------------------------------------------------------------

  void emitBlock(Block &b) {
    for (Operation &op : b)
      emitOp(&op);
  }

  bool named(Value v) { return names.count(v); }

  /// The op's destination: `dst` when it writes one, else its result,
  /// declared here. A fresh declaration of a loop-carried accumulator would
  /// shadow it and keep only the last iteration.
  Value dest(Value dst, Value result) {
    if (dst)
      return dst;
    w(cpp(result.getType()) + " " + name(result) + ";");
    return result;
  }

  void call(StringRef fn, Value dst, ArrayRef<std::string> args) {
    std::vector<std::string> all{name(dst)};
    all.insert(all.end(), args.begin(), args.end());
    w("kittens::" + fn.str() + "(" + join(all) + ");");
  }

  std::string coordArg(Value idx, const std::string &tileCpp) {
    auto ct = cast<CoordType>(idx.getType());
    if (!ct.getElement())
      return "kittens::coord<" + tileCpp + ">(" + name(idx) + ")";
    return name(idx);
  }

  std::string svEntry(Value sv, Value index) {
    if (cast<SharedVecType>(sv.getType()).getCount() == 1)
      return name(sv);
    return name(sv) + "[" + name(index) + "]";
  }

  std::string elemCpp(Type t) {
    if (auto r = dyn_cast<RegTileType>(t))
      return dtypeCpp(r.getElementType()).str();
    if (auto v = dyn_cast<RegVecType>(t))
      return dtypeCpp(v.getElementType()).str();
    return "float";
  }

  void scalarStmt(Operation *op, const std::string &expr) {
    w("const int " + name(op->getResult(0)) + " = " + expr + ";");
  }

  void emitArith(Operation *op) {
    if (op->getNumResults() == 1 && !named(op->getResult(0)))
      return; // written inline where it is used
    auto a = [&](unsigned i) { return name(op->getOperand(i)); };
    llvm::TypeSwitch<Operation *>(op)
        .Case([&](arith::ConstantOp c) {
          std::string v;
          if (auto i = dyn_cast<IntegerAttr>(c.getValue()))
            v = std::to_string(i.getInt());
          else if (auto f = dyn_cast<FloatAttr>(c.getValue()))
            v = pyFloat(f.getValueAsDouble());
          w("const int " + name(c.getResult()) + " = " + v + ";");
        })
        .Case([&](arith::AddIOp) { scalarStmt(op, a(0) + " + " + a(1)); })
        .Case([&](arith::SubIOp) { scalarStmt(op, a(0) + " - " + a(1)); })
        .Case([&](arith::MulIOp) { scalarStmt(op, a(0) + " * " + a(1)); })
        .Case([&](arith::DivSIOp) { scalarStmt(op, a(0) + " / " + a(1)); })
        .Case([&](arith::RemSIOp) { scalarStmt(op, a(0) + " % " + a(1)); })
        .Case([&](arith::AndIOp) { scalarStmt(op, a(0) + " & " + a(1)); })
        .Case([&](arith::OrIOp) { scalarStmt(op, a(0) + " | " + a(1)); })
        .Case([&](arith::XOrIOp) { scalarStmt(op, a(0) + " ^ " + a(1)); })
        .Case([&](arith::MinSIOp) {
          scalarStmt(op, "(" + a(0) + " < " + a(1) + ") ? " + a(0) + " : " +
                             a(1));
        })
        .Case([&](arith::MaxSIOp) {
          scalarStmt(op, "(" + a(0) + " > " + a(1) + ") ? " + a(0) + " : " +
                             a(1));
        })
        .Case([&](arith::CeilDivSIOp) {
          scalarStmt(op, "(" + a(0) + " + " + a(1) + " - 1) / " + a(1));
        })
        .Case([&](arith::CmpIOp c) {
          static const char *sym[] = {"==", "!=", "<", "<=", ">", ">=",
                                      "<",  "<=", ">", ">="};
          scalarStmt(op, a(0) + " " + sym[int(c.getPredicate())] + " " + a(1));
        })
        .Case<arith::ExtUIOp, arith::ExtSIOp, arith::TruncIOp,
              arith::IndexCastOp>(
            [&](Operation *) { scalarStmt(op, a(0)); })
        .Default([&](Operation *) {
          fail(op, "no C++ for this arith op: " +
                       op->getName().getStringRef());
        });
  }

  /// The C++ type of a mutable local: a scalar is an int, whatever its width.
  std::string varCpp(Type t) {
    if (t.isInteger())
      return "int";
    return cpp(t);
  }

  /// `var = value;` for each yielded value that is not already the variable.
  void assignYield(Block &b, ValueRange vars) {
    auto y = dyn_cast<scf::YieldOp>(b.getTerminator());
    if (!y)
      return;
    for (auto [var, val] : llvm::zip(vars, y.getOperands()))
      if (name(var) != name(val))
        w(name(var) + " = " + name(val) + ";");
  }

  void emitFor(scf::ForOp f) {
    // A loop-carried value is a mutable local declared in front of the loop;
    // the body's block argument and the loop's result are the same local.
    for (auto [arg, init] : llvm::zip(f.getRegionIterArgs(), f.getInitArgs()))
      w(varCpp(arg.getType()) + " " + name(arg) + " = " + name(init) + ";");
    std::string n = name(f.getInductionVar());
    int64_t unroll = 1;
    if (auto u = f->getAttrOfType<IntegerAttr>(attrs::kUnroll))
      unroll = u.getInt();
    if (unroll > 1)
      w("#pragma unroll " + std::to_string(unroll));
    auto isConst = [](Value v, int64_t want) {
      if (auto c = v.getDefiningOp<arith::ConstantOp>())
        if (auto i = dyn_cast<IntegerAttr>(c.getValue()))
          return i.getInt() == want;
      return false;
    };
    std::string lb = name(f.getLowerBound()), ub = name(f.getUpperBound()),
                step = name(f.getStep());
    if (isConst(f.getStep(), 1))
      w("for (int " + n + " = " + lb + "; " + n + " < " + ub + "; ++" + n +
        ") {");
    else
      w("for (int " + n + " = " + lb + "; " + n + " < " + ub + "; " + n +
        " += " + step + ") {");
    indent++;
    emitBlock(*f.getBody());
    assignYield(*f.getBody(), f.getRegionIterArgs());
    indent--;
    w("}");
  }

  void emitWhile(scf::WhileOp wl) {
    Block &before = wl.getBefore().front(), &after = wl.getAfter().front();
    auto cond = cast<scf::ConditionOp>(before.getTerminator());
    for (auto [fwd, arg] : llvm::zip(cond.getArgs(), before.getArguments()))
      if (fwd != arg)
        return fail(wl, "a while loop's condition has to forward the loop's "
                        "carried values unchanged");
    for (auto [arg, init] : llvm::zip(before.getArguments(), wl.getInits()))
      w(varCpp(arg.getType()) + " " + name(arg) + " = " + name(init) + ";");
    w("while (true) {");
    indent++;
    emitBlock(before);
    w("if (!(" + name(cond.getCondition()) + ")) break;");
    emitBlock(after);
    assignYield(after, before.getArguments());
    indent--;
    w("}");
  }

  void emitIf(scf::IfOp i) {
    // A branch result is a local declared in front of the branch and
    // assigned at the end of each arm.
    for (Value r : i.getResults())
      w(varCpp(r.getType()) + " " + name(r) + ";");
    std::string c = name(i.getCondition());
    if (i->hasAttr(attrs::kOpaque)) {
      // The condition is `const int` everywhere else, and an "+v" operand has
      // to be writable: hand the optimizer a copy it cannot see through.
      ++opaque;
      std::string live = c + "_live" + std::to_string(opaque);
      w("int " + live + " = " + c + ";");
      w("asm volatile(\"\" : \"+v\"(" + live + "));");
      c = live;
    }
    w("if (" + c + ") {");
    indent++;
    emitBlock(i.getThenRegion().front());
    assignYield(i.getThenRegion().front(), i.getResults());
    indent--;
    if (!i.getElseRegion().empty() &&
        (!i.getElseRegion().front().without_terminator().empty() ||
         i.getNumResults())) {
      w("} else {");
      indent++;
      emitBlock(i.getElseRegion().front());
      assignYield(i.getElseRegion().front(), i.getResults());
      indent--;
    }
    w("}");
  }

  template <typename OpT>
  std::vector<std::string> names_(OpT range) {
    std::vector<std::string> out;
    for (Value v : range)
      out.push_back(name(v));
    return out;
  }

  void emitOp(Operation *op) {
    if (isa<arith::ArithDialect>(op->getDialect()))
      return emitArith(op);
    llvm::TypeSwitch<Operation *>(op)
        .Case([&](scf::ForOp f) { emitFor(f); })
        .Case([&](scf::IfOp i) { emitIf(i); })
        .Case([&](scf::WhileOp wl) { emitWhile(wl); })
        .Case<scf::YieldOp, scf::ConditionOp, func::ReturnOp>(
            [](Operation *) {})
        // indices
        .Case([&](BlockIdxOp o) {
          w("const int " + name(o.getResult()) + " = blockIdx." +
            stringifyAxis(o.getAxis()).str() + ";");
        })
        .Case([&](WarpIdOp o) {
          // Uniform: the IR hoists this into the prologue, where every lane
          // is active, so the readfirstlane is always well defined.
          w("const int " + name(o.getResult()) +
            " = kittens::warpid_uniform();");
        })
        .Case([&](LaneIdOp o) {
          w("const int " + name(o.getResult()) + " = kittens::laneid();");
        })
        .Case([&](CoordOp o) {
          std::string parts = join(names_(o->getOperands()));
          if (!o.getResult().getType().getElement())
            w("const int4 " + name(o.getResult()) + " = make_int4(" + parts +
              ");");
          else
            w("const kittens::coord<> " + name(o.getResult()) + "(" + parts +
              ");");
        })
        .Case([&](ExtentOp o) {
          if (!named(o.getResult()))
            return;
          w("const int " + name(o.getResult()) + " = " + name(o.getTensor()) +
            "." + stringifyExtent(o.getAxis()).str() + "();");
        })
        .Case([&](AnyNeOp o) {
          std::string a = name(o.getLhs()), b = name(o.getRhs()),
                      r = name(o.getResult());
          w("bool " + r + "_v = false;");
          w("#pragma unroll");
          w("for (int o = 0; o < decltype(" + a + ")::outer_dim; o++)");
          w("    #pragma unroll");
          w("    for (int i = 0; i < decltype(" + a + ")::inner_dim; i++)");
          w("        " + r + "_v |= (" + a + ".data[o][i] != " + b +
            ".data[o][i]);");
          w("const int " + r + " = __any(" + r + "_v);");
        })
        // global memory
        .Case([&](LoadOp o) {
          Value d = dest(o.getDst(), o.getResult());
          std::string fn = o.getBounded() ? "load_bounded" : "load";
          w("kittens::" + fn + "(" + name(d) + ", " + name(o.getSrc()) + ", " +
            coordArg(o.getIdx(), cpp(d.getType())) + ");");
        })
        .Case([&](StoreOp o) {
          std::string fn = o.getBounded() ? "store_bounded" : "store";
          w("kittens::" + fn + "(" + name(o.getDst()) + ", " +
            name(o.getValue()) + ", " +
            coordArg(o.getIdx(), cpp(o.getValue().getType())) + ");");
        })
        .Case([&](StoreScalarOp o) {
          w("kittens::store_scalar(" + name(o.getDst()) + ", " +
            name(o.getValue()) + ", " + name(o.getIdx()) + ");");
        })
        .Case([&](LoadScalarOp o) {
          w("const int " + name(o.getResult()) + " = kittens::load_scalar(" +
            name(o.getSrc()) + ", " + name(o.getIdx()) + ");");
        })
        .Case([&](WgArriveOp o) {
          w("const int " + name(o.getResult()) + " = kittens::wg_arrive(" +
            name(o.getCounter()) + ", " + name(o.getIdx()) + ", " +
            name(o.getExpected()) + ");");
        })
        // LDS
        .Case([&](AllocSharedOp o) {
          auto t = o.getResult().getType();
          std::string el = rawCpp(t);
          if (t.getCount() == 1)
            w(el + " &" + name(o.getResult()) + " = al.allocate<" + el +
              ">();");
          else
            w(el + " (&" + name(o.getResult()) + ")[" +
              std::to_string(t.getCount()) + "] = al.allocate<" + el + ", " +
              std::to_string(t.getCount()) + ">();");
        })
        .Case([&](AllocSharedVecOp o) {
          auto t = o.getResult().getType();
          std::string el = rawCpp(t);
          if (t.getCount() == 1)
            w(el + " &" + name(o.getResult()) + " = al.allocate<" + el +
              ">();");
          else
            w(el + " (&" + name(o.getResult()) + ")[" +
              std::to_string(t.getCount()) + "] = al.allocate<" + el + ", " +
              std::to_string(t.getCount()) + ">();");
        })
        .Case([&](SharedAtOp o) {
          w(cpp(o.getResult().getType()) + " &" + name(o.getResult()) + " = " +
            name(o.getBuf()) + "[" + name(o.getIndex()) + "];");
        })
        .Case([&](SubtileOp o) {
          auto t = o.getResult().getType();
          w("auto " + name(o.getResult()) + " = kittens::subtile_inplace<" +
            std::to_string(t.getRows()) + ", " + std::to_string(t.getCols()) +
            ">(" + name(o.getSrc()) + ", {" + name(o.getRow()) + ", " +
            name(o.getCol()) + "});");
        })
        .Case([&](LoadSharedOp o) {
          Value d = dest(o.getDst(), o.getResult());
          std::string t = o.getWait() ? "" : "<false>";
          w("kittens::load" + t + "(" + name(d) + ", " + name(o.getSrc()) +
            ");");
        })
        .Case([&](StoreSharedOp o) {
          w("kittens::store(" + name(o.getDst()) + ", " + name(o.getValue()) +
            ");");
        })
        .Case([&](GranulesOp o) {
          std::string r = name(o.getResult());
          w(cpp(o.getResult().getType()) + " " + r + ";");
          w(r + ".init(" + name(o.getSrc()) + ");");
        })
        .Case([&](LoadFragOp o) {
          Value d = dest(o.getDst(), o.getResult());
          w("kittens::lds_read_frag<" + std::to_string(16 * o.getRow()) + ", " +
            std::to_string(16 * o.getCol()) + ">(" + name(d) + ", " +
            name(o.getGranules()) + ");");
        })
        .Case([&](StoreFragOp o) {
          auto gt = o.getGranules().getType();
          std::string off, tmpl;
          if (!o.getRow()) {
            off = "(uint32_t)(" + std::to_string(16 * gt.getSwizzleBytes()) +
                  " * " + name(o.getRowIndex()) + ")";
            tmpl = "<0, " + std::to_string(16 * o.getCol()) + ">";
          } else {
            off = "0u";
            tmpl = "<" + std::to_string(16 * *o.getRow()) + ", " +
                   std::to_string(16 * o.getCol()) + ">";
          }
          w("kittens::lds_write_frag" + tmpl + "(" + name(o.getGranules()) +
            ", " + off + ", " + name(o.getValue()) + ");");
        })
        .Case([&](LdsWaitForOp o) {
          if (!o.getN()) {
            fail(o, "lds_wait_for has no count: run hk-lds-pipeline first");
            return;
          }
          w("kittens::lds_wait_for<" + std::to_string(*o.getN()) + ">(" +
            join(names_(o.getTiles())) + ");");
        })
        .Case([&](StoreSharedVecOp o) {
          w("kittens::store(" + svEntry(o.getDst(), o.getIndex()) + ", " +
            name(o.getValue()) + ");");
        })
        .Case([&](LoadSharedVecOp o) {
          std::string r = name(o.getResult());
          w(cpp(o.getResult().getType()) + " " + r + ";");
          w("kittens::load(" + r + ", " + svEntry(o.getSrc(), o.getIndex()) +
            ");");
        })
        // staging
        .Case([&](StageBufferOp o) {
          auto t = o.getResult().getType();
          std::string dim =
              t.getDepth() > 1 ? "[" + std::to_string(t.getDepth()) + "]" : "";
          w("float4 " + name(o.getResult()) + dim + "[" +
            std::to_string(t.getCalls()) + "];");
        })
        .Case([&](StageLoadOp o) {
          auto [g, expr, tmpl] = stageArgs(o.getBuf(), o.getSlot());
          auto t = o.getBuf().getType();
          w(g + "load_global_to_register_buffer(" + expr + ", " +
            std::to_string(t.getCalls()) + ", " + name(o.getSrc()) + ", " +
            coordArg(o.getIdx(), rawCpp(t.getTile())) + ", " + tmpl + ");");
        })
        .Case([&](StageCommitOp o) {
          auto [g, expr, tmpl] = stageArgs(o.getBuf(), o.getSlot());
          (void)tmpl;
          w(g + "store_register_buffer_to_shared<" +
            (o.getWait() ? "true" : "false") + ">(" + name(o.getDst()) + ", " +
            expr + ", " + std::to_string(o.getBuf().getType().getCalls()) +
            ");");
        })
        .Case([&](VmWaitOp o) {
          w("kittens::vm_wait<" + std::to_string(o.getKeep()) + ">();");
        })
        .Case([&](GroupLoadOp o) {
          std::string g = "kittens::group<" +
                          std::to_string(o.getThreads() / 32) + ">::";
          w(g + "load(" + name(o.getDst()) + ", " + name(o.getSrc()) + ", " +
            coordArg(o.getIdx(), rawCpp(o.getDst().getType())) + ");");
        })
        .Case([&](GroupStoreOp o) {
          std::string g = "kittens::group<" +
                          std::to_string(o.getThreads() / 32) + ">::";
          w(g + "store(" + name(o.getDst()) + ", " + name(o.getSrc()) + ", " +
            coordArg(o.getIdx(), rawCpp(o.getSrc().getType())) + ");");
        })
        // synchronization
        .Case([&](BarrierOp o) {
          // The drain goes before the barrier: s_barrier has no memory
          // semantics of its own. The builtin, not __syncthreads(), which on
          // gfx11 adds an L0 flush this op does not promise.
          if (o.getDrain())
            w("kittens::lds_wait<0>();");
          w("__builtin_amdgcn_s_barrier();");
        })
        .Case([&](SchedBarrierOp) {
          w("__builtin_amdgcn_sched_barrier(0);");
        })
        .Case([&](SetPrioOp o) {
          w("__builtin_amdgcn_s_setprio(" + std::to_string(o.getLevel()) +
            ");");
        })
        // creation
        .Case([&](ZeroOp o) {
          call("zero", dest(o.getDst(), o.getResult()), {});
        })
        .Case([&](NegInftyOp o) {
          call("neg_infty", dest(o.getDst(), o.getResult()), {});
        })
        .Case([&](FullOp o) {
          Value d = dest(o.getDst(), o.getResult());
          std::string n = name(d);
          double v = o.getValue().convertToDouble();
          if (v == 0)
            w("kittens::zero(" + n + ");");
          else if (v == 1)
            w("kittens::one(" + n + ");");
          else if (std::isinf(v) && v < 0)
            w("kittens::neg_infty(" + n + ");");
          else if (std::isinf(v))
            w("kittens::pos_infty(" + n + ");");
          else {
            // No fill-with-a-constant in maps.cuh; zero then add folds to the
            // same VALU sequence.
            w("kittens::zero(" + n + ");");
            w("kittens::add(" + n + ", " + n + ", (" + elemCpp(d.getType()) +
              ")" + floatLit(v) + ");");
          }
        })
        // maps
        .Case<AddOp, SubOp, MulOp, DivOp, MaxOp, MinOp>(
            [&](auto o) { emitBinary(o); })
        .Case<ExpOp, Exp2Op, LogOp, Log2Op, AbsOp, ReluOp, CopyOp, GeluOp,
              DgeluOp, SqrtOp, RsqrtOp, SiluOp>([&](auto o) {
          Value d = dest(o.getDst(), o.getResult());
          call(o->getName().stripDialect(), d, {name(o.getSrc())});
        })
        .Case<MmaABOp, MmaABtOp, MmaAtBOp, MmaAtBtOp>([&](auto o) {
          Value d = dest(o.getDst(), o.getResult());
          call(o->getName().stripDialect(), d,
               {name(o.getA()), name(o.getB()), name(o.getC())});
        })
        .Case([&](CastOp o) {
          Value d = dest(o.getDst(), o.getResult());
          call("copy", d, {name(o.getSrc())});
        })
        // reductions and broadcasts
        .Case<RowMaxOp, RowMinOp, RowSumOp, RowProdOp, ColMaxOp, ColMinOp,
              ColSumOp, ColProdOp>([&](auto o) {
          Value d = dest(o.getDst(), o.getResult());
          std::vector<std::string> args{name(o.getSrc())};
          if (o.getAccumulate())
            args.push_back(name(d));
          call(o->getName().stripDialect(), d, args);
        })
        .Case<AddRowOp, SubRowOp, MulRowOp, DivRowOp, AddColOp, SubColOp,
              MulColOp, DivColOp>([&](auto o) {
          Value d = dest(o.getDst(), o.getResult());
          call(o->getName().stripDialect(), d,
               {name(o.getSrc()), name(o.getVec())});
        })
        .Case<BroadcastRowOp, BroadcastColOp>([&](auto o) {
          Value d = dest(Value(), o.getResult());
          call(o->getName().stripDialect(), d, {name(o.getVec())});
        })
        .Case([&](RetypeVecOp o) {
          std::string a = cpp(o.getResult().getType()),
                      b = cpp(o.getSrc().getType());
          w("static_assert(std::is_same_v<" + a + ", " + b +
            ">, \"hk: as_vec between distinct rv types\");");
          w(a + " &" + name(o.getResult()) + " = " + name(o.getSrc()) + ";");
        })
        // masks
        .Case<LeftFillOp, RightFillOp, UpperFillOp, LowerFillOp, TriuOp,
              TrilOp>([&](auto o) {
          Value d = dest(o.getDst(), o.getResult());
          call(o->getName().stripDialect(), d,
               {name(o.getSrc()), name(o.getIdx()),
                "(" + elemCpp(d.getType()) + ")" +
                    floatLit(o.getValue().convertToDouble())});
        })
        .Case<MakeCausalOp, MakeCausalTOp>([&](auto o) {
          Value d = dest(o.getDst(), o.getResult());
          // The fill value is the unpacked element type; for bf16 the tile's
          // dtype is the packed pair, so no cast at all.
          call(o->getName().stripDialect(), d,
               {name(o.getSrc()), floatLit(o.getValue().convertToDouble())});
        })
        .Case([&](TransposeOp o) {
          Value d = dest(o.getDst(), o.getResult());
          call("transpose", d, {name(o.getSrc())});
        })
        .Case([&](TransposeSepOp o) {
          Value d = dest(o.getDst(), o.getResult());
          call("transpose_sep", d, {name(o.getSrc())});
        })
        .Default([&](Operation *o) {
          fail(o, "no C++ emitter for " + o->getName().getStringRef());
        });
  }

  template <typename OpT>
  void emitBinary(OpT o) {
    Value d = dest(o.getDst(), o.getResult());
    std::vector<std::string> args{name(o.getLhs())};
    if (Value rhs = o.getRhs()) {
      if (rhs.getType().isInteger(32))
        // A device scalar: cast to the element type so the map's templated
        // right operand resolves as it does for a literal.
        args.push_back("(" + elemCpp(d.getType()) + ")" + name(rhs));
      else
        args.push_back(name(rhs));
    }
    if (auto s = o.getScalar())
      args.push_back("(" + elemCpp(d.getType()) + ")" +
                     floatLit(s->convertToDouble()));
    call(o->getName().stripDialect(), d, args);
  }

  std::tuple<std::string, std::string, std::string> stageArgs(Value buf,
                                                              int64_t slot) {
    auto t = cast<StageType>(buf.getType());
    std::string g =
        "kittens::group<" + std::to_string(t.getThreads() / 32) + ">::";
    std::string expr =
        name(buf) + (t.getDepth() > 1 ? "[" + std::to_string(slot) + "]" : "");
    auto sb = buf.getDefiningOp<StageBufferOp>();
    std::string tmpl = name(sb.getDst());
    if (sb.getDst().getType().getCount() != 1)
      tmpl += "[0]";
    return {g, expr, tmpl};
  }
};

} // namespace

LogicalResult mlir::hk::translateKernelToCpp(func::FuncOp kernel,
                                             ArrayRef<std::string> includes,
                                             raw_ostream &os) {
  return Emitter(kernel).emit(includes, os);
}

void mlir::hk::nameValuesAsCpp(func::FuncOp kernel) {
  Emitter(kernel).applyNames();
}

void mlir::hk::registerToCppTranslation() {
  TranslateFromMLIRRegistration reg(
      "hk-to-cpp", "translate hk kernels to HipKittens C++",
      [](Operation *op, raw_ostream &os) -> LogicalResult {
        auto mod = dyn_cast<ModuleOp>(op);
        if (!mod)
          return op->emitError("expected a module");
        for (auto f : mod.getOps<func::FuncOp>())
          if (f->hasAttr(attrs::kKernel))
            if (failed(translateKernelToCpp(f, {"\"kittens.cuh\""}, os)))
              return failure();
        return success();
      },
      [](DialectRegistry &registry) {
        registry.insert<HKDialect, arith::ArithDialect, scf::SCFDialect,
                        func::FuncDialect>();
      });
}
