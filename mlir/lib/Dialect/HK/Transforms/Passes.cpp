// hk-verify, hk-lds-alloc, hk-check-budgets, and the kernel pipeline.
//
// The messages are the ones python/hk/ir/verify.py printed, word for word:
// they are what a kernel author reads, and a test pins them.

#include "hk/Dialect/HK/Transforms/Passes.h"

#include "hk/Support/PyRepr.h"
#include "hk/Target/Target.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <set>

namespace mlir::hk {
#define GEN_PASS_DEF_HKVERIFY
#define GEN_PASS_DEF_HKLDSALLOC
#define GEN_PASS_DEF_HKCHECKBUDGETS
#include "hk/Dialect/HK/Transforms/Passes.h.inc"
} // namespace mlir::hk

using namespace mlir;
using namespace mlir::hk;

bool mlir::hk::isKernel(func::FuncOp f) {
  return f->hasAttr(attrs::kKernel);
}

std::string mlir::hk::atLoc(Operation *op) {
  std::string l = pythonLocOf(op->getLoc());
  return l.empty() ? "" : "\n    at " + l;
}

static StringRef archOf(func::FuncOp f) {
  if (auto a = f->getAttrOfType<StringAttr>(attrs::kArch))
    return a.getValue();
  return "gfx1100";
}

static int64_t intAttr(func::FuncOp f, StringRef name, int64_t dflt = 0) {
  if (auto a = f->getAttrOfType<IntegerAttr>(name))
    return a.getInt();
  return dflt;
}

/// The name a kernel author gave a value, as the Python IR called it.
static std::string nameOf(Value v) {
  if (auto r = dyn_cast<OpResult>(v))
    return valueNameOf(r.getOwner()->getLoc()).str();
  return valueNameOf(cast<BlockArgument>(v).getLoc()).str();
}

static StringRef opcodeOf(Operation *op) {
  if (isa<HKDialect>(op->getDialect()))
    return op->getName().stripDialect();
  return op->getName().getStringRef();
}

namespace {

//===----------------------------------------------------------------------===//
// hk-verify
//===----------------------------------------------------------------------===//

struct VerifyPass : public mlir::hk::impl::HKVerifyBase<VerifyPass> {
  void runOnOperation() override {
    func::FuncOp f = getOperation();
    if (!isKernel(f))
      return;
    const TargetInfo *tgt = getTarget(archOf(f));
    if (!tgt) {
      f.emitError() << "unknown arch " << archOf(f);
      return signalPassFailure();
    }
    int64_t warps = intAttr(f, attrs::kWarps, 1);
    if (warps < 1) {
      f.emitError() << "warps=" << warps << ": a kernel needs at least one warp";
      return signalPassFailure();
    }
    int maxWarps = tgt->maxWavesPerSimd * tgt->simdsPerLdsPool;
    if (warps > maxWarps) {
      f.emitError() << "warps=" << warps << " exceeds " << maxWarps
                    << ", the most " << tgt->arch
                    << " can put in one workgroup (" << tgt->maxWavesPerSimd
                    << " waves/SIMD x " << tgt->simdsPerLdsPool << " SIMDs)";
      return signalPassFailure();
    }

    // Nothing workgroup-wide inside a branch: a warp that skips a barrier
    // deadlocks the rest. A hang rather than a wrong answer, which is worse to
    // debug and trivial to rule out.
    WalkResult div = f.walk([&](Operation *op) {
      if (!isa<BarrierOp, StageLoadOp, StageCommitOp, GroupLoadOp,
               GroupStoreOp>(op))
        return WalkResult::advance();
      if (op->getParentOfType<scf::IfOp>()) {
        op->emitError() << opcodeOf(op)
                        << " is inside an hk.if_. It is a workgroup operation "
                           "-- every warp has to reach it or the ones that do "
                           "wait forever. Hoist it out and put only the work "
                           "it guards inside the branch."
                        << atLoc(op);
        return WalkResult::interrupt();
      }
      if (op->getParentOfType<scf::WhileOp>()) {
        op->emitError() << opcodeOf(op)
                        << " is inside a runtime while loop. It is a "
                           "workgroup operation -- every warp has to reach it "
                           "the same number of times, and a test that differs "
                           "between warps leaves the ones still looping "
                           "waiting forever. Loop with a for over a bound "
                           "every warp agrees on."
                        << atLoc(op);
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (div.wasInterrupted())
      return signalPassFailure();

    // fp8 register tiles exist only where the target has an fp8 WMMA.
    WalkResult dt = f.walk([&](Operation *op) {
      auto check = [&](Value v) {
        auto t = dyn_cast<RegTileType>(v.getType());
        if (!t || !isFp8(t.getElementType()) ||
            tgt->hasMma(dtypeName(t.getElementType())))
          return true;
        StringRef arch = archOf(f);
        op->emitError() << nameOf(v) << " is an "
                        << dtypeName(t.getElementType())
                        << " register tile, and " << arch
                        << " has no fp8 WMMA or fp8 tile to put it in -- "
                           "gfx1201 (RDNA4) does. On "
                        << arch
                        << " fp8 is a storage type: load it into an fp32 or "
                           "bf16 tile, which converts on the way in."
                        << atLoc(op);
        return false;
      };
      for (Value v : op->getOperands())
        if (!check(v))
          return WalkResult::interrupt();
      for (Value v : op->getResults())
        if (!check(v))
          return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (dt.wasInterrupted())
      return signalPassFailure();
  }
};

//===----------------------------------------------------------------------===//
// hk-lds-alloc
//===----------------------------------------------------------------------===//

struct LdsAllocPass : public mlir::hk::impl::HKLdsAllocBase<LdsAllocPass> {
  void runOnOperation() override {
    func::FuncOp f = getOperation();
    if (!isKernel(f))
      return;
    Builder b(f.getContext());
    int64_t offset = 0;
    auto place = [&](Operation *op, uint64_t size) {
      // Aligned to its own size when that is a power of two, else 16 B.
      int64_t align = (size && (size & (size - 1)) == 0) ? size : 16;
      offset = (offset + align - 1) / align * align;
      op->setAttr("offset", b.getI64IntegerAttr(offset));
      op->setAttr("nbytes", b.getI64IntegerAttr(size));
      offset += size;
    };
    // Top-level allocations only, in declaration order.
    for (Operation &op : f.getBody().front()) {
      if (auto a = dyn_cast<AllocSharedOp>(op))
        place(a, a.getResult().getType().getNumBytes());
      else if (auto v = dyn_cast<AllocSharedVecOp>(op))
        place(v, v.getResult().getType().getNumBytes());
    }
    int64_t arrivals = 0;
    f.walk([&](WgArriveOp) { ++arrivals; });
    f->setAttr(attrs::kLdsBytes, b.getI64IntegerAttr(offset));
    // kittens::wg_arrive declares one `__shared__ int` per instantiation;
    // 16 B is what the static segment rounds it to.
    f->setAttr(attrs::kLdsStatic, b.getI64IntegerAttr(16 * arrivals));
  }
};

//===----------------------------------------------------------------------===//
// hk-check-budgets
//===----------------------------------------------------------------------===//

struct Warning {
  std::string code, message, loc;
  std::string str() const {
    std::string s = "[" + code + "] " + message;
    if (!loc.empty())
      s += "\n    at " + loc;
    return s;
  }
};

/// Peak live register-tile VGPRs, by linear scan over the flattened program.
/// An estimate: tile storage, staging buffers and granule sets only.
static int estimateVgprs(func::FuncOp f, const TargetInfo &tgt) {
  SmallVector<Operation *> ops;
  f.getBody().walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (op != f.getOperation())
      ops.push_back(op);
  });
  DenseMap<Value, size_t> lastUse;
  for (auto [i, op] : llvm::enumerate(ops))
    for (Value v : op->getOperands())
      lastUse[v] = i;
  llvm::MapVector<Value, int> live;
  int peak = 0;
  for (auto [i, op] : llvm::enumerate(ops)) {
    for (Value r : op->getResults()) {
      Type t = r.getType();
      if (auto rt = dyn_cast<RegTileType>(t))
        live[r] = tgt.tileVgprs(dtypeBits(rt.getElementType()), rt.getRows(),
                                rt.getCols());
      else if (auto g = dyn_cast<GranulesType>(t))
        live[r] = g.getNumAddresses();
      else if (auto s = dyn_cast<StageType>(t))
        live[r] = 4 * s.getFloats();
    }
    int sum = 0;
    for (auto &kv : live)
      sum += kv.second;
    peak = std::max(peak, sum);
    for (Value v : op->getOperands()) {
      auto it = lastUse.find(v);
      if (it != lastUse.end() && it->second == i)
        live.erase(v);
    }
  }
  return peak;
}

struct CheckBudgetsPass : public mlir::hk::impl::HKCheckBudgetsBase<CheckBudgetsPass> {
  void runOnOperation() override {
    func::FuncOp f = getOperation();
    if (!isKernel(f))
      return;
    const TargetInfo *tgt = getTarget(archOf(f));
    if (!tgt) {
      f.emitError() << "unknown arch " << archOf(f);
      return signalPassFailure();
    }
    std::vector<Warning> warns;

    // A col-layout 16-bit operand read out of LDS costs 8x the instructions
    // of a row-layout one, and nothing about the result says so.
    f.walk<WalkOrder::PreOrder>([&](LoadSharedOp op) {
      auto dst = cast<RegTileType>(op.getDst() ? op.getDst().getType()
                                               : op.getResult().getType());
      if (dtypeBits(dst.getElementType()) != 16 ||
          tgt->isFastSmemLayout(dst.getLayout()))
        return;
      std::string layouts = llvm::join(tgt->fastSmemLayouts, ", ");
      std::string msg;
      llvm::raw_string_ostream os(msg);
      os << "loading " << pyRepr(dst) << " from LDS: a "
         << dtypeName(dst.getElementType()) << " operand in '"
         << dst.getLayout()
         << "' layout does not vectorise. Expect 16 scalar ds_read_u16 per "
            "base tile instead of 2 ds_read_b128 -- 8x the instructions. "
            "Fast layouts on "
         << tgt->arch << ": " << layouts
         << ". If you need the transpose, stage it transposed rather than "
            "loading and swapping: swap_layout moves data on this "
            "architecture, it is not a relabel.";
      warns.push_back({"slow-smem-layout", msg, pythonLocOf(op->getLoc())});
    });

    int64_t ldsBytes = intAttr(f, attrs::kLdsBytes);
    int64_t ldsStatic = intAttr(f, attrs::kLdsStatic);
    int64_t warps = intAttr(f, attrs::kWarps, 1);
    int64_t total = ldsBytes + ldsStatic;
    if (total > tgt->ldsPerWorkgroup) {
      std::string st;
      if (ldsStatic)
        st = " (" + std::to_string(ldsStatic) +
             " B of it static, declared by the library for wg_arrive)";
      f.emitError() << "LDS request is " << total << " B" << st
                    << " but one workgroup on " << tgt->arch
                    << " gets at most " << tgt->ldsPerWorkgroup
                    << " B -- the launch fails with "
                       "HSA_STATUS_ERROR_INVALID_ALLOCATION. Shrink a staged "
                       "tile, drop double buffering, or move a tile back into "
                       "registers.";
      return signalPassFailure();
    }

    int est = estimateVgprs(f, *tgt);
    if (est > tgt->vgprsPerSimd) {
      std::string msg = "register tiles alone need an estimated " +
                        std::to_string(est) + " VGPRs, more than the " +
                        std::to_string(tgt->vgprsPerSimd) +
                        " a SIMD has. This cannot be scheduled.";
      if (tgt->verified) {
        f.emitError() << msg;
        return signalPassFailure();
      }
      warns.push_back({"vgpr-budget", msg + " (target unverified)", ""});
    } else if (est > 256) {
      warns.push_back(
          {"vgpr-budget",
           "estimated " + std::to_string(est) +
               " VGPRs of live register tiles (occupancy " +
               std::to_string(tgt->occupancy(est)) + " waves/SIMD, headroom " +
               std::to_string(tgt->vgprHeadroom(est)) +
               "). This is an estimate that counts tile storage only; the "
               "real number comes from the compiler after codegen. If it "
               "spills, the kernel is wrong, not slow.",
           ""});
    }

    if (ldsBytes && est) {
      int occLds = tgt->occupancyByLds(ldsBytes, warps);
      int occReg = tgt->occupancy(est);
      if (occLds < occReg)
        warns.push_back(
            {"lds-bound",
             "LDS binds occupancy below registers: " + std::to_string(occLds) +
                 " waves/SIMD from " + std::to_string(ldsBytes) +
                 " B of LDS vs " + std::to_string(occReg) +
                 " from registers. The compiler's occupancy remark reports "
                 "the register number only and will look better than what "
                 "you get.",
             ""});
    }

    Builder b(f.getContext());
    SmallVector<Attribute> strs;
    for (auto &w : warns)
      strs.push_back(b.getStringAttr(w.str()));
    f->setAttr(attrs::kWarnings, b.getArrayAttr(strs));
  }
};

} // namespace

void mlir::hk::buildKernelPipeline(OpPassManager &pm) {
  pm.addNestedPass<func::FuncOp>(createHKVerify());
  pm.addNestedPass<func::FuncOp>(createHKLdsAlloc());
  pm.addNestedPass<func::FuncOp>(createHKLdsPipeline());
  pm.addNestedPass<func::FuncOp>(createHKCheckBudgets());
}

void mlir::hk::registerKernelPipeline() {
  PassPipelineRegistration<>(
      "hk-kernel-pipeline",
      "verify, allocate LDS, derive LDS waits, check budgets",
      [](OpPassManager &pm) { buildKernelPipeline(pm); });
}
