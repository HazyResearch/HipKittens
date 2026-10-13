// hk-lds-pipeline: derive every `s_waitcnt lgkmcnt`, and prove every
// asynchronous tile is bound before it is read.
//
// `load_shared(wait=false)` issues its ds_reads through an inline-asm output,
// so LLVM believes the destination is ready the instant the asm ends, and
// nothing stops it hoisting a consuming WMMA above the wait. Binding the
// tile's registers after the wait fixes that; this pass makes sure every
// wait names the tiles it retires, and computes the N nobody should compute
// by hand: LDS returns in order, so retiring a tile means waiting until only
// what was issued after it is outstanding.
//
// Loops are walked twice, the second time from the state the first left,
// and must agree -- a body whose derived counts differ has no steady state.
//
// A port of python/hk/ir/passes/lds_pipeline.py; the messages are its.

#include "hk/Dialect/HK/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/MapVector.h"

#include <optional>
#include <set>

namespace mlir::hk {
#define GEN_PASS_DEF_HKLDSPIPELINE
#include "hk/Dialect/HK/Transforms/Passes.h.inc"
} // namespace mlir::hk

using namespace mlir;
using namespace mlir::hk;

namespace {

using Scope = SmallVector<Operation *, 4>;

struct Entry {
  const void *id;  ///< the tile for a read, the op for a write
  int64_t count;
  Value value;
  bool write;
  std::optional<Scope> scope;  ///< nullopt once a region it was issued in closed
};

struct State {
  SmallVector<Entry> queue;
  llvm::MapVector<const void *, Value> bound;
  llvm::MapVector<const void *, Operation *> unbound;

  int indexOf(Value v) const {
    for (auto [i, e] : llvm::enumerate(queue))
      if (e.id == v.getAsOpaquePointer() && !e.write)
        return i;
    return -1;
  }
  void retireAllUnbound(Operation *by) {
    for (auto &e : queue)
      if (!e.write)
        unbound[e.id] = by;
    queue.clear();
  }
};

std::string nameOf(Value v) {
  if (auto r = dyn_cast<OpResult>(v))
    return valueNameOf(r.getOwner()->getLoc()).str();
  return valueNameOf(cast<BlockArgument>(v).getLoc()).str();
}

StringRef opcodeOf(Operation *op) {
  if (isa<HKDialect>(op->getDialect()))
    return op->getName().stripDialect();
  return op->getName().getStringRef();
}

/// Whether every wave that reaches `scope` issued an op issued in `issued`.
bool live(const std::optional<Scope> &issued, const Scope &scope) {
  if (!issued || issued->size() > scope.size())
    return false;
  for (auto [i, op] : llvm::enumerate(*issued))
    if (scope[i] != op)
      return false;
  return true;
}

struct Walker {
  bool failed = false;

  bool fail(Operation *op, const Twine &msg) {
    op->emitError(msg);
    failed = true;
    return false;
  }

  /// Nothing may read a tile that is in flight or retired-but-unbound.
  bool checkReads(Operation *op, State &st) {
    for (Value v : op->getOperands()) {
      if (!isa<RegTileType>(v.getType()))
        continue;
      if (st.indexOf(v) >= 0)
        return fail(op, Twine(opcodeOf(op)) + " reads " + nameOf(v) +
                            ", whose LDS reads are still in flight. Name it "
                            "in an lds_wait_for before using it. This is an "
                            "error rather than a race you might get away "
                            "with: the reads have not landed, and the "
                            "consumer will not wait for them on its own." +
                            atLoc(op));
      auto w = st.unbound.find(v.getAsOpaquePointer());
      if (w != st.unbound.end())
        return fail(op, Twine(opcodeOf(op)) + " reads " + nameOf(v) +
                            ". Its LDS reads were retired by an lds_wait_for "
                            "that did not name it, so the data has landed "
                            "but the register dependence has not been put "
                            "back -- this consumer is free to schedule above "
                            "the wait, and will, intermittently. Add " +
                            nameOf(v) + " to that lds_wait_for." +
                            atLoc(w->second) + atLoc(op));
    }
    return true;
  }

  bool waitFor(LdsWaitForOp op, State &st, const Scope &scope) {
    SmallVector<int> positions;
    for (Value v : op.getTiles()) {
      int i = st.indexOf(v);
      if (i < 0) {
        const void *id = v.getAsOpaquePointer();
        // Already retired by an earlier wait, bound or not: naming it here
        // is correct and costs nothing, and contributes no position.
        if (st.bound.count(id) || st.unbound.count(id))
          continue;
        return fail(op, "lds_wait_for names " + nameOf(v) +
                            ", which has no LDS reads outstanding and has "
                            "never had any. Either the load is missing, or "
                            "this names the wrong tile." +
                            atLoc(op));
      }
      positions.push_back(i);
    }
    int cut = positions.empty() ? -1
                                : *std::max_element(positions.begin(),
                                                    positions.end());
    int64_t n = 0;
    bool allLive = true;
    for (size_t j = cut + 1; j < st.queue.size(); ++j) {
      n += st.queue[j].count;
      allLive &= live(st.queue[j].scope, scope);
    }
    if (n && !allLive)
      return fail(op, "this lds_wait_for needs lgkmcnt(" + Twine(n) +
                          "), but some of the LDS ops it is counting were "
                          "issued inside an hk.if_. A wave that took the "
                          "other side of the branch has fewer in flight and "
                          "this wait lets it run on past reads that have not "
                          "landed. Retire the conditional ops before the "
                          "branch closes, or use barrier(drain=True), which "
                          "is lgkmcnt(0) and does not count." +
                          atLoc(op));
    llvm::DenseSet<const void *> named;
    for (Value v : op.getTiles())
      named.insert(v.getAsOpaquePointer());
    for (int j = 0; j <= cut; ++j) {
      auto &e = st.queue[j];
      if (e.write)
        continue;
      if (!named.count(e.id))
        st.unbound[e.id] = op;
    }
    st.queue.erase(st.queue.begin(), st.queue.begin() + (cut + 1));
    for (Value v : op.getTiles()) {
      st.unbound.erase(v.getAsOpaquePointer());
      st.bound[v.getAsOpaquePointer()] = v;
    }
    if (auto prev = op.getN(); prev && *prev != uint64_t(n))
      return fail(op, "this lds_wait_for needs lgkmcnt(" + Twine(*prev) +
                          ") on one iteration and lgkmcnt(" + Twine(n) +
                          ") on the next: the loop has no steady state, so "
                          "the queue depth depends on how many iterations "
                          "have run. Peel the prologue out of the loop, or "
                          "issue the same reads every iteration." +
                          atLoc(op));
    op.setNAttr(Builder(op.getContext()).getI64IntegerAttr(n));
    return true;
  }

  bool barrier(BarrierOp op, State &st) {
    if (st.queue.empty())
      return true;
    if (!op.getDrain()) {
      std::set<std::string> kinds;
      for (auto &e : st.queue)
        kinds.insert(e.write ? "write" : "read");
      std::string what;
      for (auto &k : kinds) {
        if (!what.empty())
          what += " and ";
        what += "ds_" + k + "s";
      }
      return fail(op, "barrier with " + what +
                          " still in flight. s_barrier orders execution, not "
                          "memory: past it the other warps touch this LDS "
                          "the other way round, and nothing waits for these. "
                          "Either retire them first -- an lds_wait_for that "
                          "names the tiles, which is free if the math needed "
                          "them anyway -- or write barrier(drain=True) to "
                          "spend an lgkmcnt(0) here." +
                          atLoc(op));
    }
    st.retireAllUnbound(op);
    return true;
  }

  bool load(Operation *op, Value dst, bool wait, int64_t count, State &st,
            const Scope &scope) {
    if (st.indexOf(dst) >= 0)
      return fail(op, Twine(opcodeOf(op)) + " writes " + nameOf(dst) +
                          " while its previous LDS reads are still in "
                          "flight. The earlier reads would land on top of "
                          "the newer ones -- retire them with an "
                          "lds_wait_for first." +
                          atLoc(op));
    const void *id = dst.getAsOpaquePointer();
    if (wait) {
      // load<true> ends in lgkmcnt(0): it retires everything, binds nothing.
      st.retireAllUnbound(op);
      st.unbound.erase(id);
    } else {
      st.unbound.erase(id);
      st.bound.erase(id);
      st.queue.push_back({id, count, dst, false, scope});
    }
    return true;
  }

  State walk(Block &block, State st, const Scope &scope) {
    for (Operation &opRef : block) {
      if (failed)
        return st;
      Operation *op = &opRef;
      if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        // Once: a branch runs at most once per encounter. Ops issued inside
        // carry it in their scope, and once it closes a counted wait that
        // would count them is the error.
        Scope inner = scope;
        inner.push_back(op);
        st = walk(ifOp.getThenRegion().front(), std::move(st), inner);
        if (!ifOp.getElseRegion().empty())
          st = walk(ifOp.getElseRegion().front(), std::move(st), inner);
        for (auto &e : st.queue)
          if (e.scope && llvm::is_contained(*e.scope, op))
            e.scope = std::nullopt;
        continue;
      }
      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        State first = walk(*forOp.getBody(), st, scope);
        st = walk(*forOp.getBody(), std::move(first), scope);
        continue;
      }
      if (auto w = dyn_cast<LdsWaitForOp>(op)) {
        waitFor(w, st, scope);
        continue;
      }
      if (auto b = dyn_cast<BarrierOp>(op)) {
        barrier(b, st);
        continue;
      }
      if (auto l = dyn_cast<LoadSharedOp>(op)) {
        Value dst = l.getDst() ? l.getDst() : l.getResult();
        load(op, dst, l.getWait(), l.getLdsLoads(), st, scope);
        continue;
      }
      if (auto l = dyn_cast<LoadFragOp>(op)) {
        Value dst = l.getDst() ? l.getDst() : l.getResult();
        load(op, dst, l.getWait(), l.getLdsLoads(), st, scope);
        continue;
      }
      if (auto s = dyn_cast<StoreFragOp>(op)) {
        // Two raw ds_write_b128 with no wait of their own: they only raise
        // the immediate of every wait issued while they are in flight.
        st.queue.push_back({op, 2, s.getGranules(), true, scope});
        continue;
      }
      if (auto c = dyn_cast<StageCommitOp>(op)) {
        if (c.getWait())
          st.retireAllUnbound(op);
        else
          st.queue.push_back(
              {op, c.getBuf().getType().getCalls(), c.getDst(), true, scope});
        continue;
      }
      checkReads(op, st);
    }
    return st;
  }
};

struct LdsPipelinePass : public mlir::hk::impl::HKLdsPipelineBase<LdsPipelinePass> {
  void runOnOperation() override {
    func::FuncOp f = getOperation();
    if (!isKernel(f))
      return;
    // A stale count from an earlier run would read as a missing steady state.
    f.walk([](LdsWaitForOp op) { op.removeNAttr(); });
    Walker w;
    State fin = w.walk(f.getBody().front(), State(), Scope());
    if (w.failed)
      return signalPassFailure();
    std::string names;
    for (auto &e : fin.queue) {
      if (e.write)
        continue;
      if (!names.empty())
        names += ", ";
      names += nameOf(e.value);
    }
    if (!names.empty()) {
      f.emitError() << "the kernel ends with LDS reads still in flight for "
                    << names
                    << ". Either they are dead -- drop the load -- or a wait "
                       "is missing.";
      return signalPassFailure();
    }
  }
};

} // namespace
