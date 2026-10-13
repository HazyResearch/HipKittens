#ifndef HK_BASE_TD
#define HK_BASE_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

def HK_Dialect : Dialect {
  let name = "hk";
  let summary = "HipKittens tile IR for AMD RDNA GPUs";
  let description = [{
    A tile IR. The unit of everything is a tile -- a register tile `!hk.rt`,
    an LDS tile `!hk.st`, the vectors that come off them, and a global
    `!hk.gl` -- not a scalar and not a thread. That is the HipKittens
    abstraction: the hard parts live in include/rdna3, and this dialect's job
    is to pick the right call and prove the choice is legal.

    Tiles are buffers. An op either makes a fresh tile (it has a result) or
    writes one that already exists (`into %dst`), which is how the target
    library spells everything -- `kittens::row_sum(acc, src, acc)` -- and how
    a loop-carried accumulator stays the same register across iterations.
    Scalars are SSA values in `arith`; control flow is `scf`.
  }];
  let cppNamespace = "::mlir::hk";
  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
    void registerTypes();
  }];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::func::FuncDialect"
  ];
}

class HK_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<HK_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

class HK_Op<string mnemonic, list<Trait> traits = []>
    : Op<HK_Dialect, mnemonic, traits>;

#endif // HK_BASE_TD
