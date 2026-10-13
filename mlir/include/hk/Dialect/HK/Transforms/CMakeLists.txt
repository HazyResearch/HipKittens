set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name HK)
add_public_tablegen_target(MLIRHKPassesIncGen)
