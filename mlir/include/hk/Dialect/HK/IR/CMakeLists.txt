add_mlir_dialect(HKOps hk)
add_mlir_doc(HKOps HKOps HK/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS HKOps.td)
mlir_tablegen(HKEnums.h.inc -gen-enum-decls)
mlir_tablegen(HKEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRHKEnumsIncGen)
add_dependencies(MLIRHKOpsIncGen MLIRHKEnumsIncGen)
