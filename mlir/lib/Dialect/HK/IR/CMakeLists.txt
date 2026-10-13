add_mlir_dialect_library(MLIRHK
  HKDialect.cpp
  HKTypes.cpp
  HKOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${HK_MLIR_SOURCE_DIR}/include/hk

  DEPENDS
  MLIRHKOpsIncGen
  MLIRHKEnumsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRSCFDialect
  MLIRFuncDialect
  MLIRSideEffectInterfaces
)
