add_mlir_translation_library(MLIRHKToCpp
  HKToCpp.cpp

  LINK_LIBS PUBLIC
  MLIRHK
  HKSupport
  HKTarget
  MLIRTranslateLib
)
