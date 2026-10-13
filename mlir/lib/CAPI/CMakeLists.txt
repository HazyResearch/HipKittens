add_mlir_public_c_api_library(HKCAPI
  HK.cpp

  LINK_LIBS PUBLIC
  MLIRHK
  MLIRHKTransforms
  MLIRHKToCpp
)
