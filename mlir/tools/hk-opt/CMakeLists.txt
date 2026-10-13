add_llvm_executable(hk-opt hk-opt.cpp)
llvm_update_compile_flags(hk-opt)
target_link_libraries(hk-opt PRIVATE
  MLIRHK
  MLIRHKTransforms
  MLIRArithDialect
  MLIRSCFDialect
  MLIRFuncDialect
  MLIROptLib
  MLIRTransforms
)
mlir_check_all_link_libraries(hk-opt)
