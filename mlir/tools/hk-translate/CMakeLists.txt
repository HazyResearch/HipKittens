add_llvm_executable(hk-translate hk-translate.cpp)
llvm_update_compile_flags(hk-translate)
target_link_libraries(hk-translate PRIVATE
  MLIRHK
  MLIRHKToCpp
  MLIRParser
  MLIRTranslateLib
  MLIRSupport
)
mlir_check_link_libraries(hk-translate)
