include(AddMLIRPython)

# Every MLIR module lives under hk._mlir (the API is embedded relocatably).
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=${MLIR_PYTHON_PACKAGE_PREFIX}.")

declare_mlir_python_sources(HKPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT HKPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/hk_mlir"
  TD_FILE dialects/HKOps.td
  SOURCES
    dialects/hk.py
  DIALECT_NAME hk
  GEN_ENUM_BINDINGS)

declare_mlir_python_extension(HKPythonSources.Extension
  MODULE_NAME _hkDialects
  ADD_TO_PARENT HKPythonSources
  SOURCES
    HKExtension.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
    MLIRCAPIArith
    MLIRCAPISCF
    MLIRCAPIFunc
    MLIRCAPITransforms
    HKCAPI
)

set(_hk_declared_sources
  HKPythonSources
  MLIRPythonSources.Core
  MLIRPythonSources.Dialects.builtin
  MLIRPythonSources.Dialects.arith
  MLIRPythonSources.Dialects.scf
  MLIRPythonSources.Dialects.func
)

add_mlir_python_common_capi_library(HKPythonCAPI
  INSTALL_COMPONENT HKPythonModules
  INSTALL_DESTINATION "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../.."
  DECLARED_SOURCES ${_hk_declared_sources}
)

add_mlir_python_modules(HKPythonModules
  ROOT_PREFIX "${MLIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}"
  INSTALL_PREFIX "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}"
  DECLARED_SOURCES ${_hk_declared_sources}
  COMMON_CAPI_LINK_LIBS
    HKPythonCAPI
)
