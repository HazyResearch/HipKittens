#ifndef HK_C_HK_H
#define HK_C_HK_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(HK, hk);

/// Register the hk passes and `hk-kernel-pipeline` with the global registry,
/// so a PassManager can parse them. Idempotent.
MLIR_CAPI_EXPORTED void mlirHKRegisterPasses(void);

/// The kernel `func.func` as HipKittens C++, streamed to `callback`. On
/// failure the diagnostics go to `errorCallback`, one per call.
MLIR_CAPI_EXPORTED MlirLogicalResult mlirHKTranslateKernelToCpp(
    MlirOperation kernel, intptr_t numIncludes, MlirStringRef const *includes,
    MlirStringCallback callback, void *userData,
    MlirStringCallback errorCallback, void *errorUserData);

/// One target's facts as "key=value" lines, for the test that keeps the C++
/// table and python/hk/target in step. Empty for an unknown arch.
MLIR_CAPI_EXPORTED void mlirHKTargetFacts(MlirStringRef arch,
                                          MlirStringCallback callback,
                                          void *userData);

#ifdef __cplusplus
}
#endif

#endif // HK_C_HK_H
