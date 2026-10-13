#include "hk-c/HK.h"

#include "hk/Dialect/HK/IR/HKDialect.h"
#include "hk/Dialect/HK/Transforms/Passes.h"
#include "hk/Target/Cpp/HKToCpp.h"
#include "hk/Target/Target.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Utils.h"
#include "mlir/IR/Diagnostics.h"

#include <mutex>

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(HK, hk, mlir::hk::HKDialect)

MlirStringRef mlirHKDialectVersion(void) {
  return mlirStringRefCreateFromCString("hk-dialect-1");
}

void mlirHKRegisterPasses(void) {
  static std::once_flag once;
  std::call_once(once, [] {
    mlir::hk::registerHKPasses();
    mlir::hk::registerKernelPipeline();
  });
}

MlirLogicalResult mlirHKTranslateKernelToCpp(
    MlirOperation kernel, intptr_t numIncludes, MlirStringRef const *includes,
    MlirStringCallback callback, void *userData,
    MlirStringCallback errorCallback, void *errorUserData) {
  mlir::Operation *op = unwrap(kernel);
  auto f = llvm::dyn_cast<mlir::func::FuncOp>(op);
  if (!f) {
    std::string msg = "hk: expected the kernel func.func";
    errorCallback(mlirStringRefCreate(msg.data(), msg.size()), errorUserData);
    return mlirLogicalResultFailure();
  }
  std::vector<std::string> incs;
  for (intptr_t i = 0; i < numIncludes; ++i)
    incs.emplace_back(includes[i].data, includes[i].length);

  mlir::ScopedDiagnosticHandler handler(
      op->getContext(), [&](mlir::Diagnostic &d) {
        std::string msg = d.str();
        errorCallback(mlirStringRefCreate(msg.data(), msg.size()),
                      errorUserData);
        return mlir::success();
      });
  std::string out;
  llvm::raw_string_ostream os(out);
  if (failed(mlir::hk::translateKernelToCpp(f, incs, os)))
    return mlirLogicalResultFailure();
  os.flush();
  callback(mlirStringRefCreate(out.data(), out.size()), userData);
  return mlirLogicalResultSuccess();
}

void mlirHKTargetFacts(MlirStringRef arch, MlirStringCallback callback,
                       void *userData) {
  const mlir::hk::TargetInfo *t =
      mlir::hk::getTarget(llvm::StringRef(arch.data, arch.length));
  if (!t)
    return;
  std::string s;
  llvm::raw_string_ostream os(s);
  os << "arch=" << t->arch << "\nwave=" << t->wave
     << "\nmax_waves_per_simd=" << t->maxWavesPerSimd
     << "\nvgprs_per_simd=" << t->vgprsPerSimd
     << "\nvgpr_granule=" << t->vgprGranule
     << "\nsgprs_per_wave=" << t->sgprsPerWave
     << "\nlds_per_workgroup=" << t->ldsPerWorkgroup
     << "\nlds_pool_bytes=" << t->ldsPoolBytes
     << "\nsimds_per_lds_pool=" << t->simdsPerLdsPool
     << "\nverified=" << (t->verified ? 1 : 0);
  for (auto &[name, m] : t->mma)
    os << "\nmma." << name << "=" << m.m << "," << m.n << "," << m.k << ","
       << m.operandVgprs << "," << m.accumVgprs;
  for (auto l : t->fastSmemLayouts)
    os << "\nfast_smem_layout=" << l;
  os.flush();
  callback(mlirStringRefCreate(s.data(), s.size()), userData);
}
