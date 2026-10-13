// The native half of hk._mlir.dialects.hk: dialect registration, the pass
// registry, and the translation to HipKittens C++.

#include "hk-c/HK.h"

#include "mlir-c/Dialect/Arith.h"
#include "mlir-c/Dialect/Func.h"
#include "mlir-c/Dialect/SCF.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

#include <string>
#include <vector>

namespace nb = nanobind;

namespace {
void append(MlirStringRef part, void *userData) {
  static_cast<std::string *>(userData)->append(part.data, part.length);
}
void appendLine(MlirStringRef part, void *userData) {
  auto *s = static_cast<std::string *>(userData);
  if (!s->empty())
    s->push_back('\n');
  s->append(part.data, part.length);
}
} // namespace

NB_MODULE(_hkDialects, m) {
  auto hk = m.def_submodule("hk");

  hk.def(
      "register_dialects",
      [](MlirContext context, bool load) {
        MlirDialectHandle handles[] = {
            mlirGetDialectHandle__hk__(), mlirGetDialectHandle__arith__(),
            mlirGetDialectHandle__scf__(), mlirGetDialectHandle__func__()};
        for (MlirDialectHandle h : handles) {
          mlirDialectHandleRegisterDialect(h, context);
          if (load)
            mlirDialectHandleLoadDialect(h, context);
        }
      },
      nb::arg("context").none() = nb::none(), nb::arg("load") = true,
      "Register (and load) hk, arith, scf and func in a context.");

  hk.def("dialect_version", []() {
    MlirStringRef v = mlirHKDialectVersion();
    return std::string(v.data, v.length);
  }, "The dialect's semantic version, part of the compile cache key.");

  hk.def("register_passes", []() { mlirHKRegisterPasses(); },
         "Register the hk passes and hk-kernel-pipeline.");

  hk.def(
      "translate_to_cpp",
      [](MlirOperation kernel, const std::vector<std::string> &includes) {
        std::vector<MlirStringRef> refs;
        for (const std::string &s : includes)
          refs.push_back(mlirStringRefCreate(s.data(), s.size()));
        std::string out, errors;
        MlirLogicalResult r = mlirHKTranslateKernelToCpp(
            kernel, static_cast<intptr_t>(refs.size()), refs.data(), append,
            &out, appendLine, &errors);
        if (mlirLogicalResultIsFailure(r))
          throw nb::value_error(
              ("hk: translation to C++ failed:\n" + errors).c_str());
        return out;
      },
      nb::arg("kernel"), nb::arg("includes"),
      "The kernel func.func as HipKittens C++ (no module boundary).");

  hk.def(
      "target_facts",
      [](const std::string &arch) {
        std::string out;
        mlirHKTargetFacts(mlirStringRefCreate(arch.data(), arch.size()),
                          append, &out);
        return out;
      },
      nb::arg("arch"), "The C++ target table's entry for `arch`.");
}
