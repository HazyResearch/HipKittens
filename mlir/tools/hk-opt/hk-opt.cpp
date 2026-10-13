#include "hk/Dialect/HK/IR/HKDialect.h"
#include "hk/Dialect/HK/Transforms/Passes.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

int main(int argc, char **argv) {
  mlir::registerTransformsPasses();
  mlir::hk::registerHKPasses();
  mlir::hk::registerKernelPipeline();

  mlir::DialectRegistry registry;
  registry.insert<mlir::hk::HKDialect, mlir::arith::ArithDialect,
                  mlir::scf::SCFDialect, mlir::func::FuncDialect>();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "hk optimizer driver\n", registry));
}
