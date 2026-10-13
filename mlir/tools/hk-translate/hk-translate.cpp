#include "hk/Target/Cpp/HKToCpp.h"

#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

int main(int argc, char **argv) {
  mlir::hk::registerToCppTranslation();
  return mlir::failed(
      mlir::mlirTranslateMain(argc, argv, "hk translation driver\n"));
}
