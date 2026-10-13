#ifndef HK_TARGET_TARGET_H
#define HK_TARGET_TARGET_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <utility>

namespace mlir::hk {

/// One matrix instruction: D[m,n] += A[m,k] * B[k,n].
struct MmaShape {
  int m, n, k;
  int operandVgprs; ///< 32-bit VGPRs one operand fragment takes, per lane.
  int accumVgprs;   ///< ... and one accumulator fragment.
};

/// Architecture facts, as data. Mirrors python/hk/target; a test keeps the
/// two in step.
struct TargetInfo {
  llvm::StringRef arch;
  int wave;
  int maxWavesPerSimd;
  int vgprsPerSimd;
  int vgprGranule;
  int sgprsPerWave;
  int64_t ldsPerWorkgroup;
  int64_t ldsPoolBytes;
  int simdsPerLdsPool;
  /// In declaration order; the first entry prices a register tile.
  llvm::ArrayRef<std::pair<llvm::StringRef, MmaShape>> mma;
  llvm::ArrayRef<llvm::StringRef> fastSmemLayouts;
  bool verified;

  int vgprRounded(int vgprs) const;
  int occupancy(int vgprs) const;
  int vgprHeadroom(int vgprs) const;
  int occupancyByLds(int64_t ldsBytes, int wavesPerWorkgroup) const;
  int tileVgprs(unsigned bits, unsigned rows, unsigned cols) const;
  bool hasMma(llvm::StringRef dtype) const;
  bool isFastSmemLayout(llvm::StringRef layout) const;
};

/// By gfx name or by the GPU_TARGET name (RDNA3, gfx11, ...); null if unknown.
const TargetInfo *getTarget(llvm::StringRef arch);

} // namespace mlir::hk

#endif // HK_TARGET_TARGET_H
