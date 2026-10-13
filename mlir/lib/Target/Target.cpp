#include "hk/Target/Target.h"

#include "llvm/ADT/StringSwitch.h"

#include <algorithm>

using namespace mlir::hk;

namespace {
// gfx1100: measured on a W7900D. See python/hk/target/gfx1100.py.
const std::pair<llvm::StringRef, MmaShape> kMma1100[] = {
    {"bf16", {16, 16, 16, 8, 8}},
    {"fp16", {16, 16, 16, 8, 8}},
};
// gfx1201: written, not verified. See python/hk/target/gfx1201.py.
const std::pair<llvm::StringRef, MmaShape> kMma1201[] = {
    {"bf16", {16, 16, 16, 4, 8}},
    {"fp16", {16, 16, 16, 4, 8}},
    {"fp8e4m3", {16, 16, 16, 2, 8}},
    {"fp8e5m2", {16, 16, 16, 2, 8}},
};
const llvm::StringRef kRowOnly[] = {"row"};

const TargetInfo kGfx1100{"gfx1100", 32, 16, 1536, 24, 106, 65536, 131072, 4,
                          kMma1100, kRowOnly, true};
const TargetInfo kGfx1201{"gfx1201", 32, 16, 1536, 24, 106, 65536, 65536, 4,
                          kMma1201, kRowOnly, false};
} // namespace

const TargetInfo *mlir::hk::getTarget(llvm::StringRef arch) {
  llvm::StringRef key = llvm::StringSwitch<llvm::StringRef>(arch)
                            .Case("RDNA3", "gfx1100")
                            .Case("gfx11", "gfx1100")
                            .Case("RDNA4", "gfx1201")
                            .Case("gfx12", "gfx1201")
                            .Default(arch);
  if (key == "gfx1100")
    return &kGfx1100;
  if (key == "gfx1201")
    return &kGfx1201;
  return nullptr;
}

int TargetInfo::vgprRounded(int vgprs) const {
  return (vgprs + vgprGranule - 1) / vgprGranule * vgprGranule;
}

int TargetInfo::occupancy(int vgprs) const {
  int rounded = vgprRounded(vgprs);
  if (rounded <= 0)
    return maxWavesPerSimd;
  return std::min(maxWavesPerSimd, vgprsPerSimd / rounded);
}

int TargetInfo::vgprHeadroom(int vgprs) const {
  int occ = occupancy(vgprs);
  if (occ <= 1)
    return 0;
  int maxForLevel = (vgprsPerSimd / occ) / vgprGranule * vgprGranule;
  return maxForLevel - vgprs;
}

int TargetInfo::occupancyByLds(int64_t ldsBytes, int wavesPerWorkgroup) const {
  if (ldsBytes <= 0)
    return maxWavesPerSimd;
  int64_t workgroups = ldsPoolBytes / ldsBytes;
  int64_t waves = workgroups * wavesPerWorkgroup;
  return static_cast<int>(
      std::min<int64_t>(maxWavesPerSimd, waves / simdsPerLdsPool));
}

int TargetInfo::tileVgprs(unsigned bits, unsigned rows, unsigned cols) const {
  int base = int(rows / 16) * int(cols / 16);
  if (mma.empty())
    return base * 8;
  const MmaShape &s = mma.front().second;
  return base * (bits == 16 ? s.operandVgprs : s.accumVgprs);
}

bool TargetInfo::hasMma(llvm::StringRef dtype) const {
  return llvm::any_of(mma, [&](auto &e) { return e.first == dtype; });
}

bool TargetInfo::isFastSmemLayout(llvm::StringRef layout) const {
  return llvm::is_contained(fastSmemLayouts, layout);
}
