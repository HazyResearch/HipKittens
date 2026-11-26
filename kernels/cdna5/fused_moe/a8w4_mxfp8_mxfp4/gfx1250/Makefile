# gfx1250 fused MoE GEMM build (self-contained kernel).
# Usage: make            (= make decode; build pybind module gfx1250_moe_gemm, decode config)
#        make decode     (decode config: single-buffer, per-K-step S2R, per-warp B)
#        make prefill    (prefill config: -DMOE_PREFILL + prefill tile params)
#        make clean
# Note:  Builds with ROCm hipcc (gfx1250). Set THUNDERKITTENS_ROOT to the repo root,
#        e.g. export THUNDERKITTENS_ROOT=/workdir/HipKittens
#        (do NOT `source env.src` if the repo is on an NTFS/CRLF mount — the stray
#         carriage return corrupts the include path; export the vars directly.)
#
# Decode and prefill are two compile-time configs of the SAME kernel and the SAME pybind module;
# `make decode` / `make prefill` select the config (rebuild to switch). The decode build is the
# default and stays byte-identical to the previous default. The exact flag sets live in
# DECODE_FLAGS / PREFILL_FLAGS below, so neither has to be retyped on the command line.

GPU_TARGET ?= CDNA5
TARGET = gfx1250_moe_gemm
SRC = moe_gemm.cpp

ROCM_INSTALL_DIR := $(ROCM_PATH)
HIPCXX ?= /opt/rocm/bin/hipcc
HIP_INCLUDE_DIR  := $(shell dirname $(shell dirname $(shell which $(HIPCXX) 2>/dev/null || echo /opt/rocm/bin/hipcc)))/include/hip

ifeq ($(GPU_TARGET),CDNA5)
HIPFLAGS += -DKITTENS_CDNA5 --offload-arch=gfx1250
endif

CXX_STD   := c++20
COMMON_FLAGS := -std=$(CXX_STD) -w -Rpass-analysis=kernel-resource-usage
ICPPFLAGS := -I${THUNDERKITTENS_ROOT}/include -I$(HIP_INCLUDE_DIR)

# WORKAROUND (LLVM codegen bug), AUTO-GATED BY ROCm VERSION.
# On ROCm < 7.15, both machine schedulers mis-schedule v_wmma_scale_f32_16x16x128_f8f6f4 and produce
# wrong results at -O2/-O3; the workaround disables both schedulers. ROCm 7.15 fixes the miscompile,
# and keeping the schedulers disabled on 7.15+ is itself incorrect, so SCHED_FIX auto-disables on
# ROCm >= 7.15. Detection is conservative: if the ROCm version can't be read, the workaround is kept.
# Override on the make command line with `SCHED_FIX="...flags..."` or `SCHED_FIX=` (empty). Applied in
# the recipe (not COMMON_FLAGS) so it survives a COMMON_FLAGS override.
ROCM_VER   := $(shell (cat $(ROCM_PATH)/.info/version 2>/dev/null || hipconfig --version 2>/dev/null) | grep -oE '[0-9]+\.[0-9]+' | head -1)
ROCM_MAJOR := $(or $(firstword $(subst ., ,$(ROCM_VER))),0)
ROCM_MINOR := $(or $(word 2,$(subst ., ,$(ROCM_VER))),0)
SCHED_FIX_DEFAULT := -mllvm -enable-misched=false -mllvm -enable-post-misched=false
ifeq ($(shell { test "$(ROCM_MAJOR)" -gt 7 || { test "$(ROCM_MAJOR)" -eq 7 && test "$(ROCM_MINOR)" -ge 15; }; } && echo modern),modern)
SCHED_FIX_DEFAULT :=
endif
SCHED_FIX ?= $(SCHED_FIX_DEFAULT)

PY_LDFLAGS := $(shell python3-config --ldflags | sed 's/-lcrypt//g')
PY_FLAGS := $(PY_LDFLAGS) -I${THUNDERKITTENS_ROOT}/prototype
PY_FLAGS += $(shell python3 -m pybind11 --includes) -shared -fPIC

# Prefill and decode are two compile-time configs of the SAME kernel: they differ only in COMMON_FLAGS
# and both build the one pybind module `gfx1250_moe_gemm` (rebuild to switch). DECODE is the default.
DECODE_FLAGS  := $(COMMON_FLAGS)
PREFILL_FLAGS := $(COMMON_FLAGS) -DMOE_BLOCK_M=48 -DMOE_BLOCK_N=128 -DMOE_MTILES=1 \
                 -DMOE_KTILE=256 -DMOE_PRESHUFFLE -DMOE_PREFILL

.PHONY: all decode prefill build clean

all: decode

# Decode (default): single-buffer, per-K-step S2R, per-warp B.
decode:
	@$(MAKE) --no-print-directory build COMMON_FLAGS="$(DECODE_FLAGS)"

# Prefill (weight-load-bound): hoisted per-K-tile S2R, dense WMMA, one shared 2D TDM B, double-buffered.
prefill:
	@$(MAKE) --no-print-directory build COMMON_FLAGS="$(PREFILL_FLAGS)"

# Shared pybind-module compile. Always rebuilds, so the requested config is never served from a stale
# .so (decode and prefill write the same output file). SCHED_FIX is auto-gated and applied here so it
# survives a COMMON_FLAGS override.
build:
	$(HIPCXX) $(SRC) $(HIPFLAGS) $(COMMON_FLAGS) $(SCHED_FIX) $(PY_FLAGS) $(ICPPFLAGS) \
	    -o $(TARGET)$(shell python3-config --extension-suffix)

clean:
	rm -f $(TARGET)*.so
