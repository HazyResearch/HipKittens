# Build system for the gfx1250 (UDNA1) GEMM optimization ladder.
#
# Every rung builds to one artifact, a pybind11 module named for the rung, which is the only entry
# point it has: `test.py` checks it against torch.matmul and `gemm_ladder.py` times it. Arch flags
# and the Python discovery (PY_INCLUDES, PY_LDFLAGS, PY_EXT_SUFFIX) come from the shared
# kernels/common.mk via GPU_TARGET=UDNA1; the rules are local (CUSTOM_RULES=1) because common.mk
# builds one TARGET and this directory builds thirteen, each with its own module name.
#
# Invocation:
#   make KERNEL=00_gemm_naive            # one module, loaded as `00_gemm_naive`
#   make ladder                          # all rungs (all-kernels is the same target)
#   make test                            # the rungs, then check every one vs torch.matmul
#   make clean
#
# The four cluster kernels (`09_gemm_wgc_multicast` through `12_gemm_two_waves`) use
# static `__cluster_dims__`; their ordinary HIP launches pick up the required 4x4 cluster metadata.
# ROCm 7.2.4 and earlier lack this support and fail these four with a cascade that reads as
# `shared_allocator` errors; ROCm 7.15 (clang 23) supports it. The nine non-cluster
# kernels build on ROCm 7.2+.

GPU_TARGET ?= UDNA1
CUSTOM_RULES := 1

# The one-feature-at-a-time ladder, and every kernel on disk; see README.md.
LADDER := 00_gemm_naive 01_gemm_double_buf 02_gemm_async 03_gemm_128x128 \
          04_gemm_256x256 05_gemm_deepk 06_gemm_segment 07_gemm_tdm \
          08_gemm_split_bar 09_gemm_wgc_multicast 10_gemm_epilogue 11_gemm_one_wave \
          12_gemm_two_waves

# KERNEL names the artifact and SRC the source it is built from, separately, so one source can be
# built under a second name: `make KERNEL=nullctl SRC=11_gemm_one_wave.cpp` is how `gemm_ladder.py`
# builds its null control.
KERNEL ?= 00_gemm_naive
SRC    := $(KERNEL).cpp

# Protocol knobs from harness.h (HK_WARMUP_ITERS, HK_MEASURED_ITERS, HK_FLUSH_BETWEEN_ITERS), so
# one rung can be built under a second protocol and compared against itself:
#   make KERNEL=noflush_11 SRC=11_gemm_one_wave.cpp HK_DEFS="-DHK_FLUSH_BETWEEN_ITERS=0"
HK_DEFS ?=

include ../../../../common.mk

PYEXT := $(KERNEL)$(PY_EXT_SUFFIX)

# Rebuild a rung whenever ANY ThunderKittens header changes: an edit to include/**/*.cuh otherwise
# leaves a stale binary, which is a correctness-of-testing hazard this tree has actually hit.
TK_HEADERS := $(shell find $(THUNDERKITTENS_ROOT)/include -name '*.cuh' 2>/dev/null)
LOCAL_HEADERS := harness.h common.h pyext.h

.PHONY: all ladder all-kernels test clean

all: $(PYEXT)

# The module's name has to match the .so basename or the import fails, so both come from KERNEL.
$(PYEXT): $(SRC) $(LOCAL_HEADERS) $(TK_HEADERS)
	$(HIPCXX) $(HIPFLAGS) $(ICPPFLAGS) $(PY_INCLUDES) $(PY_LDFLAGS) -shared -fPIC \
		-DHARNESS_PYEXT -DHK_PYEXT_NAME=$(KERNEL) $(HK_DEFS) $(SRC) -o $(PYEXT)

# `all-kernels` is kept as an alias because scripts and docs name it.
ladder all-kernels:
	@for k in $(LADDER); do $(MAKE) KERNEL=$$k || exit 1; done

# The correctness gate: there is no CPU reference in the tree and no other way in.
test: ladder
	$(PYTHON) test.py

clean:
	rm -f *$(PY_EXT_SUFFIX)
	rm -rf __pycache__
