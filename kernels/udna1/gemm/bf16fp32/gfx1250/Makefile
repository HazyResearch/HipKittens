# Build system for the gfx1250 (UDNA1) GEMM optimization ladder.
#
# Every rung builds to one artifact, a pybind11 module named for the rung, which is the only entry
# point it has: `test.py` checks it against torch.matmul and `gemm_ladder.py` times it. Arch flags
# and the Python discovery (PY_INCLUDES, PY_LDFLAGS, PY_EXT_SUFFIX) come from the shared
# kernels/common.mk via GPU_TARGET=UDNA1; the rules are local (CUSTOM_RULES=1) because common.mk
# builds one TARGET and this directory builds thirteen, each with its own module name.
#
# Invocation:
#   make KERNEL=gemm_naive               # one module, `import gemm_naive`
#   make ladder                          # all thirteen rungs (all-kernels is the same target)
#   make test                            # the rungs, then check every one vs torch.matmul
#   make clean
#
# Six cluster kernels need `__cluster_dims__` and `hipLaunchAttributeClusterDimension`.
# ROCm 7.2.4 and earlier have neither and fail these four with a cascade that reads as
# `shared_allocator` errors; ROCm 7.15 (clang 23) supports both. The nine non-cluster
# kernels build on ROCm 7.2+.

GPU_TARGET ?= UDNA1
CUSTOM_RULES := 1

# The twelve rungs, listed top down. Each is named for the one hardware or algorithmic feature it
# adds to the rung below it, and the list is every kernel on disk: the ladder is a single chain that
# ends at its fastest kernel. See README.md.
# A0-safe cluster path (top); multicast rungs below are A1+ only (hang on gfx1250 A0).
LADDER := gemm_one_wave_nomc gemm_epilogue_nomc gemm_wgc_cluster \
          gemm_one_wave gemm_epilogue gemm_wgc_multicast gemm_split_bar \
          gemm_tdm gemm_segment gemm_deepk gemm_256x256 gemm_128x128 gemm_async \
          gemm_double_buf gemm_naive

# KERNEL names the artifact and SRC the source it is built from, separately, so one source can be
# built under a second name: `make KERNEL=nullctl SRC=gemm_epilogue.cpp` is how `gemm_ladder.py`
# builds its null control.
KERNEL ?= gemm_naive
SRC    := $(KERNEL).cpp

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
		-DHARNESS_PYEXT -DHK_PYEXT_NAME=$(KERNEL) $(SRC) -o $(PYEXT)

# Everything on disk, so a toolchain or header change is checked against all 15 kernels.
# `all-kernels` is kept as an alias because scripts and docs name it.
ladder all-kernels:
	@for k in $(LADDER); do $(MAKE) KERNEL=$$k || exit 1; done

# The correctness gate: there is no CPU reference in the tree and no other way in.
test: ladder
	$(PYTHON) test.py

clean:
	rm -f *$(PY_EXT_SUFFIX)
	rm -rf __pycache__
