# Build system for the gfx1250 (UDNA1) GEMM optimization ladder.
#
# Arch flags (-DKITTENS_UDNA1 --offload-arch=gfx1250, std, opt level) come from
# the shared kernels/common.mk via GPU_TARGET=UDNA1. This Makefile keeps the
# ladder's per-rung standalone-exe build by defining its own rules
# (CUSTOM_RULES=1).
#
# Invocation:
#   make KERNEL=gemm_naive               # build only `gemm_naive` as a standalone exe
#   make KERNEL=gemm_naive run M=256 N=256 K=256
#   make ladder                          # build every kernel in LADDER
#   make clean
#
# Requires `hipcc` from ROCm 7.2+ (clang 22+). On a host without that, run
# the make inside the `rocm/dev-ubuntu-24.04:7.2` docker image.

GPU_TARGET ?= UDNA1
BUILD_MODE ?= standalone
CUSTOM_RULES := 1

# The ten rungs, listed top down. Each is named for the one hardware or algorithmic feature it
# adds to the rung below it. Anything else that used to be listed here now lives in archive/.
LADDER := gemm_wgc_multicast gemm_split_bar gemm_tdm gemm_segment gemm_deepk \
          gemm_256x256 gemm_128x128 gemm_async gemm_double_buf gemm_naive

KERNEL ?= gemm_naive
SRC    := $(KERNEL).cpp
BIN    := $(KERNEL).out

M ?= 256
N ?= 256
K ?= 256
ITERS  ?= 1
VERIFY ?= 1

include ../../../../common.mk

# Rebuild a rung whenever ANY ThunderKittens header changes -- otherwise an edit
# to include/**/*.cuh silently leaves a stale .out (a correctness-of-testing
# hazard we actually hit). Enumerated once at parse time.
TK_HEADERS := $(shell find $(THUNDERKITTENS_ROOT)/include -name '*.cuh' 2>/dev/null)

.PHONY: ladder run clean all

# Default target: build the currently selected KERNEL as a standalone exe.
all: $(BIN)

# `-fopenmp` is not optional: without it the reference pragma is ignored, the CPU check runs on
# one core, and an 8192^3 verify goes from ~4 s to over 4 minutes. That reads as a hung GPU when
# the GPU finished in milliseconds. Device code is unaffected.
$(BIN): $(SRC) harness.h common.h $(TK_HEADERS)
	$(HIPCXX) $(HIPFLAGS) $(ICPPFLAGS) -fopenmp -DHARNESS_MAIN $(SRC) -o $(BIN)

# Loop variant -- `make ladder` rebuilds every rung.
ladder:
	@for k in $(LADDER); do $(MAKE) KERNEL=$$k || exit 1; done

run: $(BIN)
	./$(BIN) $(M) $(N) $(K) $(ITERS) $(VERIFY)

clean:
	rm -f *.out
