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

LADDER := gemm_naive gemm_double_buf gemm_async gemm_split_bar gemm_segment gemm_expert gemm_tdm_arrive

KERNEL ?= gemm_naive
SRC    := $(KERNEL).cpp
BIN    := $(KERNEL).out

M ?= 256
N ?= 256
K ?= 256
ITERS  ?= 1
VERIFY ?= 1

include ../../../common.mk

.PHONY: ladder run clean all

# Default target: build the currently selected KERNEL as a standalone exe.
all: $(BIN)

$(BIN): $(SRC) harness.h common.h
	$(HIPCXX) $(HIPFLAGS) $(ICPPFLAGS) -DHARNESS_MAIN $(SRC) -o $(BIN)

# Loop variant -- `make ladder` rebuilds every rung.
ladder:
	@for k in $(LADDER); do $(MAKE) KERNEL=$$k || exit 1; done

run: $(BIN)
	./$(BIN) $(M) $(N) $(K) $(ITERS) $(VERIFY)

clean:
	rm -f *.out
