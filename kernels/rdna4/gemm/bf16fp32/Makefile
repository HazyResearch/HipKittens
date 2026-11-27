# gfx1201 (RDNA4) BF16 GEMM kernels.
#
# Each kernel's file name states its configuration (block M, N, K step, k-substep).
#
#   make                 # build the default kernel (KERNEL below)
#   make KERNEL=128_192_64_64_pingpong
#   make run             # build, then verify + time at 4096^3
#   make bench           # the shape sweep the README quotes
#   make clean
#   make EXTRA_HIPFLAGS=-DGROUP_M=16   # wider L2 rasterisation groups, for power-of-two K

GPU_TARGET ?= RDNA4
BUILD_MODE ?= standalone
CUSTOM_RULES := 1

KERNEL ?= 128_512_64_16_ws8p16c
SRC    := $(KERNEL).cpp
BIN    := $(KERNEL)

M ?= 4096
N ?= 4096
K ?= 4096
ITERS  ?= 30
VERIFY ?= 1

include ../../../common.mk

.PHONY: all run bench clean

all: $(BIN)

$(BIN): $(SRC) common.h harness.h
	$(HIPCXX) $(HIPFLAGS) $(ICPPFLAGS) -DHARNESS_MAIN $(SRC) -o $(BIN)

# A short run measures the clock ramp, not the kernel -- see README.
run: $(BIN)
	rocm-smi --setperflevel high > /dev/null 2>&1 || true
	./$(BIN) $(M) $(N) $(K) $(ITERS) $(VERIFY)
	rocm-smi --setperflevel auto > /dev/null 2>&1 || true

bench: $(BIN)
	rocm-smi --setperflevel high > /dev/null 2>&1 || true
	@for s in 4096 6144 8192 12288; do \
	  printf '%6s^3  ' $$s; ./$(BIN) $$s $$s $$s 20 0 | grep -o '[0-9.]* GFLOP/s'; \
	done
	rocm-smi --setperflevel auto > /dev/null 2>&1 || true

clean:
	rm -f $(KERNEL) 128_512_64_16_ws8p16c 128_192_64_64_pingpong *.o
