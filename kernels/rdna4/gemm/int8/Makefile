# gfx1201 (RDNA4) INT8 GEMM: C (int32) = A (int8, MxK) * B^T (int8, NxK).
#
#   make                 # build gemm_i8 and the primitive test mma_i8_test
#   make test            # exact primitive test, then an exact GEMM gate at 2048^3
#   make run             # verify + time at M x N x K (defaults 4096^3)
#   make bench           # time the shapes the README quotes
#   make clean

GPU_TARGET ?= RDNA4
BUILD_MODE ?= standalone
CUSTOM_RULES := 1

M ?= 4096
N ?= 4096
K ?= 4096
ITERS  ?= 50
VERIFY ?= 1

include ../../../common.mk

.PHONY: all test run bench clean

all: gemm_i8 mma_i8_test

gemm_i8: gemm_i8.cpp gemm_i8_kernel.h
	$(HIPCXX) $(HIPFLAGS) $(ICPPFLAGS) gemm_i8.cpp -o $@

mma_i8_test: mma_i8_test.cpp
	$(HIPCXX) $(HIPFLAGS) $(ICPPFLAGS) mma_i8_test.cpp -o $@

test: all
	./mma_i8_test
	./gemm_i8 2048 2048 2048 5 1

# A short run measures the clock ramp, not the kernel: pin clocks while timing.
run: gemm_i8
	rocm-smi --setperflevel high > /dev/null 2>&1 || true
	./gemm_i8 $(M) $(N) $(K) $(ITERS) $(VERIFY)
	rocm-smi --setperflevel auto > /dev/null 2>&1 || true

bench: gemm_i8
	rocm-smi --setperflevel high > /dev/null 2>&1 || true
	@for s in 2048 4096 6144 8192 12288; do ./gemm_i8 $$s $$s $$s $(ITERS) 0 | grep TOPS; done
	rocm-smi --setperflevel auto > /dev/null 2>&1 || true

clean:
	rm -f gemm_i8 mma_i8_test
