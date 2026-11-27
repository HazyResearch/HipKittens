# gfx1201 (RDNA4) attention forward: bf16, head dim 64, non-causal, GQA.
# Problem size is compile-time (ATTN_B, ATTN_H, ATTN_H_KV, ATTN_N); the file's
# defaults are B=16, H=64, H_KV=8, N=1024.
#
#   make                 # build fwd/kernel_d64.cpp -> attn_fwd
#   make test            # run all seven correctness modes against a CPU reference
#   make run             # time it (ITERS launches, clocks pinned)
#   make clean
#   make EXTRA_HIPFLAGS="-DATTN_N=4096"   # a different sequence length

GPU_TARGET ?= RDNA4
BUILD_MODE ?= standalone
CUSTOM_RULES := 1

ITERS ?= 50

include ../../common.mk

.PHONY: all test run clean

all: attn_fwd

attn_fwd: fwd/kernel_d64.cpp
	$(HIPCXX) $(HIPFLAGS) $(ICPPFLAGS) -DHARNESS_MAIN fwd/kernel_d64.cpp -o $@

test: attn_fwd
	@for m in 0 1 2 3 4 5 6; do echo "mode $$m"; ./attn_fwd 0 1 $$m || exit 1; done

run: attn_fwd
	rocm-smi --setperflevel high > /dev/null 2>&1 || true
	./attn_fwd $(ITERS) 0
	rocm-smi --setperflevel auto > /dev/null 2>&1 || true

clean:
	rm -f attn_fwd
