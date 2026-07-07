# gfx1250 128x128 GEMM ladder (path to 256x256).
#
#   make KERNEL=gemm_naive
#   make KERNEL=gemm_naive run M=256 N=256 K=256

GPU_TARGET ?= UDNA1
BUILD_MODE ?= standalone
CUSTOM_RULES := 1

LADDER := gemm_naive

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

all: $(BIN)

$(BIN): $(SRC) harness.h common.h
	$(HIPCXX) $(HIPFLAGS) $(ICPPFLAGS) -DHARNESS_MAIN $(SRC) -o $(BIN)

ladder:
	@for k in $(LADDER); do $(MAKE) KERNEL=$$k || exit 1; done

run: $(BIN)
	./$(BIN) $(M) $(N) $(K) $(ITERS) $(VERIFY)

clean:
	rm -f *.out
