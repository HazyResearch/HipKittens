# gfx1250 256x256 GEMM — MI350X 8-wave / 128x64-slice layout.
#
#   make KERNEL=gemm_sched run M=512 N=512 K=512

GPU_TARGET ?= UDNA1
BUILD_MODE ?= standalone
CUSTOM_RULES := 1

LADDER := gemm_sched

KERNEL ?= gemm_sched
SRC    := $(KERNEL).cpp
BIN    := $(KERNEL).out

M ?= 512
N ?= 512
K ?= 512
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
