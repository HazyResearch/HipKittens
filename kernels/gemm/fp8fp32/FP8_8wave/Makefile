GPU_TARGET ?= CDNA4
COMP_LEVEL ?= profile
BUILD_MODE := standalone
TARGET ?= tk_kernel
SRC ?= 8_wave.cu

EXTRA_HIPFLAGS += -fopenmp -Wl,--allow-multiple-definition
EXTRA_HIPFLAGS += -Rpass-analysis=kernel-resource-usage

include ../../../common.mk
