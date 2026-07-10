GPU_TARGET ?= CDNA4
COMP_LEVEL ?= profile
BUILD_MODE := standalone
TARGET ?= attn_kernel
SRC ?= attn_kernel.cpp

EXTRA_HIPFLAGS += -fopenmp -Wl,--allow-multiple-definition
EXTRA_HIPFLAGS += -Rpass-analysis=kernel-resource-usage --save-temps

include ../../../common.mk