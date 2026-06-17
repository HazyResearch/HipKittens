GPU_TARGET ?= CDNA4
TARGET ?= tk_kernel_bkwd
SRC ?= GQA_bkwd_8warps.cpp
PYTHON ?= python3

EXTRA_HIPFLAGS += -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math

include ../../../common.mk
