GPU_TARGET ?= CDNA4
TARGET ?= tk_kernel
SRC ?= 192_256_64_32_with16x32.cpp
PYTHON ?= python3

EXTRA_HIPFLAGS += -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math

include ../../../../../common.mk
