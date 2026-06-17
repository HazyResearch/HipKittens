GPU_TARGET ?= CDNA4
TARGET ?= tk_kernel
SRC ?= kernel.cpp
PYTHON ?= python3

EXTRA_HIPFLAGS += -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math -I$(ROCM_INSTALL_DIR)/include/rocrand

include ../common.mk
