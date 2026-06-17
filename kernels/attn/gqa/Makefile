GPU_TARGET ?= CDNA4
TARGET ?= tk_kernel
PYTHON ?= python3.12

ATTN_B ?= 16
ATTN_H ?= 64
ATTN_H_KV ?= 8
ATTN_N ?= 2048
ATTN_D ?= 128

CPPFLAGS += -DATTN_B=$(ATTN_B)
CPPFLAGS += -DATTN_H=$(ATTN_H)
CPPFLAGS += -DATTN_H_KV=$(ATTN_H_KV)
CPPFLAGS += -DATTN_N=$(ATTN_N)

ifeq ($(ATTN_D),64)
SRC ?= kernel_d64.cpp
else
SRC ?= kernel.cpp
endif

EXTRA_HIPFLAGS += -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math
EXTRA_ICXXFLAGS += --save-temps

include ../../common.mk
