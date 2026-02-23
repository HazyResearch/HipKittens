# CDNA3 Flash Attention Forward (Causal) Makefile
GPU_TARGET=CDNA3

THUNDERKITTENS_ROOT ?= $(abspath ../../../)

ATTN_B     ?= 4
ATTN_H     ?= 32
ATTN_H_KV  ?= 8
ATTN_N     ?= 1024

ROCM_INSTALL_DIR := $(ROCM_PATH)
HIP_INCLUDE_DIR  := $(ROCM_INSTALL_DIR)/include/hip
ROCM_INCLUDE_DIR := $(ROCM_INSTALL_DIR)/include

HIPCXX ?= $(ROCM_INSTALL_DIR)/bin/hipcc

ifeq ($(GPU_TARGET),CDNA3)
HIPFLAGS += -DKITTENS_CDNA3 --offload-arch=gfx942
endif

CXX_STD   := c++20
ICXXFLAGS := -std=$(CXX_STD)
ICPPFLAGS := -I${THUNDERKITTENS_ROOT}/include -I$(ROCM_INCLUDE_DIR) -I$(HIP_INCLUDE_DIR)
ILDFLAGS  :=
ILDLIBS   :=

CXXFLAGS := -w
ICXXFLAGS += $(CXXFLAGS)
ICPPFLAGS += $(CPPFLAGS)
ILDFLAGS  += $(LDFLAGS)
ILDLIBS   += $(LDLIBS)

ICXXFLAGS += $(shell python3 -m pybind11 --includes) $(shell python3-config --ldflags) -shared -fPIC
ICXXFLAGS += -Rpass-analysis=kernel-resource-usage

SHAPE_FLAGS = -DATTN_B=$(ATTN_B) -DATTN_H=$(ATTN_H) -DATTN_H_KV=$(ATTN_H_KV) -DATTN_N=$(ATTN_N)

PYEXT := $(shell python3-config --extension-suffix)

all: attn_causal_fwd attn_causal_fwd_d64

attn_causal_fwd: kernel.cpp
	$(HIPCXX) kernel.cpp $(HIPFLAGS) $(SHAPE_FLAGS) $(ICXXFLAGS) $(ICPPFLAGS) $(ILDFLAGS) \
	    -o attn_causal_fwd$(PYEXT) 2>&1

attn_causal_fwd_d64: kernel_d64.cpp
	$(HIPCXX) kernel_d64.cpp $(HIPFLAGS) $(SHAPE_FLAGS) $(ICXXFLAGS) $(ICPPFLAGS) $(ILDFLAGS) \
	    -o attn_causal_fwd_d64$(PYEXT) 2>&1

clean:
	rm -f attn_causal_fwd$(PYEXT) attn_causal_fwd_d64$(PYEXT)

.PHONY: all clean
