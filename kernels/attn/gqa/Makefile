# CDNA3 Flash Attention Forward (Non-Causal) Makefile
GPU_TARGET=CDNA3

# ThunderKittens root (override with env)
THUNDERKITTENS_ROOT ?= $(abspath ../../../)

# Compile-time attention shape parameters
ATTN_B     ?= 4
ATTN_H     ?= 32
ATTN_H_KV  ?= 8
ATTN_N     ?= 1024

# HIP variables
ROCM_INSTALL_DIR := $(ROCM_PATH)
HIP_INCLUDE_DIR  := $(ROCM_INSTALL_DIR)/include/hip
ROCM_INCLUDE_DIR := $(ROCM_INSTALL_DIR)/include

HIPCXX ?= $(ROCM_INSTALL_DIR)/bin/hipcc

# Compiler flags based on GPU target
ifeq ($(GPU_TARGET),CDNA3)
HIPFLAGS += -DKITTENS_CDNA3 --offload-arch=gfx942
endif

# Common flags
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

# Pybind11
ICXXFLAGS += $(shell python3 -m pybind11 --includes) $(shell python3-config --ldflags) -shared -fPIC
ICXXFLAGS += -Rpass-analysis=kernel-resource-usage

# Shape defines
SHAPE_FLAGS = -DATTN_B=$(ATTN_B) -DATTN_H=$(ATTN_H) -DATTN_H_KV=$(ATTN_H_KV) -DATTN_N=$(ATTN_N)

PYEXT := $(shell python3-config --extension-suffix)

# Default target: build both D=128 and D=64
all: attn_fwd attn_fwd_d64

attn_fwd: kernel.cpp
	$(HIPCXX) kernel.cpp $(HIPFLAGS) $(SHAPE_FLAGS) $(ICXXFLAGS) $(ICPPFLAGS) $(ILDFLAGS) \
	    -o attn_fwd$(PYEXT) 2>&1

attn_fwd_d64: kernel_d64.cpp
	$(HIPCXX) kernel_d64.cpp $(HIPFLAGS) $(SHAPE_FLAGS) $(ICXXFLAGS) $(ICPPFLAGS) $(ILDFLAGS) \
	    -o attn_fwd_d64$(PYEXT) 2>&1

clean:
	rm -f attn_fwd$(PYEXT) attn_fwd_d64$(PYEXT)

.PHONY: all clean
