# Compiler
GPU_TARGET=CDNA3

SRC=kernel_tiled.cpp kernel_muhammad.cpp kernel_baseline.cpp 
SRC_EXE=check_tiling.cpp

# HIP variables
ROCM_INSTALL_DIR := $(ROCM_PATH)
HIP_INCLUDE_DIR  := $(ROCM_INSTALL_DIR)/include/hip

HIPCXX ?= $(ROCM_INSTALL_DIR)/bin/hipcc

# Compiler flags based on GPU target
ifeq ($(GPU_TARGET),CDNA2)
HIPFLAGS+= -DKITTENS_CDNA2 --offload-arch=gfx90a
else ifeq ($(GPU_TARGET),CDNA3)
HIPFLAGS+= -DKITTENS_CDNA3 --offload-arch=gfx942
endif

# Common variables and flags
CXX_STD   := c++20
ICXXFLAGS := -std=$(CXX_STD)
ICPPFLAGS := -I${THUNDERKITTENS_ROOT}/include -I$(HIP_INCLUDE_DIR)
ILDFLAGS  :=
ILDLIBS   :=

CXXFLAGS ?= -Wall -Wextra
CXXFLAGS := -w

ICXXFLAGS += $(CXXFLAGS)
ICPPFLAGS += $(CPPFLAGS)
ILDFLAGS  += $(LDFLAGS)
ILDLIBS   += $(LDLIBS)

ICXXFLAGS+= -I${THUNDERKITTENS_ROOT}/include -I${THUNDERKITTENS_ROOT}/prototype $(shell python3 -m pybind11 --includes) $(shell python3-config --ldflags) -shared -fPIC -Rpass-analysis=kernel-resource-usage

EXT := $(shell python3-config --extension-suffix)
TARGETS_EXE := $(patsubst %.cpp,%,$(SRC_EXE))
TARGETS := $(patsubst %.cpp,%$(EXT),$(SRC)) $(TARGETS_EXE)

# Default target
all: $(TARGETS)

LOGDIR := /scratch/users/ssiddens/data_logs/$(shell date +%m%d_%H%M)_outputs
LOGFILE := $(LOGDIR)/make_build.log

%$(EXT): %.cpp
	@mkdir -p $(LOGDIR)
	@echo "Building $< -> $@" | tee -a $(LOGFILE)
	$(HIPCXX) $< $(HIPFLAGS) $(ICXXFLAGS) $(ICPPFLAGS) $(ILDFLAGS) -o $@ 2>&1 | tee -a $(LOGFILE)


# Rule for standalone executable
%: %.cpp
	$(HIPCXX) $< $(HIPFLAGS) $(ICPPFLAGS) $(ILDFLAGS) -std=c++20 -Wall -Wextra -w -I${THUNDERKITTENS_ROOT}/include -I${THUNDERKITTENS_ROOT}/prototype $(shell python3 -m pybind11 --includes) $(shell python3-config --ldflags) -fPIC -Rpass-analysis=kernel-resource-usage -o $@ 


# Clean target
clean:
	rm -f $(TARGETS) *.so
