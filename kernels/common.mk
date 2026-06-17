# Shared build rules for kernels/**.
# Kernel-local Makefiles should only define local knobs (SRC/TARGET/etc)
# then include this file.

KERNELS_COMMON_MK := $(lastword $(MAKEFILE_LIST))
KERNELS_DIR := $(dir $(abspath $(KERNELS_COMMON_MK)))

THUNDERKITTENS_ROOT ?= $(abspath $(KERNELS_DIR)/..)

ROCM_PATH ?= /opt/rocm
ROCM_INSTALL_DIR ?= $(ROCM_PATH)
HIP_INCLUDE_DIR ?= $(ROCM_INSTALL_DIR)/include/hip
HIPCXX ?= $(ROCM_INSTALL_DIR)/bin/hipcc

GPU_TARGET ?= CDNA4
ifeq ($(GPU_TARGET),CDNA4)
  KITTENS_ARCH_DEFINE := -DKITTENS_CDNA4
  KITTENS_OFFLOAD_ARCH := gfx950
else ifeq ($(GPU_TARGET),UDNA1)
  KITTENS_ARCH_DEFINE := -DKITTENS_UDNA1
  KITTENS_OFFLOAD_ARCH := gfx1250
else
  $(error Unsupported GPU_TARGET '$(GPU_TARGET)'. Supported: CDNA4, UDNA1)
endif

PYTHON ?= python3
BUILD_MODE ?= pyext
TARGET ?= tk_kernel

CXX_STD ?= c++20
COMP_LEVEL ?= profile
KITTENS_WARNING_FLAGS ?= -w

BASE_HIPFLAGS := $(KITTENS_ARCH_DEFINE) --offload-arch=$(KITTENS_OFFLOAD_ARCH)
BASE_HIPFLAGS += -std=$(CXX_STD) $(KITTENS_WARNING_FLAGS)

ifeq ($(COMP_LEVEL),safe)
  OPT_HIPFLAGS := -O0
else ifeq ($(COMP_LEVEL),debug)
  OPT_HIPFLAGS := -g -O0
else ifeq ($(COMP_LEVEL),profile)
  OPT_HIPFLAGS := -O3
else
  OPT_HIPFLAGS := -O3
endif

HIPFLAGS += $(BASE_HIPFLAGS) $(OPT_HIPFLAGS)

ICPPFLAGS += -I$(THUNDERKITTENS_ROOT)/include -I$(HIP_INCLUDE_DIR)
ICPPFLAGS += $(CPPFLAGS) $(EXTRA_CPPFLAGS)

ICXXFLAGS += $(EXTRA_ICXXFLAGS)
ILDFLAGS += $(LDFLAGS) $(EXTRA_LDFLAGS)
ILDLIBS += $(LDLIBS) $(EXTRA_LDLIBS)

PY_LDFLAGS := $(shell $(PYTHON)-config --ldflags | sed 's/-lcrypt//g')
PY_EXT_SUFFIX := $(shell $(PYTHON)-config --extension-suffix)
PY_INCLUDES := $(shell $(PYTHON) -m pybind11 --includes)

ifeq ($(BUILD_MODE),pyext)
  ICXXFLAGS += $(PY_LDFLAGS)
  ICXXFLAGS += -I$(THUNDERKITTENS_ROOT)/include -I$(THUNDERKITTENS_ROOT)/prototype
  ICXXFLAGS += $(PY_INCLUDES) -shared -fPIC
  ICXXFLAGS += -Rpass-analysis=kernel-resource-usage
endif

BUILD_DIR ?= build

.PHONY: all clean

ifneq ($(CUSTOM_RULES),1)
ifeq ($(BUILD_MODE),pyext)
all: $(TARGET)

$(TARGET): $(SRC)
	$(HIPCXX) $(SRC) $(HIPFLAGS) $(ICXXFLAGS) $(ICPPFLAGS) $(ILDFLAGS) $(ILDLIBS) \
		-o $(TARGET)$(PY_EXT_SUFFIX)

clean:
	rm -f $(TARGET) $(TARGET).*so
else ifeq ($(BUILD_MODE),standalone)
OBJ ?= $(BUILD_DIR)/$(notdir $(basename $(SRC))).o

all: $(TARGET)

$(TARGET): $(SRC)
	mkdir -p $(BUILD_DIR)
	$(HIPCXX) $(HIPFLAGS) $(ICXXFLAGS) $(ICPPFLAGS) -c $(SRC) -o $(OBJ)
	$(HIPCXX) $(HIPFLAGS) $(ILDFLAGS) $(ILDLIBS) $(OBJ) -o $(TARGET)

clean:
	rm -rf $(BUILD_DIR) $(TARGET)
else
$(error Unsupported BUILD_MODE '$(BUILD_MODE)'. Expected pyext or standalone)
endif
endif
