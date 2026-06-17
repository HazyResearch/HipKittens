# Shared build rules for tests/unit/<arch>.
#
# Each arch Makefile (cdna4/, udna1/) sets a few knobs then includes this:
#   GPU_TARGET    : CDNA4 | UDNA1            (required)
#   TEST_DEFINES  : -DTEST_... selection      (required)
#   TEST_INTENSITY: 1..4                      (optional, default 2)
#   COMP_LEVEL    : safe | debug | profile    (optional, default profile)
#
# The arch-agnostic harness (unit_tests.cu + testing_commons/) lives here in
# ../common and is compiled into build/common/. The arch-specific warp/group
# test trees are discovered locally with `find` and compiled into build/.

# NOTE: WE HIGHLY RECOMMEND RUNNING WITH `make -j32` (or however many threads
# your machine has). Single-threaded compilation of the full sweep is slow.

COMMON_DIR := ../common
TK_INCLUDE := ../../../include

# HIP toolchain
ROCM_INSTALL_DIR := $(ROCM_PATH)
HIP_INCLUDE_DIR  := $(ROCM_INSTALL_DIR)/include/hip
HIPCXX ?= $(ROCM_INSTALL_DIR)/bin/hipcc

COMP_LEVEL     ?= profile
TEST_INTENSITY ?= 2

ifndef GPU_TARGET
$(error GPU_TARGET is not set. Expected CDNA4 or UDNA1)
endif
ifndef TEST_DEFINES
$(error TEST_DEFINES is not set, e.g. -DTEST_WARP_MEMORY_TILE_GLOBAL_TO_REGISTER)
endif

# Include paths:
#   -I.                          so common/unit_tests.cu can find warp/warp.cuh in the arch dir
#   -I$(COMMON_DIR)/testing_commons   for testing_flags/commons/utils headers
#   -I$(TK_INCLUDE)              ThunderKittens core headers
HIPFLAGS := -std=c++20 -I$(HIP_INCLUDE_DIR) -I. -I$(COMMON_DIR)/testing_commons -I$(TK_INCLUDE)
HIPFLAGS += -Wall -Wextra -Wl,--allow-multiple-definition
HIPFLAGS += -DTEST_INTENSITY=$(TEST_INTENSITY)
HIPFLAGS += $(TEST_DEFINES)

ifeq ($(COMP_LEVEL),safe)
HIPFLAGS += -O0
else ifeq ($(COMP_LEVEL),debug)
HIPFLAGS += -g
else ifeq ($(COMP_LEVEL),profile)
HIPFLAGS += -O3
endif

ifeq ($(GPU_TARGET),CDNA4)
HIPFLAGS += -DKITTENS_CDNA4 --offload-arch=gfx950 -DHIP_ENABLE_WARP_SYNC_BUILTINS
else ifeq ($(GPU_TARGET),UDNA1)
HIPFLAGS += -DKITTENS_UDNA1 --offload-arch=gfx1250
else
$(error Unsupported GPU_TARGET '$(GPU_TARGET)'. Supported: CDNA4, UDNA1)
endif

# Suppress warnings
HIPFLAGS += -w
HIPFLAGS += -Wno-pass-failed

TARGET    := unit_tests
BUILD_DIR := build

# Arch-local test sources (the warp/group trees in this directory).
LOCAL_SRC  := $(shell find . -name '*.cu')
# Shared sources compiled out of ../common.
COMMON_SRC := unit_tests.cu testing_commons/testing_utils.cu

LOCAL_OBJS  := $(patsubst ./%.cu,$(BUILD_DIR)/%.o,$(LOCAL_SRC))
COMMON_OBJS := $(patsubst %.cu,$(BUILD_DIR)/common/%.o,$(COMMON_SRC))

OBJS := $(LOCAL_OBJS) $(COMMON_OBJS)

.PHONY: all run clean
all: $(TARGET)

# Shared objects from ../common.
$(BUILD_DIR)/common/%.o: $(COMMON_DIR)/%.cu
	mkdir -p $(@D)
	$(HIPCXX) $(HIPFLAGS) -c $< -o $@

# Arch-local objects.
$(BUILD_DIR)/%.o: %.cu
	mkdir -p $(@D)
	$(HIPCXX) $(HIPFLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	$(HIPCXX) $(HIPFLAGS) $^ -o $(TARGET)

run: all
	./$(TARGET)

clean:
	rm -rf $(BUILD_DIR) $(TARGET)
