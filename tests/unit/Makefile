# NOTE: WE HIGHLY RECOMMEND RUNNING WITH `make -j32` or however many threads your machine has.
# If you want to run lots of tests without parallel compilation, be prepared
# to spend a *long* time compiling. (like, HOURS single-threaded.)

# GPU Selection: Set to MI300X
GPU_TARGET=CDNA3
COMP_LEVEL=profile

GPU_RUNTIME := HIP

# HIP variables
ROCM_INSTALL_DIR := $(ROCM_PATH)
HIP_INCLUDE_DIR  := $(ROCM_INSTALL_DIR)/include/hip

HIPCXX ?= $(ROCM_INSTALL_DIR)/bin/hipcc
HIPFLAGS=-std=c++20 -I$(HIP_INCLUDE_DIR) -Itesting_commons -I../../include -Wall -Wextra -Wl,--allow-multiple-definition

# At what level to run tests?
# Level 1 takes 15 seconds to compile 1,000 tests.
# (Default) Level 2 takes a minute to compile 3,000 tests.
# Level 3 takes 5 minutes to compile 10,000 tests.
# Level 4 takes 15 while to compile 25,000 tests
HIPFLAGS+= -DTEST_INTENSITY=2
# Which tests to run?
# -DTEST_ALL means run all tests.
# You can also specify subsections, e.g. -DTEST_WARP_MEMORY
# Or individual tests, like -DTEST_WARP_MEMORY_VEC_DSMEM. Useful for debugging!
# -DTEST_WARP_MEMORY_TILE_SHARED_TO_REGISTER -DTEST_WARP_MEMORY_TILE_GLOBAL_TO_SHARED -DTEST_WARP_MEMORY_TILE_GLOBAL_TO_REGISTER -DTEST_WARP_REGISTER_TILE_MAPS -DTEST_WARP_REGISTER_TILE_MMA
HIPFLAGS+= -DTEST_ALL_WARP

ifeq ($(COMP_LEVEL),safe)
HIPFLAGS+= -O0
else ifeq ($(COMP_LEVEL),debug)
HIPFLAGS+= -g -G
else ifeq ($(COMP_LEVEL),profile)
HIPFLAGS+= -O3 -save-temps=obj
endif

# Compiler flags based on GPU target
ifeq ($(GPU_TARGET),CDNA2)
HIPFLAGS+= -DKITTENS_CDNA2 --offload-arch=gfx90a
else ifeq ($(GPU_TARGET),CDNA3)
HIPFLAGS+= -DKITTENS_CDNA3 --offload-arch=gfx942
endif

# Target binary name
TARGET=unit_tests
BUILD_DIR=build

# Test source files
TESTS_SRC=$(shell find . -name '*.cu')

# Object files
OBJS=$(patsubst %.cu,$(BUILD_DIR)/%.o,$(TESTS_SRC))

# Default target
all: build $(TARGET)

# Create the build directory
build:
	mkdir -p $(BUILD_DIR)

# Rule to compile each CU file
$(BUILD_DIR)/%.o: %.cu
	mkdir -p $(@D)
	$(HIPCXX) $(HIPFLAGS) -c $< -o $@

# Link object files to create the final executable
$(TARGET): $(OBJS)
	$(HIPCXX) $(HIPFLAGS) $^ -o $(TARGET)

# Run target
run: all
	./$(TARGET)

# Clean target
clean:
	rm -rf $(BUILD_DIR) $(TARGET)