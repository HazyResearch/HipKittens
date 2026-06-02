GPU_TARGET=CDNA4
COMP_LEVEL=profile

GPU_RUNTIME := HIP

ROCM_PATH ?= /opt/rocm
ROCM_INSTALL_DIR := $(ROCM_PATH)
HIP_INCLUDE_DIR  := $(ROCM_INSTALL_DIR)/include/hip

HIPCXX ?= $(ROCM_INSTALL_DIR)/bin/hipcc
HIPFLAGS=-std=c++20 -I$(HIP_INCLUDE_DIR) -I../../../../include -Wall -Wextra -Wl,--allow-multiple-definition -DKITTENS_CDNA4 --offload-arch=gfx950 -fopenmp

ifeq ($(COMP_LEVEL),safe)
HIPFLAGS+= -O0
else ifeq ($(COMP_LEVEL),debug)
HIPFLAGS+= -g -O0
else ifeq ($(COMP_LEVEL),profile)
HIPFLAGS+= -O3 -Rpass-analysis=kernel-resource-usage
endif

CXXFLAGS := -w

BUILD_DIR=build2

all: 8_wave_TN 8_wave_NN 8_wave_NT

8_wave_TN: 8_wave_TN.cu
	mkdir -p $(BUILD_DIR)
	$(HIPCXX) $(HIPFLAGS) $(CXXFLAGS) -c $< -o $(BUILD_DIR)/$@.o
	$(HIPCXX) $(HIPFLAGS) $(BUILD_DIR)/$@.o -o $@

8_wave_NN: 8_wave_NN.cu
	mkdir -p $(BUILD_DIR)
	$(HIPCXX) $(HIPFLAGS) $(CXXFLAGS) -c $< -o $(BUILD_DIR)/$@.o
	$(HIPCXX) $(HIPFLAGS) $(BUILD_DIR)/$@.o -o $@

8_wave_NT: 8_wave_NT.cu
	mkdir -p $(BUILD_DIR)
	$(HIPCXX) $(HIPFLAGS) $(CXXFLAGS) -c $< -o $(BUILD_DIR)/$@.o
	$(HIPCXX) $(HIPFLAGS) $(BUILD_DIR)/$@.o -o $@

clean:
	rm -rf $(BUILD_DIR) 8_wave_TN 8_wave_NN 8_wave_NT
