GPU_TARGET ?= CDNA4
COMP_LEVEL ?= profile
BUILD_MODE := standalone
CUSTOM_RULES := 1
BUILD_DIR := build

include ../../../common.mk

HIPFLAGS += -fopenmp -Wl,--allow-multiple-definition -Rpass-analysis=kernel-resource-usage

all: 8_wave_TN 8_wave_NN 8_wave_NT

8_wave_TN 8_wave_NN 8_wave_NT: %: %.cu
	mkdir -p $(BUILD_DIR)
	$(HIPCXX) $(HIPFLAGS) $(ICXXFLAGS) $(ICPPFLAGS) -c $< -o $(BUILD_DIR)/$@.o
	$(HIPCXX) $(HIPFLAGS) $(ILDFLAGS) $(BUILD_DIR)/$@.o -o $@

clean:
	rm -rf $(BUILD_DIR) 8_wave_TN 8_wave_NN 8_wave_NT
