GPU_TARGET ?= CDNA4
PYTHON ?= python3
CUSTOM_RULES := 1

EXTRA_HIPFLAGS += -Wl,--allow-multiple-definition -ffast-math -fno-gpu-rdc

include ../../../common.mk

BUILD_FLAGS = $(HIPFLAGS) $(ICXXFLAGS) $(ICPPFLAGS) $(ILDFLAGS) $(ILDLIBS)

.PHONY: all clean
all: tk_running tk_pow2

tk_running: blockwise_running.cu utils.cpp
	$(HIPCXX) blockwise_running.cu $(BUILD_FLAGS) -o tk_running$(PY_EXT_SUFFIX)

tk_pow2: blockwise_pow2.cu utils.cpp
	$(HIPCXX) blockwise_pow2.cu $(BUILD_FLAGS) -o tk_pow2$(PY_EXT_SUFFIX)

clean:
	rm -f tk_running*.so tk_pow2*.so
