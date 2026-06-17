GPU_TARGET ?= CDNA4
PYTHON ?= python3
CUSTOM_RULES := 1

ATTN_B ?= 16
ATTN_H ?= 64
ATTN_H_KV ?= 8
ATTN_N ?= 8192

CPPFLAGS += -DATTN_B=$(ATTN_B)
CPPFLAGS += -DATTN_H=$(ATTN_H)
CPPFLAGS += -DATTN_H_KV=$(ATTN_H_KV)
CPPFLAGS += -DATTN_N=$(ATTN_N)

EXTRA_HIPFLAGS += -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math

include ../../common.mk

.PHONY: all clean

all: tk_kernel_bkwd tk_kernel_bkwd_prep tk_kernel_fwd

tk_kernel_bkwd: attn_bkwd_non_causal.cpp
	$(HIPCXX) $< $(HIPFLAGS) $(ICXXFLAGS) $(ICPPFLAGS) $(ILDFLAGS) $(ILDLIBS) \
		-o $@$(PY_EXT_SUFFIX)

tk_kernel_bkwd_prep: attn_bkwd_prep.cpp
	$(HIPCXX) $< $(HIPFLAGS) $(ICXXFLAGS) $(ICPPFLAGS) $(ILDFLAGS) $(ILDLIBS) \
		-o $@$(PY_EXT_SUFFIX)

tk_kernel_fwd: attn_fwd_non_causal.cpp
	$(HIPCXX) $< $(HIPFLAGS) $(ICXXFLAGS) $(ICPPFLAGS) $(ILDFLAGS) $(ILDLIBS) \
		-o $@$(PY_EXT_SUFFIX)

clean:
	rm -f tk_kernel_bkwd* tk_kernel_bkwd_prep* tk_kernel_fwd*

