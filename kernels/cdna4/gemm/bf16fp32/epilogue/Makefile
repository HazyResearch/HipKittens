# Build one epilogue binding as a Python extension: make KERNEL=<noop|silu|...>
GPU_TARGET ?= CDNA4
KERNEL     ?= noop
TARGET     ?= tk_$(KERNEL)
SRC        := bindings/gemm_$(KERNEL).cpp
PYTHON     ?= python3

EXTRA_CPPFLAGS += -I. -Iops -Iepilogues
EXTRA_HIPFLAGS += -DTK_MODULE_NAME=$(TARGET)

include ../../../../common.mk
