# Compiler
GPU_TARGET=CDNA3

TARGET=matmul_benchmark
SRC=kernel.cpp

# HIP variables
ROCM_INSTALL_DIR := $(ROCM_PATH)
HIP_INCLUDE_DIR  := $(ROCM_INSTALL_DIR)/include/hip

HIPCXX ?= $(ROCM_INSTALL_DIR)/bin/hipcc

# Compiler flags based on GPU target
ifeq ($(GPU_TARGET),CDNA2)
HIPFLAGS+= -DKITTENS_CDNA2 --offload-arch=gfx90a -mcpu=gfx90a
else ifeq ($(GPU_TARGET),CDNA3)
HIPFLAGS+= -DKITTENS_CDNA3 --offload-arch=gfx942 -mcpu=gfx942
endif

# Performance flags
PERF_FLAGS := -O3 -DNDEBUG -ffast-math -funroll-loops -fvectorize
PERF_FLAGS += -Rpass-analysis=kernel-resource-usage -Rpass=loop-vectorize

# Add performance flags
HIPFLAGS += $(PERF_FLAGS)


# Common variables and flags
CXX_STD   := c++20
ICXXFLAGS := -std=$(CXX_STD)
ICPPFLAGS := -I${THUNDERKITTENS_ROOT}/include -I$(HIP_INCLUDE_DIR)
ILDFLAGS  :=
ILDLIBS   :=

CXXFLAGS ?= -Wall -Wextra
CXXFLAGS := -w

ICXXFLAGS += $(CXXFLAGS) --save-temps
ICPPFLAGS += $(CPPFLAGS)
ILDFLAGS  += $(LDFLAGS)
ILDLIBS   += $(LDLIBS)

ICXXFLAGS+= -I${THUNDERKITTENS_ROOT}/include -I${THUNDERKITTENS_ROOT}/prototype -Rpass-analysis=kernel-resource-usage -Xclang -Rpass=kernel-resource-usage

# Add OpenMP linking for CPU benchmark
ILDLIBS += -fopenmp

# Add rocBLAS library
ILDLIBS += -lrocblas

# Default target
all: $(TARGET)

LOGDIR := /shared/amdgpu/home/tech_ops_amd_xqh/simran/data_logs/$(shell date +%m%d_%H%M)_outputs
LOGFILE := $(LOGDIR)/make_build.log

$(TARGET): $(SRC)
	@mkdir -p $(LOGDIR)
	$(HIPCXX) $(SRC) $(HIPFLAGS) $(ICXXFLAGS) $(ICPPFLAGS) $(ILDFLAGS) $(ILDLIBS) \
	    -o $(TARGET) 2>&1 | tee $(LOGFILE)

# Clean target
clean:
	rm -f $(TARGET)
