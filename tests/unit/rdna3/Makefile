# rdna3 (gfx1100, RX 7900 / W7900) unit tests.
# Shared rules live in ../common/tests.mk; this file only sets local knobs.

GPU_TARGET     ?= RDNA3
COMP_LEVEL     ?= profile
TEST_INTENSITY ?= 2

# Which tests to run? -DTEST_ALL runs everything; or pick subsections, e.g.
# -DTEST_WARP_MEMORY, or individual tests like -DTEST_WARP_MEMORY_TILE_SHARED_TO_REGISTER.
TEST_DEFINES   ?= -DTEST_ALL

include ../common/tests.mk
