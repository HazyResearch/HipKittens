# cdna4 (gfx950, MI350) unit tests.
# Shared rules live in ../common/tests.mk; this file only sets local knobs.

GPU_TARGET     ?= CDNA4
COMP_LEVEL     ?= profile
TEST_INTENSITY ?= 2

# Which tests to run? -DTEST_ALL runs everything; or pick subsections, e.g.
# -DTEST_WARP_MEMORY, or individual tests like -DTEST_WARP_MEMORY_TILE_SHARED_TO_REGISTER.
TEST_DEFINES   ?= -DTEST_WARP_MEMORY_TILE_SHARED_TO_REGISTER

include ../common/tests.mk
