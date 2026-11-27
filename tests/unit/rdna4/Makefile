# rdna4 (gfx1201, Radeon RX 9070 / R9700) unit tests.
# Shared rules live in ../common/tests.mk; this file only sets local knobs.
#
# rdna4 has no group/ tests: its group ops are store-only, and cdna5 has no
# group tests either, so the default below is -DTEST_ALL_WARP.

GPU_TARGET     ?= RDNA4
COMP_LEVEL     ?= profile
TEST_INTENSITY ?= 2

# Which tests to run? -DTEST_ALL runs everything; or pick subsections, e.g.
# -DTEST_WARP_MEMORY, or individual tests like -DTEST_WARP_MEMORY_TILE_SHARED_TO_REGISTER.
TEST_DEFINES   ?= -DTEST_ALL_WARP

include ../common/tests.mk
