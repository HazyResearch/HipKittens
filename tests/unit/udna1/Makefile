# udna1 (gfx1250, MI400/MI450) unit tests.
# Shared rules live in ../common/tests.mk; this file only sets local knobs.

GPU_TARGET     ?= UDNA1
COMP_LEVEL     ?= profile
TEST_INTENSITY ?= 2

# Which tests to run? -DTEST_ALL runs everything; or pick subsections, e.g.
# -DTEST_ALL_WARP_MEMORY_TILE (all tile memory tests), or individual tests like
# -DTEST_WARP_MEMORY_TILE_GLOBAL_TO_REGISTER.
#
# NOTE: the gfx1250 toolchain (ROCm 7.2) cannot codegen the CDNA
# `buffer_load_lds` global->shared path, so that path was removed. The tests
# below exercise the gfx1250-native paths instead:
#   - global_to_shared : the register-mediated `load(st, gl, idx, row_stride)`,
#     verified by reading the tile back through its own `lds_offset()` map.
#   - shared_to_register : the XOR-swizzle `load(rt,st)`/`store(st,rt)` ops,
#     with the shared tile filled by a per-element global->LDS copy.
TEST_DEFINES   ?= -DTEST_WARP_MEMORY_TILE_GLOBAL_TO_REGISTER \
                  -DTEST_WARP_MEMORY_TILE_GLOBAL_TO_SHARED \
                  -DTEST_WARP_MEMORY_TILE_SHARED_TO_REGISTER

include ../common/tests.mk
