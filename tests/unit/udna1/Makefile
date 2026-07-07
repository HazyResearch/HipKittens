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
#   - global_to_shared : register-mediated `load`, async `load_async`, and
#     hardware `load_tdm` (bf16/half + st_16x32_padded only), verified by
#     `store_tdm` on a dense ST-sized GL (row_stride == tile width).
#   - shared_to_register : GEMM-chain padded s2r (g2s -> WMMA load(rt,st,origin)
#     on st_16x32_padded, verified via g2r store) plus XOR-swizzle st_16x16 path.
#   - register/tile/mma : bf16 WMMA `mma_ABt` (rt_16x16) plus MFMA transpose
#     variants on rt_16x16 and rt_32x32.
TEST_DEFINES   ?= -DTEST_WARP_MEMORY_TILE_GLOBAL_TO_REGISTER \
                  -DTEST_WARP_MEMORY_TILE_GLOBAL_TO_SHARED \
                  -DTEST_WARP_MEMORY_TILE_SHARED_TO_REGISTER \
                  -DTEST_WARP_REGISTER_TILE_MMA

include ../common/tests.mk
