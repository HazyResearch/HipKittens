"""gfx1201 -- RDNA4, RX 9070 / W9000-class.

READ THIS AS "WRITTEN, NOT VERIFIED." No gfx12 part was available. The mma
shape is read out of builtin signatures; the register file and LDS sizes are
carried over from gfx11 because nothing contradicted them. The VGPR allocation
granule in particular has NOT been measured here -- on gfx1100 it turned out to
be 24 rather than the 8 everyone assumes, and that was only discovered by
compiling a sweep and diffing against LLVM's occupancy remark.

`verified=False` makes hk-check-budgets report budget violations as warnings rather
than hard errors on this target. A wrong hard limit is worse than no limit.
Before trusting anything here, run tools/rdna-probes on the hardware and the
granule sweep in /kernel-resource-check.
"""

from .base import MmaShape, Target

GFX1201 = Target(
    arch="gfx1201",
    gpu_target="RDNA4",
    kittens_define="KITTENS_RDNA4",

    wave=32,
    max_waves_per_simd=16,

    vgprs_per_simd=1536,
    vgpr_granule=24,  # UNVERIFIED -- carried over from gfx1100, not measured.
    sgprs_per_wave=106,

    lds_per_workgroup=65536,
    lds_pool_bytes=65536,
    simds_per_lds_pool=4,

    mma={
        # gfx12 splits K between the wave halves instead of mirroring: an
        # operand is v8bf16, 4 VGPRs, and lane l holds k = 8*(l/16) .. +7 of
        # row l%16. Half the operand registers of gfx11 for the same tile --
        # that is the point of RDNA4 for this library. It also makes
        # accumulator<->operand conversion *more* expensive, not less, because
        # neither half holds everything and both directions must cross.
        "bf16": MmaShape(m=16, n=16, k=16, operand_vgprs=4, accum_vgprs=8),
        "fp16": MmaShape(m=16, n=16, k=16, operand_vgprs=4, accum_vgprs=8),
        # v_wmma_f32_16x16x16_{fp8,bf8}_{fp8,bf8}_w32_gfx12: a 16x16 fp8
        # operand is 256 B over 32 lanes, unmirrored -- 2 VGPRs. Read out of
        # the builtin signatures in include/rdna4 mma.cuh, not measured.
        "fp8e4m3": MmaShape(m=16, n=16, k=16, operand_vgprs=2, accum_vgprs=8),
        "fp8e5m2": MmaShape(m=16, n=16, k=16, operand_vgprs=2, accum_vgprs=8),
    },

    # Absent on gfx12 as well as gfx11.
    has_global_to_lds_dma=False,

    mma_operands_mirrored=False,

    fast_smem_layouts=("row",),

    verified=False,
)
