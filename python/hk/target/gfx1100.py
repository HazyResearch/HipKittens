"""gfx1100 -- RDNA3, RX 7900 / W7900.

Every claim below is either measured on a W7900D or read out of the compiler.
The three that drive essentially every design decision in include/rdna3 are
marked (1)(2)(3); the long version is in RDNA.md.
"""

from .base import MmaShape, Target

GFX1100 = Target(
    arch="gfx1100",
    gpu_target="RDNA3",
    kittens_define="KITTENS_RDNA3",

    wave=32,
    max_waves_per_simd=16,

    vgprs_per_simd=1536,
    # Measured, not documented: 240 VGPRs buys 6 waves/SIMD and 241 buys 5.
    # Verified against LLVM's own Occupancy remark at 42 compiled points, 6 of
    # which distinguish granule 24 from granule 8; LLVM agreed with 24 at all
    # six. The sweep is /kernel-resource-check's hip_resources.py.
    vgpr_granule=24,
    sgprs_per_wave=106,

    # common/util.cuh MAX_SHARED_MEMORY.
    lds_per_workgroup=65536,
    # A WGP is 2 CUs / 4 SIMD32 sharing 128 KB, of which one workgroup may
    # take 64. Measured on hk.ops.decode_gemm (16-wave workgroups, occupancy
    # 16 by registers): reserving 32 KB per workgroup left the timing at the
    # 4-per-WGP figure; 48 KB and 64 KB both dropped it to the 2-per-WGP one,
    # identical to each other and to a build held to 2 by registers. Only a
    # 128 KB pool gives that pattern.
    lds_pool_bytes=131072,
    simds_per_lds_pool=4,

    mma={
        # v_wmma_f32_16x16x16_bf16 / _f16. (1) On wave32 an operand is v16bf16
        # -- 8 VGPRs -- and lanes l and l+16 must hold *identical* data: a lane
        # holds the entire K=16 vector. So a bf16 operand tile costs exactly as
        # many registers as an fp32 accumulator tile, and on RDNA3 it is the
        # operands that bound the tile shape. That is the reverse of CDNA, and
        # it is why the GEMM cannot grow past a 128x128 block.
        "bf16": MmaShape(m=16, n=16, k=16, operand_vgprs=8, accum_vgprs=8),
        "fp16": MmaShape(m=16, n=16, k=16, operand_vgprs=8, accum_vgprs=8),
    },

    # (3) vmem-to-lds-load-insts is absent on gfx11. There is no global->LDS
    # DMA, so every staged byte passes through VGPRs and HipKittens' async-load
    # pillar does not apply. Any pass that wants to stage must emit a register
    # buffer, not a copy.
    has_global_to_lds_dma=False,

    # (1), restated for the register-budget model.
    mma_operands_mirrored=True,

    # Only a row-layout bf16 operand gets the vectorised ds_read_b128 (2
    # instructions per base tile). A col-layout one degrades to 16 scalar
    # ds_read_u16 -- 8x the instructions -- see
    # ops/warp/memory/tile/shared_to_register.cuh.
    fast_smem_layouts=("row",),
)

#: Measured ceilings on a W7900D, for reporting a result as a percentage of
#: what the part can actually do rather than of its datasheet number.
#: Conditions: bf16 in / fp32 accumulate, single GPU, one process.
CEILINGS = {
    "wmma_bf16_tflops": 100.5,
    "hbm_gbps": 864.0,
}

#: The compiler can hoist a WMMA above an s_waitcnt because the wait carries no
#: register dependence. The result is randomly wrong with ScratchSize=0 and no
#: diagnostic anywhere. Any emitted wait must be followed by an empty volatile
#: asm binding the fragments it was meant to protect; the translation's
#: LdsWaitForOp case (mlir/lib/Target/Cpp/HKToCpp.cpp) is the only place that
#: emits one.
LDS_WAIT_NEEDS_FRAGMENT_BIND = True
