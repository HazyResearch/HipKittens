"""The shape of an architecture description.

A Target is pure data plus the arithmetic that follows from it. It holds no
opinions about any particular kernel -- those live in the hk passes
(mlir/lib/Dialect/HK/Transforms), which read this table's C++ twin.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass(frozen=True)
class MmaShape:
    """One matrix instruction: D[m,n] += A[m,k] * B[k,n]."""

    m: int
    n: int
    k: int
    #: Registers one operand fragment occupies, per lane, in 32-bit VGPRs.
    operand_vgprs: int
    #: Registers one accumulator fragment occupies, per lane.
    accum_vgprs: int


@dataclass(frozen=True)
class Target:
    arch: str
    #: GPU_TARGET value kernels/common.mk expects.
    gpu_target: str
    #: -DKITTENS_* define that selects the include/ tree.
    kittens_define: str

    wave: int
    max_waves_per_simd: int

    vgprs_per_simd: int
    #: VGPRs are allocated in multiples of this. Occupancy is charged on the
    #: rounded count, which is the whole reason this field exists.
    vgpr_granule: int
    sgprs_per_wave: int

    #: Hard cap on one workgroup's LDS request (MAX_SHARED_MEMORY in
    #: common/util.cuh).
    lds_per_workgroup: int
    #: LDS bytes shared by one scheduling pool, and the number of SIMDs that
    #: pool feeds. On RDNA3 the pool is a WGP (2 CUs, 4 SIMD32), not a CU --
    #: getting this wrong reports twice the achievable occupancy.
    lds_pool_bytes: int
    simds_per_lds_pool: int

    #: Matrix instruction shapes, keyed by dtype name.
    mma: Dict[str, MmaShape] = field(default_factory=dict)

    #: True when the part can move global memory into LDS without passing
    #: through VGPRs. False on all of gfx11/gfx12, which is why every staging
    #: path in this library is a register buffer.
    has_global_to_lds_dma: bool = False

    #: Whether a bf16/fp16 mma operand fragment is mirrored across the two wave
    #: halves. When True, a 16-wide operand costs a full 8 VGPRs per lane and an
    #: operand tile is exactly as expensive as an fp32 accumulator tile.
    mma_operands_mirrored: bool = False

    #: Layouts whose shared->register load hits the vectorised ds_read_b128
    #: path. Anything else degrades to scalar ds_read_u16, 8x the instructions.
    fast_smem_layouts: Tuple[str, ...] = ("row",)

    #: False when no part was available and the numbers are inferred from
    #: builtin signatures rather than measured. Consumers must degrade rather
    #: than present an inferred number as a measured one: see
    #: the hk-check-budgets pass, which downgrades budget *errors* to warnings on an
    #: unverified target because a wrong hard limit is worse than none.
    verified: bool = True

    # ---- derived ----

    def vgpr_rounded(self, vgprs: int) -> int:
        """VGPRs actually charged for a kernel that uses `vgprs`."""
        g = self.vgpr_granule
        return -(-vgprs // g) * g

    def occupancy(self, vgprs: int) -> int:
        """Waves/SIMD the register footprint allows. Register-only: LDS can and
        does bind below this."""
        rounded = self.vgpr_rounded(vgprs)
        if rounded <= 0:
            return self.max_waves_per_simd
        return min(self.max_waves_per_simd, self.vgprs_per_simd // rounded)

    def vgpr_headroom(self, vgprs: int) -> int:
        """VGPRs addable before occupancy drops a step. 0 means the kernel is
        sitting exactly on the edge."""
        occ = self.occupancy(vgprs)
        if occ <= 1:
            return 0
        g = self.vgpr_granule
        max_for_level = (self.vgprs_per_simd // occ) // g * g
        return max_for_level - vgprs

    def occupancy_by_lds(self, lds_bytes: int, waves_per_workgroup: int) -> int:
        """Waves/SIMD the LDS footprint allows, for comparison against
        occupancy(). The binding limit is the smaller of the two, and for the
        GEMM in this repo it was this one -- which is why a kernel is never
        judged on the compiler's register-only occupancy alone.

        Worked example, the shipped attention config on gfx1100: two 8 KB
        tiles double buffered is 32 KB, so four workgroups of 12 waves fit the
        128 KB pool -> 48 waves / 4 SIMDs = 12 waves/SIMD, so its 238 VGPRs (6
        waves/SIMD) are what bind, not the LDS.
        """
        if lds_bytes <= 0:
            return self.max_waves_per_simd
        workgroups = self.lds_pool_bytes // lds_bytes
        waves = workgroups * waves_per_workgroup
        return min(self.max_waves_per_simd, waves // self.simds_per_lds_pool)

    def tile_vgprs(self, dtype_bits: int, rows: int, cols: int) -> int:
        """VGPRs one register tile occupies, per lane.

        A 16-bit tile is stored in the mma operand fragment layout and a 32-bit
        tile in the accumulator layout, so the cost follows from the mma shape
        rather than from the element count. On gfx1100 both come to 8 VGPRs per
        16x16 base tile -- that is the mirrored-halves fact (1) restated: a bf16
        operand tile is exactly as expensive as an fp32 accumulator tile, which
        is what bounds tile shapes on this architecture.
        """
        shape = next(iter(self.mma.values()), None)
        if shape is None:
            return (rows // 16) * (cols // 16) * 8
        per_base = shape.operand_vgprs if dtype_bits == 16 else shape.accum_vgprs
        return (rows // 16) * (cols // 16) * per_base

    def occupancy_achieved(self, vgprs: int, lds_bytes: int, waves_per_workgroup: int) -> int:
        """What the kernel actually gets: the smaller of the two limits."""
        return min(
            self.occupancy(vgprs),
            self.occupancy_by_lds(lds_bytes, waves_per_workgroup),
        )
