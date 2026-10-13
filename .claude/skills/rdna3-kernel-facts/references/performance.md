# gfx11 performance facts

Architecture-specific facts for gfx1100 (RDNA3) / gfx1201 (RDNA4), each paid for once on real hardware.
Not applicable to CDNA or NVIDIA; the method in `/kernel-bringup` is.

These decide the shape of the tiling, not just its constants.

## Performance

**Only `row`-layout bf16 operands get the vectorised `ds_read_b128`** (2
instructions per base tile). `col` layout degrades to 16 scalar `ds_read_u16` —
8× the instruction count. This single fact determines the tiling: arrange LDS so
every matmul operand is `row`, including by staging V transposed rather than
transposing after the load.

**`swap_layout` on gfx11 moves data** (~20 lane swaps per base tile), it is not
a relabel. "Load then transpose" is not free; stage it in the layout you need.

**Inline asm is mandatory for LDS access** in the hand-scheduled inner loop —
the intrinsics do not let you place the waits. The buffer resource descriptor
word that works here is `0x31004000`.

**No global→LDS DMA on gfx11.** Every staged byte goes through VGPRs, so a
"double-buffered LDS" pipeline is really a register-prefetch pipeline; budget
registers for it up front.

**WMMA requires operand mirroring across lanes `l` and `l^16`** (wave32,
wave-half mirroring). Violating it does not fault; it returns undefined data
that can look self-consistent. Any layout probe must mirror, or it will
confidently report a wrong layout.

**64 KB LDS per workgroup**, not 160 like CDNA4. Any CDNA kernel's tiling has to
be re-derived, not ported — this is usually the constraint that forces a
different algorithm shape rather than a different constant.

**The LDS pool is 128 KB per WGP, not 64.** One workgroup may take 64 KB, but
the WGP (2 CUs, 4 SIMD32) shares 128. Measured 2026-10-13 on
`hk/ops/decode_gemm.py` with 16-wave workgroups at register occupancy 16:
reserving 32 KB per workgroup left the 4-per-WGP timing, 48 KB and 64 KB both
gave the 2-per-WGP timing and matched each other -- only a 128 KB pool fits.
Any occupancy estimate or residency cap derived from 64 KB is off by 2x;
`hk/target/gfx1100.py` had it wrong until then.

**A streaming kernel can want *fewer* resident workgroups.** Same kernel: two
16-wave workgroups per WGP read the weights 7-11% faster than four, same
code, set by an LDS reservation. Higher occupancy is not automatically better
when every wave is a separate DRAM stream. A slowdown that coincides with an
occupancy change is a residency question first: the epilogue it was first
blamed on (fp32 LDS reads compile to `flat_load`, since `kittens::load(rt,st)`
only vectorises bf16 `row`) was rebuilt on `ds_read_b128` and was not the
cause.

**Predicate a ragged tile with the buffer range, not with code.** A raw
buffer descriptor (config word `0x31014000`) whose range ends at the tensor's
last row makes loads past it return 0 and stores past it vanish -- no branch,
no masked lane, same instruction count, so hand-placed `vmcnt` immediates
stay valid. Three ways to get it wrong, all measured 2026-10-13 (HipKittens
`include/rdna3/ops/warp/memory/tile/global_to_register.cuh`): (1) any
predicate threaded through a pointer-based load/store loop, even a
compile-time-constant one, can make LLVM stop folding addresses into
base+immediate (36 -> 91 VGPRs on an elementwise op); (2) a descriptor
computed from a value LLVM thinks is per-lane (anything derived from
threadIdx, including the warp index) gets every buffer op wrapped in a
waterfall loop -- `readfirstlane` the descriptor words; (3) keep the moving
part of the address in `soffset` and the descriptor loop-invariant, or the
descriptor is rebuilt every iteration (the GEMM's staging got 15 VGPRs
lighter from this alone).

**gfx11 runs with unaligned global/buffer access enabled.** LLVM merges a
loop of 16-bit loads at unknown alignment into `global_load_b128`, and a
`buffer_load_b128` from a tensor 2 bytes off a 16-byte boundary returns the
right data (HipKittens `tests/hk/gpu/test_bounded.py`). An alignment
fallback around buffer intrinsics is not free: LLVM cannot merge the
intrinsics, and reserves the fallback's registers whether it runs or not.

**Decode GEMV: W straight to registers, K split across 16 warps,
interleaved.** Same file: `W*X^T` with W loaded global->register, one 16-row
block per workgroup, warp w reading K-chunk `t*16 + w` (2 KB contiguous per
row per round), two workgroups per WGP, streams 663-802 GB/s on Qwen3-8B's
layers. The LDS-staged GEMM template at a 16-row tiling: 418-755.

**Hardware counters are dead.** Under `rocprofv3` on gfx1100 every `SQ` counter
except `SQ_WAVES` reads 0 — `SQ_INSTS_VALU`, `SQ_INSTS_LDS`, `LDSBankConflict`,
`ALUStalledByLDS` included. Attribution must come from ablation switches
(`measurement.md` §3).

**The part is clock- and power-limited under any load**: 122.6 TFLOPs
architectural, ~100.5 measured back-to-back WMMA at 2.18 GHz / 101 W. Use the
measured number as the denominator.

**Peer bandwidth is bimodal per process launch** (~15.5 or ~24–27 GB/s, nothing
between, stable within a process). Communication and compute must be timed in
the same process or the comparison describes two different machines.

## Benchmarking on this hardware

**Compare against rocBLAS/hipBLASLt tuned, not torch's default path.** Torch's
default is not the vendor's best and beating it proves nothing.

**Cross-process A/B is invalid on `wx-ms-w7900d-0043`** — it manufactures 13–16%
differences. See `measurement.md` §1.

**The torch profiler's kernel durations include queue time.** On this ROCm
build, in an eager vLLM step whose launches run ahead of the GPU, the chrome
trace's kernel durations summed to 4.4x the step's wall time (a 3-call H2D
memcpy "took" 618 ms). They measure from enqueue, not from start. Time steps
or calls with a synchronize on each side instead (HipKittens
`tools/hk-bench/vllm_steps.py --time-linear`).

**A library GEMM's speed can depend on the process it runs in.** Measured
2026-10-13: hipBLASLt ran the same 36 Qwen3-8B layers at M=4101 in 837 ms
inside vLLM, 849-870 standalone, and 859 inside vLLM when alternating step by
step with another GEMM (qkv alone +21%). Data, timing method and call path
were ruled out. The HK GEMM beside it timed identically everywhere. A
baseline measured in a microbenchmark is a range, not a number; compare
against the engine in place.

**Infinity Cache will fake a decode benchmark.** Re-using one weight buffer
measures cache bandwidth and reports figures above the card's memory peak.
Rotate buffers.

**Rotating a few copies is not enough either; time a pass over distinct
weights.** Measured 2026-10-12 (HipKittens `tools/hk-bench/vllm_gemm.py`,
Qwen3-8B shapes): per-call event timing over 3 rotating copies of a 100.7 MB
matrix (302 MB, 3x the cache) had upstream vLLM's `wvSplitK` reading "at"
1917 GB/s, 2.2x HBM peak; 6 copies of a 50 MB matrix gave 1639. Over 36
distinct weights timed as one back-to-back pass -- what a decode step does --
the same kernel reads at 769 and 766 GB/s, every output checked. hipBLASLt
(~40% of HBM) measured the same both ways, so a slow baseline will not reveal
the problem; only a fast candidate does. Achievable streaming read on this
card: 680-820 GB/s by size (`hbm_probe.hip`, one kernel per read), up to
835 GB/s for a back-to-back sequence.
