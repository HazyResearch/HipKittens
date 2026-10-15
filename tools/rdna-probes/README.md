# RDNA hardware probes

Standalone HIP programs that measure what the RDNA3 port is built on.
They do not use the library (except where noted) and they do not need it to be
correct — that is the point. When something in `include/rdna3` disagrees with
the hardware, these are what tell you which side is wrong.

```bash
./build.sh wmma_layout.hip gfx1100        # compile
./wmma_layout                              # run on the card
./build.sh mma_gate.hip gfx1201 --asm      # or just look at the ISA
./run_features.sh gfx1201                  # compile-probe the feature matrix
```

`build.sh <file.hip> [arch] [--asm]` — `arch` is `gfx1100` or `gfx1201` and sets
`-DKITTENS_RDNA3`. `--asm` stops at assembly, which is how
the gfx12 claims were checked without a gfx12 card.

## What each one establishes

| | |
|---|---|
| `wmma_layout.hip` | **The foundational measurement.** One-hot probes plus gauge analysis, recovering the physical register layout of every WMMA operand and accumulator. Everything in `rt_base_coord()` comes from here. Has a `PROBE_GFX12_W32` variant, written but never run. |
| `features.hip` + `run_features.sh` | The hardware feature matrix — `vmem-to-lds`, `s_setprio`, `sched_group_barrier`, `permlane*`, WMMA variants, SWMMAC, and the maximum static LDS allocation, found by bisecting the compiler error. |
| `mma_gate.hip` | Phase 3 gate: does the `rt_base` layout derivation agree with real WMMA output, for a single 16×16×16 and for blocked GEMMs. |
| `mem_gate.hip` | Phase 4 gate: every memory path round-tripped — global↔shared, global↔register, shared↔register, tiles and vectors, both layouts. |
| `red_gate.hip`, `conv_gate.hip` | Phase 5 gates: tile reductions, and the layout conversions that have to move data across the wave halves. |
| `srsrc.hip`, `srsrc2.hip`, `srsrc3.hip` | How the gfx10+ buffer resource descriptor actually behaves. CDNA's `0x110000` config word is a gfx9 format; these sweep word 3 against in-bounds and out-of-bounds accesses and land on `0x31004000`, valid for gfx11 and gfx12 both. |
| `wmma_peak.hip` | **The performance denominator.** Back-to-back `v_wmma_f32_16x16x16_bf16` with no memory access in the inner loop, over a sweep of independent accumulator chains. Measures 100.5 TFLOPs on a W7900D at a sampled 2.18 GHz, against a spec-sheet 122.6 that assumes 2.495 GHz boost. Flat from `ACC=2` to `ACC=16`, so it is an issue-rate ceiling, not a latency artifact. Every "% of peak" in this tree is taken against this number. |
| `lds_peak.hip` | **The other denominator.** LDS delivered bandwidth under the exact access patterns a GEMM inner loop uses, reported in bytes/clk/CU so it can be compared directly against what a kernel's inner loop demands. Seven patterns, same instruction count each. The one that matters is `ST`, which reproduces `st<bf16,128,32>::idx()` addressing including the XOR swizzle: it measures 64.8 B/clk/CU, tied with a hand-constructed conflict-free reference and 4.1x an unswizzled tile read. That is the evidence that the rdna3 swizzle constant is right. |
| `empty.hip` | The minimal "does the toolchain work for this target" file. |

## Do not measure against the spec sheet

The W7900 data sheet says 122.6 TFLOPs bf16, which is 96 CU x 512 FLOP/clk x
2.495 GHz. Sampling `rocm-smi` during each workload says the card does not get
there and cannot:

| workload | sclk | power | TFLOPs |
|---|---|---|---|
| `wmma_peak.hip` (no memory at all) | 2181 MHz | 101 W | 100.5 |
| rocBLAS GEMM 4096^3 | 2089 MHz | 227 W | 86.4 |
| hipBLASLt GEMM 4096^3 | 2062 MHz | 241 W | 68.3 |

The pure-WMMA loop reaches 480 FLOP/clk/CU against the architectural 512, i.e.
the issue rate is 94% of spec; the whole rest of the gap to 122.6 is clock. Add
memory traffic and the card hits its 241 W limit and drops another 100 MHz. So
the reachable peak on this part is ~104 TFLOPs, not 122.6, and a percentage
computed against 122.6 understates a kernel by about 18%.

## One warning, learned the hard way

A gfx11 WMMA operand is **mirrored across the wave halves** — lanes `l` and
`l+16` must hold identical data. The first version of `wmma_layout.hip` set a
one-hot in a single lane without mirroring it to `l^16`. That violates the
hardware constraint, so what came back was undefined behaviour, and it did not
look like garbage: it looked like a coherent layout in which half the lanes
contributed nothing. The probe now sets `l` and `l^16` together, and feeds
operands from global memory so that dynamic `insertelement` codegen cannot
confuse the picture either.

If a probe result surprises you, suspect the probe first.

## What LDS actually delivers

Same instruction count in every row; `bytes/clk/CU` is the comparable column and
`ms` is not, because `U16` moves 2 bytes per lane per instruction where the rest
move 16.

| pattern | bytes/clk/CU | what it is |
|---|---|---|
| `LINEAR` | 27.6 | `lane*16`, fully coalesced across 32 distinct lanes |
| `TILE16` | 15.7 | 16 mirrored lanes at a 64 B row stride, no swizzle -- only banks 0-3 and 16-19 are ever touched, an 8-way conflict |
| `TILE16X` | 15.6 | the same plus the *CDNA* swizzle, which flips address bit 3. A `ds_read_b128` ignores the low four address bits, so this is invisible to it and measures identical to `TILE16`, as it must |
| `XOR4` | 37.7 | 16 B granules permuted within a 64 B row by `row&3`: four bank groups |
| `K64` | 64.4 | the same with a 128 B row and `row&7`: eight bank groups, conflict-free by construction |
| **`ST`** | **64.8** | **what this tree actually does.** `off ^= ((off % 1024) >> 7) << 4` over a 64 B row stride sends sixteen lanes to bank groups 0,16,4,20,8,24,12,28 and then the same eight again -- two lanes per group, the 2-cycle minimum for a 256 B request |
| `U16` | 35.8 | one bf16 per instruction at `lane*2`, K strided 256 B apart. This is what rocBLAS/Tensile does, and it is conflict-free too |

`ST` tying `K64` is the whole point of the table: the LDS reads in this tree are
already conflict-free, and `swizzle_bytes` does not need retuning. Anything left
on the table in the GEMM is bytes moved, not cycles lost to banking.

Read the demand side like this. One HipKittens inner-loop iteration is 24
`ds_read_b128` and 16 WMMAs per wave. The WMMAs are 16 x 8192 FLOP, which at
512 FLOP/clk/CU is 256 CU-cycles, and a CU retires two wave-iterations in that
window, so the LDS has to supply `2 * 24 * 32 * 16 / 512` = **48 bytes/clk/CU**
against a 64.8 ceiling: 74% utilised. rocBLAS's 64x64 warp tile asks for 32 B/clk
for the same 16 WMMAs. That 1.5x is not a banking problem and no swizzle fixes
it; it is `(M*N)/(M+N)` for a 32x64 tile versus a 64x64 one, exactly 21.3 vs 32.
