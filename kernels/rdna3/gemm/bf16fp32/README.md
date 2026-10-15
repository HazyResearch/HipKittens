# bf16 GEMM, fp32 accumulate — RDNA3 (gfx1100)

C = A · Bᵀ, bf16 in, fp32 accumulate, bf16 out. B is passed pre-transposed as
`(n, k)`, which keeps both shared loads contiguous and makes `mma_ABt` the right
primitive.

This is the kernel the RDNA3 port was validated against. Everything below is
measured.

## Measured

W7900D (gfx1100, 96 CU), ROCm 7.2.4. TFLOPs, HipKittens against **both** AMD
BLAS libraries. The vendor gets two columns because the comparison genuinely has
two answers:

- **best** — best of {NN, NT, TN, TT}. The vendor picks its favourite data
  layout; HipKittens has only one.
- **NT** — the vendor restricted to HipKittens' layout: A is (m,k), B is (n,k),
  C = A·Bᵀ, both operands K-contiguous.

| shape (M×N×K) | hipBLASLt best | rocBLAS best | vendor NT | HK | HK / best | HK / NT | HK / ceiling |
|---|---|---|---|---|---|---|---|
| 4096³ | 69.7 | **82.9** | 69.7 | 74.4 | 90% | **107%** | 74% |
| 8192×8192×4096 | 64.9 | **88.8** | 64.9 | 77.2 | 87% | **119%** | 77% |
| 4096×8192×2048 | 71.3 | **88.4** | 70.0 | 75.4 | 85% | **108%** | 75% |
| 8192×4096×2048 | 69.9 | **88.6** | 69.9 | 75.3 | 85% | **108%** | 75% |
| 2048×4096×4096 | 67.8 | **84.6** | 67.8 | 73.7 | 87% | **109%** | 73% |
| 2048×2048×4096 | 63.8 | **75.8** | 57.7 | 67.4 | 89% | **117%** | 67% |
| 2048³ | 60.2 | **69.8** | 55.1 | 64.5 | 92% | **117%** | 64% |

**In its own layout this kernel beats the best AMD library on every shape, by
7-19%. Against the vendor's best layout it is 8-15% behind.** Both statements
are true and they are about different problems: a GEMM with both operands
K-contiguous is a harder memory problem than one where the vendor got to choose.
Pick the column that matches your data. Attention and MoE pipelines hand you the
NT column. The section below takes the vendor's spread apart and shows exactly
what the NT column is paying for — it is one number, it is measurable in this
kernel too, and this kernel does not pay it.

One further note on reading this: an early version of this table compared only
against torch's default backend and claimed the kernel won two shapes of four.
That was wrong twice over — torch on ROCm defaults to hipBLASLt, and on gfx1100
hipBLASLt is the *slower* library by up to 33%. `bench.py` benchmarks both,
across all four layouts, every time.

Percentages are against **100.5 TFLOPs**, the measured back-to-back WMMA issue
ceiling ([`tools/rdna-probes/wmma_peak.hip`](../../../../tools/rdna-probes/wmma_peak.hip)),
not against the 122.6 TFLOPs on the data sheet. 122.6 assumes 2.495 GHz boost;
sampling `rocm-smi` during a GEMM shows this card at 2.0-2.1 GHz against its
241 W limit. The reachable peak here is ~104 TFLOPs.

### Exactly one of the vendor's four layouts is slow, and it is a K-tile artifact

rocBLAS on this card, 8192×8192×4096, same math four ways:

| layout | A in memory | B in memory | TFLOPs |
|---|---|---|---|
| NN | K-contiguous | N-contiguous | 85.7 |
| **NT** | **K-contiguous** | **K-contiguous** | **56.0** |
| TN | M-contiguous | N-contiguous | 90.2 |
| TT | M-contiguous | K-contiguous | 89.9 |

Three of the four are within 5% of each other. NT — the one with *both* operands
K-contiguous, which is the one HipKittens implements — falls off a cliff. That
single outlier is where the whole "HK / best" versus "HK / NT" spread in the
table above comes from, so it is worth knowing what it is.

**It is not the inner loop.** Unbundling `TensileLibrary_Type_BB_HPA_*_gfx1100.co`
(they are `CCOB` compressed offload bundles; `clang-offload-bundler --unbundle`
gets the ELF out) and disassembling the two kernels `rocprofv3` actually
dispatches for NT and TT, they are the same kernel to a remarkable degree:
`MT128x128x16`, `MI16x16x16x1`, `WG32_4_1` — 4 waves, 64×64 each — `PGR1 PLR1`,
`TT4_64`, 256 VGPRs, 128 of them accumulators, and no operand double-buffering in
either. Per K-tile per wave:

| | NT (56.0) | TT (89.9) |
|---|---|---|
| operand reads | 16 × `ds_load_b128` | 8 × `ds_load_b128` + 64 × `ds_load_u16` |
| lane-bytes read | 8192 | 8192 |
| LDS writes | 4 × `ds_store_b128` | 4 × `ds_store_b128` |
| global loads | 4 × `buffer_load_b128` | 4 × `buffer_load_b128` |
| WMMAs | 16 | 16 |

The *slow* kernel is the one with the clean inner loop. TT cannot store its B
operand straight into LDS, so it gathers it back 2 bytes at a time — 3× the LDS
instructions for the same bytes — and still wins by 60%. Whatever separates them
is not in there.

**It is global-load granularity, and `K_STEP` is the knob.** A K-contiguous
operand tiled at K=16 is 128 rows of 32 bytes: a quarter of a 128-byte cache
line per row, so three quarters of every line fetched is thrown away, on *both*
operands. NN/TN/TT each have at least one operand contiguous along a free
dimension, where a tile row is 128 elements = 256 bytes and coalesces perfectly.

This kernel can measure that penalty directly, because `K_STEP` is a `-D`. Same
tiling, same layout, same code, only the K-tile depth moving (128×128, 8 warps,
8192×8192×4096; `ABLATE_GLOBAL=1` deletes the global→LDS stage and leaves the
math running on stale tiles):

| `K_STEP` | bytes per tile row | full | `ABLATE_GLOBAL` | global costs |
|---|---|---|---|---|
| 16 | 32 | 59 | 81 | **22** |
| 32 | 64 | 77 | 87 | **10** |
| 64 | **128** | 78 | 83 | **5** |

The cost collapses as the row reaches one cache line, and `GPREFETCH` 1→8 does
not move it at all (77 / 77 / 76 / 76 / 74 / 65 at `K_STEP=32`), so it is
granularity and not latency.

So: rocBLAS's 56 is what the NT layout costs *at a 16-deep K-tile*, and Tensile's
gfx1100 tuning has no deeper NT entry to pick. This kernel runs a 64-deep one and
pays 5 instead of 22. **That is the entire reason the HK/NT column is above 100%
— it is not a better inner loop, it is the one tuning decision the layout
demands.** It also sets the honest ceiling for this layout: even with the global
stage deleted entirely the kernel only reaches 83, so the vendor's 88-90 is not
sitting in global memory waiting to be collected.

### Could this kernel offer all four layouts?

Yes, and it is a small change, but it is an API-completeness change and not a
performance one — worth stating plainly because the measured table invites the
opposite conclusion.

The inner loop is layout-independent: LDS always holds the operands K-major, and
`mma_ABt` always sees the same thing. Only the global→LDS stage differs, and it
has exactly two forms — a straight copy when the operand's global layout matches
the LDS one, or a transposing store when it does not, which is Tensile's
`UMLDSA`/`UMLDSB` flag and nothing more. `load_global_to_register_buffer` already
splits the load from the store, so a transposing variant of
`store_register_buffer_to_shared` plus a layout tag on the tile types covers it.

What it would not do is produce another 88. The reason NN/TN/TT are fast for the
vendor is that at least one operand is free-dimension-contiguous, which is a
property of the *caller's* data — porting the layouts does not conjure it. Each
non-NT layout this kernel added would be slower than the NT path it already has,
because it would trade a straight LDS store for a transposing one while the
global side stays whatever the caller handed it. The value is that callers stop
having to pre-transpose; there is no throughput in it.

### Where the remaining gap goes

It is LDS *bytes*, not LDS instructions and not bank conflicts. Both of those
were checked and neither is the answer.

Disassembling the rocBLAS TT kernel — the 89.9 above, and therefore the one the
"HK / best" column is measured against — and histogramming the two inner loops,
normalized so both do 16 WMMAs:

| per 16 WMMAs | rocBLAS | this kernel |
|---|---|---|
| LDS read instructions | **128** × `ds_load_u16` (2 B) | 24 × `ds_read_b128` (16 B) |
| LDS bytes read per lane | **256** | **384** |
| non-WMMA VALU | **0** | 28 × `v_add_nc_u32` |
| total instructions in the loop | 188 | 122 |

rocBLAS issues 5.3× more LDS instructions and still wins, which invites a
bank-conflict story. There is not one.
[`tools/rdna-probes/lds_peak.hip`](../../../../tools/rdna-probes/lds_peak.hip)
measures the delivered bandwidth of each access pattern at equal instruction
count: the pattern this kernel uses reaches **64.8 bytes/clk/CU**, tied with a
hand-built conflict-free reference and 4.1× the same tile read with the swizzle
removed. rocBLAS's 2-byte pattern reaches 35.8. Both are conflict-free; the
`swizzle_bytes` constant in `st.cuh` does not need retuning.

What is left is LDS bytes, and the amount left is now accounted for exactly
rather than estimated.

**LDS charges lane-bytes, not distinct bytes.** That is the whole story, and it
was worth measuring rather than assuming: a gfx11 WMMA operand is mirrored
across the wave halves, so lanes `l` and `l^16` issue `ds_read_b128` against the
*same* address, and the question is whether the hardware broadcasts that for
free. It does not. `lds_peak.hip` mode `ST` reproduces this kernel's addressing
exactly and reaches **64.8 lane-bytes/clk/CU = 129.6/WGP**, which is the
hardware's 128 B/clk/WGP. The duplicate half is paid for in full.

So, per WGP per K-tile at 128×128×64:

| | LDS cycles | as a fraction of the 2048 SIMD cycles of WMMA in the same window |
|---|---|---|
| reads: 8 waves × 48 `ds_read_b128` × 512 lane-B ÷ 128 B/clk | 1536 | 75% |
| writes: 64 `ds_write_b128` × 512 lane-B ÷ 128 B/clk | 256 | 12% |
| **total** | **1792** | **87%** |

87% is not a number a pipeline can hide behind. It also matches where the
kernel actually sits: 79 / 100.5 = 78%, with the remaining 9 points going to the
barrier and to the windows where neither resident wave has math ready.

The `(M+N)/(M·N)` intensity argument says the fix is a bigger warp tile —
reads per WMMA are `2·(16/M + 16/N)`, so 32×64 costs 1.5 and 64×64 costs 1.0,
and rocBLAS's Tensile kernel for this shape
(`Cijk_..._MT128x128x16_MI16x16x16x1_..._PGR1_PLR1_..._WG32_4_1`) is exactly
**4 waves of 64×64** against this kernel's 8 waves of 32×64.

Every way of copying that measures *slower*. Swept at 8192×8192×4096, with the
compiler's spill count, because the spills are the point:

| config | warp tile | VGPRs | spill | waves/SIMD | TFLOPs |
|---|---|---|---|---|---|
| **128×128×64, 8 warps (default)** | **32×64** | **208** | **0** | **7** | **79** |
| 128×128×16, 4 warps | 64×64 | 232 | 0 | 6 | 58 |
| 128×128×32, 4 warps | 64×64 | 256 | 14 | 5 | 56 |
| 128×128×64, 4 warps | 64×64 | 256 | 91 | 5 | 14 |
| 256×128×16, 8 warps | 64×64 | 227 | 0 | 6 | 71 |
| 256×128×32, 8 warps | 64×64 | 256 | 4 | 5 | 75 |
| 128×256×32, 8 warps | 64×64 | 256 | 5 | 5 | 74 |
| 256×256×16, 16 warps | 64×64 | 222 | 0 | 6 | 66 |
| 256×256×32, 16 warps | 64×64 | 249 | 0 | 5 | 72 |

Accumulator VGPRs are `M·N/32` regardless of shape, so 64×64 is 128 accumulator
registers plus 64 operand registers before a single byte of global staging. Every
row above either spills or falls to 5-6 waves/SIMD, and the ones that avoid both
do it by taking `K_STEP` down to 16 — straight back into the 22-TFLOP global
penalty measured two sections up. **That is the trade the default tiling makes:
1.5× the LDS traffic per WMMA, bought with the registers to run a 64-deep K-tile
at 7 waves/SIMD.** Measured, it is worth 4-21 TFLOPs over every 64×64 variant.

rocBLAS runs 64×64 at 256 VGPRs with no spills because it is not carrying a
64-deep K-tile's staging — and it pays for that with the 56 in the NT column.

There is no intermediate tile either — among powers of two the square tile is
optimal, and a 48-row tile needs `BLOCK_M=192`, which divides none of the
benchmark shapes (the kernel has no bounds predication).

### Levers that were measured and are not levers

Each of these was a plausible story about the missing 20%. All of them are
closed, and the measurement is recorded so none is retried blind.

**Occupancy.** The compiler's occupancy remark counts registers only; LDS is the
binding limit here, and `hipOccupancyMaxActiveBlocksPerMultiprocessor` (whose
"multiprocessor" is a WGP: 4 SIMDs, 2048 threads, 64 KB usable LDS,
`multiProcessorCount` = 48) gives the real number. Build with `HK_OCC=1` to
print it. Raising it makes things monotonically worse:

| config | LDS | waves/SIMD | 8192×8192×4096 |
|---|---|---|---|
| 128×128×64, 8 warps | 64 KB | **2.0** | **79** |
| 128×256×32, 16 warps | 48 KB | 4.0 | 78 |
| 128×128×32, 8 warps | 32 KB | 4.0 | 71 |
| 256×256×32, 32 warps | 64 KB | 8.0 | 63 |
| 128×512×16, 32 warps | 48 KB | 8.0 | 46 |

Latency hiding here comes from the software pipeline, not from resident waves.
Two waves per SIMD with a good schedule beat eight with a bad one, and once LDS
is 87% busy more waves only add contention.

**Deeper global prefetch** (`GPREFETCH`). Keeping 2-3 K-tiles of global data in
flight, retired with `vm_wait<(GP-1)*(stage_a+stage_b)>()`, is completely flat at
K_STEP ≥ 32 (79 / 79 / 71 for GP = 1 / 2 / 3, the last collapsing at 256 VGPRs).
It recovers 52 → 58 at K_STEP=16, where the K-tile is too short to cover a global
round trip — but K_STEP=16 is 20 TFLOPs behind anyway. Global latency is already
covered.

**Clocks.** Under a 20 s steady-state run `rocm-smi` reports **2171 MHz mean at
197 W** — the same clock the `wmma_peak` probe ran at when it established
100.5 TFLOPs. The denominator is honest; there is no thermal headroom being lost.

**wave64.** The one structural escape from the mirror would be a WMMA whose
operands are not replicated. gfx1100 does have
`__builtin_amdgcn_wmma_f32_16x16x16_bf16_w64`, and its signature settles it: A
and B are `v16i16` — 8 VGPRs, the full K=16 per lane, *identical to wave32* —
with a `v4f32` accumulator. 64 lanes × 16 halves for a 256-element matrix is **4×
replication, twice as bad as wave32's 2×**. It cannot reduce LDS bytes. Dead end,
and cheap to confirm: it is one compile.

**Interleaving the LDS writes into the math** (`WRITE_POS=2`, issuing the store
at the point the tile's last `ds_read` goes out so the final K-slice's WMMAs
cover it). Correct, and worth nothing: 78 against `WRITE_POS=1`'s 79. Writes are
1/8 of the tile's LDS traffic, so moving them around inside a pipe that is
already the bottleneck does not help.

**Hardware counters** are not available to diagnose any of this. Every `SQ`
counter except `SQ_WAVES` reads 0 under `rocprofv3` on this part —
`SQ_WAVE_CYCLES`, `SQ_WAIT_ANY`, `SQ_WAIT_INST_LDS`, `SQ_INSTS_VALU`,
`SQ_INSTS_LDS`, `LDSBankConflict`, `ALUStalledByLDS`, `GRBM_GUI_ACTIVE`. The
ablation builds below exist because of this.

**So the wall is architectural**, and it is the 2× operand mirroring: a gfx11
`bf16` `rt_base` costs the same 8 VGPRs — and the same LDS bytes — as an `fp32`
one. gfx12 splits K across the halves instead and halves both, so a gfx12 port
would have headroom this kernel does not.

### The LDS reads run underneath the WMMAs, for free

Ablating one pipeline stage at a time (`sweep.sh` with `ABLATE_*`, at 4096³ /
8192×8192×4096) is how the cost was located:

| build | TFLOPs |
|---|---|
| full | 74 / 79 |
| no `s_barrier` | 77 / 83 |
| no LDS writes | 76 / 83 |
| no barrier *and* no LDS writes | 81 / 86 |
| no global prefetch | 79 / 83 |
| no LDS reads | 88 / 92 |
| WMMAs only | 95 / 101 |

Read down that list: the reads are worth 13 TFLOPs, the writes and the barrier
together are worth 7, and global prefetch is worth 4. The reads are the wall
analysed above. The 7 points in writes-plus-barrier are the only part that ever
looked addressable, and `WRITE_POS=2` — which is precisely the schedule that
should have collected them — does not.

Before the change described here the full build was 66/71 and the no-LDS-read
build was 84/95 — the read stage was the dominant cost, and perfectly overlapping
it was worth 30%.

The textbook fix is to double-buffer the operand tiles so slice `k+1`'s reads
issue before slice `k`'s WMMAs. That was tried, and it is *slower*: the second
set of tiles costs 48 VGPRs, occupancy goes 9 → 7 waves/SIMD, and at 9 waves the
SIMD was already covering some LDS latency by switching waves. The overlap does
not pay for the registers.

What does work is getting the overlap without allocating anything, and
`lds_wait<N>` is what makes that possible. It waits until *at most N* LDS
operations are still outstanding, and LDS retires in order, so a partial wait
retires a prefix of a batch and leaves the rest moving. Two changes follow:

1. **Split along N.** `B` and the accumulator are cut into `N_SPLIT` chunks of
   `SPLIT_N` columns. The whole slice's reads — `A` plus every `B` chunk — go in
   flight at once, then the waits step down: `lds_wait<(N_SPLIT-1-c)*CHUNK_READS>`
   before chunk `c` retires exactly what chunk `c` needs. Total registers are
   unchanged; this is a re-association, not a buffer. Worth 65 → 69 / 71 → 75.
2. **Rotate across slices** (`ROTATE=1`). `B_tile[c]` is dead the instant chunk
   `c`'s WMMAs have issued, so the *next* slice's chunk `c` goes in flight right
   there, a full chunk of math early. `A_tile` is dead only after the slice's
   last chunk, so `A` is issued at the boundary and *before* that chunk's `B`,
   which keeps the next slice's first chunk gated on `A` rather than on a read
   queued behind it. The steady state then has a uniform wait depth, and fifteen
   of a K-tile's sixteen chunks have reads underneath them. Worth another 69 → 70
   / 75 → 76.

Neither step moves the register count: 161 VGPRs before the split, after the
split, and after the rotation. The compiler does not rename around the
write-after-read on `B_tile[c]`, which was the thing most likely to go wrong.

**This also changed what the rest of the tuning wants**, which is the part worth
remembering. More K-slices means more rotation points, so a deeper K-tile is
suddenly worth its staging registers even at lower occupancy:

| K_STEP | slices | VGPRs | waves/SIMD | 4096³ / 8192×8192×4096 |
|---|---|---|---|---|
| 32 | 2 | 161 | 4.0 | 70 / 76 |
| 64 | 4 | 208 | 2.0 | **74 / 79** |

`K_STEP=64` was measured before the rotation landed and lost (66/69 against
66/72); it wins now. `K_STEP=128` does not fit in 64 KB of LDS with two buffers.

### Small shapes get their own tiling

The inner loop above is not what bounds small shapes; grid quantization is. The
128×128×64 block uses 48 KB of LDS, so only one workgroup fits per WGP and 48
WGPs hold 48 at once. A grid that is a poor fit for the coarse tiling idles
through a mostly-empty final pass.

So `dispatch_micro` compiles two tilings and picks by workgroup count:

| M,N,K | blocks | 128×128×64, 32×64/warp | 128×64×64, 32×32/warp |
|---|---|---|---|
| 512×512×4096 | 16 | 19 | **33** |
| 1024×1024×4096 | 64 | 49 | **51** |
| 2048×1024×4096 | 128 | **63** | 58 |
| 2048×2048×2048 | 256 | **64** | 60 |
| 4096×1024×4096 | 256 | **65** | 60 |
| 4096³ | 1024 | **73** | — |
| 8192×8192×4096 | 4096 | **79** | — |

so the rule is `blocks <= 64`. **That threshold moved by 4× when the rotated
schedule landed** — it used to be 256. Overlapping the LDS reads made the coarse
tiling much better at mid-size grids, and it now wins everywhere it has enough
workgroups to fill the machine once. The small tiling gives up half the
arithmetic intensity and gets back twice the workgroups and three more
waves/SIMD (134 VGPRs and 10 waves against 208 and 7).

A third config exists purely for correctness: `K_STEP=64` requires K to be a
multiple of 64, and the kernel has no K remainder handling, so `dispatch_micro`
falls back to a `K_STEP=32` build when it is not.

At 512×512 even the small tiling only reaches 33 TFLOPs. That regime wants
split-K, which is still missing and is the next thing here.

How it got there, each step measured at 4096³ / 8192×8192×4096:

| | TFLOPs |
|---|---|
| first correct version | 50 / 54 |
| one barrier per K-tile | 53 / 58 |
| 16-byte LDS swizzle, 2× b128 (still `flat_load`) | 59 / 64 |
| `ds_read_b128` / `ds_write_b128` via inline asm | 66 / 72 |
| split B along N, step the waits down | 69 / 75 |
| rotate the next slice's reads into the freed slots | 70 / 76 |
| K_STEP 32 → 64, which the rotation made affordable | 73 / 78 |
| write the staged tile *after* the math, and drop the unconditional `lgkmcnt(0)` drain from `store_register_buffer_to_shared` | **74 / 79** |

(The last row and the first table are separate runs of the same build; ±1 TFLOP
run to run is normal on this part.)

The `ds_read_b128` row is worth reading twice. A shared tile reached through
`shared_allocator` is a *generic* pointer as far as the compiler is concerned,
so `*(float4*)p` compiles to `flat_load_b128` — right width, wrong pipe, and it
counts against `vmcnt` instead of `lgkmcnt`. Inline asm is not an optimization
here, it is the only way to name the instruction. Worth 7 TFLOPs on its own.

## Running it

```bash
make                 # builds tk_kernel.so
python test.py       # correctness against torch
python bench.py      # the table above
./sweep.sh           # tiling sweep, and the per-stage ablations
```

## Tuning

`BLOCK_M`, `BLOCK_N`, `K_STEP`, `DOT_SLICE`, `NUM_WARPS`, `WARP_ROWS`, `WGM`,
`N_SPLIT`, `ROTATE`, `GPREFETCH`, `WRITE_POS` are
`-D`-overridable, but only take effect with `-DHK_MULTI_CONFIG=0`, which pins the
build to exactly that tiling instead of the two-way selection above. `sweep.sh`
passes it; without it a sweep would measure the selection rule rather than the
configuration it names. `ABLATE_GLOBAL` / `ABLATE_LDS_READ` / `ABLATE_LDS_WRITE` /
`ABLATE_BARRIER` / `ABLATE_MMA` delete one pipeline
stage each — they make the answer wrong on purpose, and they are what located
the cost above. Run them with `QB_ARGS=--no-check`, or the correctness check in
`quickbench.py` reports WRONG and no timing comes out. `HK_OCC=1` in the
environment prints the runtime's real occupancy, which is the LDS-limited one
and not what the compiler's remark says.

One trap worth naming, because it cost a debugging cycle: an `s_waitcnt
lgkmcnt(N)` immediate that is too *large* does not over-wait, it fails to wait
at all. `dot_tile` takes `has_tail` as a compile-time argument for exactly this
reason — the epilogue passes an empty tail, and counting writes that were never
issued would let the last K-tile's WMMAs run on operands still in flight.

Things that were tried and did *not* work, recorded so they are not retried
blind:

1. **Double-buffering the operand tiles.** 48 VGPRs, occupancy 9 → 7, measures
   63/68 against 66/72. The zero-register version above is what replaced it —
   see "The LDS reads run underneath the WMMAs".
2. **Bigger register blocks.** Every 4×4-base-tile shape hits the 256-VGPR
   ceiling and spills, and the ones that fit are latency-bound. See the warp
   tile table above.
3. **Forcing occupancy with `__launch_bounds__`.** The second argument is min
   waves per EU in HIP, not min blocks per CU. Asking for 8 at `K_STEP=64`
   spills 21 registers and drops to 66/69. 208 VGPRs is the genuine floor for
   this tile; there is no free occupancy to take.
4. **Tuning `WGM`.** 1 / 4 / 8 / 16 measure 72/75, 73/78, 72/78, 71/76. Within
   noise between 4 and 8.

## Structure

Follows `kernels/cdna4/gemm/bf16fp32`, with the schedule redesigned for three
RDNA3 facts:

- **wave32**, and the operand mirroring above.
- **No global→LDS DMA.** `vmem-to-lds-load-insts` does not exist on gfx11, so
  every prefetched byte passes through VGPRs. The "async copy" is
  `load_global_to_register_buffer` now and `store_register_buffer_to_shared` at
  the point the buffer is needed. This is the one HipKittens pillar that does
  not survive the port, and it caps how much of the paper's ping-pong schedule
  is reachable.
- **One XCD**, so there is no chiplet swizzle — only the L2 group swizzle.
