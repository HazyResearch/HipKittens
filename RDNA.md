# HipKittens on RDNA3

A port of HipKittens to AMD's consumer / workstation architecture — gfx11
(RDNA3: RX 7900, W7900). Upstream supports CDNA3/4/5 only.

| | RDNA3 (gfx1100) |
|---|---|
| core library | `include/rdna3`, 68 files |
| unit tests | **1659 passed, 0 failed** on a W7900D |
| bf16 GEMM | **77.2 TFLOPs** peak, 77% of the measured WMMA ceiling. 107-119% of the best AMD library *in HipKittens' own layout*; 85-93% of that library when it is free to pick its own |
| distributed | fused GEMM→all-reduce / reduce-scatter, as a torch operator: **1.22-1.31x** and **1.31-1.59x** over hipBLASLt + RCCL at prefill, on 2x W7900D. [Full writeup](kernels/rdna3/distributed/README.md) |
| attention | single-GPU SDPA forward, as a torch operator and an `F.scaled_dot_product_attention` drop-in: **64-65 TFLOPs**, **2.8-3.3x** the aotriton kernel every Radeon framework actually reaches, and 4-14% ahead of a hand-written Triton FA-2 (causal: 4.4x and 3-12%). [Full writeup](kernels/rdna3/attn/README.md) |
| fp8 | not available in gfx11 hardware |

RDNA4 (gfx12) is **not** part of this; it is being added separately in
[#91](https://github.com/HazyResearch/HipKittens/pull/91). Where gfx12 differs
in a way that shaped a gfx11 decision, this document says so and marks it as
unverified — no gfx12 part was available here, and nothing in `include/rdna3`
depends on one.

## Why this is a separate tree and not a `#ifdef`

RDNA's matrix unit is WMMA, not MFMA, and the fragment layout is different
enough that sharing code would have meant an `#ifdef` in every load, store and
conversion. The repo already splits by architecture (`include/cdna3`,
`cdna4`, `cdna5`), so this follows that.

Three hardware facts drove essentially every design decision, all three
established by probe rather than by documentation:

**1. gfx11 WMMA operands are mirrored across the wave halves.** On gfx1100
wave32, a `v_wmma_f32_16x16x16_bf16` operand is `v16bf16` — 8 VGPRs — and lanes
`l` and `l+16` must hold *identical* data. A lane holds the entire K=16 vector.
Compare CDNA3's MFMA, where a lane holds 4 halves, K is split across lane
groups, and there is no redundancy at all.

The consequence is counterintuitive and it is the single most important thing
about this port: **a bf16 operand tile costs the same registers as an fp32
accumulator tile.** On RDNA3 the operand registers bound the tile shape, which
is the reverse of CDNA. It is why the GEMM cannot grow past a 128×128 block, and
why software-pipelining the operand loads measures *slower*.

(This also cost a false start. The first layout probe put a one-hot value in a
single lane without mirroring it, violating the hardware constraint, and read
back undefined behaviour that looked like a coherent — and wrong — layout.)

**2. gfx12 splits K between the halves instead.** Operands are `v8bf16`, 4
VGPRs, no redundancy: lane `l` holds `k = 8·(l/16) … +7` of row `l%16`. Half the
operand registers for the same tile — which is why the gfx11 layout below is
gfx11's and not a shared one. It also makes accumulator↔operand conversion
*more* expensive, not
less — on gfx11 one direction was a free discard because the operand halves
already held everything; on gfx12 neither half does, so both directions cross
the halves at 8 `permlanex16` each.

**3. There is no global→LDS DMA on either.** `vmem-to-lds-load-insts` is absent
on gfx11 and gfx12 alike. HipKittens' third pillar — "asynchronous loads/stores
using direct buffer loads to shared memory" — has no hardware to stand on here.
Every prefetched byte passes through VGPRs and into the wave's instruction
stream. This is why `include/rdna3` is derived from `cdna3` rather than `cdna4`:
cdna3's `global_to_shared.cuh` already staged through `float4` buffers, which is
the only path RDNA has. It also caps how much of the paper's ping-pong schedule
is reachable, and the GEMM numbers should be read with that in mind.

## Coverage

**Library** (`include/rdna3` — 68 files, the same shape as `include/cdna3`):

- Types: `rt_base` / `rt` / `st` / `st_subtile` / `sv` / `rv` / `gl`, both
  layouts, all three `rv_layout`s, `shared_allocator`, `group<N>`.
- Ops: warp and group, register and shared and global, tile and vector — maps,
  reductions, conversions, `mma_{AB,ABt,AtB,AtBt}`, and every
  `{global,shared}_to_{shared,register}` transfer.
- gfx11 WMMA (`f16`, `bf16`). gfx11 has no fp8 matrix instruction.

**Tests** (`tests/unit/rdna3` — 64 files, ported from `tests/unit/cdna4`), at
`TEST_INTENSITY=2`, on a W7900D:

```
1659 tests passed
0 tests failed
509 tests skipped (invalid template parameters)
```

**Kernels**: `kernels/rdna3/gemm/bf16fp32` (measured, see its README),
`kernels/rdna3/distributed` (measured on two GPUs, see its README),
and `kernels/rdna3/attn/fwd` (measured on one GPU, see its README).

### Not covered

- **Attention backward, paged/KV-cache attention, and fp16.** The forward
  covers causal and non-causal, MHA and GQA, head_dim 64 and 128, bf16 — which
  is inference, and only inference, for a dense model. Nothing here trains, and
  nothing here reads a block table.
- **Any end-to-end pipeline.** The attention kernel is compared against the
  backends diffusers / comfyUI / vLLM / SGLang dispatch to, at the operator
  level, on the shapes MiniMax H3 produces. No framework was run: the pod has
  no network and no H3 weights.
- **fp8 as anything but a WMMA operand.** No maps, reductions or vector ops on
  fp8 register tiles. `constants<fp8e4m3>` deliberately has no infinity, so a
  reduction over one is a compile error rather than a silently wrong answer.
- **A 64×64 warp tile in the GEMM.** The LDS→register stage is now pipelined
  (at no register cost — see below), but the wider warp tile rocBLAS uses still
  does not fit in 256 VGPRs on gfx1100. That is the last 10%.
- **Distributed beyond two GPUs.** The kernels carry no peer count and TP=4/8
  satisfy their shape constraints, but only two GPUs were free on the shared
  node, and on a ~36 GB/s PCIe fabric TP>2 is expected to lose to RCCL anyway.
- **MoE, fp8 and column-parallel** in the distributed path; and its vLLM /
  SGLang patches are written against the upstream APIs but **never run inside a
  server**.

## Performance

W7900D (gfx1100, 96 CU). bf16 GEMM, fp32 accumulate. HipKittens implements one
layout — A (m,k), B (n,k), C = A·Bᵀ, both operands K-contiguous, torch's "NT" —
so the vendor gets two columns: its best of {NN, NT, TN, TT}, and the same
library restricted to that one layout.

| shape (M×N×K) | hipBLASLt best | rocBLAS best | vendor NT | HK | HK / best | HK / NT | HK / ceiling |
|---|---|---|---|---|---|---|---|
| 4096³ | 69.7 | **82.9** | 69.7 | 74.4 | 90% | **107%** | 74% |
| 8192×8192×4096 | 64.9 | **88.8** | 64.9 | 77.2 | 87% | **119%** | 77% |
| 4096×8192×2048 | 71.3 | **88.4** | 70.0 | 75.4 | 85% | **108%** | 75% |
| 8192×4096×2048 | 69.9 | **88.6** | 69.9 | 75.3 | 85% | **108%** | 75% |
| 2048×4096×4096 | 67.8 | **84.6** | 67.8 | 73.7 | 87% | **109%** | 73% |
| 2048×2048×4096 | 63.8 | **75.8** | 57.7 | 67.4 | 89% | **117%** | 67% |
| 2048³ | 60.2 | **69.8** | 55.1 | 64.5 | 92% | **117%** | 64% |

**Three corrections to numbers this file previously carried.** The first two
made the kernel look better than it is; the third made it look worse.

*The baseline was the wrong library.* torch on ROCm defaults to hipBLASLt, and
on gfx1100 hipBLASLt is the slower of the two by up to 33% — rocBLAS wins every
shape here. Benchmarking against torch's default is benchmarking against the
library AMD has not tuned for this architecture. The honest statement is "beats
hipBLASLt everywhere", not the "wins 2 of 4" that the hipBLASLt-only table
showed.

*The denominator was unreachable.* 122.6 TFLOPs is 96 CU × 512 FLOP/clk ×
2.495 GHz boost. This card does not run a GEMM at 2.495 GHz: `rocm-smi` sampled
under load shows 2.0-2.1 GHz against a 241 W limit. Back-to-back WMMA with no
memory at all measures **100.5 TFLOPs**
([`tools/rdna-probes/wmma_peak.hip`](tools/rdna-probes/wmma_peak.hip)) — 94% of
the architectural issue rate, at 2.18 GHz and only 101 W. The reachable peak is
~104, so percentages against 122.6 understate by about 18%.

*The comparison was against a different problem.* rocBLAS's 88-90 TFLOPs
headline is reached in NN, TN or TT. Restricted to the layout this kernel
implements — NT, both operands K-contiguous — it reaches 55-70, and HipKittens
is **107-119% of the best AMD library** there, on every shape. Neither column is
the "real" one; they answer different questions, and both are now printed.

### What the vendor's layout spread actually is

rocBLAS at 8192×8192×4096 measures NN 85.7, **NT 56.0**, TN 90.2, TT 89.9.
Three of four are within 5%; the outlier is the one with *both* operands
K-contiguous, which is the one HipKittens implements. Disassembling the two
kernels rocprofv3 dispatches (unbundle the `CCOB` `.co` files with
`clang-offload-bundler --unbundle`) shows NT and TT are the *same* Tensile
kernel — `MT128x128x16`, 4 waves of 64×64, `PGR1 PLR1`, 256 VGPRs — and that the
slow one has the cleaner inner loop: 16 × `ds_load_b128` against TT's 8 ×
`ds_load_b128` + 64 × `ds_load_u16` for the same lane-bytes. The difference is
not in there.

It is global-load granularity. A K-contiguous operand tiled at K=16 is 128 rows
of 32 bytes — a quarter of a cache line each, on both operands. This kernel can
measure the penalty because `K_STEP` is a `-D`; same tiling, same layout, only
the K-tile depth moving (128×128, 8 warps, 8192×8192×4096, against
`ABLATE_GLOBAL`):

| `K_STEP` | bytes per tile row | full | no global stage | global costs |
|---|---|---|---|---|
| 16 | 32 | 59 | 81 | **22** |
| 32 | 64 | 77 | 87 | **10** |
| 64 | **128** | 78 | 83 | **5** |

and `GPREFETCH` 1→8 does not move it, so it is granularity, not latency. **The
vendor's 56 in the NT column is what that layout costs at a 16-deep K-tile, and
Tensile's gfx1100 tuning has no deeper NT entry to pick. This kernel runs a
64-deep one and pays 5 instead of 22 — that, and not a better inner loop, is the
whole of the >100% in the HK/NT column.**

The remaining gap to the vendor's *best* layout is a different thing, and it is
now accounted for exactly rather than estimated: **LDS lane-bytes**.

A gfx11 WMMA operand is mirrored across the wave halves, so lanes `l` and `l^16`
issue `ds_read_b128` against the same address. LDS charges for that duplicate in
full — `lds_peak.hip` mode `ST` reproduces the kernel's exact addressing and
measures 64.8 lane-bytes/clk/CU, i.e. 129.6 per WGP, which *is* the hardware's
128 B/clk/WGP. So per WGP per K-tile at 128×128×64: reads are 8 waves × 48
`ds_read_b128` × 512 lane-B ÷ 128 B/clk = **1536 cycles against 2048 cycles of
WMMA in the same window (75%)**, and writes add 256 more (**87% total**). The
kernel sits at 77% of the WMMA ceiling; there is no room under 87% for it to sit
much higher.

It is worth saying what it is *not*, because both alternatives look plausible
from the disassembly and both were checked. It is not instruction count: rocBLAS
issues 5.3× more LDS instructions (128 × `ds_load_u16` against 24 ×
`ds_read_b128` per 16 WMMAs) and still wins. And it is not bank conflicts:
[`tools/rdna-probes/lds_peak.hip`](tools/rdna-probes/lds_peak.hip) measures this
tree's access pattern at 64.8 bytes/clk/CU, tied with a hand-built conflict-free
reference and 4.1× the same read with the swizzle removed. The swizzle constant
is right.

Every escape from that was measured and none works. **Occupancy** is not the
lever — 2, 4 and 8 waves/SIMD measure 79, 78 and 63, because latency hiding here
comes from the software pipeline and not from resident waves. **Deeper global
prefetch** is flat at K_STEP ≥ 32. **Clocks** are not a hidden denominator: 20 s
of steady state holds 2171 MHz, the same clock the WMMA probe ran at. And
**wave64**, the one structural way to kill the mirror, is worse: gfx1100's
`__builtin_amdgcn_wmma_f32_16x16x16_bf16_w64` takes A and B as `v16i16` — 8
VGPRs, the full K=16 per lane, identical to wave32 — so 64 lanes hold 1024 halves
for a 256-element matrix, **4× replication against wave32's 2×**.

Half of the gap that *was* closeable has been closed, and how is the part worth
carrying forward. rocBLAS pays for its wider tile with a software pipeline (`PLR1`) that
costs registers; the textbook version of that here — double-buffering the
operand tiles — costs 48 VGPRs, drops occupancy from 9 waves/SIMD to 7, and
measures *slower*. The version that works costs nothing, because `lds_wait<N>()`
waits until at most N LDS ops are still outstanding and LDS retires in order.
Split the accumulator and the B operand into chunks along N, issue the whole
slice's reads at once, and step the waits down: each chunk's WMMAs then run on
top of the reads for the chunks after it. Then rotate — a B chunk's registers
are dead the moment its WMMAs have issued, so the next K-slice's reads for that
chunk go in flight right there. Register count is identical at every step (161
VGPRs before the split, after the split, after the rotation), and it is worth
65 → 70 TFLOPs at 4096³.

It also moved two other optima, which is the usual reason to redo a sweep after
a scheduling change rather than assume it composes. A deeper K-tile became worth
its staging registers (`K_STEP` 32 → 64, 70 → 73, despite occupancy 9 → 7),
because more K-slices means more rotation points. And the small-shape crossover
moved by 4×, from 256 workgroups to 64.

What is left is the tile itself: 64×64 needs 128 VGPRs of accumulator plus 64 of
operands, and the operands are already 2× larger than they would be on CDNA
because of the wave-half mirroring. There is no intermediate tile — accumulator
VGPRs are `M·N/32` regardless of shape, so square is strictly optimal among
powers of two, and 48-row tiles need `BLOCK_M=192`, which divides none of the
benchmark shapes. **That is the wall, and it is an architectural one.** The full
config sweep is in
[`kernels/rdna3/gemm/bf16fp32/README.md`](kernels/rdna3/gemm/bf16fp32/README.md).

Small shapes are a separate problem: grid quantization, not the inner loop.
`dispatch_any` compiles seven tilings and picks by workgroup count and by M.
Below about 64 workgroups even the small tiling falls off (33 TFLOPs at
512×512×4096).

The extreme of that regime — decode, M ∈ [1, 32] — is now covered by four thin
configs with split-K, and it is not a compute problem at all but a weight
streaming one: the whole `mlp_down` WMMA is ~6 µs against a 114 µs floor set by
reading the weights. Those reach 627–637 GB/s at K_local=8704 (~95% of tuned
hipBLASLt) and 329–380 at K_local=3072 (~65%); the short-K column is the
remaining gap. **Split-K is a decode-only tool** — its `[SPLIT_K][M][N]` fp32
workspace is written and read while the weight traffic is M-independent, so the
split that wins by 28% at M=8 loses by 22% at M=32. Details and the sweep are in
[`kernels/rdna3/distributed/README.md`](kernels/rdna3/distributed/README.md#62-decode-is-weight-streaming-not-a-gemv).

## Distributed

A second kernel family fuses the tensor-parallel collective into the GEMM's
epilogue: each output tile, at the moment its accumulator is complete and still
in registers, goes either to local memory or straight into its owner's inbox on
another GPU. Against `F.linear` + RCCL on two W7900D over PCIe, TP=2, Qwen3-27B
shapes — 30 shapes, every one checked elementwise before it was timed:

| | prefill 2048-8192 | decode 1-32 |
|---|---|---|
| all-reduce | **1.22-1.31x** | 1.21-1.38x (mlp_down), 0.92-1.06x (attn_out) |
| reduce-scatter | **1.31-1.59x** | n/a — M=1 has no row axis to shard |

It is packaged as `torch.ops.hk_dist` with a drop-in `HKRowParallelLinear`, one
process per rank over IPC-mapped peer memory, which is the model vLLM and SGLang
actually run. The decode `attn_out` row is a wash and the writeup separates the
two reasons rather than averaging them.

Three gfx1100 facts drove that design as much as WMMA drove the GEMM, and all
three are worth knowing before writing any multi-GPU kernel for this part:
**nothing a shader can execute invalidates L2**, so peer-written data needs a
host sync to be readable; **device-memory flags are useless as barrier flags
over PCIe**, measuring round trips in seconds; and **this node's peer bandwidth
is bimodal per process launch**, which forbids comparing any two numbers
measured in different runs. Full derivation, measurements and the register-spill
trap that silently NaNs one accumulator tile:
[`kernels/rdna3/distributed/README.md`](kernels/rdna3/distributed/README.md).

## Attention

A third kernel family: single-GPU SDPA forward, bf16 in and fp32 accumulate,
targeting [MiniMax H3](kernels/rdna3/attn/README.md#1-the-shape-that-drove-every-decision) —
a video+audio diffusion transformer, where softmax attention is over 85% of the
transformer's runtime at the sequence lengths a real clip produces (a 480p/5s
generation is 49 920 tokens).

The comparison that matters is not against a paper. Every framework that runs
attention on a Radeon — diffusers, comfyUI, vLLM, SGLang — calls
`F.scaled_dot_product_attention`, and on ROCm/gfx1100 both the `FLASH` and the
`EFFICIENT` backend land in the same aotriton kernel, `attn_fwd` (confirmed with
the profiler, not inferred). That kernel is **flat at 20.0–21.1 TFLOPs from
N=4096 to N=65536** — 21% of this chip's measured WMMA ceiling, and it does not
improve with length. B=1, H=56, D=128, all three backends in one process:

| | N=4096 | N=16384 | 49920 (480p/5s) | N=65536 |
|---|---|---|---|---|
| HipKittens | 59.8 TF | 63.9 TF | 65.0 TF | **64.8 TF** |
| vs aotriton | 2.83× | 3.07× | 3.25× | 3.21× |
| vs a hand-written Triton FA-2 | 1.04× | 1.09× | 1.13× | 1.14× |

At 63–65 TF this is 65% of the chip's WMMA ceiling and within 4% of the best
bf16 GEMM measured on it — for an operation with a softmax in the middle. The
margin over Triton widens with length as staging amortizes; the two are close to
level at N=4096. The honest summary is "2.8–3.3× the backend a Radeon actually
uses today, and 4–14% ahead of the best thing you could write in Triton."

**With causal masking the picture is now the same:** 3.9–4.4× aotriton (which
loses efficiency per surviving FLOP when the mask goes on) and **1.03–1.12× the
Triton baseline** — 49.1 / 59.5 / 61.8 / 61.7 TF at those four lengths. Causal
used to be the weak result at 1.00–1.03× Triton; three things moved it, and the
largest was a one-line change to dispatch order. Under causal the grid is a
ramp (workgroup *i* does *i*+1 KV blocks) and workgroups reach the WGPs roughly
in index order, so the machine drains with a tail of the longest ones.
Reversing the q index is longest-processing-time-first and is worth **+3.6%**,
at zero register cost and provably inert for non-causal.

Both tables are a correction of earlier ones that had non-causal at 59–60 TF and
causal *behind* Triton. The kernel was leaving ~5% on the floor to an occupancy
cliff that only the causal instantiation fell off — gfx1100's VGPR allocation
granule is 24, so 6 waves/SIMD needs ≤ 240 registers, and the tiling had been
chosen by a sweep that only ever timed the non-causal build. Finding it also
meant discarding a set of A/B measurements taken across separate processes,
which on this node invents differences of 13–16%; the
[writeup](kernels/rdna3/attn/README.md#the-sweep-measured-the-wrong-instantiation)
describes both mistakes, since both are easy to repeat.

The second general lesson from tuning this, which cost two wrong predictions: an
**ablation share measures work deleted, not time recoverable.** Staging is 18%
of the runtime by ablation, and halving the staged bytes returned ~1%, because
the co-resident workgroup on the same WGP was already covering it. The
prediction from the same table that *did* pay out was the softmax line — most
of which is one rescale multiply over all 64 accumulator registers, on every KV
block, that is exactly a no-op whenever the running max did not grow. Guarding
it with an exact equality test (not a tolerance — the output is bit-identical)
returned ~2%.

Almost none of this is a transcription of the CUDA or CDNA flash attention.
gfx11's WMMA fragment layouts are different enough that the *shape of the
algorithm* changes: **S is computed transposed** (Sᵀ = K·Qᵀ), because in the
fp32 `col` accumulator the kv axis is the cheap element axis and the q axis is
the lane axis, so online-softmax's max and sum over kv become eight in-lane
steps plus one `permlanex16` instead of a four-step butterfly; **O is
accumulated transposed** for the same reason; and **V is transposed on the way
into LDS**, because only `row`-layout bf16 operands reach the vectorized
`ds_read_b128` path and the `col` fallback costs 8× the instructions. Each of
those falls out of a table in the library, not out of a choice. The derivation
is [§2](kernels/rdna3/attn/README.md#2-the-derivation).

Coverage is causal and non-causal, MHA and GQA, head_dim 64 and 128, sequence
lengths that divide nothing — 24 shapes checked elementwise against an fp32
reference. It ships as `torch.ops.hk_attn` plus a drop-in whose signature
matches `F.scaled_dot_product_attention` exactly and which forwards to torch for
anything it cannot take (masks, dropout, fp16, cross-attention, backward), so a
framework integration is one import. Full writeup, including the tiling sweep
and the per-stage ablation:
[`kernels/rdna3/attn/README.md`](kernels/rdna3/attn/README.md).

## Notes for anyone continuing this

Things that cost time and are written down so they cost it only once:

- **`s_waitcnt` carries no register dependence, so `lds_wait<0>()` does not
  keep a `v_wmma` below it.** This is the most expensive thing found in the
  whole port. `lds_wait` is a bare `asm volatile("s_waitcnt lgkmcnt(0)")` with
  no operands; the scheduler is free to hoist the consuming WMMA above it, and
  then the fragment is read while the `ds_read` filling it is still in flight.
  The symptom is a result that is wrong *sometimes*, with `ScratchSize` 0 and
  nothing in the ISA looking out of place. The fix is a compiler-level data
  dependence — `lds_bind` / `lds_wait_for` in `include/rdna3`, an empty
  `asm volatile` tying the wait to the fragment registers — and it costs zero
  instructions. It was silently corrupting the GEMM here too. Derivation and
  the ISA diff: [`kernels/rdna3/attn/README.md#4`](kernels/rdna3/attn/README.md#4-the-bug-that-was-not-in-this-kernel).
- **`s_waitcnt` is split on gfx12** into `s_wait_dscnt` / `s_wait_loadcnt` /
  `s_wait_storecnt` / `s_wait_kmcnt`. The trap is that the combined legacy
  `s_waitcnt lgkmcnt(0)` still *assembles* for gfx1201 — LLVM just never emits
  it — so inline asm inherited from a CDNA tree looks fine and silently waits on
  the wrong thing.
- **Inline asm for LDS is mandatory, not stylistic.** A `shared_allocator` tile
  is a generic pointer, so `*(float4*)p` emits `flat_load_b128` rather than
  `ds_load_b128`. Right width, wrong pipe. Worth 7 TFLOPs in the GEMM.
- **The V# buffer descriptor word is `0x31004000`** for gfx11 *and* gfx12, not
  CDNA's `0x110000` — gfx10+ replaced `DATA_FORMAT`/`NUM_FORMAT` with `FORMAT`.
- **fp8 on gfx12 is OCP, on gfx942 it is fnuz**, and HIP gates them by target.
  The fnuz types are still declared on gfx12, so a typedef inherited from the
  CDNA tree compiles until the first thing instantiates it.
- **Benchmark against rocBLAS, not against torch's default.** torch on ROCm
  picks hipBLASLt, and on gfx1100 that is the library with the thinner Tensile
  tuning — rocBLAS is up to 33% faster on the same shape. Switch with
  `torch.backends.cuda.preferred_blas_library("cublas")`, and take best-of-four
  transposes while you are there: at 8192×8192×4096 rocBLAS measures NN 85.7,
  NT 56.0, TN 90.2, TT 89.9 — a 1.6× spread, all of it in the single outlier.
- **Do not use the data-sheet peak as a denominator.** Measure the ceiling with
  `tools/rdna-probes/wmma_peak.hip` and sample `rocm-smi -c -P` during the run.
  This card is clock- and power-limited well below boost under any real load.
- **The layout probe is the ground truth**, not the ISA doc:
  [`tools/rdna-probes/`](tools/rdna-probes/), and `wmma_layout.hip` in
  particular, which has a `PROBE_GFX12_W32` variant already written for whenever
  a gfx12 card turns up. Mirror the one-hot across `l` and `l^16` on gfx11 or
  you will measure undefined behaviour.

Per-area detail lives next to the code: `include/rdna3/ops/warp/register/tile/
conversions.cuh` opens with the layout derivation the whole port rests on, and
each of the five READMEs (`tests/unit/rdna{3,4}`, `kernels/rdna{3,4}/gemm/
bf16fp32`, `kernels/rdna3/distributed`) says what is verified and what is not.
The distributed one carries its own set of these notes, for the memory model
rather than the matrix unit.
