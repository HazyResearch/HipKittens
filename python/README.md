# hk — HipKittens kernels in Python, over an MLIR dialect

Write Radeon kernels in Python, get HipKittens C++, get a `.so`, call it from
torch. The point is not convenience: it is that three classes of *silently
wrong* kernel on RDNA3 stop being expressible.

```python
import hk

@hk.kernel(grid=lambda p: (hk.cdiv(p.o.cols, 64), hk.cdiv(p.o.rows, 16), p.o.batch))
def add(a: hk.Tensor[hk.bf16], b: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]):
    idx = hk.tile_coord(hk.arch.block_idx.z, 0, hk.arch.block_idx.y,
                        hk.arch.block_idx.x)
    t = hk.RT(hk.bf16, 16, 64)
    o[idx] = a[idx].load(t) + b[idx].load(t)

add(a, b, out)          # traces, verifies, translates, compiles (once), launches
```

Three entry points, in increasing order of what they need:

| call | needs | does |
|---|---|---|
| `k.trace()` | nothing | runs the body once into an hk MLIR module and runs the passes |
| `k.source()` | nothing | the HipKittens C++ translated from that module |
| `k.build()` | hipcc | compiles and **gates on the resource remarks** |
| `k(*tensors)` | a Radeon + torch | launches |

`build()` needing no GPU is deliberate: the failure most likely to bite —
generated code that compiles but spills — is caught on a laptop.

## Why this exists

`include/rdna3` is 9494 lines of verified C++ hitting 77.2 TFLOPs on GEMM and
64–65 on attention. Using it meant writing C++: `kernels/rdna3/attn/fwd/attn.cpp`
is 1237 lines plus 20 `#define` knobs, and retuning meant editing macros,
`make clean`, rebuild, remeasure. vLLM and SGLang could not touch it.

But the real argument is the three traps, each of which cost days:

1. **`s_waitcnt` carries no register dependence.** The compiler will hoist a
   WMMA above the wait that was supposed to guard it. Results are randomly
   wrong, `ScratchSize` is 0, and nothing is reported. The translation binds
   the fragment with a volatile asm after every wait — you cannot forget to.
2. **A VGPR spill is silent corruption, not slowness**, because this library
   hand-places `s_waitcnt` and scratch traffic reorders against it.
   `k.build()` parses `-Rpass-analysis=kernel-resource-usage` and *refuses to
   return* a spilling kernel.
3. **Only row-layout 16-bit operands reach `ds_read_b128`**; column layout
   degrades to 16 scalar `ds_read_u16`, eight times the instructions. The
   passes flag it while tracing, not at benchmark time.

Plus the arch facts that used to live in `RDNA.md`, kernel comments and
notebooks, now data: `target/gfx1100.py` for Python and
`mlir/lib/Target/Target.cpp` for the passes, with a test that the two tables
agree field by field — the VGPR granule is 24, so occupancy 6 needs ≤240
registers, not ≤256.

An op that the library does not implement for the requested dtype is the same
kind of late failure, so it is caught the same way: `gelu` on a bf16 tile
raises while tracing rather than producing `undefined hidden symbol` at the end
of a 30-second compile.

## The language

A kernel is a Python function. Positional parameters are tensors
(`hk.Tensor[dtype]`, 4-D `(batch, depth, rows, cols)`); keyword-only parameters
are constexprs, baked into the source and ordinary Python values in the body,
so `range(D // 16)` unrolls and `if KIND == "softmax"` picks a branch at trace
time.

```python
T = hk.RT(hk.fp32, 16, 64)

@hk.kernel(warps=4, grid=lambda p: (1, hk.cdiv(p.x.rows, 16), p.x.batch))
def row_sumsq(x: hk.Tensor[hk.bf16], o: hk.Tensor[hk.fp32]):
    row = hk.arch.block_idx.y * 16
    acc = hk.zeros(hk.col_vec(T))
    for cb in range(hk.arch.warp_id(), hk.cdiv(x.cols, 64), 4):
        v = x.at[row, cb * 64].load(T)       # global -> registers, bf16 -> fp32
        acc += (v * v).sum(axis=1)           # one row_sum, accumulated in place
    o.at[row, hk.arch.warp_id()] = acc
```

* **Types are values you build**: `hk.RT(dtype, rows, cols, layout)` a register
  tile, `hk.ST` an LDS tile, `hk.col_vec(t)` / `hk.row_vec(t)` the vectors a
  reduction of `t` gives, `hk.SV` an LDS vector.
* **Device values are objects with operators.** `a + b`, `t * v` (a vector
  broadcasts along the axis it came off), `t.exp()`, `t.to(hk.bf16)`,
  `t.sum(axis=1)`, `acc += a @ b.T` (one WMMA into `acc`; the variant follows
  from the transposes), `-t`, `hk.min(a, b)`; scalars are `Int`/`Bool` with
  Python arithmetic and comparisons.
* **Memory is indexed.** `x[coord]`, `x.at[r, c]` (element coordinate) and
  `x.tile[r, c]` are views; `.load(type)`, `.store(value)`, `x[idx] = value`,
  `.load_scalar()`. LDS tiles come from `hk.lds.alloc`, and `s[i]`,
  `s.subtile(...)`, `s.load(type, wait=False)`, `s.granules()` follow.
* **Tiles are buffers.** An operation either makes a new tile or writes an
  existing one — `+=`, `t.exp_()`, `into=` — which is how include/rdna3 spells
  everything and what keeps a loop's registers the same registers.
* **Control flow is Python.** The body is rewritten at the AST level, as CuTe
  DSL does: a `for` over a device bound, an `if` on a device condition and a
  `while` on a device test become `scf.for` / `scf.if` / `scf.while`. A scalar
  the loop or branch rebinds is carried through it (iter_args, branch
  results); rebinding a *tile* inside a runtime loop is refused, with the
  in-place spelling in the message. `hk.range(n, unroll=4)` asks for a runtime
  loop with a pragma, `hk.range_constexpr` insists on unrolling, and
  `hk.scope()` / `hk.if_(c, opaque=True)` keep their register-allocator meaning.
* **The rest has namespaces**: `hk.arch` (block and warp ids, `barrier`,
  `sched_barrier`, `setprio`, `vm_wait`, `wg_arrive`), `hk.lds` (`alloc`,
  `wait_for`), `hk.math` (elementwise functions, and `add`/`mul`/... with
  `into=`), `hk.group(n)` (workgroup loads, stores and the staging pipeline),
  `hk.collective` (`cross_warp`, `fold_rows`). `@hk.func` marks a helper that
  is traced inline, with its control flow rewritten too.

A device value used where Python needs a number — a tile extent, a list index,
`range()` outside a traced function — raises at that line, saying which of
those it was.

## The IR

`k.trace()` is an MLIR module in the `hk` dialect (`mlir/`), and
`k.trace().dump()` prints it with the names the C++ will use:

```mlir
func.func @row_sumsq(%x: !hk.gl<bf16> {hk.name = "x"}, %o: !hk.gl<f32> {hk.name = "o"})
    attributes {hk.arch = "gfx1100", hk.kernel, hk.warps = 4 : i64, hk.lds_bytes = 0 : i64, ...} {
  %warp_id = hk.warp_id
  %bid_y = hk.block_idx y
  %mul = arith.muli %bid_y, %c_1 : i32
  %zv = hk.zero -> !hk.rv<f32, 16x64, row, col>
  %cols = hk.extent %x cols : !hk.gl<bf16>
  %cdiv = arith.ceildivsi %cols, %c_0 : i32
  scf.for %cb = %warp_id to %cdiv step %c4_i32 : i32 {
    %idx = hk.coord %c, %c, %mul, %mul_2 : !hk.coord<element>
    %ld = hk.load %x[%idx] : !hk.gl<bf16>, !hk.coord<element> -> !hk.rt<f32, 16x64, row>
    %mul_4 = hk.mul %ld, %ld : !hk.rt<f32, 16x64, row>, !hk.rt<f32, 16x64, row> -> !hk.rt<f32, 16x64, row>
    hk.row_sum %mul_4 into %zv {accumulate} : !hk.rt<f32, 16x64, row> into !hk.rv<f32, 16x64, row, col>
  } {hk.iv_name = "cb"}
  ...
}
func.func @row_sumsq__grid(...) -> (i32, i32, i32) attributes {hk.grid_of = @row_sumsq}
```

* **Types**: `!hk.rt<f32, 16x64, row>` register tile, `!hk.st<bf16, 128x64, 2>`
  LDS tile (a double buffer), `!hk.rv<..., col>` register vector,
  `!hk.sv`, `!hk.gl<bf16>` global, `!hk.coord<tile|element>`,
  `!hk.granules`, `!hk.stage`. All print in full, so a dump reads on its own.
* **Ops**, about seventy, each with an optional destination: `hk.add %a, %b`
  makes a tile, `hk.add %a, %b into %d` writes one. Scalars are `arith`,
  control flow is `scf`, a kernel is a `func.func` with `hk.*` attributes, and
  the launch grid is a second, host-side function of the tensors.
* **Passes**, in C++ (`mlir/lib/Dialect/HK/Transforms`), run by `k.trace()` as
  `hk-kernel-pipeline`: `hk-verify` (op legality, dtype support, no workgroup
  operation inside a branch or a runtime while), `hk-lds-alloc` (offsets, the
  64 KB ceiling), `hk-lds-pipeline` (every `lgkmcnt` derived, every
  asynchronously loaded tile proven waited for before it is read, loops walked
  until they reach a steady state) and `hk-check-budgets` (the register
  estimate against the occupancy the kernel asked for). Their errors come out
  of `k.trace()` as `hk.VerifyError`, ending `at file:line in function` —
  the kernel author's line.
* **Translation** (`mlir/lib/Target/Cpp/HKToCpp.cpp`): hk + scf + arith to
  HipKittens C++ — named tile aliases, the value names of the dump, a
  provenance header — because it is the artifact you take to the disassembler
  when something spills. `codegen/scaffold.py` adds the module boundary
  (pybind, a `TORCH_LIBRARY` registration, or none) around the identical body.

The C++ for the kernel above:

```cpp
    const int cdiv = (cols + c1 - 1) / c1;
    for (int cb = warp_id; cb < cdiv; cb += 4) {
        const int mul1 = cb * c1;
        const kittens::coord<> idx(c, c, mul, mul1);
        rt_fp32_16x64_row ld;
        kittens::load(ld, g.x, idx);
        rt_fp32_16x64_row mul2;
        kittens::mul(mul2, ld, ld);
        kittens::row_sum(zv, mul2, zv);
    }
```

`hk-opt` and `hk-translate` run the same passes and translation on a `.mlir`
file, which is how the lit tests in `mlir/test` are written.

## Building

The backend is the stock hipcc compiling the translated C++ against
`include/rdna3`, so nothing at run time needs LLVM. The dialect does, once:

```bash
scripts/build_mlir.sh       # MLIR 23.1.3 only, X86 target, Python bindings (~20 min on 64 cores)
scripts/build_hk_mlir.sh    # the dialect, passes, translation and hk._mlir (~1 min)
```

`build_mlir.sh` takes `LLVM_TAG`, `PYTHON`, `HK_LLVM_ROOT`, `JOBS` and
`LINK_JOBS`; keep `JOBS` × ~2 GB under the memory limit (a 48 Gi pod is
`JOBS=6 LINK_JOBS=1`). The bindings land in `python/hk/_mlir`, about 36 MB.

## Layout

```
hk/dsl/         the language: @kernel / @func, values and operators, the AST
                rewrite, control flow, hk.arch / lds / math / group / collective
hk/types.py     dtypes and the tile / vector / tensor types a kernel is written in
hk/compiler/    the MLIR context, the pass pipeline, the translation entry point
hk/_mlir/       the built bindings (hk dialect + MLIR core), not checked in
hk/target/      gfx1100 / gfx1201 facts, as data
hk/codegen/     the module boundary around a translated kernel (scaffold.py)
hk/runtime/     arch detection, hipcc + content-hash cache, the resource gate,
                and the torch.ops registration (torch_ext.py)
hk/autotune/    schedule search: build the space, keep what passes the gate, time
hk/tuned/       the schedules that search picked, and the measured routing
                tables, as shipped data
hk/ops/         kernels written in the DSL, plus the SDPA drop-in (sdpa.py)
hk/integration/ patching a running vLLM or SGLang; engine.py is what every
                patch shares (op, routes, counters, CUDA-graph contract)
mlir/           the hk dialect (ODS), its passes, the C++ translation,
                hk-opt / hk-translate, lit tests
```

## Caching

Key = hipcc version + a content hash of `include/rdna3` + the full flag list +
the dialect version + the generated source. A build lands in
`~/.cache/hk/<key>/` via a rename, so a killed compile never leaves an
importable half-written `.so`. Failed builds are kept as `failed-<key>/` with
their source and log.

The thresholds (`max_vgprs`, `min_occupancy`) are *not* in the key — they are
re-applied on every cache hit, so a kernel with a strict occupancy requirement
is not let through by whoever happened to compile the same bytes first.

First compile of a small kernel is ~5 s; an attention kernel is 20–40 s.

`python3 -m hk.runtime.warm -j 32` fills the cache in parallel (19x on 24
kernels), and `python3 -m hk.autotune --aot` does the same for every schedule a
tuning record can reach. Those two are how the serving path ends up with no
hipcc in it.

## Tests

Four tiers, by what they need. The first three run anywhere:

```bash
ninja -C $HK_MLIR_BUILD check-hk                     # lit: the dialect, passes, translation
python3 -m pytest tests/hk/ir tests/hk/codegen -q    # no GPU, no torch
```

| tier | needs | what it catches |
|---|---|---|
| `mlir/test/` | the MLIR build | op verifiers, pass results and errors, the C++ for hand-written IR |
| `tests/hk/ir/` | hk._mlir | tracing, the language, pass errors from Python, generated C++ |
| `tests/hk/codegen/` | hipcc | compiles; spill/occupancy gates fire |
| `tests/hk/gpu/` | Radeon + torch | numerics vs torch |

A tier that cannot run is skipped with a reason, never silently passed.

## Status

The sections below are a record, in the order the work happened, and they name
the API of their time: before the last section the IR was Python and kernels
were written as flat calls (`hk.load(x, idx, ty, bounded=True)` is now
`x[idx].load(ty, bounded=True)`). The kernels themselves have all been ported;
the numbers are as measured then, and the last section re-measures the ones
that matter against the port.

**Phases 1 and 2 are done.** End to end: Python → IR → C++ → hipcc → `.so` →
torch. 291 shipped kernels — elementwise (5 binary × 3 dtypes plus the unary ops
the library specialises), RMSNorm / LayerNorm / softmax, RoPE, SiLU-mul, and
per-row int8 quantize/dequantize. Every one builds with `scratch=0 spill=0`
(a hard gate, not a check) and matches torch elementwise on a W7900D, including
shapes that do not divide the tile.

Phase 2's gate was "do not require a win". It won anyway: against
`torch.compile` on a W7900D, 13 of 15 cases are faster and the other two
(`quantize 16384x4096` and `layernorm 16384x5120`) land within 1% behind --
which is inside the same noise band Phase 5 later had to make explicit, and so
is reported as a tie rather than as a win for either side.

```
case                        hk      compile    eager
rmsnorm   4096x4096      0.060 ms   0.093     0.418
rmsnorm  16384x5120      0.486      0.775     2.823
layernorm 4096x4096      0.061      0.081     0.063
softmax   8192x16384     0.707      0.874     0.738
silu_mul  8192x11008     0.761      0.763     1.218
rope     128x1024x128    0.076      0.107     0.583
quantize 16384x5120      0.367      0.369     6.118
```

Getting there was mostly *not* kernel work. A 4096x4096 quantize is 60 µs of
GPU; the Python wrapper around it was 13.9 µs. Three rounds — memoising the
launcher on the kwargs as passed, caching the plan on the `torch.dtype` object
rather than `str(dtype)`, and a pybind converter that interns the attribute
names and reads a tensor in five Python calls (`pyutils/hk_bind.cuh`) — took the
worst wrapper to 9.9 µs and `rmsnorm` from 11.3 to 6.6. See
`tools/hk-bench/host_overhead.py`, which measures host time, queue depth and
device time separately so the three cannot be confused for each other.

**Phase 3 is done: the generated GEMM matches the handwritten one.** The hard
gate was three numbers at once -- same `ScratchSize`, same occupancy, and
77.2 TFLOPs. On the pod's ROCm 7.2.4, `hk.ops.matmul` and
`kernels/rdna3/gemm/bf16fp32/gemm.cpp`'s `config<128,128,64,16,8,4,8,4>` compile
to *the same* 209 VGPRs, 0 scratch, 0 spill, 7 waves/SIMD, 64 KB LDS, and run
within noise of each other:

```
shape                 variant         ms    TFLOPs   vs C++
4096x4096x4096        hk (DSL)     1.831      75.0    1.00x
4096x4096x4096        C++          1.833      75.0    1.00x
4096x4096x4096        rocBLAS      1.975      69.6    0.93x

8192x8192x4096        hk (DSL)     7.055      77.9    1.00x
8192x8192x4096        C++          7.047      78.0    1.00x
8192x8192x4096        rocBLAS      7.661      71.8    0.92x
```

Same-process interleaved, bit-identity checked before timing
(`tools/hk-bench/gemm_ab.py`). Disassembled, the two kernels issue the identical
64 WMMAs, 96 `ds_read`s, 16 `ds_write`s and 24 global loads; the generated one
has 52 *fewer* scalar and vector ALU instructions.

It did not start there -- the first measurement was 2.4% and 3.0% behind, and
the whole of it was one instruction. `hk.ops.barrier` is documented as ordering
execution and not memory, but the emitter was lowering it to `__syncthreads()`,
which promises to order global memory too and on gfx11 pays for that with a
`buffer_gl0_inv`: a per-WGP vector L0 flush, once per K-tile, throwing away the
B-tile lines the other workgroup on the WGP was about to read. Emitting
`__builtin_amdgcn_s_barrier()` instead -- which is what the handwritten kernel
calls, and what the op always meant -- closed the entire gap. The LDS side is
not weakened by this: `lds_pipeline` proves it separately, either by finding a
wait that already retired the queue or by making the author write
`barrier(drain=True)`.

**Phase 4 is done: the generated attention kernel matches the handwritten one,
and beats it under causal.** The gate was 64-65 TFLOPs plus elementwise
agreement with `F.scaled_dot_product_attention` across every H3 shape, a
non-dividing N, B=2, GQA and causal. Same process, interleaved, both kernels
loaded at once (`hk/ops/_attn_ab.py`):

```
shape           dsl ms   dsl TF   c++ ms   c++ TF  dsl/c++
N=4096           7.880     61.0    7.895     60.9   1.002x
N=8192          30.528     63.0   30.527     63.0   1.000x
N=16384        120.334     64.0  120.300     64.0   1.000x
N=32768        475.309     64.8  475.132     64.8   1.000x
480p/5s       1100.228     64.9 1099.149     65.0   0.999x
causal 8192     16.628     57.9   16.754     57.4   1.008x
```

Against torch SDPA on the same chip that is 2.9-3.2x, and 4.0x causal. 31
numerical cases pass (`tests/hk/gpu/test_attn.py`). The generated kernel also
runs at **occupancy 6 against the handwritten kernel's 5** -- 224 VGPRs versus
247 -- which is the one place the IR is structurally ahead rather than level.

Three findings came out of getting there, and all three now live in the library
or the IR rather than in a kernel:

* **`warpid()` costs a divergent branch.** It is `threadIdx.x >> 5`, which the
  compiler keeps in a VGPR because it cannot see that a wave is 32 consecutive
  threads. Anything derived from it inherits the VGPR, so a comparison on it
  becomes `v_cmpx` with exec save and restore rather than one `s_cbranch`, and
  every value the branch needs is pinned in a vector register across it. The
  causal kernel spilled 90 VGPRs that way and none with
  `kittens::warpid_uniform()`.

* **An index inside a tile mask costs one register per element.** Written the
  obvious way -- ask `rt_base_coord` for (row, col) and evaluate the predicate
  -- a mask holds 8 loop-invariant registers, which LLVM hoists into the
  preheader and spills. Writing it as `dbase >= thr` with a scalar threshold is
  *worse*: LLVM reassociates straight back. The fix is to fold the index into
  the lane term so the per-element part is an integer literal after unrolling,
  which is one register for the whole tile. `detail::diag_fill` and
  `detail::axis_fill` in `conversions.cuh` now do this for all eight fills.

* **Branches in the hot region are a register-allocation lever, and the trade
  is asymmetric.** A mask that fires on one block in the whole loop (the
  backed-up last KV block) is worth a uniform branch: 224 VGPRs and no spill
  with it, 116 B/lane without. A mask that fires on nearly every block (causal)
  is worth no branch at all: 226 and none without it, 156 B/lane with it.

The last 2% was one thing the IR could not say. The handwritten kernel skips
the online-softmax rescale when the running max did not grow -- 64 `v_mul_f32`
per KV block at head_dim 128, against that block's 32 WMMAs -- and that needs a
predicate that reduces *out of* the register file and into control flow, which
no elementwise op can produce. `hk.s_any_ne` is that op, and it closed the gap
exactly. Causal then went ahead on longest-processing-time-first dispatch: the
work per workgroup is a ramp, so reversing the q index starts the long ones
while there is still short work to fill the slots they free.

**Phase 5 is done: `hk.autotune` replaces `sweep.sh`, and most of what it
measured turned out not to be worth acting on.** A `Tuner` is a kernel factory, a schedule
space and a record. Searching it is two stages, and the first one never touches
the GPU:

```
python3 -m hk.autotune --list
python3 -m hk.autotune --tune gemm_bf16 -j 32      # search
python3 -m hk.autotune --ship                      # promote what is real
python3 -m hk.autotune --aot  -j 32                # prebuild it all
```

Stage 1 builds every candidate through `runtime.warm` -- 32 hipcc invocations
at a time -- and three kinds of candidate die there for free: the factory's own
arithmetic refuses a tiling that does not divide, the verifier refuses an
illegal layout, and the resource gate refuses anything that spills. Of
attention's 32 schedules, 26 never reach the GPU. Stage 2 times what is left in
*one process*, interleaved, reversing the order on odd rounds and scoring each
candidate by its minimum, because cross-process A/B on this chip manufactures
13-16% differences out of nothing.

The production path never searches. `Tuner.kernel(key)` is a dict probe and a
memoised factory call, so a record costs nothing per launch; a missing, stale
or foreign record costs performance and never correctness.

`min` over a column of noisy numbers always names a winner, so a tuner that
ships its winner unconditionally ships noise as a finding. `tune()` therefore
also records `default_ms`, and a record changes nothing unless it beat the
default by `MIN_GAIN` (1%). The bar is applied at `--ship` *and* on the lookup
path, so a record measured here is held to the same standard as one that
arrived in a wheel; `HK_AUTOTUNE_MIN_GAIN` moves it and `HK_AUTOTUNE=0` ignores
records entirely.

Out of 16 keys across five tuners, four clear it:

```
tuner                  key                 winner                  gain
gemm_bf16              4096x4096x4096      k_step32 wgm16 pos1    +1.2%
attn_fwd_d128          n16384              qk1 pv2                +1.1%
attn_fwd_d128          n65536              qk1 pv2                +1.1%
attn_fwd_d128_causal   n4096               qk1 pv1                +2.8%
```

The other twelve stay on the default, including every `gemm_bf16` shape but
one and all six `attn_fwd_d64` keys. That is the result and not a gap: a space
whose winners are inside the noise is a space whose default was already right.

The bar had to be checked against itself before any of this was believable.
The first attention sweep reported gains of 0.10-0.21% with four *different*
winners across three lengths -- a coin. A second sweep on the same machine
reported 0.26% to 2.8%. Two runs of one tuner disagreeing by 10x is not a
result, so the three keys above were re-measured three times each at
`--rounds 5`:

```
key                        repeat 1        repeat 2        repeat 3
d128 n16384          qk1 pv2 +1.04%  qk1 pv2 +0.99%  qk1 pv2 +1.00%
d128 n65536          qk1 pv2 +1.12%  qk1 pv2 +1.12%  qk1 pv2 +1.09%
d128_causal n4096    qk2 pv1 +1.81%  qk1 pv4 +2.13%  qk1 pv2 +1.29%
```

The first two reproduce exactly -- same schedule, same gain to within 0.05 --
so `qk1 pv2` is a real if small win at head_dim 128, and the original sweep's
0.1% was the measurement that was wrong. The third says something different
and more useful: at `n4096` causal the gain is reproducible but the *winner* is
not, because the default (`qk2 pv2`) is specifically the slow one there and all
three alternatives beat it by about the same amount. The shipped record names
one of them, and this paragraph is what stops the next reader from concluding
that `qk1 pv1` is special.

`--aot` prebuilds every schedule production can reach -- the recorded ones *and*
the default, since the default is what an unrecorded key gets -- into the
content-addressed cache, which is what takes hipcc's 20-40 s off the first
request. `pip install -e .` works from the repo root; `hk/tuned/*.json` and
`hk/include/**` are package data, because the generated C++ `#include`s the
headers at JIT time.

**Phase 6 is done: a DSL kernel becomes a `torch.ops.hk.*` op, and the SDPA
entry point is a drop-in.**

```python
import hk
from hk.ops import sdpa

sdpa.patch()                 # F.scaled_dot_product_attention -> hk, where it fits
torch.ops.hk.attn_fwd_d128(q, k, v, o)
```

`hk.torch_op(kernel)` emits a third scaffold beside `pybind` and `bare`, builds
it with plain `hipcc -shared -fPIC` and loads it with `torch.ops.load_library`
-- not `cpp_extension.load`, which would hipify source that is already HIP. The
schema is derived from the IR rather than from the author: `written_tensors`
walks the ops that write memory the caller can see (`store_global`,
`store_scalar`, `group_store`, and deliberately not `store_frag`, which writes
a shared tile), and those tensors become `Tensor(a!)` out-parameters. A schema
that forgets to mark an output mutable compiles, loads, runs, and is wrong only
under `torch.compile`'s functionalization -- which is exactly the kind of bug a
derivation removes and a convention does not.

It costs nothing. Same process, interleaved, bit-identity checked before timing
(`tools/hk-bench/sdpa_paths.py`); `aotriton` is the backend torch's own SDPA
picks on this chip, and is the thing being replaced:

```
shape             path             ms      TF   host us   vs pybind
1x56x4096x128     pybind        7.875    61.1       3.7       1.00x
                  torch.ops     7.857    61.2       3.5       1.00x
                  sdpa drop-in  7.872    61.1       9.4       1.00x
                  aotriton     22.612    21.3      11.5       0.35x

1x56x16384x128    pybind      120.204    64.0       3.8       1.00x
                  torch.ops   120.449    63.9       3.1       1.00x
                  sdpa drop-in 120.466   63.9       9.5       1.00x
                  aotriton    368.805    20.9      11.7       0.33x

1x32x8192x128 c   pybind        9.769    56.3       3.3       1.00x
                  torch.ops     9.525    57.7       2.9       1.03x
                  sdpa drop-in  9.599    57.3       9.6       1.02x
                  aotriton     40.467    13.6      15.0       0.24x
```

Device time is identical across the three hk paths, which is what the shared
kernel body predicts. The surprise is the host column: torch's dispatcher is
*cheaper* than the pybind converter it replaces (3.1 us against 3.8), so the
registration is not a tax even on a 60 us elementwise op. The drop-in's 9.4 us
is the entry point's own work -- the support check and the output allocation --
and is the price of being callable as `F.scaled_dot_product_attention`.

Three details are load-bearing:

* **The kernel launches on torch's stream.** `cpp.py` now emits
  `launch_on(const globals&, hipStream_t)` with `launch(g)` as the default-stream
  wrapper, and the torch impl passes `getCurrentHIPStream()`. A default-stream
  launch breaks CUDA-graph capture and overlapped copies, silently.
* **`TORCH_LIBRARY_FRAGMENT`, not `TORCH_LIBRARY`.** Every kernel is its own
  `.so` and they share one namespace; the non-fragment form claims the namespace
  exclusively, and the second `.so` to load throws.
* **A `Meta` implementation** makes the op traceable by `torch.compile`. It is a
  no-op, because the op allocates nothing -- which is also what a serving
  framework wants.

`hk.integration` patches a running vLLM or SGLang. It rebinds **by identity**,
not by name: `_common.rebind` walks `sys.modules` under a prefix and replaces
every attribute that *is* `torch.nn.functional.scaled_dot_product_attention`,
which catches the modules that did `from torch.nn.functional import ...` at
import time and cannot be fooled by a same-named impostor. `python3 -m
hk.integration --warm -j 32` fills the cache for the torch half specifically:
libtorch's include and ABI flags are part of the cache key, so a `--aot` pass
built for pybind does not spare a server its first attention compile.

Two things are deliberately *not* patched, and the reason is the same both
times -- the patch would be slower than what it replaces:

* **RMSNorm.** Every framework's RMSNorm applies a learned weight;
  `hk.ops.rmsnorm` is unweighted, so the patch would add an elementwise pass to
  save one.
* **SiLU-mul.** `SiluAndMul` slices one `(..., 2d)` tensor into two
  non-contiguous halves and `hk.ops.silu_mul` has no stride support, so the
  patch would add two copies to save one pass.

Both are written down as missing *kernel* work rather than attempted as
patches.

**What the patch can and cannot reach in vLLM**, read against vLLM at 1dfe9fb
and checked at runtime by `hk.integration.vllm.probe()`:

* **The decoder's attention is out of reach, structurally.** vLLM V1 keeps KV
  in a paged cache and dispatches through `AttentionImpl.forward(..., kv_cache,
  block_table, ...)` -- `RocmAttentionImpl` or `TritonAttentionImpl` on this
  chip. There is no dense `(B, H, N, D)` tensor on that path for an SDPA
  drop-in to intercept. Reaching it needs a *paged* kernel registered through
  `vllm.v1.attention.backends.registry.register_backend`, which is kernel work
  `hk.ops.attn` has not done.
* **The vision tower's attention is in reach, exactly.**
  `mm_encoder_attention._forward_sdpa` -> `torch.ops.vllm.torch_sdpa_wrapper`
  -> `apply_sdpa` is a literal `F.scaled_dot_product_attention` on a dense
  `(B, H, N, D)` tensor, and patching `F` reaches it because the attribute is
  resolved on the module at call time.
* **It is not the default on RDNA3.** `RocmPlatform.get_vit_attn_backend`
  prefers the Triton flash-attention whenever `flash_attn_triton_available()`;
  `--mm-encoder-attn-backend TORCH_SDPA` selects it deliberately, and that is a
  supported configuration rather than a trick.

An earlier version of this file said `VLLM_ATTENTION_BACKEND=TORCH_SDPA` was
the lever for the decode path. That was wrong twice over: the variable no
longer exists in current vLLM, and `AttentionBackendEnum.TORCH_SDPA` carries
the comment "this tag is only used for ViT".

Measured on the exact call the ViT path makes -- `(b, s, h, d)` permuted to
`(b, h, s, d)`, so never contiguous -- against aotriton on a W7900D:

```
tower                      B,H,N,D        path           ms   vs fp32   speedup
CLIP ViT-L/14-336 (LLaVA)  1,16,577,64    hk (kernel) 0.0978   0.00369     1.69x
                                          torch       0.1657   0.00185     1.00x
InternViT-300M             1,16,1025,64   hk (kernel) 0.1508   0.00243     2.49x
                                          torch       0.3759   0.00105     1.00x
SigLIP so400m              1,16,729,72    hk (back)   0.1945   0.00128     0.95x
                                          torch       0.1850   0.00128     1.00x
Qwen2.5-VL ViT             1,16,1024,80   hk (back)   0.3406   0.00113     0.97x
                                          torch       0.3295   0.00113     1.00x
```

`hk.ops.attn` has head_dim 64 and 128, so CLIP-L/14 (LLaVA, Pixtral,
InternViT) fits and SigLIP's 72 and Qwen2-VL's 80 fall back. The fallback rows
are the reason this table exists: the first measurement of them was 0.84x and
0.90x, because `scaled_dot_product_attention` copied `q`, `k` and `v`
contiguous *before* asking whether the kernel could take them -- three copies
thrown away, and torch handed the original strided tensors anyway. Deciding
first and memoising the verdict took it to 0.95x and 0.97x, the remainder
being the Python predicate itself on a 0.19 ms call.

The drop-in now counts which branch it took (`hk.ops.sdpa.STATS`), because a
patch that silently falls back produces identical answers and no speedup,
which is indistinguishable from one that works unless something counts.

`tools/hk-bench/vllm_verify.py` is the verification, and it has now been run
inside a real vLLM (0.16.1.dev0, ROCm 7.2.1, W7900D) serving
`OpenGVLab/InternVL2-2B` with `--mm-encoder-attn-backend TORCH_SDPA`. The
engine reaches the kernel: **120 kernel calls, 0 fallbacks** over five
requests, which is 24 InternViT layers each, and the server answers normally.

Getting there found two failures that nothing short of a live engine would
have shown, both of which looked like success:

* **vLLM V1 runs the model in a spawned process.** `apply()` in the process
  that built `LLM(...)` patches a process the model never runs in. The first
  end-to-end run reported `{'kernel': 0, 'fallback': 0}` with byte-identical
  answers and a plausible-looking 1.001x. Production therefore needs the
  `vllm.general_plugins` entry point (`hk/integration/_vllm_plugin.py`), which
  vLLM loads in the driver *and* every worker; it is opt-in on
  `HK_PATCH_VLLM=1`, because installing a kernel library must not silently
  change the numerics of every vLLM on the machine.
* **The vision tower runs once per distinct image.** With
  `mm_processor_cache` and prefix caching on, a warm-up request consumed the
  only ViT pass and every later request was served from cache -- counters at
  zero again, for a different reason.

And one measurement was wrong until a control caught it. Running the arms in
blocks gave "1.18x end to end"; running the *same* arm twice under a different
ViT backend gave 1.16x with the kernel never called at all. The order effect
was the whole number. Interleaved, the end-to-end spread is +-25% on a ~250 ms
request of which the vision tower is ~4 ms, so an end-to-end request simply
cannot resolve this -- the tower's own number is the `--unit` table above, and
`--e2e` is a correctness and reachability check that now says so.

**And it found a real bug, which is now fixed.** Patched, InternVL2-2B
described the test image differently from unpatched -- while the unpatched run
repeated itself exactly, and vLLM's Triton flash-attention ViT backend agreed
with unpatched word for word. Two independent implementations agreeing and
ours disagreeing is the shape of a bug, so `tools/hk-bench/attn_precision.py`
went after it. Three hypotheses died first:

* **not** the non-dividing KV tail: `hk/torch` was the same 1.5-2.4x at
  N = 512, 544, 576, 608, 1024, 1056 as at 577, 578, 1000, 1025, 1026;
* **not** an artifact of the reference: torch's flash, mem-efficient and math
  backends all land within 3e-6 of each other here;
* **not** concentrated: p50, p90 and p99 were each ~2.5x torch's with the
  worst rows scattered, which is rounding and not a mishandled block.

Emulating the kernel's own algorithm in torch then ruled out the algorithm:
with the probability tile rounded to bf16, a 32-wide KV block and a 1-ulp
exp2, the emulation landed on *torch's* number exactly, not the kernel's. So
the kernel was not doing what its algorithm said.

The probe that found it has an exact answer. Set `V` to all ones: the softmax
weights sum to one, so attention must return exactly 1.0 for every element,
whatever Q and K were. torch returned exactly 1.0. **This kernel returned 1.0
off by 2^-8 -- one whole bf16 ulp -- on every element.**

`convertor<bf16, float>` in `include/rdna3/common/base_types.cuh` was
`bit_cast<uint32_t>(u) >> 16`, labelled "fast unsafe conversion (truncation
only)". Truncation toward zero is *biased*, half an ulp, always the same
direction -- and attention rounds a tile of positive probabilities through it
on every KV block while summing the fp32 ones for the denominator, so the bias
does not cancel. It accumulates into the ratio.

Rounding instead of truncating closes it completely. Measured, four ways in
one process (`tools/hk-bench/bf16_round_cost.py`), cost on attention at
head_dim 128 / 64:

```
0  truncate             1 instr      --          biased; not a supported mode
1  nearest, ties away   2 instr   +2.1% / +3.4%  unbiased          <- default
2  nearest even         3 instr   +3.5% / +6.4%  unbiased
3  nearest even + NaN   5 instr   +6.0% / +12.5% unbiased and total
```

GEMM pays 0.2% in every mode, because its conversion is once per output
element rather than once per probability. No mode changes the register count
(215 / 217 / 147 VGPRs, zero scratch, all four).

The default is ties-away: it is the cheapest unbiased rounding there is, and
it is **numerically indistinguishable from the expensive one** here -- `V = 1`
returns exactly 1.0, and hk's error against fp32 is now `1.00x` torch's at
every N in the sweep, not 1.5-2.4x. The NaN guard doubles the cost to protect
against a value attention cannot produce, so it is opt-in
(`-DHK_BF16_ROUND=3`); without it a NaN whose low mantissa bits are near
all-ones can come back as an infinity.

Two tests now hold the line, both with exact answers rather than a tolerance
(`tests/hk/gpu/test_attn.py`): `V = 1` must come back bit-exact, and `Q = 0`
must give the mean of V. The file's existing `RTOL = 5e-2` could never have
seen this -- it is sized to catch a shifted mask, and a half-ulp bias passed
all 31 of its cases while changing what a model said.

**What is left is a tie, and it was measured rather than assumed.** hk and
aotriton now differ on 0.02-0.04% of the vision tower's output elements, by
exactly one bf16 ulp -- two correctly-rounded kernels breaking ties from
different intermediates, which is as close as independent implementations get.
Whether that changes a caption is a property of how close the model's top two
tokens were, so the e2e runs six images under all three implementations:

```
image   aotriton (TORCH_SDPA)      Triton FA        hk
  1     "mosaic of shapes"         same             "crowd of people"
  2     "mosaic of shapes"         same             same
  3     "crowd of people"          same             same
  4     "crowd of people"          "digital noise"  "digital noise"
  5     "blurred and distorted"    same             same
  6     "shades of green, red"     same             same
```

vLLM's own two backends already disagree with each other on 1 of 6; hk
disagrees with aotriton on 2 of 6, and on one of those (image 4) it agrees
with Triton FA *against* aotriton. Each arm repeated itself 6/6, so the engine
is deterministic and these are real disagreements -- between three
implementations, on inputs chosen to be maximally ambiguous (synthetic noise).
hk is inside the spread that already exists between shipped backends, not
outside it. A real-image benchmark would say more; this says enough to stop
treating the remaining difference as a defect.

The layer underneath is covered by the GPU tier -- 225 tests including
`torch.compile(fullgraph=True)`, CUDA-graph capture and replay, and the
strided-ViT-input contract in both directions.

`python3 -m hk.integration --warm -j 32` builds the six distinct attention
registrations in 17.8 s, which is the number that matters for a server's first
request.

**Phase 7: the decoder, which the SDPA patch structurally could not reach.**
vLLM V1 keeps KV in a paged cache and dispatches through `AttentionImpl`, so
the way in is to *be* a backend. `hk/ops/paged.py` is the kernel and
`hk/integration/vllm_backend.py` registers it.

The cache layout decides the kernel. `gl` derives its strides from its dims,
so vLLM's default `(block, token, head, dim)` cannot yield a contiguous
`(block_size, head_dim)` tile for one head. A backend declares its own shape,
so this one declares `(2, num_blocks, num_kv_heads, block_size, head_size)` --
exactly `gl`'s `(b, d, r, c)`, so one page of one head is a plain tile load.
The price is that the cache *update* becomes a strided scatter, and that is
the right way round: a decode step reads `seq_len` pages per layer and writes
one.

**It loses, and an earlier version of this file said it won.** That claim came
from Qwen3-0.6B, four short prompts, measured as total wall time. Total wall
time mixes prefill with decode and is neither latency nor throughput, and a
0.6B model spends so little of a step in attention that the number was mostly
measuring something else. Retracted.

The measurement that means something isolates one decode step. Each point runs
the same requests at two output lengths and takes the *slope*,
`(T(72) - T(8)) / 64`, which cancels prefill and every other fixed cost.
Qwen3-8B (36 layers, 32 query heads over 8 KV heads, head_dim 128), bf16,
eager, three reps per endpoint, `tools/hk-bench/vllm_sweep.py`:

```
                 ms / decode step            tokens / s
ctx   batch     hk     triton   ratio      hk    triton
1024      1   54.43    42.01    1.30x    18.4     23.8
1024      4   54.60    42.83    1.27x    73.3     93.4
1024     16   57.65    46.54    1.24x   277.5    343.8
1024     32   59.23    61.53    0.96x   540.3    520.1   <- the only win
4096      1   82.68    42.97    1.92x    12.1     23.3
4096      4   83.09    45.97    1.81x    48.1     87.0
4096      8   83.86    49.13    1.71x    95.4    162.8
4096     16   91.74    56.78    1.62x   174.4    281.8
```

Measurement bands were under 1% of the value at every point except hk at
ctx 1024 batch 1, so the ordering is not in question.

**Why, in one number.** Going from 1024 to 4096 tokens of context at batch 1
costs Triton **+0.97 ms/step** and this kernel **+28.2 ms/step**. That extra
3072 tokens is 12.6 MB of K and V per layer, so per layer:

```
triton   0.027 ms   469 GB/s    54% of the W7900D's 864 GB/s
hk       0.785 ms    16 GB/s     1.9%
```

Decode attention is a bandwidth problem and this kernel is getting 2% of the
bandwidth. The cause is the limitation already written down when it was built:
**no split along the KV axis**. One workgroup per (request, kv head) means
batch 1 launches 8 workgroups of one warp each on a 48-WGP GPU, and they walk
the whole context serially. Triton splits the KV range across the machine.

That also explains the shape of the whole table. hk is nearly flat in batch --
54.4 ms at batch 1, 59.2 at batch 32 -- because at low batch the GPU is idle
and extra requests fill it for free. Triton starts near the bandwidth limit
and degrades as requests compete for it (42.0 -> 61.5). The two cross at
batch 32, which is the only cell hk wins, and it wins it by 4%.

**Splitting the KV axis closes most of it.** `paged_decode_split_kernel` cuts
the page range `splits` ways and writes a partial `(O, m, l)` per split;
`merge_splits` combines them. How many ways was measured rather than reasoned
(`tools/hk-bench/paged_splits.py`, fifteen shapes), and the first guess was
wrong by an order of magnitude: the best split at every point puts
`n_reqs * n_kv_heads * splits` near **a thousand** workgroups, not the 96 that
"two per WGP" suggests -- which is the right instinct for a twelve-warp
workgroup and wrong for a one-warp one. At batch 16 the old rule asked for one
split, i.e. 128 workgroups of a single warp each walking 128 pages in series.

```
                 ms / decode step
ctx   batch   before   split   +planner   triton   ratio
1024      1    54.43   48.60     49.16     42.01   1.17x
1024      4    54.60   51.66     51.10     42.83   1.19x
1024     16    57.65   57.65     54.46     46.54   1.17x
1024     32    59.23   59.68     59.59     61.53   0.97x   <- win
4096      1    82.68   51.10     50.67     42.97   1.18x
4096      4    83.09   61.65     53.51     45.97   1.16x
4096      8    83.86   71.20     58.06     49.13   1.18x
4096     16    91.74   91.29     64.54     56.78   1.14x
```

Worst case 1.92x -> 1.19x. Standalone, the kernel now reaches 56% of the
card's 864 GB/s at ctx 4096 batch 16 and 76% at ctx 16384 batch 32, against
1.9% before.

**And the remaining gap is flat, which says what it is.** 1.14-1.19x at every
shape is not a kernel that is uniformly slower -- a kernel-speed difference
would grow with context, as the 1.92x did. It is a constant ~7-9 ms per step,
or ~0.2 ms per layer, and the backend's `forward` issues about a dozen small
torch ops per layer around the kernel: the query pad copy, the output copy,
two scatters for the cache update, and six or seven elementwise passes in
`merge_splits`. At ROCm's launch cost that is the whole difference. It is the
same class of mistake as the `.tolist()` that made the first version slower
than Triton, one level down.

So: the kernel is no longer the problem and the wrapper is. Folding the merge
and the cache update into kernels, and keeping the pad and the output copy out
of the per-layer path, is what a later round has to do -- and the flatness of
that column is the evidence that it is worth doing.

## Against the engine, which is the number that matters

Everything above measures a decode step against vLLM's Triton attention. That
is the wrong target twice over: the point of this IR is to patch kernels an
engine is *missing* a good one for and make the request faster, so the
baseline is the engine as it ships, and Triton attention is the best-tuned
kernel on the ROCm board rather than a gap.

A profile says where the time is. Qwen3-8B, batch 16, context 4096, by device
time (`tools/hk-bench/vllm_profile.py`):

```
57.6%  hipBLASLt GEMM
39.1%  Triton prefill attention
 1.0%  silu-and-mul
 0.9%  Triton decode attention        <- what the whole section above is about
 0.6%  fused_add_rms_norm
 0.4%  rms_norm
 0.2%  rotary embedding
 0.1%  reshape_and_cache
```

The decode attention is **0.9%** of this workload. The prefill attention in
front of **39%** is served by the dense kernel, which Phase 4 already put at
parity with its handwritten C++ and 2.9-3.2x over aotriton -- and the backend
routes prefill to it for any prompt that fills a query tile.

Total request time, arms alternating in separate processes, stock vLLM
choosing its own backend as the 1.00x (`tools/hk-bench/vllm_e2e.py`, 72 output
tokens, two rounds, min; the two hk rounds agreed to 0.3%):

```
   shape    stock ms      hk ms   speedup
  1024x1      3261.2     3753.4    0.87x
  1024x4      4077.5     4565.0    0.89x
 1024x16      7515.4     7730.0    0.97x
  4096x1      4491.8     4589.1    0.98x
  4096x4      9085.3     7906.2    1.15x
 4096x16     27455.9    21196.0    1.29x
```

**1.29x end to end on the prefill-heavy shape**, and the ordering across the
table is exactly what the profile predicts: the win tracks prefill's share of
the request, so it grows with context and with batch, and at short context
with a small batch -- where the request is almost all decode -- the 1.15x
decode deficit shows through as 0.87x.

What that says about the tool rather than the kernel: the thing worth patching
was not the kernel that was hardest to beat, it was the one with 39% of the
time in front of it, and finding that took a profile rather than an argument.
The remaining 57.6% is GEMM, which is where the next look goes.

Four facts about vLLM's plugin surface are written down in
`vllm_backend.py` because each cost a run to find: `AttentionBackendEnum` is
closed and `register_backend` overrides rather than adds; the reserved
`CUSTOM` member is rejected by `RocmPlatform`'s allowlist, so the override has
to target a member the platform accepts (`TRITON_ATTN`, which is also ROCm's
default -- hence the separate `HK_VLLM_BACKEND=1` opt-in); `get_name()` must
answer with the member it overrides because vLLM round-trips the name through
the enum; and short prompts cannot use the dense kernel and cannot be padded
into it, because zero-padded keys are not masked out, they are keys with score
zero and `exp(0)` is a real weight in the softmax sum.

Scope, stated rather than discovered: no paged *prefill* kernel, so prefix
caching and chunked prefill are refused rather than silently attending over
part of a context. The backend also declares that it cannot run inside a full
CUDA graph -- it plans its KV split on the host from `max_seq_len`, which a
graph would freeze at the capture batch -- so with graphs on, vLLM captures
the model piecewise and runs this attention eagerly between the pieces.

### The other 57.6%: the linear layers

On gfx1100 every one of those GEMMs ends in `torch.nn.functional.linear`, i.e.
hipBLASLt. The installed vLLM has a skinny-GEMM path for decode, but it is
gated on gfx9, and correctly: its `wvSplitK` is compiled for gfx9 only, and
called on gfx1100 it fails a device-side assert and takes the process with it.
So the engine brings no GEMM of its own here, and the question is only whether
the DSL GEMM beats the library at the shapes this model runs.

Phase 0 on exactly those shapes against exactly that call
(`tools/hk-bench/vllm_gemm.py`: one process, variants interleaved, weights
rotated through 3x the 96 MB Infinity Cache, every variant checked against
fp32 before it is timed). The kernel is the Phase 3 DSL GEMM, unchanged:

```
hk / F.linear              M=1024   M=4096   M=16384
qkv      6144 x 4096        1.15x    1.10x    1.10x
o        4096 x 4096        1.18x    1.09x    1.09x
gate_up 24576 x 4096        1.11x    1.16x    1.23x
down     4096 x 12288       1.17x    1.10x    1.17x
```

All 36 layers' GEMMs for one forward step at M=4096: 838.5 ms on hipBLASLt,
744.4 on hk, 1.13x. rocBLAS, the other library torch can be told to use, is
slower than hipBLASLt at every prefill shape.

Below 1024 rows it is a comb, not a curve:

```
M          128    192    256    384    512    768
qkv       1.38   0.88   1.13   1.43   1.20   1.17
o         1.15   0.75   0.95   1.11   1.24   1.17
gate_up   1.56   0.95   1.21   1.18   1.19   1.09
down      1.16   0.71   0.96   1.15   1.33   1.18
```

The last M block backs up to end at M and recomputes a full 128 rows, and at
small grids two workgroups then share a WGP. The rule that comes out of it:
**route when M >= 1024, or M >= 384 in whole 128-row blocks**; everything else
-- including all of decode, where M is the batch -- stays on vLLM's path.
(Decode got its own kernel afterwards; see "The decode kernel" below.)
Steps that mix a new prompt with the running requests' decode tokens land just
above a multiple of 128, so those were measured too: M=4097..4111 costs hk
~18 ms per step over M=4096 for the extra block, and it still wins 1.09-1.10x.

`hk/integration/vllm_gemm.py` installs it, behind its own opt-in
(`HK_PATCH_VLLM=1 HK_VLLM_GEMM=1`). It rebinds `dispatch_unquantized_gemm` by
identity in every vLLM module that imported it, and what that returns calls a
single opaque custom op, `hk_vllm::linear`, with a fake implementation. Under
vLLM's torch.compile the token count is symbolic, so branching on it inside
traced code would be a guard and a recompile; inside the op it is neither, and
CUDA-graph capture sees a launch on torch's stream. Below the rule the op calls
vLLM's own `rocm_unquantized_gemm`.

End to end, eager like every earlier number in this file, arms alternating in
separate processes, two rounds, min; the rounds agreed to 0.3%, and every arm
produced the same 72 greedy tokens as stock at every shape:

```
                     stock ms     gemm       hk    hk+gemm
total request
  1024 ctx x 1         3268.5   1.010x   0.870x   0.877x
  1024 x 4             4085.5   1.025x   0.893x   0.912x
  1024 x 16            7533.5   1.070x   0.971x   1.024x
  4096 x 1             4504.8   1.019x   0.979x   1.000x
  4096 x 4             9117.3   1.040x   1.149x   1.205x
  4096 x 16           27542.9   1.044x   1.296x   1.366x
prefill (the same batch, one output token)
  1024 x 1              274.0   1.111x   1.059x   1.190x
  1024 x 4             1019.3   1.093x   1.081x   1.188x
  1024 x 16            4288.3   1.152x   1.104x   1.257x
  4096 x 1             1445.7   1.054x   1.452x   1.584x
  4096 x 4             5920.8   1.083x   1.437x   1.625x
  4096 x 16           23788.4   1.088x   1.438x   1.632x
```

The GEMM patch alone is never slower, and everything it buys is in prefill:
5-15% of the prefill, 1-7% of the request. The two patches compose -- the
prefill gains roughly multiply -- and together they are 1.37x on the
long-context batch.
What they cannot fix is the left half of the table, where the request is
nearly all decode and the attention backend's decode deficit shows through.

**And with vLLM as it ships** -- torch.compile and CUDA graphs on, the compile
cache off so no arm can load another's graph -- the same table to within a
percent. The attention backend runs under piecewise graphs (it declares it
cannot be captured whole; vLLM logs the downgrade), the GEMM patch under
vLLM's default full-and-piecewise. Decode is measured directly here, as the
slope `(T(72) - T(24)) / 48`, which cancels every step that admits a prompt:

```
                     stock ms     gemm       hk    hk+gemm
total request
  1024 ctx x 1         3216.2   1.007x   0.864x   0.870x
  1024 x 4             4035.1   1.024x   0.887x   0.904x
  1024 x 16            7556.9   1.073x   0.965x   1.025x
  4096 x 1             4452.6   1.016x   0.973x   0.993x
  4096 x 4             9063.7   1.040x   1.149x   1.201x
  4096 x 16           27517.9   1.044x   1.293x   1.364x
prefill
  1024 x 1              273.8   1.111x   1.064x   1.192x
  4096 x 16           23829.6   1.090x   1.436x   1.630x
decode, ms per step at the full batch
  1024 x 1              41.46   0.998x   0.849x   0.848x
  1024 x 16             46.02   1.003x   0.834x   0.845x
  4096 x 1              42.33   0.998x   0.840x   0.842x
  4096 x 16             56.23   0.999x   0.837x   0.856x
```

Same tokens as stock, 72 of 72, at every shape in both modes. Two things in
it are worth a sentence:

* **Graphs barely move stock's decode**: 41.5 ms a step against 42.2 eager at
  batch 1. A decode step on this card is not launch-bound, it is GEMV-bound --
  which is the next paragraph -- so the eager numbers above were not flattering
  the patches by more than a percent or two.
* **Steps that admit a prompt: the ~51 ms they seemed to cost was the
  comparison, not the patch.** A first version of this bullet reported that
  such a step (one new prompt plus the running requests' decode tokens,
  M = 4096 + r) cost the patch ~51 ms more end to end than the GEMMs explained,
  inferred by subtracting runs of different lengths. Timed directly instead
  (`tools/hk-bench/vllm_steps.py`, 4096 x 16, the same 87 steps in both arms,
  two rounds), the patch makes every step faster:

  ```
  step                                     stock ms    gemm ms    saved
  first: two prompts, 8192 tokens            2937.8     2728.4    209.4
  admit one prompt + decode (x14, avg)       1467.8     1411.8     56.0
  decode at batch 16 (x57, avg)                57.6       37.3     20.3
  ```

  The decode saving is exactly what the GEMMs predict (19.9 ms of layers plus
  0.4 of LM head). The ~51 ms was a mixed step measured against *half the
  first step*: there the patch's mixed step costs 48.7 ms more and stock's
  1 ms less. The GEMMs, timed in place with a synchronize around each call,
  account for 45 of those 48.7 ms. The patched GEMM's mixed step costs 29 ms
  more than half its M = 8192 step, because the last M block backs up and
  recomputes 128 rows. hipBLASLt's costs 16 ms *less* than half of its own.

  What is **not** explained: hipBLASLt is faster inside vLLM than in any
  standalone run of the same 36 layers -- 837 ms of GEMM per mixed step,
  against 849-870 -- so the patch's lead there is 51 ms where the benchmarks
  say 76-88. The patched GEMM times the same everywhere (785-786 ms, shape by
  shape). Ruled out: the timing method (per call with a synchronize vs back to
  back), the operands (Qwen3-8B's own weights with RMSNorm- and SiLU-shaped
  activations time like random ones), and vLLM's GEMM path (on gfx1100 it ends
  in the same `F.linear`). And it moves: in one process alternating the two
  GEMMs step by step, hipBLASLt's mixed step took 859 ms, its qkv layers 21%
  slower than in a process of its own. So the variation is real and depends
  on context; which part of the context, these runs do not say.

  One tool lesson came out of it: on this ROCm build the torch profiler's
  kernel durations include time queued behind earlier launches -- in an eager
  step they summed to 4.4x the step's wall time. `vllm_profile.py` ran eager,
  so the shares it printed at the top of this section are queue-weighted. The
  ordering they gave (GEMM, then prefill attention, then the rest) has since
  been borne out end to end; the percentages should not be quoted.

**What is left is decode, and decode is GEMV.** At batch 1 the 36 layers' four
GEMVs plus the LM head take 39.7 ms of a 41.5 ms step on hipBLASLt, at 37-46%
of the card's 864 GB/s for the layers and 75% for the LM head. What a kernel
can reach is measured, not the datasheet: `tools/hk-bench/hbm_probe.hip`
streams 680-820 GB/s depending on size, and upstream vLLM's `wvSplitK` --
which upstream has since enabled on gfx11, for one to five tokens -- reaches
743-835 GB/s on these exact matrices, built standalone for gfx1100. Per decode
step, in GEMM milliseconds (`vllm_gemm.py --upstream-skinny`):

```
tokens     installed vLLM   upstream vLLM   best of everything
             (hipBLASLt)    (wvSplitK <= 5)   incl. rocBLAS, hk
  1             39.7            19.1             19.1
  4             39.7            20.4             20.4
  8             39.5            39.5             34.8
 16             41.4            41.4             36.5
 32             41.1            41.1             36.4
 64             42.0            42.0             38.2
128             55.0            55.0             37.5   <- hk GEMM, 1.36x
```

So the gap has a precise shape. At one to five tokens a near-ceiling GEMV
exists and the installed engine cannot run it. From six tokens up nobody has
one: every library lands at 35-42 ms, about half of what `wvSplitK` shows is
reachable. That band -- the batch sizes a serving engine spends most of its
decode steps in -- is the next kernel (below); and at batch 128 the existing
GEMM is already 1.36x, which the routing rule, written for prefill, still
leaves on the table.

One measurement lesson cost a run and is worth keeping: timing each call
separately over three rotating weight copies let the Infinity Cache serve most
of a 100 MB matrix to `wvSplitK`, which then read "at" 1917 GB/s, 2.2x the HBM
peak. hipBLASLt, too slow to notice the cache, measured the same either way.
The harness now times what a decode step does: one pass over 36 distinct
weights, back to back.

### The decode kernel

**Why the GEMM cannot do it.** The DSL GEMM at a 16-row tiling already beats
hipBLASLt on every decode shape -- 1.19-1.62x, which shipped first and was
worth 1.36-1.44x on decode end to end -- but it streams only 418-755 GB/s.
Its schedule is the reason. Each workgroup keeps exactly one 18 KB batch of
loads in flight, and the grid is N/128 workgroups: 32 on a 4096-wide layer.
At about a microsecond of latency, Little's law caps that near 32 x 18 KB/us,
which is where `o` and `down` land. Split-K on the same schedule was measured
in the C++ (`thin_config`) and topped out at 372-633 GB/s, so the template was
not worth extending.

`hk/ops/decode_gemm.py` is laid out for the read instead, in about sixty
lines of DSL:

* **W goes straight to registers**, never through LDS -- it is used once.
  W*X^T puts W's rows on the WMMA's 16-row side, the way the paged decode
  kernel puts K there; the accumulator is C^T and its transpose is free.
* **One 16-row block of W per workgroup, K split across its warps**: N/16
  workgroups of 16 waves, 6144 waves on a 6144-wide layer.
* **The split is interleaved**: on round t, warp w reads K-chunk t*16 + w, so
  a workgroup reads 2 KB of each of its rows contiguously at once instead of
  128 B islands 8 KB apart.
* The warps' partial sums meet once, through LDS, at the end.

It was right the first time it ran: 32 cases across 1-64 rows, six warp
splits, strided input. Timed as a step reads the weights, M=16, microseconds:

```
              F.linear   GEMM 16-row   decode w4   w8      w16     w32
qkv             156.7       119.1        105.8    85.4    75.9    76.3
o               105.9        65.5         65.6    59.3    47.5    51.4
gate_up         592.4       354.4        311.5   278.1   257.5   261.0
down            250.6       166.6        168.9   168.1   145.5   150.0
lm_head        1980.3      1649.2       1632.6  1584.2  1551.2  1549.3
per step, ms     41.8        27.1         25.1    22.9    20.5    20.9
```

At 16 warps that is 663-802 GB/s against hipBLASLt's 317-638. More token
tiles cost registers -- each is another X operand and another accumulator per
wave -- so the schedule depends on the batch width (`decode_gemm.SCHEDULES`).
GEMM milliseconds per decode step, 36 layers plus the LM head:

```
tokens    installed vLLM   upstream wvSplitK   hk decode     vs installed
1-16           39.7-41.8      19.1-20.4 (<=5)       20.5      1.94-2.04x
17-32               40.4                  --        22.3           1.81x
33-64               42.2                  --        28.6           1.47x
```

So the band where nobody had a kernel now runs 7% behind upstream's GEMV at
one token, and the kernel covers the batch widths upstream's does not. The
last row is register-bound (four token tiles at 8 warps run at occupancy 5),
and it is the one with the most left.

End to end, vLLM as it ships (torch.compile, CUDA graphs, compile cache off),
arms alternating in separate processes, two rounds, min. The `gemm` arm is
now the prefill route above plus this kernel for M <= 64:

```
                     stock ms     gemm    hk+gemm
total request (72 output tokens)
  1024 ctx x 1         3202.4   1.717x   1.382x
  1024 x 4             4035.5   1.552x   1.309x
  1024 x 16            7625.0   1.354x   1.297x
  4096 x 1             4440.6   1.457x   1.436x
  4096 x 4             9059.0   1.227x   1.464x
  4096 x 16           27588.0   1.104x   1.476x
decode, ms per step at the full batch
  1024 x 1              41.24   1.808x   1.402x
  1024 x 16             46.99   1.807x   1.417x
  4096 x 1              42.09   1.784x   1.371x
  4096 x 16             57.18   1.580x   1.281x
wider batches (1024 ctx, 128 output tokens; decode over T(128) - T(64))
  total, x 32         16512.7   1.267x   1.370x
  total, x 64         28509.3   1.162x   1.362x
  decode, x 32          62.98   1.447x   1.591x
  decode, x 64          89.22   1.195x   1.668x
```

Prefill is unchanged from the table above (1.06-1.15x for `gemm`). Every arm
produced the same greedy tokens as stock at every shape, 72 of 72 and 128 of
128. Where each configuration wins and loses:

* **`gemm` alone is the best patch whenever decode dominates** -- short
  context, small batch: 1.72x on a whole 1024 x 1 request. Its decode gain
  shrinks as attention's share of the step grows, 1.81x at 1024 context to
  1.58x at 4096 x 16, and as the batch widens past one token tile.
* **`hk+gemm` is best on long-context prefill and on wide decode.** At batch
  1-16 the attention backend's decode is slower than stock's, which leaves
  decode at 0.75-0.81x of what `gemm` alone gets. At batch 32 and 64 that
  reverses: the same backend makes decode faster, 1.59x and 1.67x with the
  GEMMs, against 1.45x and 1.20x without it.
* **Not covered:** decode steps of 65 to 383 tokens stay on hipBLASLt (the
  wide runs route 2304 such calls), and batch 128 -- where the prefill GEMM
  is already 1.36x per step -- is still excluded by the prefill rule.

**Residency is a lever, and in the direction that looks wrong.** At 16 warps,
two workgroups per WGP read faster than four: 145 against 163 us on `down`,
47.5 against 50.8 on `o`, the same code either way. More concurrent 16-row
streams cost more than the latency they hide -- the likely cause is DRAM row
conflicts, but with the counters dead that is an inference, not a
measurement. The schedule therefore asks for two (`resident=2`, which sizes
the combine's LDS so a third does not fit).

Finding that took two wrong turns, both worth not repeating:

* **The first residency test measured nothing.** It reserved LDS for 2, 3 and
  4 workgroups per WGP against the 64 KB pool in `hk/target/gfx1100.py`, and
  nothing changed. Reserving 48 KB and 64 KB later both gave the
  two-per-WGP time, identical to each other; 32 KB had given the four-per-WGP
  one. Only a **128 KB pool** produces that pattern, and that is what RDNA3
  has; the target now says so. Every "LDS binds occupancy" estimate the
  verifier printed for gfx1100 before this was off by 2x.
* **The combine was blamed first.** The fp32 combine reads LDS with flat
  loads (the library vectorises only bf16 shared tiles), and batching those
  reads coincided with the slowdown, so a combine moved onto `ds_read_b128`
  -- partials as bf16 hi/lo pairs -- was built and measured. Same speed as
  the fp32 one at the same residency, 21.6 ms a step at four per WGP and
  20.6 at two. The flat loads were never the cost. (The DSL's LDS verifier
  had to learn that a counted wait inside an `hk.if_` may count reads issued
  in the same, still-open branch before that kernel would build; that fix
  stayed.)

Measured and not levers: 32 warps (20.9 ms), batching the combine's reads,
the combine's LDS path. What is left to 800 GB/s -- about 1.5 ms a step -- is
not explained yet.

### Row-predicated loads and stores

Until now a tile could not hang off the end of a tensor. The decode GEMM padded
every batch to 16 rows with an extra launch, and the GEMM backed its last M
block up over its neighbour, so M had to be at least one block. Both were
there because the staging path had no bounds check. Predicating the issue
would make the right `vmcnt` a runtime value, and no instruction accepts that.

The way out is the buffer instruction's own range check. A buffer resource
whose range ends at the tensor's last row makes the hardware return zero for
a load past it and drop a store past it, with no branch, no masked lane, and
exactly the instructions a whole tile issues. So every hand-placed wait count
stays an immediate:

* **Staging** (global -> registers -> LDS) is always bounded this way, and the
  synchronous global -> LDS load and LDS -> global store skip the same rows.
  Free: the staging descriptor now starts at the tile's row and carries the
  K offset in `soffset`, which makes it loop-invariant. The default GEMM came
  out 15 VGPRs *lighter* (226 -> 211, occupancy 6 -> 7).
* **Register loads and stores** are opt-in, `hk.load(..., bounded=True)` and
  `hk.store(..., bounded=True)`, and compile to separate buffer-instruction
  functions (`kittens::load_bounded` / `store_bounded`). Plain `load` and
  `store` compile to exactly what they did before.

What it bought:

* **The GEMM takes any M.** The last block runs past M -- its A rows past M
  stage as zeros and its bounded store writes only rows below M -- and
  `gemm.matmul` no longer refuses M < 128; at M <= 16 it picks the 16-row
  tiling.
* **The decode GEMM stops padding.** Same kernel, one process, 36 layers back
  to back, eager: 19.85 -> 18.13 ms at batch 1, 1.6-1.7 ms a step at batch
  1-8 (1.08-1.10x) -- a pad launch and a sliced 16-row output per linear call,
  144 of each per step.

Gates. All 303 shipped kernel instantiations rebuilt against HEAD:
no occupancy dropped, no scratch, no build error. Performance against the
shipped GEMM, both built and timed in one process, interleaved, at 4096^3:
the default schedule is 1.0% faster in every run. The tuned schedule `matmul`
uses there ranged from 1.7% slower to 0.9% faster across five runs, always
inside the 3-7% round-to-round spread, and its main loop is 121 instructions
against 136 shipped, wait for wait otherwise the same. 343 GPU tests pass,
including a tensor that starts 2 bytes past a 16-byte boundary.

It took five wrong versions, each measured, and the reasons are what the next
person needs:

* **Always on in the pointer path:** paged decode spilled 66 VGPRs, for tiles
  that are never ragged. Hence opt-in.
* **A predicate that is a compile-time constant** on the unbounded side still
  changed how LLVM addressed every store: 32 base+offset stores became 32
  separate address pairs, an elementwise op went from 36 to 91 VGPRs, and the
  GEMM spilled. Hence separate functions, not a flag.
* **A descriptor built from a warp-derived coordinate** looked per-lane to
  LLVM, which wrapped each of the GEMM epilogue's 64 buffer stores in a
  waterfall loop -- 8% of the kernel. `make_srsrc` now `readfirstlane`s it;
  it is uniform by construction.
* **An unaligned fallback** in the bounded load cost the decode GEMM 10
  VGPRs and an occupancy step: LLVM merges a pointer loop into wide loads but
  cannot merge buffer intrinsics, and reserves the fallback's registers
  whether it runs or not. Dropped -- gfx11 runs with unaligned access on, and
  the 2-byte-misaligned test covers it.
* **A whole-tile fast path** in the bounded store, falling back to the old
  pointer store, made the default GEMM spill 76 VGPRs. Reverted.

Scope: rows only. N and K still have to tile exactly. Also seen in passing
and not touched: both the shipped and the new GEMM wait `vmcnt(0)` right
before every iteration's barrier, so under `write_pos=1` the prefetch for
tile t+2 never overlaps tile t's math -- a lever for the GEMM itself.

### A reduction across workgroups, and the decode GEMV that did not need it

The IR had no way for workgroups to combine results inside one launch, which
is what split-K across workgroups needs. `hk.wg_arrive(counter, idx,
expected)` is that primitive, as a counter rather than fp32 atomics:

* Each workgroup stores its partial and arrives. The one that gets back
  `expected - 1` arrived last, and finishes by reading the partials in a
  fixed order. So the answer does not depend on which workgroup that was --
  20 repeats are bit-identical, which a greedy-token comparison against the
  engine needs and fp32 atomics would not give.
* The fences (release before the increment, acquire after) make every store
  issued before the call visible to the last arrival.
* The last arrival resets its counter, so one zeroed int32 tensor serves
  every launch and every replay of a captured graph with no memset.
* Its broadcast slot is 16 B of static LDS outside the allocator, and the
  first kernel that used it asked for 65,552 B and failed to launch
  (`HSA_STATUS_ERROR_INVALID_ALLOCATION`). The IR now counts static LDS and
  the verifier refuses the sum over 64 KB at build time, with that error in
  the message.

`decode_gemm(splits=S)` puts it to work: workgroup (b, s) takes the s-th
slice of K, and the last of the S finishes the block. It is correct, and it
is slower on every shape at every split, M=1 and M=16 alike (one process,
>= 1.5 GB of distinct weights per shape):

```
us at M=1      splits 1    splits 2    splits 3    splits 4
qkv                66.4        76.9          --        81.8
o                  46.5        51.4          --        55.0
gate_up           252.4       271.2          --       294.2
down              139.5       144.3       153.6       146.9
lm_head          1515.5      1663.0          --      1768.4
```

The case for it was `down`: 256 blocks over 96 resident slots leave a
third-full last round. The workspace round trip and the second pass cost
more than that tail. So the decode GEMV ships unsplit, and `wg_arrive`
ships as a capability for kernels whose grid really is too small -- a
negative result, measured, not a guess.

Where the decode GEMV stands now, in GEMM milliseconds per decode step (36
layers plus the LM head, `tools/hk-bench/vllm_gemm.py --upstream-skinny`,
>= 1.5 GB of distinct weights per shape, every variant gated against fp32):

```
tokens   hipBLASLt   rocBLAS   upstream wvSplitK   hk decode   vs hipBLASLt
     1       39.98     37.02               19.11       19.66        2.03x
     2       39.68     37.57               19.87       19.69        2.01x
     4       39.72     39.02               20.40       19.77        2.01x
     8       39.50     40.86                  --       19.89        1.99x
    16       41.00     41.75                  --       20.23        2.03x
    24       39.74     43.93                  --       20.56        1.93x
    32       40.48     46.81                  --       20.84        1.94x
```

Against the ceiling a plain read reaches (`hbm_probe.hip`, one kernel per
read: 676 GB/s for `o`'s 34 MB up to 821 for 302 MB), at one token: `qkv`
762 GB/s, `o` 724, `gate_up` 800, `down` 720, LM head 819. Upstream's
`wvSplitK`, which handles one to five tokens, is level on `qkv`, 1.5-2.6%
ahead on `o`, `gate_up` and the LM head, and 6.9% ahead on `down` (770
GB/s). That last one is where what is left lives. The row predicate is
also worth seeing here: without the pad, two and four tokens now beat
`wvSplitK` per step, and 32 tokens dropped from 22.3 to 20.8 ms.

**End to end.** Qwen3-8B, bf16, one W7900D. The baseline (1.00x) is stock
vLLM 0.16.1.dev0 choosing its own backends; the patched arm is the same
engine with `HK_VLLM_GEMM=1`, so every unquantized linear runs on the decode
GEMV (M <= 64) or the generated GEMM and everything else is stock. Prefix
caching and chunked prefill are off, so a prefill step is whole prompts. The
arms alternate in separate processes, min over rounds (two in graph mode,
one in eager). `total` is the whole batch from prompt to last token,
`prefill` is the same batch generating one token, and `decode` is one step at
the full batch, from the difference of two output lengths
(`tools/hk-bench/vllm_e2e.py`).

Graph mode (torch.compile plus CUDA graphs, vLLM's default), 72 output
tokens (128 at batch 32 and 64):

```
             total ms              prefill ms            decode ms per step
ctx x batch  stock     hk   x      stock     hk   x      stock     hk   x
1024x1      3211.6  1750.2 1.84    273.9   243.0 1.13    41.37  21.23 1.95
1024x4      4029.4  2489.2 1.62   1025.1   922.7 1.11    42.30  22.05 1.92
1024x16     7551.4  5600.6 1.35   4291.5  3706.4 1.16    45.90  26.04 1.76
1024x32    16346.1 12836.7 1.27   8616.1  7403.7 1.16    61.59  42.56 1.45
1024x64    28382.1 24055.6 1.18  17341.0 14881.4 1.17    88.22  71.72 1.23
4096x1      4452.9  2930.6 1.52   1446.3  1364.0 1.06    42.35  22.03 1.92
4096x4      9056.0  7237.1 1.25   5929.3  5439.1 1.09    44.97  24.45 1.84
4096x16    27523.8 24837.4 1.11  23860.2 21779.3 1.10    56.11  36.38 1.54
```

Eager (`enforce_eager=True`), 72 output tokens, a subset:

```
1024x1      3258.2  1792.3 1.82    274.5   243.8 1.13    42.09  21.83 1.93
1024x16     7545.4  5604.0 1.35   4283.1  3699.6 1.16    46.45  26.15 1.78
1024x32    12858.9 10483.0 1.23   8623.4  7423.4 1.16    61.01  42.10 1.45
4096x1      4483.2  2971.2 1.51   1440.2  1365.4 1.06    42.93  22.65 1.90
4096x16    27452.4 24869.2 1.10  23775.4 21748.2 1.09    55.43  37.14 1.49
```

How to read it:

* **Batch 1 is a latency number.** The whole request is 1.52-1.84x faster,
  and nearly all of that is decode, whose GEMMs halve (41-43 ms to 21-23 ms
  per token). Prefill latency improves 6-13%.
* **Batches are throughput numbers.** The batch finishes 1.10-1.35x sooner
  at 16 requests and 1.18-1.27x sooner at 32-64. The gain shrinks with batch
  and with context, as the profile predicts. Attention, which this patch
  does not touch, grows with both. And at 64 tokens the decode GEMV leads
  hipBLASLt by 1.65x, not 2x.
* **Nothing loses:** no shape, no phase, in either mode. On every shape the
  first request's greedy output is token-identical to stock (72/72,
  128/128).
* **Graph and eager agree to within 2.5% for both arms.** On gfx1100 stock
  vLLM's step is GEMM-bound, and capturing it saves launch overhead that is
  small next to 40 ms of GEMMs.
* **Not every call was routed.** At batch 32-64, vLLM captures graphs for
  every batch size up to 512. Under the hand rule, the capture sizes between
  64 and 1024 that are not whole 128-row blocks fell back (2,304 calls). The
  measured routing table below routes the ones that measured faster.

### Strided globals, and the two patches they were for that lost

A global used to be a base pointer plus the strides implied by its shape, so
any non-contiguous tensor was refused, and an engine's views had to be
copied first. The case that motivated the change is vLLM's fused gate/up
output (M, 2I), whose two halves `x[:, :I]` and `x[:, I:]` are what SiLU-mul
reads.

A global now carries one runtime row stride. It defaults to the column count,
and `operator[]` and `stride<axis>()` use it, which every load and store goes
through. Both entry points -- the pybind launcher and the torch-op scaffold
-- accept a slice of the last axis: contiguous along it, rows evenly strided,
outer axes compact over the rows. Anything else is refused with a message
saying why, rather than read as if it were that. Contiguous tensors take the
same path and pay the same one call as before. Rebuilt against the previous
commit, 63 of 66 kernel families are identical; dequantize is +1 VGPR at the
same rounded count and occupancy.

What it was for did not survive measurement. The plan was SiLU-mul and a
weighted RMSNorm in vLLM, each opt-in and kept only if the end-to-end did not
lose. Before writing either, the kernels were timed against what the engine
runs instead. In the mode vLLM ships, that is not its HIP kernels: under
torch.compile both ops come from Inductor, fused with their neighbours.
Kernel time on Qwen3-8B shapes, one process:

```
                               Inductor    hk
SiLU-mul (4096, 2*12288)        409.5 us   429.7 us   (contiguous halves)
RMSNorm  (4096, 4096)            38.8 us    55.8 us   (hk unweighted)
RMSNorm  q/k heads (163840, 128) 42.9 us    70.4 us   (hk unweighted)
```

The library's RMSNorm loses by 44-64% before it even has a weight to
multiply. An opaque op in that position would also cost Inductor the fusion
of the residual add and the weight. Replacing it could only make the request
slower, so both integrations were dropped rather than built and then
measured losing. (At M=16-64 hk looks 2x faster in the same table; that is
the compiled function's Python overhead, which does not exist inside vLLM's
graph.) Not measured: vLLM's own eager kernels, the baseline only in
`enforce_eager` mode. The strided global stays: it is what the next patch
that receives a view instead of a tensor needs.

### gfx1201: no arch baked into `hk.ops`, and fp8 WMMA tiles

Every kernel in `hk.ops` named `arch="gfx1100"`. They now name none, and a
kernel that names none is built for `hk.target.current_arch()`:
`$HK_ARCH`, else the GPU in the machine (`hk.runtime.arch.detect`), else
gfx1100. It is resolved on first use, so importing `hk.ops` still builds no
IR and probes no GPU. The two LDS limits that were spelled `65536` in a
gfx1100 error message read the target instead.

RDNA4 has an fp8 WMMA and RDNA3 does not, so fp8 support depends on the
target:

* `hk.fp8e4m3` and `hk.fp8e5m2` are dtypes. Register tiles of them, and
  `mma_*` on them -- including mixed e4m3/e5m2, which the four gfx12
  builtins allow -- are accepted where the target lists an fp8 WMMA shape.
  The gfx1201 target lists one: 2 VGPRs per 16x16 operand, from the builtin
  signature, unmirrored.
* On gfx1100 the `mma` says "fp8 operands need an fp8 WMMA, and gfx1100 has
  none -- gfx1201 (RDNA4) does". A kernel that only *holds* an fp8 tile is
  refused by the verifier with the alternative, load into fp32 or bf16. Both
  happen at trace time, not as a template error forty lines deep.
* A gfx1201 fp8 kernel (two fp8 loads, `mma_ABt`, an fp32 store) compiles
  with hipcc for gfx1201 at 37 VGPRs, no scratch, and its ISA is four
  `v_wmma_f32_16x16x16_fp8_bf8`. The strided global had to be mirrored into
  the RDNA4 layout first: that was the only thing that did not compile.

**Not verified on hardware.** There is no gfx12 part here, so none of this
has run: what is checked is the type rules, the build gate and the ISA. The
gfx1201 target's register granule and LDS pool are still carried over from
gfx1100 and marked unverified in `hk/target/gfx1201.py`.

Every `hk.ops` kernel now generates for gfx1201 (`HK_ARCH=gfx1201`). Under
gfx1100 all 306 instantiations generate byte-identical source and resource
caps to the previous commit, so no gfx1100 build changed.

### One patch layer, and a routing table that is measured rather than written

The GEMM patch and the attention backend each wrote the same parts by hand.
`hk.integration.engine` writes them once:

* **`Op`** is an opaque torch custom op with a fake implementation.
* **`Patch`** counts each path a call takes, and holds the patch's CUDA-graph
  contract.
* **`RouteTable`** maps a shape to a kernel, and replaces the transcribed
  rule.

`vllm_gemm` is an `Op` routed by a table. `vllm_backend` is a `Patch`: its
metadata builder's `_cudagraph_support` is derived from its declaration
instead of being a second statement of it.

**The table.** `tools/hk-bench/route_tune.py` measures each (N, K) of the
model at every M up to 64, every 16 rows up to 1024, and coarser above. At
each point it times the engine's path, the prefill GEMM and the decode GEMV,
each gated against fp32 first, interleaved in one process, with weights
rotated through 3x the Infinity Cache. A kernel takes an M only where it
beat the engine by 3%; a tie stays the engine's. The table answers an M as
follows:

* inside a run of measured Ms with the same winner, that winner;
* between two runs, which means between two measured Ms that disagreed, the
  engine's path;
* above the largest measured M, the winner measured there;
* below the smallest, or for a shape never measured, the engine's path, and
  the shape is recorded as a miss.

Tables are per arch and per model. `hk/tuned/routes_vllm_linear_gfx1100.json`
is Qwen3-8B's, raw timings included, and `$HK_VLLM_ROUTES` names another.
On qkv it reads:

```
     1..64     hk_decode
    80..128    hk          one 128-row block, full enough
   144..192    engine      a second block, mostly empty
   208..256    hk
   272..288    engine
   304..512    hk
   528..576    engine
   592..768    hk
   784         engine
   800..16384  hk
```

That is the comb the hand rule could only approximate: the teeth are the Ms
just past a multiple of 128. Scored on the tuning run itself, over its 59 Ms
from 65 to 1023, the table is 1.12x the hand rule in GEMM time per step
(geometric mean). That is in-sample, so the Ms where the two disagree were
re-measured in a fresh process. GEMM milliseconds per step, 36 layers plus
the LM head:

```
   M   engine   hand rule    table    table vs rule
  96    43.57      43.57     37.57    1.16x
 128    53.80      53.80     40.10    1.34x
 208    70.14      70.14     64.26    1.09x
 256    70.97      70.97     64.53    1.10x
 320    90.02      90.02     76.05    1.18x
 448   119.87     119.87    101.36    1.18x
 576   133.74     133.74    125.04    1.07x
 784   180.39     180.39    167.86    1.07x
1216   250.02     235.30    236.51    0.995x
1728   342.98     329.55    336.65    0.979x
```

Out of sample, the table never loses to the engine, and beats the hand rule
by 7-34% wherever the rule fell back. The last two rows are what the 3% bar
costs. There the true gain sits just under it, the fresh run disagreed with
the tuning run, and the blanket rule's "everything above 1024" happened to
be right: 0.5% and 2.1% of a step.

**End to end**, where the difference reaches a request: decode at batch 96
and 128, sizes a busy server runs, where a step's M is the batch. Graph mode,
Qwen3-8B, 512-token prompts, 160 output tokens, one round. The arms are
stock (1.00x), the patch routed by the hand rule written as a table
(`gemm@rule`, which reproduces the rule at every M from 0 to 19,999), and
the patch routed by the measured table:

```
             total ms                           decode ms per step
ctx x batch  stock    hand rule     table       stock  hand rule     table
512x96      27349.9  25561.4 1.07  24523.9 1.12  97.48  97.98 0.99  91.48 1.07
512x128     37156.1  34813.6 1.07  32419.1 1.15 136.11 136.92 0.99 121.14 1.12
```

These are throughput numbers (whole batches):

* Prefill is the same under both patched arms, 1.17-1.18x, because every
  prefill step here has M >= 1024, which both route.
* Decode is where they part. The hand rule sent M=96 and M=128 back to
  hipBLASLt, so its decode is stock's. The table routes them: 1.07x and
  1.12x per step, the 6 and 16 ms a step that the GEMM measurement predicted.
* The batch finishes 4.2% and 7.4% sooner than under the rule.
* Greedy output is identical to stock in every arm (160/160).
* The counters say the same thing. Under the rule, none of the graphs vLLM
  captured for batch sizes 72-128 used the GEMM; under the table, 1,944
  captured calls did.

**The graph check.** A patch declares `graph="safe"` or `graph="never"`, and
"never" must give its reason. Under capture:

* A "never" patch raises `GraphCaptureError` with that reason, instead of
  being frozen at the capture batch's values.
* A "safe" one runs with torch's sync debug mode set to error. A host sync
  in its wrapper then raises at the line that does it, naming the patch,
  instead of failing the capture somewhere else, or succeeding.
* Calls made while capturing are counted apart from eager calls, since a
  replay runs no Python.
* Changing the routes after a capture warns, because graphs captured before
  keep the old routes.

The limit is torch's: its sync debug mode is a prototype that sees torch's
own syncs (`.item()`, `.tolist()`, blocking copies, `synchronize()`), not a
raw HIP call inside an extension.

Not ported: the two SDPA rebinding patches (`vllm.py`, `sglang.py`) still
keep their own counters.

See `.claude/plans/` for the full plan and its gates.

### The IR as an MLIR dialect, and kernels that read like the math

Until here a kernel was a list of flat calls — `_ops.store(o, _ops.mul_row(
_ops.load(x, _ops.elem_coord(...), ty), r), idx)` — recorded into a Python
IR. That is a builder API, not an IR anyone would recognise, and nothing
else could read it. Now the IR is MLIR: an `hk` dialect in C++ (`mlir/`)
with its verifiers, the four passes and the translation to HipKittens C++,
and kernels are written in a Python that reads like the operations it
performs (`acc += a @ b.T`, `o[idx] = (v * r).to(hk.bf16)`, a plain `for`
over a device bound). The Python IR, its builder, verifier and passes, the
Python emitter and the flat API are deleted. The backend did not change:
hipcc and include/rdna3, so nothing at run time needs LLVM.

The move was gated three times, each against the code it replaced.

**Gate A — the dialect against the Python IR.** A temporary converter
turned every kernel the old front end traced into the dialect, and ran it
through the C++ passes and the C++ translation. For all 354 shipped
instantiations, and the constexpr points the tests use (708 translations),
the C++ was byte-identical to the Python emitter's. The whole no-GPU suite
ran under the gate as its corpus — 394 tests, the error paths included,
where the C++ passes had to refuse what the Python ones refused, in the same
words. Then the converter was deleted with the rest.

**Gate B — each kernel family against its old self** (elementwise, norm,
quant, fused, decode GEMV, GEMM, paged, attention):

* All 422 instantiations rendered both ways. 49 are byte-identical, 281
  differ only in a loop variable's name (`i` became the `kb` the source
  wrote), and 92 differ in constant folding, hoist order or names. Those 92
  were compiled: VGPRs, SGPRs, scratch, LDS and occupancy are identical in
  every one.
* tests/hk/gpu: 379 passed on gfx1100.
* Old and new in one process, interleaved for 7 rounds, the old as control:

  | kernel | shape | new / old |
  |---|---|---|
  | GEMM | 4096³ | 1.002x (71.6 TF) |
  | attention d128 | N=4096, non-causal / causal | 1.001x / 0.993x |
  | decode GEMV | M=1 N=4096, M=1 N=12288, M=4 N=4096 | 1.010x / 0.999x / 1.000x |
  | rmsnorm | 8192x4096, 8192x8192, 32768x1024 | 0.998x / 1.002x / 1.002x |

  Every gap is smaller than that row's own round-to-round spread, which is
  the expected answer for kernels whose C++ differs in names.

After the old front end was deleted, the 422 sources were rendered again:
byte-identical to the migration commit's, so the numbers above are the
shipped code's.

**Gate C — the engine.** vLLM 0.16.1 serving Qwen3-8B on the W7900D with the
GEMM patch (`vllm_e2e.py`, eager, context 1024, batch 1 and 16, 72 output
tokens, three rounds of each arm alternating), once from the tree before the
move and once after, each against stock vLLM in its own run:

| | before | after |
|---|---|---|
| total, batch 1 / 16 | 1.820x / 1.338x | 1.824x / 1.340x |
| prefill | 1.125x / 1.160x | 1.125x / 1.160x |
| decode per step | 1.928x / 1.735x | 1.933x / 1.746x |

The routing counters match call for call (5,472 prefill GEMMs, 102,553
decode GEMVs, no fallback). At both shapes the first request's greedy output
is identical to stock, and so to the other run's (72/72).

The first attempt at this gate measured nothing. With vLLM's engine in its
own process, the patch installed in the driver never reached the model, and
both arms timed as stock (0.995–1.000x, `hk 0, hk_decode 0`). The tool prints
the routing counters for exactly this reason. The gate is run with
`VLLM_ENABLE_V1_MULTIPROCESSING=0`, as every earlier measurement here was.

**Tests, now.** `mlir/test` (lit): 8, for the op verifiers, a round trip
through text, each of the four passes on hand-written IR (including the
workgroup-operation checks and the budget warning), and the translation of
a kernel and of a `while`. tests/hk/ir and
tests/hk/codegen: 984, no xfails. tests/hk/gpu: 383 passed on
gfx1100 (the 379 above and four for `while`).

**Wheels.** `scripts/build_wheel.sh` builds one per interpreter, because
`hk._mlir` is nanobind extension modules for one CPython:
`hipkittens-0.2.0-cp310-cp310-linux_x86_64.whl` (11.2 MB) and
`hipkittens-0.2.0-cp312-cp312-linux_x86_64.whl` (11.3 MB). Each
carries the package, include/ (the generated kernels include it at JIT
time) and the bindings. A wheel built for another interpreter installs and
then fails at import, so the script refuses to build one. The link-time
`libHKPythonCAPI.so` symlink is dropped, because a wheel would carry it as a
second 34 MB copy. Installed outside the checkout, each wheel
traces, translates, compiles against its own headers and launches on the
W7900D, with a result equal to torch's bit for bit.

**What porting the tests found.** About fifteen test files used the flat
API, and porting them found real problems in the new front end, not just
spelling:

* *Error locations were wrong two ways.* The frame filter skipped every
  file whose path contained `/hk/`, which includes `hk/ops` and `tests/hk`,
  so an error in a shipped kernel pointed at its caller. That bug predates
  the DSL. And an op that recorded its location directly (`barrier`,
  `lds.alloc`, `lds.wait_for`, a store) skipped one frame too many, landing
  on the line that called `k.trace()`. The filter is now the front end's own
  directories, with no fixed frame count. `hk.scope()` no longer reports
  contextlib, and a body lifted into a loop function reports the kernel
  function's name.
* *The dump and the C++ named values differently* (`%ld_0` vs `ld1`).
  `dump()` now prints a copy renamed by the translation's own naming, so the
  two cannot drift.
* *Seven of the Python-side messages had lost information* — which mma
  variant the layouts select, that an alternative is `x[idx].store(v)`, why a
  subtile has to divide its parent, that `//` on a tile is not a rounding
  division. The old wording is back. The C++ passes never lost theirs.

**What the language does not do.** Each of these raises, with the
spelling that works:

* A loop or branch body that `return`s, `break`s or `continue`s is left as
  Python, so its condition has to be known at trace time. `while ... else`
  is not rewritten.
* A workgroup operation (`barrier`, group loads and stores, staging) cannot
  sit inside a runtime branch or a runtime `while`: a warp that leaves early
  leaves the rest waiting. A `for` over a bound every warp agrees on is fine.
* A tile rebound inside a runtime loop is refused; it is updated in place
  (`+=`, `into=`), so that every iteration writes the same registers.
* A kernel body must be source `inspect` can read. Code built with `exec`
  or a lambda traces, but its loops cannot be runtime loops.
* Pass errors name values by their base name (`lds`), not the C++'s
  deduplicated one (`lds1`), so "add lds to that lds_wait_for" can be
  ambiguous with several LDS tiles in flight. The old front end had the same
  limit.
* Nothing is eliminated before the passes, so an unused allocation still
  counts in `lds_bytes`. The old front end behaved the same way.
* MLIR is built from source (23.1.3). Wheels are Linux x86_64 only, one
  CPython each; gfx1201 is still compile-only.

**Reproducing.**

```bash
scripts/build_mlir.sh && scripts/build_hk_mlir.sh
ninja -C $HK_MLIR_BUILD check-hk
python3 -m pytest tests/hk/ir tests/hk/codegen -q
python3 -m pytest tests/hk/gpu -q                       # gfx1100
scripts/build_wheel.sh
VLLM_ENABLE_V1_MULTIPROCESSING=0 HK_GPU_UTIL=0.85 python3 tools/hk-bench/vllm_e2e.py \
    --context 1024 --batch 1,16 --tokens 72 --short 24 --reps 3 --arms stock,gemm
```

**For whoever continues**, in order of value:

1. Name values in pass errors the way `dump()` does, by the translation's
   naming, so a message names one tile.
2. `break` and `continue` in runtime loops: a flag carried through an
   `scf.while`, or the `cf` dialect.
3. A canonicalize/DCE pass in front of the hk passes. It will change the
   C++, so it needs Gate B again, resources first.
4. Build MLIR in CI and publish the wheels; today they are built by hand on
   the machines that have the toolchain.
