# Epilogue benchmarks (gfx950)

Benchmarks for the fused GEMM-epilogue kernels in this directory, against PyTorch eager and
`torch.compile`. Every script checks its outputs against an fp32 PyTorch reference before timing,
prints a table, and exits non-zero if any row fails.

| Script | Question |
|---|---|
| `ab_noop_vs_base.py` | Does the epilogue chassis (`tk_noop`) cost anything versus upstream's original GEMM? |
| `bench_epilogues.py` | How much time and HBM traffic does each fused epilogue save versus eager PyTorch? |
| `bench_layer.py --phases` | Per phase of one forward layer: HK vs eager vs `torch.compile`. |
| `bench_layer.py --full` | The whole layer: HK vs eager vs `torch.compile`. |

## Build

From this directory's parent (the epilogue directory). Building needs a ROCm `hipcc` that targets gfx950
but no GPU; running needs a gfx950 GPU and PyTorch.

```bash
export THUNDERKITTENS_ROOT=$(cd ../../../../.. && pwd)   # this repository; an inherited root breaks every make
for m in noop silu scale residual_add rmsnorm_scale swiglu rmsnorm_swiglu rope rmsnorm_rope \
         partialrms residual_rms rms_reduce; do make KERNEL=$m; done
make -C ..                                  # upstream base GEMM, for ab_noop_vs_base.py
bench/build_attn.sh 32 8 128 2048 4096 8192 # attention for llama3-8b / pr1 (H, H_KV, Dh, M...)
bench/build_attn.sh 64 8 128 2048 4096 8192 # attention for llama3-70b
```

Build on a local disk: a network filesystem can make each module's build many times slower.

## Run

```bash
python3 bench/ab_noop_vs_base.py
python3 bench/bench_epilogues.py
python3 bench/bench_layer.py --phases --full
```

Common flags: `--config llama3-8b`, `--M 2048,4096`, `--iters 50`, `--warmup 10`, `--json out.json`,
`--markdown`. `torch.compile` autotuning can pick different kernels in different processes, so run each
script in 3 fresh processes and report the spread, not the best run.

## Config

Single-sequence causal prefill (batch 1) at M ∈ {2048, 4096, 8192} tokens. Configs, in `CONFIGS` in
`common.py`:

| `--config` | Model | d | d_ff | H | H_KV | Dh | Attention build |
|---|---|---:|---:|---:|---:|---:|---|
| `llama3-8b` (default) | Llama-3.1-8B, same shape as Mistral-7B | 4096 | 14336 | 32 | 8 | 128 | `build_attn.sh 32 8 128 ...` |
| `llama3-70b` | Llama-3.1-70B | 8192 | 28672 | 64 | 8 | 128 | `build_attn.sh 64 8 128 ...` |
| `pr1` | PR #1's shape: Llama-2-7B widths with GQA (not a real model) | 4096 | 11008 | 32 | 8 | 128 | `build_attn.sh 32 8 128 ...` |

Both Llama-3.1 models fit on one MI355X (288 GB), so the shapes are unsharded (no tensor parallelism).

## How timing works

- **Cold cache.** Before every timed call, the timer overwrites a buffer twice the last-level cache
  (2 × 256 MiB on gfx950), so each call reads its inputs from HBM. GPU events bracket only the call.
- **Median of 50 calls** after 10 warm-up calls; JSON also records p10 and p90.
- **Outside the timer:** input creation, weight preparation (`hk.prepare`: gamma folding, column
  permutation, transpose), and `torch.compile` compilation and autotuning.
- **One compile mode:** `torch.compile(mode="max-autotune-no-cudagraphs", dynamic=False)`. CUDA graphs
  are not used: both sides are limited by kernel time, not launch overhead.
- Eager and compiled PyTorch are separate columns; there is no best-of number.

## Columns in `bench_epilogues.py`

- **HK:** one fused HipKittens call.
- **eager:** `torch.matmul` (hipBLASLt) followed by the epilogue written as lean bf16 eager PyTorch.
- **fusion-only:** HK's `noop` GEMM followed by the same eager PyTorch. HK and fusion-only share one
  GEMM, so `fusion-only/HK` is the speedup from fusion alone; `eager/HK` also includes the GEMM difference.
- **HBM saved:** an analytical estimate, not a hardware measurement. Each epilogue has one formula in
  `bench_epilogues.py` (`saved=` in `make_case`): the bytes of every [M, N] or [M, K] intermediate the
  eager code writes to HBM and reads back, counted once for the write and once for the read. Vectors
  ([M] or [N]) are ignored. It ignores cache: a small intermediate may stay in the 256 MiB Infinity Cache,
  so the real saving can be lower. With bf16 sizes `MN = M·N·2` and `MK = M·K·2` bytes:

  | Epilogue | Eager intermediates | Saved |
  |---|---|---|
  | noop | none | 0 |
  | silu, scale, residual_add | D | 2·MN |
  | rmsnorm_scale | D, D·r | 4·MN |
  | rmsnorm | x·r, x·r·g | 4·MK |
  | swiglu | D, silu(gate) (half width) | 3·MN |
  | rmsnorm_swiglu | rmsnorm + swiglu | 4·MK + 3·MN |
  | rope | D, six half-width products and sums | 8·MN |
  | rmsnorm_rope | rmsnorm + rope | 4·MK + 8·MN |
  | inv_rms | D | 2·MN − HK's fp32 partials, 2·(N/64)·M·4 |
  | residual_rms | D, second read of h | 3·MN − HK's fp32 partials |

## Layer phases and boundary rules

| # | Phase | HK | PyTorch |
|---|---|---|---|
| 1 | qkv | `hk.qkv`: one `rmsnorm_rope` GEMM over `[Wq ∣ Wk ∣ Wv]` + 3 copies (4 launches) | `rms_norm`, three matmuls, RoPE on q and k |
| 2 | attention | `gqa_causal` on contiguous q, k, v | transposes, then SDPA |
| 3 | out_proj | `residual_rms(o, Wo, x)`: 2 launches | `x + o @ Wo` |
| 4 | gate_up | `rmsnorm_swiglu`: 1 launch | `rms_norm`, matmul, `silu(gate) * up` |
| 5 | down | `residual_rms(g, Wd, h)`: 2 launches | `h + g @ Wd`, then 1/rms for the next layer |

- HK splits RMSNorm across phases: the producing GEMM emits r = 1/rms, the consuming GEMM applies it.
  PyTorch normalizes in the consumer. Each side does each RMS reduction once per layer, so phase *sums*
  compare fairly, but a single phase does not compare op-for-op.
- HK's layer input `r_attn` is computed once outside timing (a previous layer would emit it).
- Each phase is timed on intermediates captured from a full run of the same implementation.
- `--phases` compiles each PyTorch phase separately; `--full` compiles the whole layer as one function.
  `compile full - phases` shows what cross-phase fusion buys PyTorch.
- **Attention baseline:** the script uses the strongest fused SDPA backend available (flash, then
  memory-efficient), with native GQA if that backend supports it, else KV expanded to all heads. The math
  backend is disabled. The chosen backend is printed and saved in the JSON.
