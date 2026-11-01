# Epilogue benchmarks (gfx950)

Fused GEMM-epilogue kernels vs PyTorch eager and `torch.compile`. Each script checks outputs against an
fp32 PyTorch reference before timing, prints a table, and exits non-zero if any row fails. In `bench_layer.py`
a phase whose HK outputs pass but whose eager or compiled PyTorch output misses is still timed and marked
`TORCH_MISS` (`ok_eager` / `ok_compile` in the JSON).

| Script | Question |
|---|---|
| `ab_noop_vs_base.py` | Does `tk_noop` cost anything versus the upstream base GEMM? |
| `bench_epilogues.py` | Time and HBM traffic each fused epilogue saves versus eager and `torch.compile`. |
| `bench_layer.py --phases` | Per layer phase: HK vs eager vs `torch.compile`, plus a fusion-only control per GEMM phase. |
| `bench_layer.py --full` | The whole layer: HK vs eager vs `torch.compile`. |

## Build

From the epilogue directory. Building needs a gfx950 `hipcc` (no GPU); running needs a gfx950 GPU and PyTorch.

```bash
export THUNDERKITTENS_ROOT=$(cd ../../../../.. && pwd)   # override any inherited root
for m in noop noop_qkv silu scale residual_add rmsnorm_scale swiglu rmsnorm_swiglu rope rmsnorm_rope \
         rmsnorm_rope_qkv partialrms residual_rms residual_rms_gamma rms_reduce; do make KERNEL=$m; done
make -C ..                                  # base GEMM, for ab_noop_vs_base.py
bench/build_attn.sh 32 8 128 2048 4096 8192 # attention for llama3-8b / pr1 (H, H_KV, Dh, M...)
bench/build_attn.sh 64 8 128 2048 4096 8192 # attention for llama3-70b
```

Build on a local disk; network filesystems are much slower.

## Run

```bash
python3 bench/ab_noop_vs_base.py
python3 bench/bench_epilogues.py
python3 bench/bench_layer.py --phases --full
```

Common flags: `--config llama3-8b`, `--M 2048,4096`, `--iters 50`, `--warmup 10`, `--json out.json`,
`--markdown`. `torch.compile` autotuning varies across processes: run each script in 3 fresh processes and
report the spread, not the best run.

## Config

Batch-1 causal prefill at M ∈ {2048, 4096, 8192} tokens. `CONFIGS` in `common.py`:

| `--config` | Model | d | d_ff | H | H_KV | Dh | Attention build |
|---|---|---:|---:|---:|---:|---:|---|
| `llama3.2-1b` | Llama-3.2-1B | 2048 | 8192 | 32 | 8 | 64 | `build_attn.sh 32 8 64 ...` |
| `llama3.2-3b` | Llama-3.2-3B | 3072 | 8192 | 24 | 8 | 128 | `build_attn.sh 24 8 128 ...` |
| `llama3-8b` (default) | Llama-3.1-8B, same shape as Mistral-7B | 4096 | 14336 | 32 | 8 | 128 | `build_attn.sh 32 8 128 ...` |
| `llama3-70b` | Llama-3.1-70B | 8192 | 28672 | 64 | 8 | 128 | `build_attn.sh 64 8 128 ...` |
| `llama3-405b` | Llama-3.1-405B (one layer) | 16384 | 53248 | 128 | 8 | 128 | `build_attn.sh 128 8 128 ...` |
| `pr1` | PR #1's shape: Llama-2-7B widths with GQA (not a real model) | 4096 | 11008 | 32 | 8 | 128 | `build_attn.sh 32 8 128 ...` |

Widths are from each model's Hugging Face `config.json`, unsharded (one layer fits on one 288 GB MI355X). The fp32
attention reference holds `[H, M, M]` scores: 405B at M = 16384 needs about 128 GiB, plus a copy in softmax.

## How timing works

- **Cold cache.** Before every timed call the timer overwrites 2 × LLC (2 × 256 MiB on gfx950). GPU events
  bracket only the call.
- **Median of 50 calls** after 10 warm-up calls; JSON also records p10 and p90.
- **Outside the timer:** input creation (including the one-head `[M, Dh]` RoPE table, broadcast over heads by
  every timed arm), HK's RMSNorm inputs from a previous GEMM (`r`, `x · gamma`), and `torch.compile` compilation
  and autotuning.
- **Same plain `[N, K]` weights for every arm**, never folded, permuted, transposed or concatenated. HK reads
  them in place (QKV via three pointers); PyTorch multiplies by the `W.t()` view.
- **RMSNorm gamma:** HK applies it in the producer (`hk.residual_rms(..., gamma=g)` writes `h · g`); PyTorch per
  call (`F.rms_norm(x, ..., g)` in the layer, `x · r · g` in `bench_epilogues.py`).
- **One compile mode:** `torch.compile(mode="max-autotune-no-cudagraphs", dynamic=False)`. No CUDA graphs:
  both sides are kernel-bound.
- Eager and compiled PyTorch are separate columns; there is no best-of number.

## Columns in `bench_epilogues.py`

- **HK:** one fused HipKittens call.
- **eager:** `torch.matmul(x, W.t())` (hipBLASLt), then the epilogue in bf16 eager PyTorch.
- **compile:** the eager function under `torch.compile`; Dynamo is reset per row so no row hits the recompile
  limit and silently runs eager.
- **fusion-only:** HK's `noop` GEMM, then the same eager PyTorch. `fusion-only/HK` isolates fusion;
  `eager/HK` also includes the GEMM difference.
- **HBM saved:** an analytical upper bound (`saved=` in `make_case`): each [M, N] or [M, K] intermediate the
  eager code writes and reads back, counted once per write and read. Vectors ([M], [N]) and caching (256 MiB
  Infinity Cache) are ignored. With bf16 sizes `MN = M·N·2` and `MK = M·K·2` bytes:

  | Epilogue | Eager intermediates | Saved |
  |---|---|---|
  | noop | none | 0 |
  | silu, scale, residual_add | D | 2·MN |
  | rmsnorm_scale | D, D·r | 4·MN |
  | rmsnorm | x·r, x·r·g, minus HK's producer store of x·g | 3·MK |
  | swiglu | D, silu(gate) (half width) | 3·MN |
  | rmsnorm_swiglu | rmsnorm + swiglu | 3·MK + 3·MN |
  | rope | D, six half-width products and sums | 8·MN |
  | rmsnorm_rope | rmsnorm + rope | 3·MK + 8·MN |
  | inv_rms | D | 2·MN − HK's fp32 partials, 2·(N/64)·M·4 |
  | residual_rms | D, second read of h (1/rms), third read of h (h·g) | 4·MN − HK's fp32 partials |

  `rmsnorm*` rows give HK `x · g` and `r` as its producer writes them; bounds are net of that producer's extra
  `h · g` store (one MK).

## Layer phases and boundary rules

| # | Phase | HK | PyTorch |
|---|---|---|---|
| 1 | qkv | `hk.qkv`: one `rmsnorm_rope` GEMM over Wq, Wk, Wv (three pointers) + 3 copies (4 launches) | `rms_norm(x, g_attn)`, three matmuls, RoPE on q and k |
| 2 | attention | `gqa_causal` on contiguous q, k, v | transposes, then SDPA |
| 3 | out_proj | `residual_rms(o, Wo, x, gamma=g_mlp)`: 2 launches, writes h, h·g_mlp, r | `x + o @ Wo.t()` |
| 4 | gate_up | `rmsnorm_swiglu` on h·g_mlp: 1 launch | `rms_norm(h, g_mlp)`, matmul, `silu(gate) * up` |
| 5 | down | `residual_rms(g, Wd, h, gamma=g_attn)`: 2 launches, writes x_out, x_out·g_attn, r | `h + g @ Wd.t()` |

- HK's producing GEMM emits r = 1/rms and h·gamma; the consumer applies r. PyTorch normalizes in the consumer,
  so single phases do not compare op-for-op.
- HK's `r_attn` and `x · g_attn` are computed untimed (a previous down phase would emit them, as this layer's
  down emits the next layer's). Each timed layer does two RMS reductions: HK in its producers (`r_mlp` in
  out_proj, `r_next` in down), PyTorch in its consumers (`rms_norm` in qkv and gate_up).
- **fusion-only control (`--phases`):** per GEMM phase, the same HK GEMM(s) with `noop` on HK's inputs, then
  the epilogue as eager PyTorch (QKV: `(x · g) · r`, one noop GEMM over the three weights, RoPE, V copy;
  out_proj/down: residual add, `h · g`, 1/rms; gate_up: `(h · g) · r`, SwiGLU).
- **Control tolerance:** the control's GEMM output passes through HK's truncating bf16 store, and SwiGLU
  multiplies two such values, so its bf16 outputs are gated at `rtol 2.5e-2` (`CONTROL_TOL`) in both scripts;
  HK, eager and compiled PyTorch keep `rtol 1e-2`.
- Each phase is timed on intermediates captured from a full run of the same implementation.
- `--phases` compiles each PyTorch phase separately; `--full` compiles the layer as one function.
  `compile full - phases` shows what cross-phase fusion buys PyTorch.
- **Attention baseline:** the strongest available fused SDPA backend (flash, then memory-efficient), native GQA
  if supported, else KV expanded. The math backend is disabled. The backend is printed and saved in the JSON.
- **Known PyTorch miss:** with torch 2.13 + ROCm 10 (AOTriton 0.13), flash SDPA at head dim 64 (Llama-3.2-1B)
  and M ≥ 8192 returns different outputs for identical inputs; about 1 call in 8 exceeds `STAGE_TOL`, so those
  attention rows can read `TORCH_MISS`. Head dim 128, M ≤ 4096, the CK and math SDPA backends, and torch 2.11 +
  ROCm 7.13 (AOTriton 0.11) are deterministic.
