# gfx1250 128×128 GEMM ladder (path to 256×256)

Progressive bf16 → fp32 WMMA GEMM for gfx1250 (UDNA1), scaled from the
64×64 teaching ladder in `../gfx1250/`. This directory validates larger
block tiles before growing to MI350-class 256×256 geometry.

## Tile geometry (phase 1 — current)

| Parameter | 64 ladder | **128 ladder (here)** | 256 target |
|-----------|-----------|----------------------|------------|
| Block output | 64×64 | **128×128** | 256×256 |
| K step | 32 | 32 → 64 | 64 |
| Warps (M×N) | 2×2 | **4×4** | 4×4 or 8×4 |
| Threads | 128 | **512** | 512–1024 |
| Warp output | 32×32 | **32×32** (unchanged) | 32×32 or 64×64 |
| LDS layout | single slab | single slab | **2×2 half-tile split** |
| g2s | reg / async / TDM | same rungs TBD | async + TDM |

**Design choice:** keep the proven 32×32 warp micro-tile from the 64 ladder
and scale the block by adding warps (2×2 → 4×4). That reuses `rt_16x32`
operands, `rt_16x16` accumulators, and the existing s2r / store helpers
without new register layouts.

LDS budget at 128×128, K_STEP=32, double-buffered:
~2 × (128×32 + 128×32) bf16 ≈ 32 KiB logical (+ padding) — fits in one
64 KiB segment with room to grow K_STEP to 64.

## Rungs

| File | Status | New feature |
|------|--------|-------------|
| `gemm_naive.cpp` | **implemented** | Baseline reg-mediated g2s, 128×128 block |
| `gemm_double_buf.cpp` | planned | Double-buffered LDS |
| `gemm_async.cpp` | planned | `load_async` g2s |
| `gemm_split_bar.cpp` | planned | Split `arrive` / `wait` |
| `gemm_segment.cpp` | planned | A @ seg0, B @ seg1 |
| `gemm_expert.cpp` | planned | `sched::expert_scope` |
| `gemm_tdm_arrive.cpp` | planned | TDM + LDS barriers (silicon only) |

## Phase 2 — 256×256 (after 128 rungs pass)

Mirror MI350 `256_256_64_32_with16x32.cpp` structure adapted for WMMA:

1. **BLOCK 256**, **K_STEP 64**
2. **2×2 LDS half-tiles** — four 128×64 slabs (`As[2][2]`, `Bs[2][2]`)
3. **8 warps** (2×4) with 128×64 warp tiles, or 16 warps with 64×64
4. **Deep K-pipeline** — 2 K-tiles per loop iteration (manual waitcnt)
5. **Workgroup swizzle** — chiplet + WGM grouping when NUM_XCDS > 1

## Build & run

```bash
make KERNEL=gemm_naive
make KERNEL=gemm_naive run M=256 N=256 K=256
```

Requires ROCm 7.2+ hipcc (`--offload-arch=gfx1250`). On FFM-Lite:

```bash
~/run_gfx1250.sh -- bash -lc 'cd kernels/gemm/bf16fp32/gfx1250_128 && make run'
```

M, N, K must be multiples of 128.
