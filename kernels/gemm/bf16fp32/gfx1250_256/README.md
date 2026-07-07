# gfx1250 256×256 GEMM — MI350X 8-wave layout

Mirrors MI350X [`256_256_64_32_with16x32.cpp`](../256_256_64_32_with16x32.cpp) wave
geometry, adapted for gfx1250 WMMA (wave-32).

## Kernels

| File | Status |
|------|--------|
| `gemm_sched.cpp` | **Verified** — MI350 ping-pong + `wait_ds`/`s_setprio`/`sched_barrier` scaffolding |
| `gemm_naive.cpp` | planned — coarse `wait_async` + `sync` baseline (not in tree yet) |

## MI350X ↔ gfx1250 mapping

| | **MI350X (CDNA4)** | **gfx1250 (this kernel)** |
|---|---|---|
| Waves / WG | **8** (2×4 grid) | **8** (2×4 grid) |
| Threads | 512 (wave-64) | 256 (wave-32) |
| Block tile | 256×256 | 256×256 |
| Wave C slice | **128×64** | **128×64** |
| C per wave | `C_accum[2][2]` of **64×32** | same |
| LDS quarters | 4× **128×64** (`As[2][2]`, `Bs[2][2]`) | 4× **128×32** (`As[2][2]`, `Bs[2][2]`) |
| K step | 64 | **32** (2 gfx1250 steps = 1 MI350 step) |
| MMA | MFMA `mma_ABt` | WMMA `mma_ABt` |
| g2s | `group<8>` buffer_load_lds | `load_async<256>` |
| Scheduling | `vmcnt` / `lgkmcnt` / `s_setprio` | `wait_async` / `wait_ds` / `s_setprio` (sched rung) |

## Build & run

```bash
make KERNEL=gemm_sched run M=512 N=512 K=512
```

M, N multiples of **256**; K multiple of **32**.

## Roadmap

- [x] MI350 ping-pong (`As[2][2]`, `Bs[2][2]`, `tic`/`toc`, `tile+=2`)
- [x] `load_async` g2s
- [ ] Calibrate `wait_async<N>` ↔ MI350 `vmcnt(N)` (sched rung)
- [ ] K_STEP=64 via two WMMA passes per slab
- [ ] WGM / chiplet swizzle from MI350 prologue
