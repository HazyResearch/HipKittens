# gfx1250 GEMM optimization ladder

A progressive series of bf16 -> fp32 GEMM kernels for gfx1250. Each
rung adds one gfx1250-specific feature on top of the previous, written
against the library surface in `include/ops/warp/{sync,sched,cluster}/` and
the gfx1250-only extensions to `global_to_shared.cuh`, `shared_to_register.cuh`,
and `register/tile/mma.cuh`.

Output dtype is bf16; accumulation is in fp32; default tile is 64x64 with
`K_STEP = 32`. Each kernel takes globals `{a, b, c}` where `a` is `[M, K]`,
`b` is `[N, K]` (so the kernel computes `C = A . B^T` via `mma_ABt`), and
`c` is `[M, N]`.

## Rungs

| File                  | New feature                                                                                    |
|-----------------------|------------------------------------------------------------------------------------------------|
| `gemm_naive.cpp`      | Baseline: `kittens::load`, `kittens::sync::sync`, `mma_ABt`, register-mediated copy. |
| `gemm_double_buf.cpp` | Double-buffered LDS.                                                                           |
| `gemm_async.cpp`      | `__builtin_amdgcn_global_load_async_to_lds_b128` via `kittens::load_async`.                    |
| `gemm_split_bar.cpp`  | Explicit `sync::arrive()` / `sync::wait()` split.                                              |
| `gemm_segment.cpp`    | A in `segment<0>`, B in `segment<1>` (distinct LDS read ports).                                |
| `gemm_expert.cpp`     | `sched::expert_scope` (reuse-B handled by `mma_ABt`'s default zigzag traversal).               |
| `gemm_tdm_arrive.cpp` | `load_tdm` + per-operand `barrier_lds`: fine-grained TDM ordering via `DS_ATOMIC_ASYNC_BARRIER_ARRIVE_B64` and a phase-flip wait. |

## Build

The kernels target `gfx1250` and require **clang 22+** (ROCm 7.2 hipcc).
On a host without that toolchain, run the make inside the
`rocm/dev-ubuntu-24.04:7.2` docker image; bind-mount the repo at `/work`.

The `Makefile` defines `KITTENS_UDNA1` and sets `--offload-arch=gfx1250`
automatically. From inside this directory:

```
make KERNEL=gemm_naive               # build one rung
make ladder                          # build every rung
```
