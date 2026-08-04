# gfx1250 GEMM optimization ladder

Ten bf16 -> fp32 GEMM rungs, ordered upward from a naive kernel to the fastest. Each rung adds one
hardware or algorithmic feature -- plus whatever that feature's API or latency structure forces
along with it -- and changes nothing else. Accumulation is fp32 throughout, and each rung is named
for the feature it adds.

Every rung shares `common.h`, `harness.h`, the TN operand contract (`a` is `[M, K]`, `b` is
`[N, K]`, both K-contiguous, so the kernel computes `C = A . B^T` via `mma_ABt`) and a column-major
`c`. They are written against `include/udna1/ops/warp/{sync,sched,cluster}/` and the gfx1250-only
extensions to `memory/tile/global_to_shared.cuh`, `memory/tile/shared_to_register.cuh` and
`register/tile/mma.cuh`.

## The ladder

Measured 2026-08-03 at 8192^3, 10 rounds, arm order rotated per round, every cell verified in the
invocation that timed it. The climb is monotone: 6.7x from rung 1 to rung 10.

| #  | rung                 | TFLOP/s | adds over the rung below |
|----|----------------------|--------:|--------------------------|
| 1  | `gemm_naive`         |   366.9 | baseline: one LDS slab, two barriers |
| 2  | `gemm_double_buf`    |   369.1 | a second LDS stage, so a fill overlaps the previous block's compute |
| 3  | `gemm_async`         |   606.0 | the cooperative async fill, `global_load_async_to_lds_b128` |
| 4  | `gemm_128x128`       |  1087.3 | macro tile 64x64 -> 128x128, and the warp grid 2x2 -> 4x4 with it |
| 5  | `gemm_256x256`       |  1363.9 | macro tile 128x128 -> 256x256 |
| 6  | `gemm_deepk`         |  1596.1 | the deep LDS stage, BLOCK_K 32 -> 128 |
| 7  | `gemm_segment`       |  1599.1 | LDS segment separation -- flat at this shape, kept for position |
| 8  | `gemm_tdm`           |  2198.1 | the hardware tile-DMA fill, `load_tdm` |
| 9  | `gemm_split_bar`     |  2304.1 | the barrier actually split, the last matrix op moving between signal and wait |
| 10 | `gemm_wgc_multicast` |  2450.3 | a 4x4 workgroup cluster multicasting both operands |

Two things are deliberately not rungs. Wave count is co-designed with rung 9 rather than free to
set on its own: with the barrier halves fused, 8 waves beats 4. Workgroup swizzles have nothing to
steer, because placement on this part is not a function of `blockIdx` and is not reproducible
across two identical launches.

## Reproducing it

`gemm_ladder.py` syncs the tree, builds every rung on the box, runs the rotated paired campaign and
prints the table. Rungs are positional and default to the ladder in order.

```
./gemm_ladder.py --host <hostname> -r 10 -i 10
./gemm_ladder.py --host <hostname> -r 20 --no-null gemm_deepk gemm_segment
```

The second form is how a single close step gets extra rounds. `--no-null` skips the null control,
so that run has no resolution of its own.

## Build

The kernels target `gfx1250` and need clang 22+ (ROCm 7.2 hipcc). Without that toolchain, run make
inside the `rocm/dev-ubuntu-24.04:7.2` image.

```
make KERNEL=gemm_naive               # build one rung
make ladder                          # build every rung
```

Each rung is a standalone executable taking `M N K iters verify`. At `verify=1` the harness checks
against an OpenMP CPU reference in the same invocation that timed it, which is why `make` always
passes `-fopenmp`.
