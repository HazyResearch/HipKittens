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

## Benchmark

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
