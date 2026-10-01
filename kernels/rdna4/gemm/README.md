# Matrix Multiply Kernels — RDNA4 / gfx1201

GEMM kernels for the Radeon RX 9070 XT (gfx1201):

- `bf16fp32/` — bf16 in, fp32 accumulate, bf16 out. Two kernels: a
  warp-specialised one (the default) and a symmetric ping-pong one.
- `int8/` — int8 in, exact int32 out.

Each directory has a Makefile (`make`, `make test` or `make run`) and a README
describing how to use the kernels and how their schedules work.

RDNA4 differs from the CDNA trees in ways that shape every kernel here: waves are
32 lanes; the matrix instruction is a 16×16×16 WMMA; there is no direct
global→LDS copy, so every LDS fill passes through registers; and a wave can
address at most 256 VGPRs.
