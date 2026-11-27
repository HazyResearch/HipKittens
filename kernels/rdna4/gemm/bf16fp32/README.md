# gfx1201 BF16 GEMM

`C (bf16) = A (bf16, M×K) · Bᵀ (bf16, N×K)`, fp32 accumulation. A and B are both
K-contiguous (row-major, BLAS "TN"). M, N and K need not be multiples of the tile.

## Use

```
make                                  # builds 128_512_64_16_ws8p16c (the default kernel)
make KERNEL=128_192_64_64_pingpong    # the other kernel
make run                              # verify against a CPU reference + time, 4096³
make run M=8192 N=1024 K=8192         # any shape
make bench                            # time a few square shapes
```

Each binary also runs directly: `./<kernel> M N K iters verify`. `verify=1` checks
every output against an exact fp32 CPU reference, `verify=2` checks 512 sampled
outputs (use it for large shapes), and `verify=0` only times.

Each file's configuration is fixed and stated by its name. The one build option
is `GROUP_M` (see below): `make EXTRA_HIPFLAGS=-DGROUP_M=16`. `make run` pins
clocks while timing; without that, short runs measure the clock ramp, not the
kernel.

To call a kernel from your own code, build without `-DHARNESS_MAIN` and use the
file's `gemm_globals` and `dispatch(gemm_globals)`.

## Kernels

Both use 16×16×16 bf16 WMMA, stage A through a double-buffered LDS tile, and
load B straight from global memory into registers (Tensile's "DirectToVgpr"),
since every wave reads a different slice of B. Workgroups are ordered in groups
of `GROUP_M` tile-rows so neighbouring workgroups share A and B in L2. It is 8 by
default; 16 is a few percent faster when K is a large power of two (A's rows then
alias in L2) and within noise elsewhere.

### `128_512_64_16_ws8p16c.cpp` — warp-specialised

Block tile 128×512, K-step 64, one 24-wave workgroup per WGP.

```
waves 0-7   producers   fetch A(k+1) global→regs ─┐   commit to LDS buf (k+1)&1
waves 8-23  consumers   fetch B(k) global→regs    │   A(k) from LDS buf k&1 → WMMA
                                                   └── one block barrier per K-step
```

- The **8 producer waves** only move A. Each K-step they fetch the *next* A tile
  from global into registers and commit it into the LDS buffer the consumers are
  not reading. They run no WMMA.
- The **16 consumer waves** form a 2×8 grid, each owning a 64×64 slice of C. Each
  K-step they fetch their B slice into registers, meet the producers at the
  block barrier, then read A from LDS in 16-wide K sub-steps and issue the WMMAs.
- One barrier per K-step is enough. The producer only overwrites a buffer two
  steps after the consumers last read it, and by then those reads have completed.
- The epilogue converts the fp32 accumulators to bf16 and stores them straight
  to global memory.

Producer waves keep the A-load bookkeeping (addresses, staging registers) out of
the waves that issue WMMAs.

### `128_192_64_64_pingpong.cpp` — symmetric ping-pong

Block tile 128×192, K-step 64, 8 waves in two groups of four. Simpler, and the
better starting point for a new shape.

```
             phase 0                         phase 1
group 0      compute on A(k)                 load its half of A(k+1)
group 1      load its half of A(k+1)         compute on A(k)
             ── barrier ──                   ── barrier ──
```

- Group *g* owns rows `[64g, 64g+64)` of the block, so the two groups read
  disjoint halves of A and have no data dependency on each other.
- Each K-step has two phases. In phase *p*, group *p* computes: it reads its A
  half from LDS, streams B into registers in double-buffered chunks, and issues
  the WMMAs. The other group meanwhile loads its half of the *next* A tile into
  the other LDS buffer. The groups swap at each barrier, so memory and matrix
  work overlap across waves, not within one wave's instruction stream.
- The barrier only keeps the groups in anti-phase and orders each group's own
  LDS writes before its reads. It never waits on the other group's data.

## Correctness

`make run` verifies by default. Gate every configuration change: a config can
build, run faster, and be wrong.
