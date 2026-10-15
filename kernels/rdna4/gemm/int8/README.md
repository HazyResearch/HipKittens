# gfx1201 INT8 GEMM

`C (int32) = A (int8, M×K) · Bᵀ (int8, N×K)`, exact int32 accumulation, on the
`v_wmma_i32_16x16x16_iu8` path. A and B are both K-contiguous.

## Use

```
make                          # builds gemm_i8 and mma_i8_test
make test                     # exact primitive test, then an exact GEMM check at 2048³
make run M=6144 N=6144 K=6144 # verify + time one shape
make bench                    # time the square shapes 2048³-12288³
```

`./gemm_i8 M N K iters verify` runs directly. `verify=1` compares 2048 sampled
outputs against an exact int32 dot product.

**Shapes must tile exactly.** Neither configuration handles partial tiles, so M,
N and K must be multiples of the chosen tile (below). `gemm_i8` refuses shapes
that don't fit and says which multiples it needs.

`mma_i8_test` checks the `wmma161616_acc(int2[4], int8_4[2], int8_4[2])` primitive
against a signed reference. The test must be signed: an unsigned build is wrong on
every negative product and produces plausible-looking large values.

## Configurations

`gemm_i8_kernel.h` defines the kernel, templated on a `gemm_i8_config<...>`.
`gemm_i8.cpp` instantiates two configurations and its `dispatch()` picks one per
shape:

| config | block tile | per-wave C tile | K-step | LDS | used when |
|---|---|---|---|---|---|
| `cfg_alt`  | 192×256 | 48×64 | 64 | double buffer, split barrier | M % 192 == 0, N % 256 == 0, K % 64 == 0 |
| `cfg_main` | 256×256 | 64×64 | 32 | 4-slot ring | otherwise (e.g. powers of two) |

## Schedule

Both configurations use 24 waves per workgroup, which is one workgroup per WGP:

- **8 producer waves** load A and B tiles from global into registers and commit
  them to LDS. They run no WMMA.
- **16 consumer waves** in a 4×4 grid each own a block of C (the per-wave tile
  above). They read 16×16 fragments of A and B from LDS and issue the WMMAs.

The two configurations differ in how producers and consumers stay out of each
other's way:

- **`cfg_main`: 4-slot LDS ring.** Producers run several K-steps ahead, writing
  slot `(k + lead) % 4` while consumers read slot `k % 4`. A block barrier every
  two K-steps is enough to keep the writer from overtaking the reader, which
  halves the barrier count compared with a double buffer.
- **`cfg_alt`: double buffer with a split barrier.** Consumers *signal* the
  barrier as soon as they have issued their LDS reads for a K-step, keep doing
  WMMAs, and *wait* only at the end of the step. Producers are released early to
  refill that buffer.

LDS holds A and B in 512-byte fragment units: lane `l` reads bytes `[16l, 16l+16)`
and gets one 16-row fragment for two consecutive k-steps, so every fragment read
is one contiguous 16-byte access and bank-conflict-free without padding or
swizzle. Workgroups are ordered in groups of `GROUP_M` tile-rows for L2 reuse.

## Tuning

The tuning parameters are template parameters of `gemm_i8_config` (in
`gemm_i8_kernel.h`), documented where they are declared. A configuration is one
line, e.g. `gemm_i8_config<256, 256, 32, true, 0>`, and `static_assert`s reject
combinations that would produce wrong results.
