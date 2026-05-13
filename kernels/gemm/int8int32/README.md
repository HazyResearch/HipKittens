# MI350X INT8 GEMM Kernel

INT8 x INT8 -> INT32 GEMM on CDNA4 (gfx950), built around `v_mfma_i32_16x16x64_i8`. Currently one variant: 4 wave per threadblock, modeled on `FP8_4wave` / `MXFP4_4wave` (BLOCK 256x256, 2x2 warp grid, `c[2][2]` per warp, double-buffered shared with vmcnt-paced prefetch, XCD-aware grid swizzle).

The per-kernel load/store helpers live in `INT8_4wave/utils.cpp` and are specific to this kernel (notably `load_one` uses `ds_read_b64` for the rt_16x64 base, vs. `ds_read_b128` in FP8/MXFP4 for rt_16x128).

### To run

Runs the kernel with correctness check and performance reporting.

```
cd INT8_4wave
ROCM_PATH=/opt/rocm make
./tk_kernel
```

Expected output:

```
INT8 x INT8 -> INT32 GEMM (M=4096, N=4096, K=4096)
...
  Kernel time (best): 0.063 ms,  TOPS: ~2180
  Kernel time (avg ): 0.065 ms,  TOPS: ~2090
Correctness: PASSED
```

### Changing the problem size

Edit one line at the top of `INT8_4wave/4_wave.cu`:

```c
#define SIZE 8192   // or 1024, 2048, etc.
```

then rebuild with `make && ./tk_kernel`. The CPU reference is OpenMP-parallel but still slow — at 8192³ the run waits ~20s on the reference, not the kernel. Comment out the `#pragma omp parallel for` block for pure perf runs.

### Constraints

- `M`, `N`, `K` must be multiples of `BLOCK_SIZE` (256). Other shapes will silently produce wrong output (the grid math assumes evenly tiled blocks).
- Inputs are int8 in `[-3, 3]` (seed 42), which keeps the int32 accumulator well below overflow even for K=8192.
- `make COMP_LEVEL=debug` builds with `-O0 -g` for stepping.

### Verifying the right MFMA is being emitted

```
ROCM_PATH=/opt/rocm /opt/rocm/bin/hipcc -std=c++20 -I/opt/rocm/include/hip \
  -I../../../../include -DKITTENS_CDNA4 --offload-arch=gfx950 -O3 --save-temps \
  -c 4_wave.cu -o /tmp/4_wave.o
grep mfma_i32 /tmp/4_wave-hip-amdgcn-amd-amdhsa-gfx950.s | head
```

Should print a stream of `v_mfma_i32_16x16x64_i8` lines.

### Measured throughput (M=N=K, MI355X, gfx950)

| Size  | best ms | TOPS  | % of 5 POPs peak |
|-------|---------|-------|------------------|
| 1024  | 0.019   |  113  |  2% (launch-bound) |
| 2048  | 0.030   |  573  | 11% |
| 4096  | 0.063   | 2187  | 44% |
| 8192  | 0.408   | 2696  | 54% |
