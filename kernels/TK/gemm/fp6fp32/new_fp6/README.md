
# FP6 8192x8192x8192 GEMM with 256x256 tile & K-step 128
Kernel is in file: `variants/16x16x128_4wave.cpp` and the name is `micro_tk`

To run: in `new_fp6/`: `make && ./tk_kernel`

This kernel is a FP6 matrix multiply with size 8192x8192x8192. It accumulates in fp32 and writes out in bfloat16. There is no persistent grid, and we launch a total of 1024 thread blocks of 4 waves each. Each thread block computes a 256x256 tile of the result.

Inside each thread block, we use an M & N - sliced schedule, where each wave owns 4 64x64 tiles of the result - one in each quadrant of the 256x256 result tile.

We double buffer shared memory (2 256x128 A shared tiles and 2 256x128 B shared tiles - both row-major) and load from global in the following order inside the hot loop:
- top half of A tile
- top half of B tile
- bottom half of B tile
- bottom half of A tile

Each of the above loads is actually 4 `load_dwordx3` instructions. In this kernel, we've manually interleaved each individual `load_dwordx3` with the `ds_read_b96` and `mfma` instructions.


## Some notes to self for development

Check available ROCM intrinsics:

```bash

grep -r -i "fp6" /opt/rocm-7.0.0/include/ > out.log
```

https://rocm.docs.amd.com/en/docs-7.0-rc1/preview/install/rocm.html

```bash
# measure bank conflicts
rocprofv3 --pmc SQ_INSTS_LDS SQ_LDS_BANK_CONFLICT --output-format csv --output-file lds_conflict -d out -- ./tk_kernel

# view bank conflicts
python out/analyze_conflicts.py
```



