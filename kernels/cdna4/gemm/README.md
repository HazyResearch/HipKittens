# Matrix Multiply Kernels
The `main` branch of `HK` supports GEMMs using `BF16` and `FP8` datatypes. These kernels can be found in `bf16fp32/` and `fp8fp32/`, respectively. We have written a BF16 GEMM kernel for each of MI325X and MI350X. Our fastest BF16 GEMM for MI350X is `bf16fp32/mi350x/256_256_64_32_with16x32.cpp`.

See the READMEs in each subdirectory for information on running the kernels.

### FP6 GEMM
We also have experimental FP6 support and an FP6 GEMM kernel on branch `fp6_experimental`.
