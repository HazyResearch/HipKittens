# MI350X FP8 GEMM Kernels

We have two FP8 GEMM kernels: 4 wave per threadblock and 8 wave per threadblock

The 4 wave per threadblock version is faster by around 3%, but 8 wave is more programmable. That is, developers will likely have an easier time extending the 8 wave gemm concept to additional kernels. Our 4 wave GEMM is built on load/store functions specific to this kernel (in `FP8_4wave/utils.cpp`), while our 8 wave GEMM is built on the standard `HK` memory functions.

### To run 4 wave
This runs the kernel with correctness check and performance reporting
```
cd FP8_4wave
make
./tk_kernel
```

### To run 8 wave
This runs the kernel with correctness check and performance reporting
```
cd FP8_8wave
make
./tk_kernel
```