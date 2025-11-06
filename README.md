## Experimental FP6 Support

There are two versions of HipKittens FP6: dwordx3 and dwordx4. Each contains a GEMM kernel using `buffer_load_dwordx3` or `buffer_load_dwordx4`, respectively, for global to LDS memory loads. Each also contains a version of the HipKittens headers which supports the corresponding approach. These are separate because there are a few differences in the library for each. For example, the shared allocator and shared subtile implementations are different.

`fp6_dwordx4` is currently faster and is likely the correct approach because these GEMM kernels are global-to-LDS bottlenecked.
