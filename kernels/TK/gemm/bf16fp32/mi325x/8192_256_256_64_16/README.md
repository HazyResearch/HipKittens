```
root@gpu-10:/workdir/AMD-benchmarking-harness/kernels/TK/bf16fp32/mi325x/256_256_64_16# python test_python.py
src: 256_256_64_16.cpp
Warning: tk_kernel.cpython-313-x86_64-linux-gnu.so not found at /workdir/AMD-benchmarking-harness/kernels/TK/bf16fp32/mi325x/256_256_64_16/tk_kernel.cpython-313-x86_64-linux-gnu.so, skipping.
C_ref.dtype=torch.bfloat16
PyTorch reference average execution time: 1.3800 ms
PyTorch reference performance: 796.75 TFLOPS for 8192x8192 matrix multiplication.

C.dtype=torch.bfloat16
Average execution time: 1.4396 ms
Performance: 763.74 TFLOPS for 8192x8192 matrix multiplication.

Max error between kernel and reference: 0.03125
Max error: 0.03125
Mean error: 0.0020266701467335224
Number of large errors (>0.1): 0

diff[:32, :32].max() tensor(0.0156, device='cuda:0')
diff[:32, 32:64].max() tensor(0.0156, device='cuda:0')
diff[32:64, :32].max() tensor(0.0156, device='cuda:0')
diff[32:64, 32:64].max() tensor(0.0156, device='cuda:0')

diff[7168:7232, 7168:7232].max() tensor(0.0156, device='cuda:0')
diff[7232:7296, 7232:7296].max() tensor(0.0156, device='cuda:0')
diff[7296:7360, 7296:7360].max() tensor(0.0156, device='cuda:0')
diff[7360:7424, 7360:7424].max() tensor(0.0156, device='cuda:0')
```