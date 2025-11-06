```
256_128_64_32.cpp:41:1: remark:     SGPRs: 34 [-Rpass-analysis=kernel-resource-usage]
256_128_64_32.cpp:41:1: remark:     VGPRs: 207 [-Rpass-analysis=kernel-resource-usage]
256_128_64_32.cpp:41:1: remark:     AGPRs: 0 [-Rpass-analysis=kernel-resource-usage]
256_128_64_32.cpp:41:1: remark:     ScratchSize [bytes/lane]: 0 [-Rpass-analysis=kernel-resource-usage]
256_128_64_32.cpp:41:1: remark:     Dynamic Stack: False [-Rpass-analysis=kernel-resource-usage]
256_128_64_32.cpp:41:1: remark:     Occupancy [waves/SIMD]: 2 [-Rpass-analysis=kernel-resource-usage]
256_128_64_32.cpp:41:1: remark:     SGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]
256_128_64_32.cpp:41:1: remark:     VGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]
256_128_64_32.cpp:41:1: remark:     LDS Size [bytes/block]: 0 [-Rpass-analysis=kernel-resource-usage]

root@gpu-10:/workdir/AMD-benchmarking-harness/kernels/TK/bf16fp32/mi325x/256_128_64_32_outbf16# python test_python.py 
src: 256_128_64_32.cpp
Warning: tk_kernel.cpython-313-x86_64-linux-gnu.so not found at /workdir/AMD-benchmarking-harness/kernels/TK/bf16fp32/mi325x/256_128_64_32_outbf16/tk_kernel.cpython-313-x86_64-linux-gnu.so, skipping.
C_ref.dtype=torch.bfloat16
PyTorch reference average execution time: 1.3812 ms
PyTorch reference performance: 796.03 TFLOPS for 8192x8192 matrix multiplication.

C.dtype=torch.bfloat16
Average execution time: 1.6948 ms
Performance: 648.74 TFLOPS for 8192x8192 matrix multiplication.

Max error between kernel and reference: 0.015625
Max error: 0.015625
Mean error: 2.178580871259328e-07
Number of large errors (>0.1): 0

diff[:32, :32].max() tensor(0., device='cuda:0')
diff[:32, 32:64].max() tensor(0., device='cuda:0')
diff[32:64, :32].max() tensor(0., device='cuda:0')
diff[32:64, 32:64].max() tensor(0., device='cuda:0')

diff[7168:7232, 7168:7232].max() tensor(0., device='cuda:0')
diff[7232:7296, 7232:7296].max() tensor(0., device='cuda:0')
diff[7296:7360, 7296:7360].max() tensor(0., device='cuda:0')
diff[7360:7424, 7360:7424].max() tensor(0., device='cuda:0')
```