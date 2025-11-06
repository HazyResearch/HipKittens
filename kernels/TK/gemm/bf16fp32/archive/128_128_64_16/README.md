```
128_128_64_16.cpp:39:1: remark:     SGPRs: 34 [-Rpass-analysis=kernel-resource-usage]
128_128_64_16.cpp:39:1: remark:     VGPRs: 126 [-Rpass-analysis=kernel-resource-usage]
128_128_64_16.cpp:39:1: remark:     AGPRs: 0 [-Rpass-analysis=kernel-resource-usage]
128_128_64_16.cpp:39:1: remark:     ScratchSize [bytes/lane]: 0 [-Rpass-analysis=kernel-resource-usage]
128_128_64_16.cpp:39:1: remark:     Dynamic Stack: False [-Rpass-analysis=kernel-resource-usage]
128_128_64_16.cpp:39:1: remark:     Occupancy [waves/SIMD]: 4 [-Rpass-analysis=kernel-resource-usage]
128_128_64_16.cpp:39:1: remark:     SGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]
128_128_64_16.cpp:39:1: remark:     VGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]
128_128_64_16.cpp:39:1: remark:     LDS Size [bytes/block]: 0 [-Rpass-analysis=kernel-resource-usage]

src: 128_128_64_16.cpp
Warning: tk_kernel.cpython-313-x86_64-linux-gnu.so not found at /workdir/AMD-benchmarking-harness/kernels/TK/bf16fp32/mi325x/tk_kernel.cpython-313-x86_64-linux-gnu.so, skipping.
PyTorch reference average execution time: 1.3724 ms
PyTorch reference performance: 801.18 TFLOPS for 8192x8192 matrix multiplication.

Average execution time: 1.8508 ms
Performance: 594.08 TFLOPS for 8192x8192 matrix multiplication.

Max error between kernel and reference: 0.015614509582519531
Max error: 0.015614509582519531
Mean error: 0.001016465830616653
Number of large errors (>0.1): 0

diff[:32, :32].max() tensor(0.0078, device='cuda:0')
diff[:32, 32:64].max() tensor(0.0076, device='cuda:0')
diff[32:64, :32].max() tensor(0.0075, device='cuda:0')
diff[32:64, 32:64].max() tensor(0.0075, device='cuda:0')

diff[7168:7232, 7168:7232].max() tensor(0.0078, device='cuda:0')
diff[7232:7296, 7232:7296].max() tensor(0.0077, device='cuda:0')
diff[7296:7360, 7296:7360].max() tensor(0.0078, device='cuda:0')
diff[7360:7424, 7360:7424].max() tensor(0.0078, device='cuda:0')
```