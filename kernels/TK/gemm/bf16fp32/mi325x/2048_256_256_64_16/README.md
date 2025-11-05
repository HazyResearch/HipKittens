```
root@gpu-10:/workdir/AMD-benchmarking-harness/kernels/TK/gemm/bf16fp32/mi325x/2048_256_256_64_16# make
/opt/rocm/bin/hipcc 256_256_64_16.cpp -DKITTENS_CDNA3 --offload-arch=gfx942 -std=c++20 -w -I/workdir/AMD-benchmarking-harness/ThunderKittens-HIP/include -I/workdir/AMD-benchmarking-harness/ThunderKittens-HIP/prototype -I/opt/conda/envs/py_3.12/include/python3.12 -I/opt/conda/envs/py_3.12/lib/python3.12/site-packages/pybind11/include -L/opt/conda/envs/py_3.12/lib/python3.12/config-3.12-x86_64-linux-gnu -L/opt/conda/envs/py_3.12/lib  -lpthread -ldl  -lutil -lm  -shared -fPIC -Rpass-analysis=kernel-resource-usage -I/workdir/AMD-benchmarking-harness/ThunderKittens-HIP/include -I/opt/rocm/include/hip  \
    -o tk_kernel.cpython-312-x86_64-linux-gnu.so 2>&1 | tee /workdir/data_logs/0715_222240_outputs/make_build.log
256_256_64_16.cpp:38:1: remark: Function Name: _Z8micro_tk13micro_globals [-Rpass-analysis=kernel-resource-usage]
   38 | void micro_tk(const micro_globals g) {
      | ^
256_256_64_16.cpp:38:1: remark:     SGPRs: 36 [-Rpass-analysis=kernel-resource-usage]
256_256_64_16.cpp:38:1: remark:     VGPRs: 102 [-Rpass-analysis=kernel-resource-usage]
256_256_64_16.cpp:38:1: remark:     AGPRs: 0 [-Rpass-analysis=kernel-resource-usage]
256_256_64_16.cpp:38:1: remark:     ScratchSize [bytes/lane]: 0 [-Rpass-analysis=kernel-resource-usage]
256_256_64_16.cpp:38:1: remark:     Dynamic Stack: False [-Rpass-analysis=kernel-resource-usage]
256_256_64_16.cpp:38:1: remark:     Occupancy [waves/SIMD]: 4 [-Rpass-analysis=kernel-resource-usage]
256_256_64_16.cpp:38:1: remark:     SGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]
256_256_64_16.cpp:38:1: remark:     VGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]
256_256_64_16.cpp:38:1: remark:     LDS Size [bytes/block]: 0 [-Rpass-analysis=kernel-resource-usage]
root@gpu-10:/workdir/AMD-benchmarking-harness/kernels/TK/gemm/bf16fp32/mi325x/2048_256_256_64_16# python test_python.py
src: 256_256_64_16.cpp
Warning: tk_kernel.cpython-313-x86_64-linux-gnu.so not found at /workdir/AMD-benchmarking-harness/kernels/TK/gemm/bf16fp32/mi325x/2048_256_256_64_16/tk_kernel.cpython-313-x86_64-linux-gnu.so, skipping.
C_ref.dtype=torch.bfloat16
PyTorch reference average execution time: 0.0455 ms
PyTorch reference performance: 377.21 TFLOPS for 2048x2048 matrix multiplication.

C.dtype=torch.bfloat16
Average execution time: 0.0518 ms
Performance: 331.63 TFLOPS for 2048x2048 matrix multiplication.

Max error between kernel and reference: 0.015625
Max error: 0.015625
Mean error: 0.0010135583579540253
Number of large errors (>0.1): 0

diff[:32, :32].max() tensor(0.0078, device='cuda:0')
diff[:32, 32:64].max() tensor(0.0078, device='cuda:0')
diff[32:64, :32].max() tensor(0.0078, device='cuda:0')
diff[32:64, 32:64].max() tensor(0.0078, device='cuda:0')
```