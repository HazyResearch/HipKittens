
```bash
# measure bank conflicts
rocprofv3 --pmc SQ_INSTS_LDS SQ_LDS_BANK_CONFLICT --output-format csv --output-file lds_conflict -d out -- python3 test_python.py

# view bank conflicts
python out/analyze_conflicts.py
```

```
root@tg-mi350-node5:/workdir/AMD-benchmarking-harness/kernels/TK/gemm/bf16fp32/mi350x/256_64# make
/opt/rocm/bin/hipcc kernel.cpp -DKITTENS_CDNA4 --offload-arch=gfx950  -std=c++20 -w -L/opt/conda/envs/py_3.10/lib/python3.10/config-3.10-x86_64-linux-gnu -L/opt/conda/envs/py_3.10/lib   -lpthread -ldl  -lutil -lm -lm  -I/workdir/AMD-benchmarking-harness/ThunderKittens-HIP/include -I/workdir/AMD-benchmarking-harness/ThunderKittens-HIP/prototype -I/opt/conda/envs/py_3.10/include/python3.10 -I/opt/conda/envs/py_3.10/lib/python3.10/site-packages/pybind11/include -shared -fPIC -Rpass-analysis=kernel-resource-usage --save-temps -I/workdir/AMD-benchmarking-harness/ThunderKittens-HIP/include -I/opt/rocm/include/hip  \
    -o tk_kernel.cpython-310-x86_64-linux-gnu.so 2>&1 | tee /workdir/data_logs/0719_230312_outputs/make_build.log
remark: kernel.cpp:38:0: Function Name: _Z8micro_tk13micro_globals [-Rpass-analysis=kernel-resource-usage]
remark: kernel.cpp:38:0:     SGPRs: 25 [-Rpass-analysis=kernel-resource-usage]
remark: kernel.cpp:38:0:     VGPRs: 184 [-Rpass-analysis=kernel-resource-usage]
remark: kernel.cpp:38:0:     AGPRs: 0 [-Rpass-analysis=kernel-resource-usage]
remark: kernel.cpp:38:0:     ScratchSize [bytes/lane]: 0 [-Rpass-analysis=kernel-resource-usage]
remark: kernel.cpp:38:0:     Dynamic Stack: False [-Rpass-analysis=kernel-resource-usage]
remark: kernel.cpp:38:0:     Occupancy [waves/SIMD]: 2 [-Rpass-analysis=kernel-resource-usage]
remark: kernel.cpp:38:0:     SGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]
remark: kernel.cpp:38:0:     VGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]
remark: kernel.cpp:38:0:     LDS Size [bytes/block]: 0 [-Rpass-analysis=kernel-resource-usage]
root@tg-mi350-node5:/workdir/AMD-benchmarking-harness/kernels/TK/gemm/bf16fp32/mi350x/256_64# python test_python.py
Out
tensor([[    -0.0152,      0.1196,      0.0454,     -0.0654,     -0.0569,      0.0349,      0.1060,      0.0188],
        [     0.0757,      0.1279,     -0.0437,     -0.0408,     -0.0500,     -0.0240,      0.0021,     -0.0615],
        [     0.0255,     -0.0107,      0.0422,      0.0095,      0.1025,     -0.0620,      0.0062,      0.0269],
        [     0.0825,      0.1924,     -0.0698,     -0.1934,      0.0962,      0.0347,     -0.0693,     -0.1157],
        [     0.0123,     -0.0035,     -0.1099,     -0.0288,      0.0530,      0.0469,     -0.0474,     -0.0452],
        [    -0.0187,      0.0040,      0.0092,     -0.0092,      0.1885,     -0.0869,     -0.0347,      0.0388],
        [    -0.0215,      0.0198,      0.0154,      0.0193,      0.0094,      0.0265,     -0.0908,      0.0255],
        [    -0.0283,      0.0669,     -0.1050,      0.0435,     -0.1504,     -0.1279,     -0.0471,     -0.0645],
        [     0.1235,     -0.0398,     -0.0250,      0.1396,     -0.0503,      0.0488,      0.0500,      0.1226],
        [     0.0923,      0.0815,     -0.0613,     -0.0250,     -0.0214,      0.0591,      0.0322,     -0.0869],
        [    -0.0688,     -0.0177,      0.0493,      0.1719,     -0.0159,     -0.0284,     -0.2080,     -0.0052],
        [     0.0825,      0.0515,     -0.0386,     -0.0796,      0.0913,     -0.0913,      0.0344,      0.0820],
        [    -0.0564,      0.0613,     -0.0850,      0.0183,      0.0383,      0.0464,      0.1348,      0.0299],
        [    -0.0075,      0.0457,     -0.0439,     -0.0219,      0.0081,     -0.0679,      0.0291,     -0.0298],
        [    -0.0001,     -0.1768,     -0.0160,      0.0364,     -0.1348,     -0.1118,     -0.0303,      0.0776],
        [    -0.0027,     -0.0898,      0.0256,     -0.0815,      0.1079,     -0.0752,     -0.1157,     -0.0203]],
       device='cuda:0', dtype=torch.bfloat16)
Ref
tensor([[    -0.0153,      0.1197,      0.0455,     -0.0655,     -0.0569,      0.0350,      0.1062,      0.0188],
        [     0.0761,      0.1282,     -0.0438,     -0.0409,     -0.0502,     -0.0241,      0.0021,     -0.0616],
        [     0.0255,     -0.0108,      0.0424,      0.0096,      0.1026,     -0.0620,      0.0063,      0.0269],
        [     0.0828,      0.1929,     -0.0702,     -0.1943,      0.0965,      0.0347,     -0.0697,     -0.1159],
        [     0.0123,     -0.0035,     -0.1101,     -0.0289,      0.0531,      0.0470,     -0.0475,     -0.0454],
        [    -0.0188,      0.0040,      0.0092,     -0.0092,      0.1893,     -0.0871,     -0.0348,      0.0390],
        [    -0.0216,      0.0199,      0.0155,      0.0193,      0.0094,      0.0266,     -0.0912,      0.0255],
        [    -0.0284,      0.0671,     -0.1050,      0.0435,     -0.1507,     -0.1287,     -0.0473,     -0.0649],
        [     0.1237,     -0.0399,     -0.0251,      0.1398,     -0.0503,      0.0489,      0.0502,      0.1226],
        [     0.0924,      0.0819,     -0.0613,     -0.0251,     -0.0215,      0.0593,      0.0323,     -0.0873],
        [    -0.0692,     -0.0177,      0.0493,      0.1720,     -0.0159,     -0.0285,     -0.2082,     -0.0052],
        [     0.0828,      0.0516,     -0.0386,     -0.0797,      0.0914,     -0.0914,      0.0345,      0.0823],
        [    -0.0565,      0.0613,     -0.0850,      0.0184,      0.0384,      0.0464,      0.1351,      0.0299],
        [    -0.0075,      0.0459,     -0.0441,     -0.0220,      0.0081,     -0.0682,      0.0291,     -0.0299],
        [    -0.0001,     -0.1768,     -0.0160,      0.0365,     -0.1353,     -0.1122,     -0.0304,      0.0780],
        [    -0.0027,     -0.0903,      0.0257,     -0.0818,      0.1082,     -0.0756,     -0.1159,     -0.0204]],
       device='cuda:0')
Max diff: 0.0019273757934570312
```

