```
256_128_64_16_rolled.cpp:41:1: remark:     SGPRs: 36 [-Rpass-analysis=kernel-resource-usage]                                                                                                               
256_128_64_16_rolled.cpp:41:1: remark:     VGPRs: 250 [-Rpass-analysis=kernel-resource-usage]                                                                                                              
256_128_64_16_rolled.cpp:41:1: remark:     AGPRs: 0 [-Rpass-analysis=kernel-resource-usage]                                                                                                                
256_128_64_16_rolled.cpp:41:1: remark:     ScratchSize [bytes/lane]: 0 [-Rpass-analysis=kernel-resource-usage]                                                                                             
256_128_64_16_rolled.cpp:41:1: remark:     Dynamic Stack: False [-Rpass-analysis=kernel-resource-usage]                                                                                                    
256_128_64_16_rolled.cpp:41:1: remark:     Occupancy [waves/SIMD]: 2 [-Rpass-analysis=kernel-resource-usage]                                                                                               
256_128_64_16_rolled.cpp:41:1: remark:     SGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]                                                                                                          
256_128_64_16_rolled.cpp:41:1: remark:     VGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]                                                                                                          
256_128_64_16_rolled.cpp:41:1: remark:     LDS Size [bytes/block]: 0 [-Rpass-analysis=kernel-resource-usage]    



Performance: 471.15 TFLOPS for 8192x8192 matrix multiplication.

Max error between kernel and reference: 0.015625
Max error: 0.015625
Mean error: 2.178580871259328e-07
Number of large errors (>0.1): 0

diff[:32, :32].max() tensor(0., device='cuda:0')
diff[:32, 32:64].max() tensor(0., device='cuda:0')
diff[32:64, :32].max() tensor(0., device='cuda:0')
diff[32:64, 32:64].max() tensor(0., device='cuda:0')

diff[:32, 64:96].max() tensor(0., device='cuda:0')
diff[:32, 96:128].mean() tensor(0., device='cuda:0')
diff[32:64, 64:96].max() tensor(0., device='cuda:0')
diff[32:64, 96:128].max() tensor(0., device='cuda:0')

diff[7168:7232, 7168:7232].max() tensor(0., device='cuda:0')
diff[7232:7296, 7232:7296].max() tensor(0., device='cuda:0')
diff[7296:7360, 7296:7360].max() tensor(0., device='cuda:0')
diff[7360:7424, 7360:7424].max() tensor(0., device='cuda:0')
```