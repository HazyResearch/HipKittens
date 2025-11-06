### AMD Instinct MI325X (gfx942) 

Run command:
```bash
make clean && make && python test_python.py
```

Hardware Overview:

| Component                    | Value                          | Notes                                              |
|-----------------------------|--------------------------------|----------------------------------------------------|
| Compute Units (CUs)         | 304                            | More than MI250/MI300 — high parallelism           |
| SIMDs per CU                | 4                              | Each SIMD handles 1 wavefront                      |
| Wavefront Size              | 64 threads                     | Fixed per SIMD                                     |
| Max Waves per CU            | 32                             | Controls concurrency                               |
| Max Threads per CU          | 2048                           | 32 waves × 64 threads                              |
| Max Workgroup Size          | 1024 threads                   | Max threads per kernel dispatch                    |
| VGPRs per SIMD              | 65,536 (32-bit)                | 256 KB per SIMD                                    |
| Register File per CU        | 4 × 256 KB = 1 MB              | 4 SIMDs                                            |
| Register File per GPU       | 304 × 1 MB = 304 MB            | Massive register footprint                         |
| Shared Memory (LDS) per CU  | 64 KB                          | Per workgroup                                      |
| L1 Cache                    | 32 KB                          | Per CU                                             |
| L2 Cache                    | 4 MB                           | Shared                                             |
| L3 Cache                    | 256 MB                         | HBM-attached                                       |
| Max VGPRs per Thread        | 256 (typical)                  | Set by compiler                                    |
| SGPRs per SIMD              | ~800–1024                      | Scalar register file                               |
| Max Clock Speed             | 2100 MHz                       | From `rocminfo`                                    |
| Memory Bandwidth            | ~5.2 TB/s                      | With HBM3 (if installed with 128 GB)               |
| ISA Name                    | amdgcn-amd-amdhsa--gfx942      | Use `--offload-arch=gfx942`                        |



