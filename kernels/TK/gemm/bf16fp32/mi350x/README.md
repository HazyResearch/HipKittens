### AMD MI350 (gfx950) 


Hardware Overview:

| Component                    | Value                     | Notes                                           |
|-----------------------------|---------------------------|-------------------------------------------------|
| Compute Units (CUs)         | 256                       | Total across GPU                                |
| SIMDs per CU                | 4                         | Each SIMD runs 1 wavefront (64 threads)         |
| Wavefront Size              | 64 threads                | Fixed on AMD GPUs                               |
| Max Waves per CU            | 32                        | Limits active wavefronts per CU                 |
| Max Threads per CU          | 2048                      | = 32 waves × 64 threads                         |
| Max Workgroup Size          | 1024 threads              | Per dispatch                                    |
| VGPRs per SIMD              | 65,536 (32-bit)           | 256 KB per SIMD                                 |
| Register File per CU        | 4 × 256 KB = 1 MB         | 4 SIMDs per CU                                  |
| Register File per GPU       | 256 × 1 MB = 256 MB       | Total GPU register file                         |
| Shared Memory (LDS) per CU  | 160 KB                    | For workgroup storage                           |
| L1 Cache                    | 32 KB                     | Per CU                                          |
| L2 Cache                    | 4 MB                      | Shared                                          |
| L3 Cache                    | 256 MB                    | Shared HBM-attached                             |
| Max VGPRs per Thread        | 256 (typical)             | Controlled by compiler                          |
| SGPRs per SIMD              | ~800–1024                 | Scalar register file                            |
| Max Clock Speed             | 2200 MHz                  | From `rocminfo`                                 |
| Memory Bandwidth            | ~5.2 TB/s                 | With 12


Notes:
- 32 XCDs, 8 CUs per XCD [CDNA4 Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf?utm_source=chatgpt.com)

