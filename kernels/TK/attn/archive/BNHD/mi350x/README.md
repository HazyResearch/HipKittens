

## Kernel Benchmarks

| Kernel file        | Description                                     | Perf. (TFLOPs) |
|--------------------|-------------------------------------------------|----------------|
| **reg-kernel.cpp** | 8 warps – everything stays **in-register**      | **82**         |
| **shared-kernel.cpp** | 8 warps – loads through **shared (LDS)**<br/>`N_STEP = 128`, `N_SUB_STEP = 32` | **59**         |

<sub>*Numbers measured with hipEvent timing; batch=16, heads=16, N=4096, D=64.*</sub>

