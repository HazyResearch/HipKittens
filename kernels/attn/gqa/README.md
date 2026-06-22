# CDNA3 Flash Attention Forward (GQA, Non-Causal)

MFMA-based flash attention forward kernel for MI300X (gfx942) using HipKittens.

## Architecture

- **MFMA tile**: `mfma_f32_16x16x16` (bf16 inputs, f32 accumulator)
- **Q tile**: 16 rows per warp, 4 warps = 64 Q rows per block
- **KV block**: 64 rows per iteration
- **Head dimensions**: D=128 (`kernel.cpp`) and D=64 (`kernel_d64.cpp`)
- **Algorithm**: FlashAttention-2 online softmax with base-2 exp

## Tensor Layout

All tensors are `[B, H, N, D]` in bf16:
- **Q**: `[B, H_Q, N, D]` — query heads
- **K**: `[B, H_KV, N, D]` — key heads (GQA: H_KV <= H_Q)
- **V**: `[B, H_KV, N, D]` — value heads
- **O**: `[B, H_Q, N, D]` — output (bf16)
- **L**: `[B, H_Q, N, 1]` — log-sum-exp (float32)

## Build

```bash
source ../../env.src  # Set ROCM_PATH
make ATTN_B=4 ATTN_H=32 ATTN_H_KV=8 ATTN_N=1024
```

Build flags:
- `ATTN_B`: Batch size
- `ATTN_H`: Number of Q heads
- `ATTN_H_KV`: Number of KV heads (GQA ratio = H/H_KV)
- `ATTN_N`: Sequence length

## Test

```bash
python3 test_python.py --B 4 --H 32 --H_KV 8 --N 1024 --D 128
python3 test_python.py --B 4 --H 32 --H_KV 8 --N 1024 --D 64
python3 test_python.py --B 4 --H 32 --H_KV 8 --N 1024 --D 128 --profile
```

## MMA Type Signatures

- **QK**: `mma_ABt(S, Q, K, S)` — `A[16,D,row] @ B[64,D,row]^T → D[16,64,col]`
- **PV**: `mma_AB(O, P, V, O)` — `A[16,64,row] @ B[64,D,col] → D[16,D,col]`

## Optimization: Register-Buffer Pipeline

Uses `load_global_to_register_buffer` / `store_register_buffer_to_shared` to overlap global memory loads with MFMA compute:
- V load overlaps with QK compute + softmax
- K[kv+1] prefetch overlaps with PV compute
- AMD scheduling hints (`__builtin_amdgcn_s_setprio`, `__builtin_amdgcn_sched_barrier`) around MMA clusters

## Shared Memory

- D=128: 16KB (single KV tile: 64 x 128 x 2B, K and V alternate via register buffer)
- D=64: 8KB (single KV tile: 64 x 64 x 2B)

## Register Pressure

**D=128** (`__launch_bounds__(256, 1)` — 1 wave/SIMD):
- 256 VGPRs + 236 AGPRs, 0 spills, 20 bytes/lane scratch
- Register buffer: `float4 kv_buf[4]` = 16 VGPRs

**D=64** (`__launch_bounds__(256, 2)` — 2 waves/SIMD):
- 254 VGPRs, 0 spills, 20 bytes/lane scratch
- Register buffer: `float4 kv_buf[2]` = 8 VGPRs

## Performance (MI300X, B=4, H=32, H_KV=8, N=1024)

| Variant | Kernel TFLOPS | PyTorch SDPA | Speedup |
|---------|--------------|--------------|---------|
| D=128 non-causal | 69.0 | 55.4 | 1.25x |
| D=64 non-causal | 72.7 | 53.0 | 1.37x |
