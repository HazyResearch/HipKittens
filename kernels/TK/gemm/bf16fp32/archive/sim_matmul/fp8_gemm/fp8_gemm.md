# FP8 GEMM Problem Solution 

## Main idea
I used a classical 2-level tiling approach. Check my [blog post](https://seb-v.github.io/optimization/update/2025/01/20/Fast-GPU-Matrix-multiplication.html) for more details, but here are the main steps:

1. Loop over the K dimension:

    - Load a tile of matrix A and matrix B into LDS, using all threads in the workgroup.
    - Load a sub-tile into VGPRs and perform smaller matrix multiplications at the wave level.

2. Accumulate results into local VGPRs.


<p align="center">
<img src="picture0.jpg" width="700" >
</p>

## LDS tiling
My solution uses a single tiling configuration across all test cases: 128×128 for each matrix. This choice was driven by:

- 128 stride matches the scaling tile size of the problem (As & Bs).
- Two FP8 tiles of 128×128 occupy 32KB of LDS, which is exactly half of the 64KB available per CU. I reserved the remaining 32KB either for double buffering or to maintain reasonable occupancy.
 
## MFMA instruction
To fully leverage the MI300X architecture, using Matrix Core (MFMA) instructions is essential. For FP8, the options are:
 - `V_MFMA_F32_16x16x32_FP8_FP8` (16 cycles) 
 - `V_MFMA_F32_32x32x16_FP8_FP8` (32 cycles).
 
I chose the first one (`16x16x32`) to start with. I intended to test the second, but got sidetracked with the other two problems 😅

My solution uses 8 waves per workgroup arranged in a 2×4 grid. Each wave performs eight 16×16 matmuls as shown below:

<p align="center">
<img src="picture3.jpg" width="450" >
</p>


Given that our LDS tile’s K dimension is 128, and each V_MFMA_F32_16x16x32_FP8_FP8 instruction consumes 32 elements along K, each matmul requires 4 iterations. Since each wave performs 8 matmuls, this results in 32 MFMA instructions per wave.

MFMA instructions are wave level intructions and require a special layout. If you look at the CDNA3 guide (or if you use the awesome [amd matrix calculator](https://github.com/ROCm/amd_matrix_instruction_calculator)), we get this :

```A i: (lane % 16)
A k: 8 * floor(lane / 16) + 4 * GPR_num + floor(GPR_bits / 8)

B j: (lane % 16)
B k: 8 * floor(lane / 16) + 4 * GPR_num + floor(GPR_bits / 8)
```
Or visually this :
<p align="center">
<img src="picture1.jpg" width="550" >
</p>

Each thread in a wave loads 8 values along the K axis (via two 32-bit VGPRs).
- For A: 8 consecutive values in a row

- For B: 8 consecutive values in a column

So, if we want to directly load these values from the LDS, we need A to be row-major and B to be column-major. Unfortunately, that’s exactly the opposite of what the problem provides. Both A and B are column-major, and since we perform A × Bᵗ, it means we are doing a matmul between a column-major and a row-major matrix. Therefore, both A and B need to be transposed.

## LDS load/write
Initially, I loaded A and B into LDS contiguously and transposed them in LDS (using padding to avoid bank conflicts) but this wasn't very efficient.

I wanted to keep the inner loop as fast as possible. Basically I wanted to just do a single `ds_read2st64_b64` instruction to load both A and B and then unroll all the MFMA instructions. 
With 128×128 tiles, that's 12 LDS loads and 32 MFMAs.

To enable this, I implemented the transpose in 8×8 tiles. I tried a couple of options there (using ds_permute_b32, DPP...) but it turns out the best performance I could get was by doing the transpose within a single thread. The HIP compiler would generate Sub-Dword Addressing instructions (SDWA) which turned out to be faster than cross-wave alternatives.

Also we don't need to transpose the full 128×128 tile, Just 8×8 blocks are sufficient if we update the LDS load addressing before the MFMA.

To further reduce register pressure, I split the A & B loading between the 8 available waves:

- 4 waves load A

- 4 waves load B

This helps stay below 128 VGPRs, maintaining theoretical occupancy of 4 waves per SIMD. In practice, this enables fully unrolled global_load_dwordx2 operations, with no s_waitcnt stalls inbetween.

<p align="center">
<img src="picture5.jpg" width="800" >
</p>


After the 8×8 tile transpose, the 16×8 tile for the first 16 lanes looks like this:

<p align="center">
<img src="picture2.jpg" width="500" >
</p>


We just need to update the col and row calculation like this :
```
row = k + laneId % 8 + (laneId / 16) * 8
col = ((laneId % 16) / 8)
```

To avoid bank conflicts, I used a swizzling trick rather than naive padding. Here's the swizzle formula, which ensures conflict-free reads/writes:

```
col_swizzled = ((col / 2) ^ (row % 8)) * 2 + col % 2;
```


# SPLIT-K

Some test cases like (1024, 512, 7168) produce only 32 blocks but have large K dimensions — underutilizing the 304 available CUs.

To address this, I simply split the computation of the matmul across the K axis and used the atomicAdd function to accumulate the results. This worked well.

I initially implemented a version using the packed instruction `GLOBAL_ATOMIC_PK_ADD_BF16` (with a tiny DPP move to permute two output values together). However, accumulating in BF16 caused precision test failures.

So, I had to fall back to doing the accumulation on a temporary float buffer and then cast to BF16 once the computation was done.

I only applied these optimizations to 3 test cases.


# Things that I tried that didn't pan out
- Double buffering increased LDS usage and register pressure, leading to lower occupancy, with no noticeable performance gain. This is likely because the relatively high occupancy of the original kernel was already doing a decent job at hiding GMEM load latency. This could make sense with a lower occupancy kernel I suppose.
- Cross lane transpose using ds_permute (huge latency on ds_permute)
- Cross lane transpose using a combinaison of DPP quad_perm & row_shl on 16 lanes (also rather high latency on these instructions)


# Room for improvement
- Increase arithmetic complexity. Right now, the ratio between MFMA latency and LDS read latency isn't ideal. Increasing this could help further hide GMEM latency.
- Better CU utilization for small test cases: I use a single tiling configuration, which limits block count. For example, (1024, 3072, 1536) only yields 192 blocks, which doesn't fully engage all 304 CUs — and doesn't benefit from Split-K.
