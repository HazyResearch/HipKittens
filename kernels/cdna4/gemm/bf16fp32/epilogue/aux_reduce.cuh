#pragma once
#include "epilogue_args.cuh"

// RMSNorm epsilon: r = 1 / sqrt(mean(x^2) + RMS_EPS). Fixed at 1e-5 (Llama-2/3).
constexpr float RMS_EPS = 1e-5f;

// Arguments of rms_reduce: partialrms's output and the per-row inverse RMS it produces.
struct RmsReduceGlobals {
    gl<float, -1, -1, -1, -1> partials;   // [N/64, M] fp32; as a 4-D gl: [1, 1, N/64, M]
    gl<float, -1, -1, -1, -1> r;          // [M] fp32;       as a 4-D gl: [1, 1, 1, M]
};

// r[row] = 1 / sqrt(sum of the row's N/64 partial sums of squares / N + RMS_EPS).
// One wavefront per row: each of its 64 lanes sums a strided share of the row's groups, a shuffle tree
// adds the 64 lane sums, and lane 0 writes r[row].
__global__ void rms_reduce(const gl<float, -1, -1, -1, -1> partials, gl<float, -1, -1, -1, -1> r) {
    constexpr int WF = kittens::WARP_THREADS;
    const int lane = threadIdx.x & (WF - 1);
    const int row  = (blockIdx.x * blockDim.x + threadIdx.x) / WF;
    const int M = r.cols();
    if (row >= M) return;                                // the whole wavefront shares one row
    const int groups = partials.rows();
    const int N = groups * REG_BLOCK_N;
    float s = 0.f;
    for (int g = lane; g < groups; g += WF) s += partials[{0, 0, g, row}];
    for (int off = WF / 2; off > 0; off >>= 1) s += __shfl_down(s, off);
    if (lane == 0) r.raw_ptr[row] = rsqrtf(s / (float)N + RMS_EPS);
}
