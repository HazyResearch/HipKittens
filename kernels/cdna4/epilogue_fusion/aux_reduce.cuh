#pragma once
#include "epilogue_args.cuh"

// r = 1 / sqrt(mean(x^2) + RMS_EPS)
constexpr float RMS_EPS = 1e-5f;

struct RmsReduceGlobals {
    gl<float, -1, -1, -1, -1> partials;   // [N/64, M] fp32; as a 4-D gl: [1, 1, N/64, M]
    gl<float, -1, -1, -1, -1> r;          // [M] fp32;       as a 4-D gl: [1, 1, 1, M]
};

// r[row] = 1 / sqrt(sum(partials[:, row]) / N + RMS_EPS). One wavefront per row.
__global__ void rms_reduce(const gl<float, -1, -1, -1, -1> partials, gl<float, -1, -1, -1, -1> r) {
    constexpr int WF = kittens::WARP_THREADS;
    const int lane = threadIdx.x & (WF - 1);
    const int row  = (blockIdx.x * blockDim.x + threadIdx.x) / WF;
    const int M = r.cols();
    if (row >= M) return;                                // wavefront-uniform
    const int groups = partials.rows();
    const int N = groups * REG_BLOCK_N;
    float s = 0.f;
    for (int g = lane; g < groups; g += WF) s += partials[{0, 0, g, row}];
    for (int off = WF / 2; off > 0; off >>= 1) s += __shfl_down(s, off);
    if (lane == 0) r.raw_ptr[row] = rsqrtf(s / (float)N + RMS_EPS);
}
