#pragma once
#include "base.cuh"
#include "reductions.cuh"

// Writes only per-row partial sums of squares of A @ B^T (no c): partials[grp, m] for N/64 groups.
// Summing a row's N/64 partials gives sum(x^2) for RMSNorm.
struct PartialRMSGlobals {
    _gl_A a; _gl_B b;
    gl<float, -1, -1, -1, -1> partials;   // [N/64, M] fp32; as a 4-D gl: [1, 1, N/64, M] (row on the last axis)
};
struct PartialRMSEpilogue {
    template<typename Globals, typename Accum>
    static __device__ inline void apply(const Globals& g, Accum& C, int row, int col, int wr, int wc) {
        partial_row_sum_sq(g, C, row, col, wr, wc);
    }
};
