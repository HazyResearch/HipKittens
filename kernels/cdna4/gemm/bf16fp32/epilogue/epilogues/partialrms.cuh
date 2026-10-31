#pragma once
#include "base.cuh"
#include "reductions.cuh"

// Writes only partials[grp, m] = per-row sum of squares of A @ B^T over 64-column group grp (no c).
struct PartialRMSGlobals {
    _gl_A a; _gl_B b;
    gl<float, -1, -1, -1, -1> partials;   // [N/64, M] fp32
};
struct PartialRMSEpilogue {
    template<typename Globals, typename Accum>
    static __device__ inline void apply(const Globals& g, Accum& C, int row, int col, int wr, int wc) {
        partial_row_sum_sq(g, C, row, col, wr, wc);
    }
};
