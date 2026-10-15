#pragma once
#include "base.cuh"
#include "vec.cuh"

// out = alpha * (A @ B^T), with alpha a 1-element fp32 tensor.
struct ScaleGlobals {
    _gl_A a; _gl_B b; _gl_C c;
    gl<float, 1, 1, 1, 1> alpha;
};
struct ScaleEpilogue {
    template<typename Globals, typename Accum>
    static __device__ inline void apply(const Globals& g, Accum& C, int row, int col, int wr, int wc) {
        apply_scale(g, C);
        store_C(g, C, row, col, wr, wc);
    }
};
