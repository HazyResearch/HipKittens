#include <cstdint>
#include "gemm_base.cuh"
#include "rope.cuh"
#include "pyutils/pyutils.cuh"
#include "checked_bind.cuh"

void dispatch(RopeGlobals g, hipStream_t stream) {
    if (g.cos_sin.rows() != g.a.rows() || g.cos_sin.cols() != g.b.rows())
        throw std::runtime_error("rope: cos_sin must be [M, N] (M = a.rows(), N = b.rows())");
    if (reinterpret_cast<uintptr_t>(g.c.raw_ptr) % alignof(bf16_2) != 0)
        throw std::runtime_error("rope: output must have 4-byte alignment");
    launch<RopeEpilogue, RopeGlobals>(g, stream);
}

PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "GEMM + RoPE: c = RoPE(a @ b.T), interleaved pairs, natural column order; a = [M,K], b = [N,K] and cos_sin = [M,N] "
              "rope_perm'd (see ops/rotary.cuh), c = [M,N] 4-byte aligned (all bf16)";
    bind_checked<dispatch>(m, "dispatch", {tensor_dtype::bf16, tensor_dtype::bf16, tensor_dtype::bf16, tensor_dtype::bf16},
        &RopeGlobals::a, &RopeGlobals::b, &RopeGlobals::c, &RopeGlobals::cos_sin);
}
