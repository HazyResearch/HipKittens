#include <cstdint>
#include "gemm_base.cuh"
#include "rmsnorm_rope.cuh"
#include "pyutils/pyutils.cuh"
#include "checked_bind.cuh"

void dispatch(RmsnormRopeGlobals g, hipStream_t stream) {
    if (g.r.rows() != 1 || g.r.cols() != g.a.rows())
        throw std::runtime_error("rmsnorm_rope: r must be [M] (M = a.rows())");
    if (g.cos_sin.rows() != g.a.rows() || g.cos_sin.cols() != g.b.rows())
        throw std::runtime_error("rmsnorm_rope: cos_sin must be [M, N] (M = a.rows(), N = b.rows())");
    if (reinterpret_cast<uintptr_t>(g.c.raw_ptr) % alignof(bf16_2) != 0)
        throw std::runtime_error("rmsnorm_rope: output must have 4-byte alignment");
    launch<RmsnormRopeEpilogue, RmsnormRopeGlobals>(g, stream);
}

PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "GEMM + RMSNorm + RoPE: c = RoPE(r[:, None] * (a @ b.T)), natural column order; a = [M,K], b = [N,K] "
              "gamma-folded and rope_perm'd, cos_sin = [M,N] rope_perm'd, c = [M,N] 4-byte aligned (bf16), r = [M] (fp32)";
    bind_checked<dispatch>(m, "dispatch", {tensor_dtype::bf16, tensor_dtype::bf16, tensor_dtype::bf16, tensor_dtype::fp32, tensor_dtype::bf16},
        &RmsnormRopeGlobals::a, &RmsnormRopeGlobals::b, &RmsnormRopeGlobals::c,
        &RmsnormRopeGlobals::r, &RmsnormRopeGlobals::cos_sin);
}
