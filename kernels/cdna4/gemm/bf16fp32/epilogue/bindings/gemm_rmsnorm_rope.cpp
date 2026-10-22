#include <cstdint>
#include "gemm_base.cuh"
#include "rmsnorm_rope.cuh"
#include "pyutils/pyutils.cuh"
#include "checked_bind.cuh"

void dispatch(RmsnormRopeGlobals g, hipStream_t stream) {
    if (g.r.rows() != 1 || g.r.cols() != g.a.rows())
        throw std::runtime_error("rmsnorm_rope: r must be [M] (M = a.rows())");
    const int N = g.b.rows(), dh = g.cos_sin.cols();
    if (g.cos_sin.rows() != g.a.rows() || dh <= 0 || dh % 2 || N % dh)
        throw std::runtime_error("rmsnorm_rope: cos_sin must be [M, Dh] with Dh even and N % Dh == 0 (M = a.rows(), N = b.rows())");
    if (g.rope_cols < 0 || g.rope_cols > N || g.rope_cols % BLOCK_SIZE)
        throw std::runtime_error("rmsnorm_rope: rope_cols must be a multiple of 256 in [0, N]");
    if (reinterpret_cast<uintptr_t>(g.c.raw_ptr) % alignof(bf16_2) != 0)
        throw std::runtime_error("rmsnorm_rope: output must have 4-byte alignment");
    launch<RmsnormRopeEpilogue, RmsnormRopeGlobals>(g, stream);
}

PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "GEMM + RMSNorm + RoPE: c = RoPE(r[:, None] * (a @ b.T)) on columns [0, rope_cols), natural column order; "
              "a = [M,K], b = [N,K] gamma-folded, cos_sin = [M,Dh] one head's natural table, c = [M,N] 4-byte "
              "aligned (bf16), r = [M] (fp32)";
    bind_checked<dispatch>(m, "dispatch", {tensor_dtype::bf16, tensor_dtype::bf16, tensor_dtype::bf16, tensor_dtype::fp32,
                                           tensor_dtype::bf16, tensor_dtype::int_scalar},
        &RmsnormRopeGlobals::a, &RmsnormRopeGlobals::b, &RmsnormRopeGlobals::c,
        &RmsnormRopeGlobals::r, &RmsnormRopeGlobals::cos_sin, &RmsnormRopeGlobals::rope_cols);
}
