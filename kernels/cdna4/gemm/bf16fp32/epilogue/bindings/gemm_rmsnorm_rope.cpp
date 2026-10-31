#include <cstdint>
#include "gemm_base.cuh"
#include "rmsnorm_rope.cuh"
#include "rope_args.cuh"
#include "pyutils/pyutils.cuh"
#include "checked_bind.cuh"

void dispatch(RmsnormRopeGlobals g, hipStream_t stream) {
    if (g.r.rows() != 1 || g.r.cols() != g.a.rows())
        throw std::runtime_error("rmsnorm_rope: r must be [M] (M = a.rows())");
    check_rope_args("rmsnorm_rope", g, g.b.rows(), {g.b.rows()});
    launch<RmsnormRopeEpilogue, RmsnormRopeGlobals>(g, stream);
}

PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "GEMM + RMSNorm + RoPE: c = RoPE(r[:, None] * (a @ b.T)) on columns [0, rope_cols), natural column order; "
              "a = [M,K] the RMSNorm input times its gamma (hg from residual_rms_gamma), b = [N,K] plain weight, "
              "cos_sin = [M,Dh] one head's natural table, c = [M,N] 4-byte "
              "aligned (bf16), r = [M] (fp32)";
    bind_checked<dispatch>(m, "dispatch", {tensor_dtype::bf16, tensor_dtype::bf16, tensor_dtype::bf16, tensor_dtype::fp32,
                                           tensor_dtype::bf16, tensor_dtype::int_scalar},
        &RmsnormRopeGlobals::a, &RmsnormRopeGlobals::b, &RmsnormRopeGlobals::c,
        &RmsnormRopeGlobals::r, &RmsnormRopeGlobals::cos_sin, &RmsnormRopeGlobals::rope_cols);
}
