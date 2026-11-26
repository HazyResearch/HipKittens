#include "gemm_base.cuh"
#include "rmsnorm_swiglu.cuh"
#include "pyutils/pyutils.cuh"
#include "checked_bind.cuh"

void dispatch(RmsnormSwigluGlobals g, hipStream_t stream) {
    if (g.r.rows() != 1 || g.r.cols() != g.a.rows())
        throw std::runtime_error("rmsnorm_swiglu: r must be [M] (M = a.rows())");
    launch<RmsnormSwigluEpilogue, RmsnormSwigluGlobals>(g, stream);
}

PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "GEMM + RMSNorm + SwiGLU: c = silu(gate) * value with [gate | value] = r[:, None] * (a @ b.T); "
              "a = [M,K] the RMSNorm input times its gamma (hg from residual_rms_gamma), b = [N,K] plain weight, "
              "rows [0, N/2) gate and [N/2, N) value, "
              "c = [M,N/2] (bf16), r = [M] (fp32)";
    bind_checked<dispatch>(m, "dispatch", {tensor_dtype::bf16, tensor_dtype::bf16, tensor_dtype::bf16, tensor_dtype::fp32},
        &RmsnormSwigluGlobals::a, &RmsnormSwigluGlobals::b, &RmsnormSwigluGlobals::c, &RmsnormSwigluGlobals::r);
}
