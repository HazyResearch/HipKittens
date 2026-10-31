#include "gemm_base.cuh"
#include "rmsnorm_scale.cuh"
#include "pyutils/pyutils.cuh"
#include "checked_bind.cuh"

void dispatch(RMSNormScaleGlobals g, hipStream_t stream) {
    if (g.r.rows() != 1 || g.r.cols() != g.a.rows())
        throw std::runtime_error("rmsnorm_scale: r must be [M] (M = a.rows())");
    if (g.gamma.rows() != 1 || g.gamma.cols() != g.b.rows())
        throw std::runtime_error("rmsnorm_scale: gamma must be [N] (N = b.rows())");
    launch<RMSNormScaleEpilogue, RMSNormScaleGlobals>(g, stream);
}

PYBIND11_MODULE(TK_MODULE_NAME, m) {
    m.doc() = "GEMM + output RMSNorm: c = (a @ b.T) * r[:, None] * gamma[None, :], normalizing the GEMM's output: "
              "r = 1/rms of a @ b.T (hk.inv_rms(a, b)), gamma = the norm's weight over the N output columns. "
              "a = [M,K] raw input, b = [N,K], c = [M,N] (bf16), r = [M] (fp32), gamma = [N] (bf16)";
    bind_checked<dispatch>(m, "dispatch", {tensor_dtype::bf16, tensor_dtype::bf16, tensor_dtype::bf16, tensor_dtype::fp32, tensor_dtype::bf16},
        &RMSNormScaleGlobals::a, &RMSNormScaleGlobals::b, &RMSNormScaleGlobals::c,
        &RMSNormScaleGlobals::r, &RMSNormScaleGlobals::gamma);
}
